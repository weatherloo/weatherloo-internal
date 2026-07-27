#!/usr/bin/env python3
"""Compute HRDPS analysis benchmark JSON for issue #20.

Fetches ECCC HRDPS 00/06/12/18Z continental 2.5 km analysis (PT000H / f000)
from MSC Datamart, interpolates 2 m temperature and 10 m wind (from u/v
components) to each station, and uses that single analysis value as the
"forecast" for every lead time.  Ground truth is the observation at each
lead time's valid time (initialization + lead_hours).

Because the analysis only represents the state at init, error vs observations
grows with lead time (RMSE/MAE increase, ACC decrease).

Also writes a consolidated ``hrdps_analysis_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).

Wind speed: sqrt(u10^2 + v10^2), converted m/s -> km/h (* 3.6).
ACC anomaly baseline: DOY + UTC-hour climatology from station observations
with a ±15-day calendar window (single-year proxy; document in metadata.json).

HRDPS uses a rotated lat-lon grid with 2D geographic coordinates; station
values use bilinear-style interpolation via ``scipy.interpolate.LinearNDInterpolator``
on geographic latitude/longitude (see metadata.json).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
from scipy.interpolate import LinearNDInterpolator

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
CACHE_DIR = REPO_ROOT / ".cache" / "hrdps_grib"

DATAMART_HOSTS = (
    "https://dd.weather.gc.ca",
    "https://dd.meteo.gc.ca",
    "http://hpfx.collab.science.gc.ca",
)
INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "hrdps_analysis"
ANALYSIS_FXX = 0

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

HRDPS_FIELDS = {
    "t2m": ("TMP", "AGL-2m"),
    "u10": ("UGRD", "AGL-10m"),
    "v10": ("VGRD", "AGL-10m"),
}

DOWNLOAD_RETRIES = 6
INTERP_WINDOW_DEG = 1.0


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_observations(station_id: str) -> dict[str, dict[str, float | None]]:
    path = OBS_ROOT / station_id / "observations_6h_2025.json"
    data = json.loads(path.read_text())
    return {
        row["valid_time"]: {
            "t2m": row.get("t2m"),
            "wind_speed": row.get("wind_speed"),
        }
        for row in data["observations"]
    }


def build_climatology(
    obs_by_time: dict[str, dict[str, float | None]], window_days: int = 15
) -> dict[str, dict[str, float]]:
    """DOY + UTC-hour climatology using a calendar-day window within 2025 obs."""
    slots: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for valid_time, vals in obs_by_time.items():
        dt = parse_utc(valid_time)
        doy = dt.timetuple().tm_yday
        hour = dt.hour
        for var in ("t2m", "wind_speed"):
            v = vals.get(var)
            if v is None:
                continue
            slots.setdefault((var, hour), []).append((doy, float(v)))

    clim: dict[str, dict[str, float]] = {"t2m": {}, "wind_speed": {}}
    for var in ("t2m", "wind_speed"):
        for hour in (0, 6, 12, 18):
            entries = slots.get((var, hour), [])
            if not entries:
                continue
            for target_doy in range(1, 367):
                vals_in_window = [
                    v
                    for doy, v in entries
                    if abs(doy - target_doy) <= window_days
                    or abs(doy - target_doy + 365) <= window_days
                    or abs(doy - target_doy - 365) <= window_days
                ]
                if vals_in_window:
                    clim[var][f"{target_doy:03d}-{hour:02d}"] = float(
                        np.mean(vals_in_window)
                    )
    return clim


def climatology_lookup(
    clim: dict[str, dict[str, float]], var: str, valid_time: datetime
) -> float | None:
    key = f"{valid_time.timetuple().tm_yday:03d}-{valid_time.hour:02d}"
    return clim.get(var, {}).get(key)


def download_bytes(url: str) -> bytes:
    headers = {"User-Agent": "weatherloo-internal/1.0"}

    last_err: BaseException | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 404:
                raise
            if exc.code not in (429, 500, 502, 503, 504) or attempt == DOWNLOAD_RETRIES - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
            if attempt == DOWNLOAD_RETRIES - 1:
                raise

        delay = min(60.0, (2**attempt) + random.uniform(0, 1))
        print(f"  download retry {attempt + 1}/{DOWNLOAD_RETRIES - 1} in {delay:.1f}s: {url[:80]}…")
        time.sleep(delay)

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"download failed: {url}")


def hrdps_grib_filename(init_dt: datetime, fxx: int, var: str, level: str) -> str:
    date = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")
    return (
        f"{date}T{hh}Z_MSC_HRDPS_{var}_{level}_RLatLon0.0225_PT{fxx:03d}H.grib2"
    )


def hrdps_grib_urls(init_dt: datetime, fxx: int, var: str, level: str) -> list[str]:
    date = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")
    fname = hrdps_grib_filename(init_dt, fxx, var, level)
    rel = f"model_hrdps/continental/2.5km/{hh}/{fxx:03d}/{fname}"
    urls: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    if date == today:
        for base in DATAMART_HOSTS:
            urls.append(f"{base}/today/{rel}")
    for base in DATAMART_HOSTS:
        urls.append(f"{base}/{date}/WXO-DD/{rel}")
    return urls


def cache_path_for_field(init_dt: datetime, fxx: int, var: str, level: str) -> Path:
    key = (
        f"{init_dt.strftime('%Y%m%d')}_{init_dt.strftime('%H')}z"
        f"_pt{fxx:03d}_{var}_{level.replace('-', '_')}"
    )
    return CACHE_DIR / f"{key}.grib2"


def download_hrdps_field(
    init_dt: datetime, fxx: int, var: str, level: str
) -> Path:
    path = cache_path_for_field(init_dt, fxx, var, level)
    if path.exists():
        return path

    last_err: BaseException | None = None
    for url in hrdps_grib_urls(init_dt, fxx, var, level):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(download_bytes(url))
            return path
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code != 404:
                raise

    if last_err is not None:
        raise last_err
    raise FileNotFoundError(
        f"HRDPS GRIB not found for {init_dt.isoformat()} {var} {level} f{fxx:03d}"
    )


def try_cache_hrdps_field(
    init_dt: datetime, fxx: int, var: str, level: str
) -> bool:
    """Download one analysis field; return False if not yet on Datamart (404)."""
    path = cache_path_for_field(init_dt, fxx, var, level)
    if path.exists():
        return True
    try:
        download_hrdps_field(init_dt, fxx, var, level)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except FileNotFoundError:
        return False


def cache_analysis_grib(init_dt: datetime) -> bool:
    """Cache PT000H t2m + 10 m u/v for one init. True if all three fields present."""
    ok = True
    for var, level in HRDPS_FIELDS.values():
        if not try_cache_hrdps_field(init_dt, ANALYSIS_FXX, var, level):
            ok = False
    return ok


def recent_init_datetimes(days: int) -> list[datetime]:
    """Inits from the last ``days`` calendar days through now (00/06/12/18 UTC)."""
    now = datetime.now(timezone.utc)
    start_day = (now - timedelta(days=days - 1)).date()
    inits: list[datetime] = []
    cur = start_day
    while cur <= now.date():
        for hour in INIT_HOURS_UTC:
            init_dt = datetime(cur.year, cur.month, cur.day, hour, tzinfo=timezone.utc)
            if init_dt <= now:
                inits.append(init_dt)
        cur += timedelta(days=1)
    return inits


def analysis_grib_cached(init_dt: datetime) -> bool:
    return all(
        cache_path_for_field(init_dt, ANALYSIS_FXX, var, level).exists()
        for var, level in HRDPS_FIELDS.values()
    )


def run_cache_only(
    days: int,
    also_compute: bool,
    resume: bool,
    year: int,
    out_dir: Path,
) -> None:
    """Download recent PT000H GRIB into ``.cache/hrdps_grib/`` (live archive)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    inits = recent_init_datetimes(days)
    cached = 0
    pending = 0
    for init_dt in inits:
        if cache_analysis_grib(init_dt):
            cached += 1
            print(f"  cached {init_dt.strftime('%Y-%m-%dT%H')}Z")
        else:
            pending += 1
            print(f"  pending {init_dt.strftime('%Y-%m-%dT%H')}Z (not on Datamart yet)")

    print(f"Cache sweep: {cached} complete, {pending} pending, dir={CACHE_DIR}")

    if not also_compute:
        return

    obs = {sid: load_observations(sid) for sid in STATIONS}
    clim = {sid: build_climatology(obs[sid]) for sid in STATIONS}
    computed = 0
    for init_dt in inits:
        if init_dt.year != year:
            continue
        if not analysis_grib_cached(init_dt):
            continue
        fname = process_init(init_dt, obs, clim, out_dir, resume)
        computed += 1
        print(f"  benchmark {fname}")

    if computed:
        write_metadata(out_dir, year)
        existing = sorted(p.name for p in all_init_json_paths(out_dir))
        write_index(out_dir, existing)
        export_npz(out_dir, year)
    print(f"Computed {computed} init JSON file(s) for {year}.")


def interp_grib_field(path: Path, lat: float, lon: float) -> float:
    """Bilinear-style interp on geographic lat/lon for HRDPS curvilinear grid."""
    ds = cfgrib.open_dataset(path)
    var_name = list(ds.data_vars)[0]
    lats = np.asarray(ds.latitude.values, dtype=float).ravel()
    lons = np.asarray(ds.longitude.values, dtype=float).ravel()
    vals = np.asarray(ds[var_name].values, dtype=float).ravel()

    window = INTERP_WINDOW_DEG
    mask = (
        (lats >= lat - window)
        & (lats <= lat + window)
        & (lons >= lon - window)
        & (lons <= lon + window)
    )
    if mask.sum() < 3:
        mask = np.isfinite(vals)

    points = np.column_stack([lats[mask], lons[mask]])
    values = vals[mask]
    interp = LinearNDInterpolator(points, values)
    result = float(interp(lat, lon))
    if not np.isfinite(result):
        dist = (lats - lat) ** 2 + (lons - lon) ** 2
        result = float(vals[dist.argmin()])
    return result


def fetch_hrdps_point(
    init_dt: datetime,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Return (t2m degC, wind_speed km/h) from PT000H analysis at a point."""
    t2m_var, t2m_level = HRDPS_FIELDS["t2m"]
    u_var, u_level = HRDPS_FIELDS["u10"]
    v_var, v_level = HRDPS_FIELDS["v10"]

    t2m_path = download_hrdps_field(init_dt, ANALYSIS_FXX, t2m_var, t2m_level)
    u_path = download_hrdps_field(init_dt, ANALYSIS_FXX, u_var, u_level)
    v_path = download_hrdps_field(init_dt, ANALYSIS_FXX, v_var, v_level)

    t2m_c = interp_grib_field(t2m_path, lat, lon) - 273.15
    u10 = interp_grib_field(u_path, lat, lon)
    v10 = interp_grib_field(v_path, lat, lon)
    wind_kmh = float(np.hypot(u10, v10) * 3.6)
    return t2m_c, wind_kmh


def point_metrics(
    forecast: float,
    obs: float,
    clim: float | None,
) -> dict[str, float | None]:
    err = forecast - obs
    rmse = abs(err)
    mae = abs(err)
    bias = err
    acc = None
    if clim is not None:
        f_anom = forecast - clim
        o_anom = obs - clim
        denom = abs(f_anom) * abs(o_anom)
        if denom > 0:
            acc = float((f_anom * o_anom) / denom)
    return {"rmse": rmse, "mae": mae, "bias": bias, "acc": acc}


def init_datetimes(year: int) -> list[datetime]:
    """All benchmark inits: every day at 00, 06, 12, 18 UTC (1460 for 2025)."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, tzinfo=timezone.utc)
    inits: list[datetime] = []
    cur = start
    while cur.date() <= end.date():
        for hour in INIT_HOURS_UTC:
            inits.append(
                datetime(cur.year, cur.month, cur.day, hour, tzinfo=timezone.utc)
            )
        cur += timedelta(days=1)
    return inits


def init_filename(init_dt: datetime) -> str:
    return f"{init_dt.strftime('%Y-%m-%dT%H')}Z.json"


def init_json_paths(out_dir: Path, year: int) -> list[Path]:
    return sorted(out_dir.glob(f"{year}-*T*Z.json"))


def all_init_json_paths(out_dir: Path) -> list[Path]:
    """Per-init JSON files across all years (for index.json)."""
    return sorted(out_dir.glob("????-??-??T??Z.json"))


def build_init_json(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
) -> dict:
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    locations: dict = {}

    for station_id, coords in STATIONS.items():
        fcst_t2m, fcst_wind = fetch_hrdps_point(
            init_dt, coords["lat"], coords["lon"]
        )

        var_metrics = {
            var: {m: [] for m in ("rmse", "mae", "bias", "acc")}
            for var in ("t2m", "wind_speed")
        }
        for lead in LEAD_TIMES:
            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)

            for var, fcst in (("t2m", fcst_t2m), ("wind_speed", fcst_wind)):
                if not obs_vals or obs_vals.get(var) is None:
                    for m in var_metrics[var]:
                        var_metrics[var][m].append(None)
                    continue
                obs_val = float(obs_vals[var])
                clim_val = climatology_lookup(clim[station_id], var, valid)
                pm = point_metrics(fcst, obs_val, clim_val)
                for m in var_metrics[var]:
                    var_metrics[var][m].append(pm[m])

        variables = {
            var: {"lead_times_hours": LEAD_TIMES, **var_metrics[var]}
            for var in ("t2m", "wind_speed")
        }

        locations[station_id] = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "variables": variables,
        }

    return {
        "method": METHOD_ID,
        "initialization": init_iso,
        "locations": locations,
    }


def process_init(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
    resume: bool,
) -> str:
    fname = init_filename(init_dt)
    out_path = out_dir / fname
    if resume and out_path.exists():
        return fname

    payload = build_init_json(init_dt, obs, clim)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return fname


def write_metadata(out_dir: Path, year: int) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": "ECCC HRDPS analysis (PT000H / f000)",
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "grid": "continental 2.5 km rotated lat-lon (RLatLon0.0225)",
        "year": year,
        "interpolation": (
            "bilinear on geographic lat/lon via scipy LinearNDInterpolator "
            f"(±{INTERP_WINDOW_DEG:g}° window; curvilinear HRDPS grid)"
        ),
        "wind": "sqrt(u10^2 + v10^2) from 10 m u/v components, m/s to km/h",
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, ±15-day window",
        "forecast_logic": (
            "Single PT000H analysis value at init reused for all 12 lead times; "
            "obs verified at init + lead_hours"
        ),
        "source": "https://dd.weather.gc.ca/model_hrdps/continental/2.5km/",
        "datamart_retention": (
            "MSC Datamart keeps ~30 days of HRDPS on rolling dated paths "
            "({YYYYMMDD}/WXO-DD/...). Full-year 2025 requires a local GRIB cache "
            f"under {CACHE_DIR} or running during the calendar year."
        ),
        "npz_file": f"{METHOD_ID}_{year}.npz",
        "npz_schema": "see benchmarking-site/AGENTS.md",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def write_index(out_dir: Path, files: list[str]) -> None:
    (out_dir / "index.json").write_text(
        json.dumps({"files": sorted(files)}, indent=2) + "\n"
    )


def export_npz(out_dir: Path, year: int) -> Path | None:
    """Consolidate per-init JSON files into one compressed NPZ for analysis."""
    json_paths = init_json_paths(out_dir, year)
    if not json_paths:
        print(f"No {year}-*T*Z.json files in {out_dir}; skipping NPZ export.")
        return None

    n_init = len(json_paths)
    n_station = len(STATION_IDS)
    n_var = len(VARIABLES)
    n_lead = len(LEAD_TIMES)
    shape = (n_init, n_station, n_var, n_lead)

    arrays = {metric: np.full(shape, np.nan, dtype=np.float64) for metric in METRICS}
    initializations: list[str] = []

    for init_idx, path in enumerate(json_paths):
        payload = json.loads(path.read_text())
        initializations.append(payload["initialization"])
        locations = payload["locations"]
        for station_idx, station_id in enumerate(STATION_IDS):
            variables = locations[station_id]["variables"]
            for var_idx, var_id in enumerate(VARIABLES):
                var_data = variables[var_id]
                for metric in METRICS:
                    for lead_idx, value in enumerate(var_data[metric]):
                        if value is not None:
                            arrays[metric][init_idx, station_idx, var_idx, lead_idx] = (
                                value
                            )

    npz_path = out_dir / f"{METHOD_ID}_{year}.npz"
    np.savez_compressed(
        npz_path,
        method_id=np.array(METHOD_ID),
        station_ids=np.array(STATION_IDS),
        variables=np.array(VARIABLES),
        metrics=np.array(METRICS),
        lead_times_hours=np.array(LEAD_TIMES, dtype=np.int16),
        initializations=np.array(initializations),
        **arrays,
    )
    print(f"Wrote {npz_path} ({n_init} inits, shape {shape} per metric).")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="YYYY-MM-DD inclusive lower bound",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="YYYY-MM-DD inclusive upper bound",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel init workers (default 2)",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=6,
        help="Retries per GRIB download with exponential backoff (default 6)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip init files that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 2025-01-15 (all four cycles)",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help="Comma-separated init hours UTC to run (default: 0,6,12,18)",
    )
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Rebuild NPZ from existing JSON files without fetching HRDPS",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Download PT000H GRIB for recent inits into .cache/hrdps_grib/ (live archive)",
    )
    parser.add_argument(
        "--cache-days",
        type=int,
        default=3,
        help="With --cache-only: calendar days of inits to fetch (default 3)",
    )
    parser.add_argument(
        "--also-compute",
        action="store_true",
        help="With --cache-only: write benchmark JSON for cached inits matching --year",
    )
    args = parser.parse_args()

    global DOWNLOAD_RETRIES
    DOWNLOAD_RETRIES = args.download_retries

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year)
        existing = sorted(p.name for p in all_init_json_paths(out_dir))
        write_index(out_dir, existing)
        return

    if args.cache_only:
        run_cache_only(
            days=args.cache_days,
            also_compute=args.also_compute,
            resume=args.resume,
            year=args.year,
            out_dir=out_dir,
        )
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    obs = {sid: load_observations(sid) for sid in STATIONS}
    clim = {sid: build_climatology(obs[sid]) for sid in STATIONS}

    cycle_hours = INIT_HOURS_UTC
    if args.cycles:
        cycle_hours = tuple(int(h.strip()) for h in args.cycles.split(","))

    inits = init_datetimes(args.year)
    inits = [d for d in inits if d.hour in cycle_hours]
    if args.dry_run:
        inits = [
            datetime(2025, 1, 15, hour, tzinfo=timezone.utc) for hour in cycle_hours
        ]
    if args.start_date:
        start = parse_utc(f"{args.start_date}T00:00:00Z")
        inits = [d for d in inits if d >= start]
    if args.end_date:
        end = parse_utc(f"{args.end_date}T18:00:00Z")
        inits = [d for d in inits if d <= end]

    print(f"Processing {len(inits)} initializations -> {out_dir}")

    files: list[str] = []
    if args.workers <= 1:
        for init_dt in inits:
            fname = process_init(init_dt, obs, clim, out_dir, args.resume)
            files.append(fname)
            print(f"  wrote {fname}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_init, init_dt, obs, clim, out_dir, args.resume
                ): init_dt
                for init_dt in inits
            }
            for fut in as_completed(futures):
                init_dt = futures[fut]
                try:
                    fname = fut.result()
                    files.append(fname)
                    print(f"  wrote {fname}")
                except Exception as exc:
                    print(f"  FAILED {init_dt.isoformat()}: {exc}")
                    raise

    write_metadata(out_dir, args.year)
    existing = sorted(p.name for p in all_init_json_paths(out_dir))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, index.json updated.")


if __name__ == "__main__":
    main()
