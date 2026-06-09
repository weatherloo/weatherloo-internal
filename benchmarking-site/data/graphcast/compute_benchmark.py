#!/usr/bin/env python3
"""Compute GraphCastGFS interpolated-at-station benchmark JSON for issue #11.

Fetches NOAA GraphCastGFS 00/06/12/18Z (0.25°) from AWS Open Data, bilinearly
interpolates 2 m temperature and 10 m wind (from u/v components) to each station,
and writes per-initialization metric files in this directory
(``data/graphcast/``).

Also writes a consolidated ``graphcast_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).

Wind speed: sqrt(u10^2 + v10^2), converted m/s -> km/h (* 3.6).
ACC anomaly baseline: DOY + UTC-hour climatology from station observations
with a ±15-day calendar window (single-year proxy; document in metadata.json).
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
from scipy.interpolate import RegularGridInterpolator

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
CACHE_DIR = REPO_ROOT / ".cache" / "graphcast_grib"

AWS_BASE = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "graphcast"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

GRIB_NEEDLES = {
    "t2m_k": ":TMP:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
}

DOWNLOAD_RETRIES = 6


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


def download_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "weatherloo-internal/1.0"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"

    last_err: BaseException | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
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


def parse_idx_ranges(idx_text: str) -> dict[str, tuple[int, int]]:
    lines = idx_text.strip().splitlines()
    ranges: dict[str, tuple[int, int]] = {}
    for i, line in enumerate(lines):
        for key, needle in GRIB_NEEDLES.items():
            if key in ranges:
                continue
            if needle in line:
                start = int(line.split(":")[1])
                end = int(lines[i + 1].split(":")[1])
                ranges[key] = (start, end)
    missing = set(GRIB_NEEDLES) - set(ranges)
    if missing:
        raise RuntimeError(f"Missing GRIB fields in idx: {missing}")
    return ranges


def interp_field(path: Path, lat: float, lon: float) -> float:
    lon_q = lon + 360.0 if lon < 0 else lon
    ds = cfgrib.open_dataset(path)
    var = list(ds.data_vars)[0]
    lats = np.asarray(ds.latitude.values, dtype=float)
    lons = np.asarray(ds.longitude.values, dtype=float)
    arr = np.asarray(ds[var].values, dtype=float)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    return float(RegularGridInterpolator((lats, lons), arr)((lat, lon_q)))


def fetch_graphcast_point(
    date_yyyymmdd: str,
    cycle_hour: int,
    fxx: int,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    """Return (t2m degC, wind_speed km/h) at a point for one init/lead."""
    cycle_tag = f"{cycle_hour:02d}z"
    grib_url = (
        f"{AWS_BASE}/graphcastgfs.{date_yyyymmdd}/{cycle_hour:02d}/"
        f"forecasts_13_levels/graphcastgfs.t{cycle_tag}.pgrb2.0p25.f{fxx:03d}"
    )
    cache_key = f"{date_yyyymmdd}_{cycle_tag}_f{fxx:03d}"
    idx_path = CACHE_DIR / f"{cache_key}.idx"
    if idx_path.exists():
        idx_text = idx_path.read_text()
    else:
        idx_text = download_bytes(grib_url + ".idx").decode("utf-8", errors="replace")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(idx_text)
    ranges = parse_idx_ranges(idx_text)

    field_paths: dict[str, Path] = {}
    for field in GRIB_NEEDLES:
        path = CACHE_DIR / f"{cache_key}_{field}.grib2"
        field_paths[field] = path
        if path.exists():
            continue
        start, end = ranges[field]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download_bytes(grib_url, start, end))

    t2m_c = interp_field(field_paths["t2m_k"], lat, lon) - 273.15
    u10 = interp_field(field_paths["u10"], lat, lon)
    v10 = interp_field(field_paths["v10"], lat, lon)
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
    """Per-init JSON files for the current naming scheme."""
    return sorted(out_dir.glob(f"{year}-*T*Z.json"))


def build_init_json(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
) -> dict:
    date_str = init_dt.strftime("%Y%m%d")
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    locations: dict = {}

    for station_id, coords in STATIONS.items():
        var_metrics = {
            var: {m: [] for m in ("rmse", "mae", "bias", "acc")}
            for var in ("t2m", "wind_speed")
        }
        for lead in LEAD_TIMES:
            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)
            fcst_t2m, fcst_wind = fetch_graphcast_point(
                date_str, init_dt.hour, lead, coords["lat"], coords["lon"]
            )
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
        default=3,
        help="Parallel init workers (default 3; use 1-2 if AWS resets persist)",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=6,
        help="Retries per GRIB/idx download with exponential backoff (default 6)",
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
        help="Rebuild NPZ from existing JSON files without fetching GraphCast data",
    )
    args = parser.parse_args()

    global DOWNLOAD_RETRIES
    DOWNLOAD_RETRIES = args.download_retries

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
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

    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, index.json updated.")


if __name__ == "__main__":
    main()
