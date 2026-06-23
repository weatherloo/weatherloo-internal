#!/usr/bin/env python3
"""Compute HRRR interpolated-at-station benchmark JSON for issue #21.

Fetches NOAA HRRR 00/06/12/18Z (3 km CONUS) surface output from AWS Open Data
(``noaa-hrrr-bdp-pds``), bilinearly interpolates 2 m temperature and 10 m wind
(from u/v components) to each station, and writes per-initialization metric files
in this directory (``data/hrrr_interpolated/``).

Also writes a consolidated ``hrrr_interpolated_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).

HRRR is a **Lambert conformal** grid: latitude/longitude are 2-D curvilinear
arrays, so we cannot interpolate on a (lat, lon) regular grid like GFS. Instead
we build the LCC CRS from the GRIB projection attrs, transform each station's
lon/lat into projected x/y, and run ``RegularGridInterpolator`` on the **regular
projected y/x axes** (verified accurate against CYYZ obs in issue #21).

Wind speed: sqrt(u10^2 + v10^2), converted m/s -> km/h (* 3.6).

Lead-time limitation (documented in metadata.json): HRRR forecasts for the
00/06/12/18Z cycles only extend to **f48**. Leads 54/60/66/72 h are therefore
**null by design** in every output file -- a known HRRR limitation, not missing
data to backfill later.

ACC anomaly baseline: DOY + UTC-hour climatology from station observations with a
+/-15-day calendar window (single-year proxy; same as gfs_interpolated).
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
CACHE_DIR = REPO_ROOT / ".cache" / "hrrr_grib"

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
INIT_HOURS_UTC = (0, 6, 12, 18)
# Full lead axis (length 12) kept for output-contract compatibility.
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
# HRRR 00/06/12/18Z runs only reach f48; leads beyond are null by design.
MAX_LEAD_HOURS = 48
METHOD_ID = "hrrr_interpolated"

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
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            # http.client.HTTPException covers IncompleteRead (truncated S3 response).
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


# --- HRRR Lambert-conformal grid (static; built once, reused across threads) ---

_GRID: dict | None = None
_GRID_LOCK = threading.Lock()


def ensure_grid(sample_path: Path) -> dict:
    """Build (once) the LCC projected axes + station projected coords.

    HRRR's CONUS grid never changes, so the regular projected x/y axes and the
    two station x/y points are computed a single time from the first GRIB message
    opened and cached for every later interpolation. Guarded by a lock so the
    one-time full-grid projection is safe under the ThreadPoolExecutor.
    """
    global _GRID
    if _GRID is not None:
        return _GRID
    with _GRID_LOCK:
        if _GRID is not None:
            return _GRID
        ds = cfgrib.open_dataset(sample_path)
        var = list(ds.data_vars)[0]
        attrs = ds[var].attrs
        lat = np.asarray(ds.latitude.values, dtype=float)
        lon = np.asarray(ds.longitude.values, dtype=float)
        lon = np.where(lon > 180.0, lon - 360.0, lon)

        crs = CRS.from_proj4(
            f"+proj=lcc +lat_1={attrs['GRIB_Latin1InDegrees']} "
            f"+lat_2={attrs['GRIB_Latin2InDegrees']} "
            f"+lat_0={attrs['GRIB_LaDInDegrees']} "
            f"+lon_0={attrs['GRIB_LoVInDegrees']} "
            f"+R=6371229 +units=m +no_defs"
        )
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tr.transform(lon, lat)
        x_axis = xs[0, :]
        y_axis = ys[:, 0]
        flip_x = x_axis[0] > x_axis[-1]
        flip_y = y_axis[0] > y_axis[-1]
        xa = x_axis[::-1] if flip_x else x_axis
        ya = y_axis[::-1] if flip_y else y_axis

        station_xy: dict[str, tuple[float, float]] = {}
        for sid, coords in STATIONS.items():
            sx, sy = tr.transform(coords["lon"], coords["lat"])
            if not (xa[0] <= sx <= xa[-1] and ya[0] <= sy <= ya[-1]):
                raise RuntimeError(f"Station {sid} falls outside HRRR domain")
            station_xy[sid] = (float(sx), float(sy))

        _GRID = {
            "xa": xa,
            "ya": ya,
            "flip_x": flip_x,
            "flip_y": flip_y,
            "station_xy": station_xy,
        }
        return _GRID


def interp_field(path: Path, station_xy: tuple[float, float]) -> float:
    """Bilinear interp of one field at a station's projected (x, y)."""
    g = ensure_grid(path)
    ds = cfgrib.open_dataset(path)
    var = list(ds.data_vars)[0]
    arr = np.asarray(ds[var].values, dtype=float)
    if g["flip_y"]:
        arr = arr[::-1, :]
    if g["flip_x"]:
        arr = arr[:, ::-1]
    rgi = RegularGridInterpolator(
        (g["ya"], g["xa"]), arr, method="linear", bounds_error=False, fill_value=np.nan
    )
    sx, sy = station_xy
    return float(rgi((sy, sx)))


def fetch_hrrr_point(
    date_yyyymmdd: str,
    cycle_hour: int,
    fxx: int,
    station_id: str,
) -> tuple[float, float]:
    """Return (t2m degC, wind_speed km/h) at a station for one init/lead."""
    cycle_tag = f"{cycle_hour:02d}z"
    grib_url = (
        f"{AWS_BASE}/hrrr.{date_yyyymmdd}/conus/"
        f"hrrr.t{cycle_tag}.wrfsfcf{fxx:02d}.grib2"
    )
    cache_key = f"{date_yyyymmdd}_{cycle_tag}_f{fxx:02d}"
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

    station_xy = ensure_grid(field_paths["t2m_k"])["station_xy"][station_id]
    t2m_c = interp_field(field_paths["t2m_k"], station_xy) - 273.15
    u10 = interp_field(field_paths["u10"], station_xy)
    v10 = interp_field(field_paths["v10"], station_xy)
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
            # HRRR has no output beyond f48 for these cycles -> null by design.
            if lead > MAX_LEAD_HOURS:
                for var in ("t2m", "wind_speed"):
                    for m in var_metrics[var]:
                        var_metrics[var][m].append(None)
                continue

            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)
            fcst_t2m, fcst_wind = fetch_hrrr_point(
                date_str, init_dt.hour, lead, station_id
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


def write_metadata(out_dir: Path, year: int) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": "NOAA HRRR",
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "grid": "3km CONUS Lambert conformal",
        "year": year,
        "interpolation": "bilinear on projected LCC x/y axes (pyproj)",
        "wind": "sqrt(u10^2 + v10^2) from 10 m u/v components, m/s to km/h",
        "lead_times_hours": LEAD_TIMES,
        "available_lead_times_hours": [h for h in LEAD_TIMES if h <= MAX_LEAD_HOURS],
        "unavailable_lead_times_hours": [h for h in LEAD_TIMES if h > MAX_LEAD_HOURS],
        "lead_time_note": (
            "HRRR forecasts for the 00/06/12/18Z cycles only extend to f48. "
            "Leads 54/60/66/72 h are null by design in every file -- a known HRRR "
            "limitation, NOT missing data to backfill later."
        ),
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, ±15-day window",
        "source": "https://registry.opendata.aws/noaa-hrrr-pds/",
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
    shape = (n_init, len(STATION_IDS), len(VARIABLES), len(LEAD_TIMES))

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
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD inclusive lower bound")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD inclusive upper bound")
    parser.add_argument("--workers", type=int, default=3, help="Parallel init workers (default 3; use 1-2 if AWS resets persist)")
    parser.add_argument("--download-retries", type=int, default=6, help="Retries per GRIB/idx download with exponential backoff (default 6)")
    parser.add_argument("--resume", action="store_true", help="Skip init files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="Process only 2025-01-15 (all four cycles)")
    parser.add_argument("--cycles", type=str, default=None, help="Comma-separated init hours UTC to run (default: 0,6,12,18)")
    parser.add_argument("--export-npz-only", action="store_true", help="Rebuild NPZ from existing JSON files without fetching HRRR")
    args = parser.parse_args()

    global DOWNLOAD_RETRIES
    DOWNLOAD_RETRIES = args.download_retries

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year)
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
        inits = [datetime(2025, 1, 15, hour, tzinfo=timezone.utc) for hour in cycle_hours]
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
                pool.submit(process_init, init_dt, obs, clim, out_dir, args.resume): init_dt
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
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, index.json updated.")


if __name__ == "__main__":
    main()
