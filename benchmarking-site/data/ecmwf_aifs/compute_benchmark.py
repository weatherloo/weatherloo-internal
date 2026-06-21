#!/usr/bin/env python3
"""Compute ECMWF AIFS Single interpolated-at-station benchmark JSON.

Fetches ECMWF AIFS Single historical forecasts from the dynamical.org catalog
(Icechunk Zarr on AWS Open Data), bilinearly interpolates 2 m temperature and
10 m wind (from u/v components) to each station, and writes per-init JSON in
this directory (``data/ecmwf_aifs/``).

Also writes a consolidated ``ecmwf_aifs_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).

Data source: https://dynamical.org/catalog/ecmwf-aifs-single-forecast/
Catalog ID: ``ecmwf-aifs-single-forecast`` (``dynamical-catalog>=0.5.0``).

Wind speed: sqrt(u10^2 + v10^2), converted m/s -> km/h (* 3.6).
ACC anomaly baseline: DOY + UTC-hour climatology from station observations
with a +/-15-day calendar window (single-year proxy; document in metadata.json).
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dynamical_catalog
import numpy as np
from scipy.interpolate import RegularGridInterpolator

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"

CATALOG_ID = "ecmwf-aifs-single-forecast"
INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "ecmwf_aifs"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

CATALOG_VARS = {
    "t2m": "temperature_2m",
    "u10": "wind_u_10m",
    "v10": "wind_v_10m",
}

# Regional window covering both stations with padding for bilinear interp.
LAT_MAX = max(coords["lat"] for coords in STATIONS.values()) + 0.5
LAT_MIN = min(coords["lat"] for coords in STATIONS.values()) - 0.5
LON_MIN = min(coords["lon"] for coords in STATIONS.values()) - 0.5
LON_MAX = max(coords["lon"] for coords in STATIONS.values()) + 0.5

LEAD_TIMEDELTAS = [np.timedelta64(h, "h") for h in LEAD_TIMES]

_dataset_lock = threading.Lock()


class MissingForecastError(RuntimeError):
    """Raised when a forecast init is absent from the dynamical.org archive."""


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
    slots: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for valid_time, vals in obs_by_time.items():
        dt = parse_utc(valid_time)
        doy = dt.timetuple().tm_yday
        hour = dt.hour
        for var in VARIABLES:
            value = vals.get(var)
            if value is None:
                continue
            slots.setdefault((var, hour), []).append((doy, float(value)))

    clim: dict[str, dict[str, float]] = {var: {} for var in VARIABLES}
    for var in VARIABLES:
        for hour in INIT_HOURS_UTC:
            entries = slots.get((var, hour), [])
            if not entries:
                continue
            for target_doy in range(1, 367):
                vals_in_window = [
                    value
                    for doy, value in entries
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


def _grid_interpolator(
    lats: np.ndarray, lons: np.ndarray, arr: np.ndarray
) -> RegularGridInterpolator:
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        arr = arr[:, ::-1]
    return RegularGridInterpolator((lats, lons), arr)


def fetch_init_fields(ds, init_dt: datetime) -> dict[str, np.ndarray]:
    """Load t2m, u10, v10 for all benchmark leads in the station window."""
    init_sel = init_dt.strftime("%Y-%m-%dT%H:00:00")
    fields: dict[str, np.ndarray] = {}
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None

    with _dataset_lock:
        for field_key, var_name in CATALOG_VARS.items():
            try:
                chunk = (
                    ds[var_name]
                    .sel(
                        init_time=init_sel,
                        lead_time=LEAD_TIMEDELTAS,
                        latitude=slice(LAT_MAX, LAT_MIN),
                        longitude=slice(LON_MIN, LON_MAX),
                    )
                    .compute()
                )
            except KeyError as exc:
                raise MissingForecastError(
                    f"missing AIFS init in catalog: {init_sel}"
                ) from exc

            if chunk.sizes.get("lead_time", 0) != len(LEAD_TIMES):
                raise MissingForecastError(
                    f"incomplete lead times for init {init_sel}"
                )

            if lats is None:
                lats = np.asarray(chunk.latitude.values, dtype=float)
                lons = np.asarray(chunk.longitude.values, dtype=float)
            fields[field_key] = np.asarray(chunk.values, dtype=float)

    if lats is None or lons is None:
        raise MissingForecastError(f"empty regional slice for init {init_sel}")

    fields["_lats"] = lats
    fields["_lons"] = lons
    return fields


def interp_station(
    fields: dict[str, np.ndarray], lat: float, lon: float, lead_idx: int
) -> tuple[float, float]:
    """Return (t2m degC, wind_speed km/h) for one station and lead index."""
    lats = fields["_lats"]
    lons = fields["_lons"]

    t2m = float(
        _grid_interpolator(lats, lons, fields["t2m"][lead_idx])(np.array([[lat, lon]]))[
            0
        ]
    )
    u10 = float(
        _grid_interpolator(lats, lons, fields["u10"][lead_idx])(np.array([[lat, lon]]))[
            0
        ]
    )
    v10 = float(
        _grid_interpolator(lats, lons, fields["v10"][lead_idx])(np.array([[lat, lon]]))[
            0
        ]
    )
    wind_kmh = float(np.hypot(u10, v10) * 3.6)
    return t2m, wind_kmh


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
    """All benchmark inits: every day at 00, 06, 12, 18 UTC."""
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


def append_null_metrics(var_metrics: dict[str, dict[str, list]], var: str) -> None:
    for metric in METRICS:
        var_metrics[var][metric].append(None)


def build_init_json(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
    ds,
) -> tuple[dict, int]:
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    locations: dict = {}
    forecast_count = 0

    try:
        fields = fetch_init_fields(ds, init_dt)
    except MissingForecastError:
        fields = None

    for station_id, coords in STATIONS.items():
        var_metrics = {var: {m: [] for m in METRICS} for var in VARIABLES}
        for lead_idx, lead in enumerate(LEAD_TIMES):
            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)

            if fields is None:
                for var in VARIABLES:
                    append_null_metrics(var_metrics, var)
                continue

            fcst_t2m, fcst_wind = interp_station(
                fields, coords["lat"], coords["lon"], lead_idx
            )
            forecast_count += 1

            for var, fcst in (("t2m", fcst_t2m), ("wind_speed", fcst_wind)):
                if not obs_vals or obs_vals.get(var) is None:
                    append_null_metrics(var_metrics, var)
                    continue
                obs_val = float(obs_vals[var])
                clim_val = climatology_lookup(clim[station_id], var, valid)
                pm = point_metrics(fcst, obs_val, clim_val)
                for metric in METRICS:
                    var_metrics[var][metric].append(pm[metric])

        variables = {
            var: {"lead_times_hours": LEAD_TIMES, **var_metrics[var]}
            for var in VARIABLES
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
    }, forecast_count


def process_init(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
    resume: bool,
    ds,
    keep_empty_inits: bool,
) -> str | None:
    fname = init_filename(init_dt)
    out_path = out_dir / fname
    if resume and out_path.exists():
        return fname

    payload, forecast_count = build_init_json(init_dt, obs, clim, ds)
    if forecast_count == 0 and not keep_empty_inits:
        return None

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return fname


def write_metadata(out_dir: Path, year: int) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": "ECMWF AIFS Single",
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "grid": "0.25 degree regular lat/lon (dynamical.org regridded archive)",
        "year": year,
        "interpolation": "bilinear",
        "wind": "sqrt(u10^2 + v10^2) from 10 m u/v components, m/s to km/h",
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, +/-15-day window",
        "source": "https://dynamical.org/catalog/ecmwf-aifs-single-forecast/",
        "catalog_id": CATALOG_ID,
        "source_notes": [
            "ECMWF AIFS Single forecast data processed by dynamical.org from ECMWF Open Data.",
            "Licensed under CC BY 4.0 and ECMWF Terms of Use.",
            "Archive spans 2024-04-01 to present at 6-hourly inits and 6-hourly lead steps (0-360 h).",
        ],
        "params": {
            "t2m": "temperature_2m",
            "u10": "wind_u_10m",
            "v10": "wind_v_10m",
        },
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
        help="Parallel init workers (default 2; catalog reads are serialized per process)",
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
        "--keep-empty-inits",
        action="store_true",
        help="Write all-null init JSON when the catalog has no forecast for that init",
    )
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Rebuild NPZ from existing JSON files without fetching AIFS data",
    )
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year)
        return

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

    print(f"Opening dynamical.org catalog: {CATALOG_ID}")
    ds = dynamical_catalog.open(CATALOG_ID)
    print(f"Processing {len(inits)} initializations -> {out_dir}")

    files: list[str] = []
    skipped = 0
    if args.workers <= 1:
        for init_dt in inits:
            fname = process_init(
                init_dt,
                obs,
                clim,
                out_dir,
                args.resume,
                ds,
                args.keep_empty_inits,
            )
            if fname is None:
                skipped += 1
                print(f"  skipped missing {init_filename(init_dt)}")
                continue
            files.append(fname)
            print(f"  wrote {fname}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_init,
                    init_dt,
                    obs,
                    clim,
                    out_dir,
                    args.resume,
                    ds,
                    args.keep_empty_inits,
                ): init_dt
                for init_dt in inits
            }
            for fut in as_completed(futures):
                init_dt = futures[fut]
                try:
                    fname = fut.result()
                    if fname is None:
                        skipped += 1
                        print(f"  skipped missing {init_filename(init_dt)}")
                        continue
                    files.append(fname)
                    print(f"  wrote {fname}")
                except Exception as exc:
                    print(f"  FAILED {init_dt.isoformat()}: {exc}")
                    raise

    write_metadata(out_dir, args.year)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    if existing:
        write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, {skipped} skipped, index.json updated.")


if __name__ == "__main__":
    main()
