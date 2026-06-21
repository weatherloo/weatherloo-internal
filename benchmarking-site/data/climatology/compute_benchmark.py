#!/usr/bin/env python3
"""Compute climatology benchmark JSON for issue #1.

Builds a multi-year (default 2010-2024) DOY + UTC-hour station climatology
from historical observations and uses it as the "forecast" for each
initialization and lead time.  The climatological mean for the valid_time's
day-of-year and UTC hour IS the forecast value.

Because the forecast equals the climatology, ACC is always null (forecast
anomaly is identically zero).

Wind speed uses station-observed km/h directly; no u/v decomposition is
needed since the climatology is built from station observations, not from
gridded model output.

Also writes a consolidated ``climatology_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_station_observations.py"

INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "climatology"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

DEFAULT_CLIM_YEARS = range(2010, 2025)  # 2010-2024 inclusive
DEFAULT_CLIM_WINDOW = 15
DEFAULT_MIN_SAMPLES = 3


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_observations(station_id: str) -> dict[str, dict[str, float | None]]:
    """Load 2025 observations for verification."""
    path = OBS_ROOT / station_id / "observations_6h_2025.json"
    data = json.loads(path.read_text())
    return {
        row["valid_time"]: {
            "t2m": row.get("t2m"),
            "wind_speed": row.get("wind_speed"),
        }
        for row in data["observations"]
    }


def load_observations_for_year(
    station_id: str, year: int
) -> dict[str, dict[str, float | None]] | None:
    """Load observations for a specific year.  Returns None if file missing."""
    path = OBS_ROOT / station_id / f"observations_6h_{year}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        row["valid_time"]: {
            "t2m": row.get("t2m"),
            "wind_speed": row.get("wind_speed"),
        }
        for row in data["observations"]
    }


def build_multiyear_climatology(
    station_id: str,
    years: range,
    window_days: int = 15,
    min_samples: int = 3,
) -> dict[str, dict[str, float]]:
    """DOY + UTC-hour climatology from multiple years of station observations.

    Returns {var: {"DDD-HH": mean_value}} where DDD is zero-padded DOY and
    HH is the UTC hour.  Bins with fewer than *min_samples* are omitted
    (treated as unavailable by the lookup function).
    """
    # Collect all (doy, value) pairs grouped by (variable, hour)
    slots: dict[tuple[str, int], list[tuple[int, float]]] = {}
    years_loaded = 0

    for year in years:
        obs = load_observations_for_year(station_id, year)
        if obs is None:
            print(f"  [climatology] {station_id}: no obs file for {year}, skipping")
            continue
        years_loaded += 1
        for valid_time, vals in obs.items():
            dt = parse_utc(valid_time)
            doy = dt.timetuple().tm_yday
            hour = dt.hour
            for var in ("t2m", "wind_speed"):
                v = vals.get(var)
                if v is None:
                    continue
                slots.setdefault((var, hour), []).append((doy, float(v)))

    if years_loaded == 0:
        raise RuntimeError(
            f"No historical observation files found for {station_id} "
            f"in years {years[0]}-{years[-1]}.  "
            f"Run with --fetch-historical to download them first."
        )
    print(f"  [climatology] {station_id}: loaded {years_loaded} years of observations")

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
                if len(vals_in_window) >= min_samples:
                    clim[var][f"{target_doy:03d}-{hour:02d}"] = float(
                        np.mean(vals_in_window)
                    )
    return clim


def climatology_lookup(
    clim: dict[str, dict[str, float]], var: str, valid_time: datetime
) -> float | None:
    key = f"{valid_time.timetuple().tm_yday:03d}-{valid_time.hour:02d}"
    return clim.get(var, {}).get(key)


def point_metrics_climatology(
    forecast: float,
    obs: float,
) -> dict[str, float | None]:
    """Metrics for a single (init, station, variable, lead) point.

    ACC is always None because the forecast IS the climatology, so the
    forecast anomaly is identically zero.
    """
    err = forecast - obs
    return {
        "rmse": abs(err),
        "mae": abs(err),
        "bias": err,
        "acc": None,
    }


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

            for var in ("t2m", "wind_speed"):
                fcst = climatology_lookup(clim[station_id], var, valid)
                if fcst is None or not obs_vals or obs_vals.get(var) is None:
                    for m in var_metrics[var]:
                        var_metrics[var][m].append(None)
                    continue
                obs_val = float(obs_vals[var])
                pm = point_metrics_climatology(fcst, obs_val)
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


def write_metadata(out_dir: Path, year: int, clim_years: range) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": f"DOY + UTC-hour station climatology ({clim_years[0]}-{clim_years[-1]})",
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "year": year,
        "forecast_logic": (
            f"Multi-year DOY + UTC-hour mean from station observations "
            f"({clim_years[0]}-{clim_years[-1]}). The climatological mean at "
            f"the valid_time's day-of-year and UTC hour is the forecast."
        ),
        "climatology_years": f"{clim_years[0]}-{clim_years[-1]}",
        "climatology_window_days": DEFAULT_CLIM_WINDOW,
        "climatology_min_samples": DEFAULT_MIN_SAMPLES,
        "wind": "Station wind speed (km/h) from observations; no u/v decomposition",
        "acc_note": (
            "ACC is null: forecast equals climatology, so all forecast "
            "anomalies are zero"
        ),
        "source_cyyz": "ECCC climate-hourly API, STN_ID 51459",
        "source_eric_d_soulis": (
            "UW Soulis archive (main logger 2010-2014, HOBO 2015-2024)"
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


def fetch_historical_observations(years: range) -> None:
    """Download historical observations by invoking fetch_station_observations.py."""
    for year in years:
        # Check if both stations already have files for this year
        cyyz_path = OBS_ROOT / "cyyz" / f"observations_6h_{year}.json"
        soulis_path = OBS_ROOT / "eric_d_soulis" / f"observations_6h_{year}.json"
        if cyyz_path.exists() and soulis_path.exists():
            print(f"  [fetch] {year}: both stations already have obs files, skipping")
            continue

        print(f"  [fetch] Downloading observations for {year}...")
        try:
            subprocess.run(
                [sys.executable, str(FETCH_SCRIPT), "--year", str(year)],
                check=True,
                cwd=str(REPO_ROOT),
            )
        except subprocess.CalledProcessError as exc:
            print(f"  [fetch] WARNING: failed to fetch {year}: {exc}")
            continue


def parse_clim_years(s: str) -> range:
    """Parse '2010-2024' or '2010,2024' into a range."""
    if "-" in s:
        parts = s.split("-")
        return range(int(parts[0]), int(parts[1]) + 1)
    parts = s.split(",")
    return range(int(parts[0]), int(parts[-1]) + 1)


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
        default=4,
        help="Parallel init workers (default 4; no downloads so higher is fine)",
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
        help="Rebuild NPZ from existing JSON files without recomputing",
    )
    parser.add_argument(
        "--clim-years",
        type=str,
        default="2010-2024",
        help="Range of years for climatology (default: 2010-2024)",
    )
    parser.add_argument(
        "--clim-window",
        type=int,
        default=DEFAULT_CLIM_WINDOW,
        help="Calendar-day half-window for climatology smoothing (default: 15)",
    )
    parser.add_argument(
        "--min-clim-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum samples per DOY-hour bin (default: 3)",
    )
    parser.add_argument(
        "--fetch-historical",
        action="store_true",
        help="Download historical observations for clim-years before computing",
    )
    args = parser.parse_args()

    clim_years = parse_clim_years(args.clim_years)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year, clim_years)
        return

    if args.fetch_historical:
        print(f"Fetching historical observations for {clim_years[0]}-{clim_years[-1]}...")
        fetch_historical_observations(clim_years)

    # Build multi-year climatology for each station
    print("Building multi-year climatology...")
    clim = {}
    for station_id in STATIONS:
        clim[station_id] = build_multiyear_climatology(
            station_id,
            clim_years,
            window_days=args.clim_window,
            min_samples=args.min_clim_samples,
        )

    # Load verification-year observations
    obs = {sid: load_observations(sid) for sid in STATIONS}

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

    write_metadata(out_dir, args.year, clim_years)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, index.json updated.")


if __name__ == "__main__":
    main()
