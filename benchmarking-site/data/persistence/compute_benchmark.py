#!/usr/bin/env python3
"""Compute the Persistence baseline benchmark JSON (method #3).

Persistence forecast: at each initialization time, take the *actual observed*
value (t2m and 10 m wind speed) at that exact moment and reuse it unchanged as
the forecast for **all 12 lead times** [6, 12, ..., 72]. Each persisted value is
then compared against the actual observation at every lead time's valid time
(``init_time + lead_hours``), yielding RMSE / MAE / Bias / ACC arrays of length
12 per station per variable.

Writes one JSON file per initialization in this directory
(``data/persistence/``) plus a consolidated ``persistence_<year>.npz`` (see
``benchmarking-site/AGENTS.md``).

No network / model downloads are needed — this baseline is built purely from the
station observation files in ``data/observations/<station_id>/``.

ACC anomaly baseline: DOY + UTC-hour climatology from 2025 station observations
with a +/-15-day calendar window (single-year proxy; documented in metadata.json).
This matches the gfs_interpolated reference so methods are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"

INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "persistence"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]


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
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    locations: dict = {}

    for station_id, coords in STATIONS.items():
        # Persisted forecast = actual observation at the initialization instant.
        init_obs = obs[station_id].get(init_iso) or {}
        persisted = {
            "t2m": init_obs.get("t2m"),
            "wind_speed": init_obs.get("wind_speed"),
        }

        var_metrics = {
            var: {m: [] for m in ("rmse", "mae", "bias", "acc")}
            for var in ("t2m", "wind_speed")
        }
        for lead in LEAD_TIMES:
            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)
            for var in ("t2m", "wind_speed"):
                fcst = persisted[var]
                obs_val = obs_vals.get(var) if obs_vals else None
                # Missing the persisted value or the verifying observation -> null.
                if fcst is None or obs_val is None:
                    for m in var_metrics[var]:
                        var_metrics[var][m].append(None)
                    continue
                clim_val = climatology_lookup(clim[station_id], var, valid)
                pm = point_metrics(float(fcst), float(obs_val), clim_val)
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
        "model": "Persistence baseline",
        "description": (
            "Forecast = observed value at initialization time, held constant "
            "across all lead times."
        ),
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "year": year,
        "source": "data/observations/<station_id>/observations_6h_2025.json",
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, +/-15-day window",
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
                            arrays[metric][
                                init_idx, station_idx, var_idx, lead_idx
                            ] = value

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
        "--start-date", type=str, default=None, help="YYYY-MM-DD inclusive lower bound"
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="YYYY-MM-DD inclusive upper bound"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N initializations (for quick checks)",
    )
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help="Comma-separated init hours UTC to run (default: 0,6,12,18)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip init files that already exist"
    )
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Rebuild NPZ from existing JSON files without recomputing",
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
    if args.start_date:
        start = parse_utc(f"{args.start_date}T00:00:00Z")
        inits = [d for d in inits if d >= start]
    if args.end_date:
        end = parse_utc(f"{args.end_date}T18:00:00Z")
        inits = [d for d in inits if d <= end]
    if args.limit is not None:
        inits = inits[: args.limit]

    print(f"Processing {len(inits)} initializations -> {out_dir}")

    written = 0
    skipped = 0
    for i, init_dt in enumerate(inits, 1):
        fname = init_filename(init_dt)
        if args.resume and (out_dir / fname).exists():
            skipped += 1
            continue
        process_init(init_dt, obs, clim, out_dir, args.resume)
        written += 1
        if written % 100 == 0 or i == len(inits):
            print(f"  [{i}/{len(inits)}] wrote {fname}")

    write_metadata(out_dir, args.year)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(
        f"Done. {written} written, {skipped} skipped, "
        f"{len(existing)} total init files; index.json updated."
    )


if __name__ == "__main__":
    main()
