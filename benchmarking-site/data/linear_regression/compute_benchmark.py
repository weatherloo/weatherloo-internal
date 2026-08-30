#!/usr/bin/env python3
"""Linear Regression bias-correction benchmark (issue #5).

For each (station, variable, lead_time) we fit one simple linear regression

    corrected = a + b * raw_gfs_forecast

and apply it to the GFS interpolated 2025 forecasts, scoring the corrected
value against the 2025 station observations.

TRAINING: leave-one-month-out cross-validation on the *real* GFS forecasts.
  ERA5 reanalysis is NOT a forecast, so training on ERA5 init->valid pairs
  created a train/test feature mismatch. Instead we train and test entirely on
  the existing gfs_interpolated 2025 output:

    for each month M in 2025:
      train on all inits whose month != M  (the other 11 months)
      fit  corrected = a + b * raw_gfs   per (station, variable, lead)
      apply to inits whose month == M, score vs observations, write their JSON

  => 48 regressions per fold (2 stations x 2 variables x 12 leads),
     12 folds (one per month). The model never sees the month it scores.

FEATURE: single predictor — the raw GFS interpolated forecast for that
  (station, variable, lead). No season/month predictors beyond the fold split.

RAW FORECAST RECOVERY (no GFS re-fetch): the gfs_interpolated files store
  bias = forecast - observation, so the raw GFS forecast is recovered exactly
  as ``raw = observation + bias`` at each lead.

Interpolation/units are inherited from gfs_interpolated (bilinear; t2m degC;
wind_speed km/h from 10 m u/v).

ACC anomaly baseline: DOY + UTC-hour climatology from 2025 station obs with a
+/-15-day calendar window (same proxy as gfs_interpolated / persistence so
methods stay comparable).

coefficients.json stores the fold-averaged (a, b) per station/variable/lead for
inspection (mean of the 12 fold fits).

Flags:
  --score-only   run folds + score (kept for compatibility; fitting is inline)
  --resume       skip init JSON files that already exist
  --export-npz-only   rebuild the consolidated NPZ from existing JSON
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
GFS_DIR = BENCHMARKING_SITE / "data" / "gfs_interpolated"
COEFFS_PATH = METHOD_DIR / "coefficients.json"

INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
MONTHS = list(range(1, 13))
METHOD_ID = "linear_regression"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

MIN_PAIRS = 30


# --------------------------------------------------------------------------- #
# Shared helpers (observations, climatology, metrics)                          #
# --------------------------------------------------------------------------- #
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
    forecast: float, obs: float, clim: float | None
) -> dict[str, float | None]:
    err = forecast - obs
    acc = None
    if clim is not None:
        f_anom = forecast - clim
        o_anom = obs - clim
        denom = abs(f_anom) * abs(o_anom)
        if denom > 0:
            acc = float((f_anom * o_anom) / denom)
    return {"rmse": abs(err), "mae": abs(err), "bias": err, "acc": acc}


def init_datetimes(year: int) -> list[datetime]:
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


# --------------------------------------------------------------------------- #
# Raw GFS forecast recovery: raw = observation + bias                          #
# --------------------------------------------------------------------------- #
def load_raw_gfs(
    inits: list[datetime],
    obs: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, dict[str, list[float | None]]]]:
    """raw[init_iso][station][var] = list of raw forecast (or None) per lead."""
    raw: dict = {}
    for init_dt in inits:
        path = GFS_DIR / init_filename(init_dt)
        init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
        raw[init_iso] = {}
        payload = json.loads(path.read_text()) if path.exists() else None
        for station_id in STATION_IDS:
            raw[init_iso][station_id] = {}
            for var in VARIABLES:
                bias_arr = (
                    payload["locations"][station_id]["variables"][var]["bias"]
                    if payload is not None
                    else [None] * len(LEAD_TIMES)
                )
                vals: list[float | None] = []
                for li, lead in enumerate(LEAD_TIMES):
                    valid = init_dt + timedelta(hours=lead)
                    valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
                    obs_vals = obs[station_id].get(valid_iso)
                    obs_val = obs_vals.get(var) if obs_vals else None
                    bias = bias_arr[li] if li < len(bias_arr) else None
                    vals.append(
                        float(obs_val) + float(bias)
                        if (obs_val is not None and bias is not None)
                        else None
                    )
                raw[init_iso][station_id][var] = vals
    return raw


# --------------------------------------------------------------------------- #
# Fit: leave-one-month-out folds                                               #
# --------------------------------------------------------------------------- #
def fit_folds(
    inits: list[datetime],
    raw: dict,
    obs: dict[str, dict[str, dict[str, float | None]]],
) -> dict[int, dict]:
    """Return fold_coef[month][station][var][str(lead)] = {a, b, n}."""
    from sklearn.linear_model import LinearRegression

    # Collect pairs grouped by month: pairs[s][v][lead_idx][month] = ([raw],[obs])
    pairs: dict = {
        s: {v: {li: {m: ([], []) for m in MONTHS} for li in range(len(LEAD_TIMES))}
            for v in VARIABLES}
        for s in STATION_IDS
    }
    for init_dt in inits:
        init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
        month = init_dt.month
        for s in STATION_IDS:
            for v in VARIABLES:
                raw_list = raw[init_iso][s][v]
                for li, lead in enumerate(LEAD_TIMES):
                    r = raw_list[li]
                    if r is None or not np.isfinite(r):
                        continue
                    valid = init_dt + timedelta(hours=lead)
                    valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
                    ov = obs[s].get(valid_iso)
                    o = ov.get(v) if ov else None
                    if o is None or not np.isfinite(o):
                        continue
                    xs, ys = pairs[s][v][li][month]
                    xs.append(r)
                    ys.append(float(o))

    fold_coef: dict[int, dict] = {}
    for fold_month in MONTHS:
        fc: dict = {s: {v: {} for v in VARIABLES} for s in STATION_IDS}
        for s in STATION_IDS:
            for v in VARIABLES:
                for li, lead in enumerate(LEAD_TIMES):
                    tr_x: list[float] = []
                    tr_y: list[float] = []
                    for m in MONTHS:
                        if m == fold_month:
                            continue
                        xs, ys = pairs[s][v][li][m]
                        tr_x.extend(xs)
                        tr_y.extend(ys)
                    if len(tr_x) < MIN_PAIRS:
                        fc[s][v][str(lead)] = {"a": None, "b": None, "n": len(tr_x)}
                        continue
                    X = np.asarray(tr_x).reshape(-1, 1)
                    y = np.asarray(tr_y)
                    model = LinearRegression().fit(X, y)
                    fc[s][v][str(lead)] = {
                        "a": float(model.intercept_),
                        "b": float(model.coef_[0]),
                        "n": len(tr_x),
                    }
        fold_coef[fold_month] = fc
    return fold_coef


def average_coefficients(fold_coef: dict[int, dict]) -> dict:
    """Average a, b across the 12 folds per station/var/lead (for inspection)."""
    coeffs: dict = {s: {v: {} for v in VARIABLES} for s in STATION_IDS}
    for s in STATION_IDS:
        for v in VARIABLES:
            for lead in LEAD_TIMES:
                a_vals, b_vals, n_vals = [], [], []
                for m in MONTHS:
                    e = fold_coef[m][s][v][str(lead)]
                    if e["a"] is not None:
                        a_vals.append(e["a"])
                        b_vals.append(e["b"])
                        n_vals.append(e["n"])
                if a_vals:
                    coeffs[s][v][str(lead)] = {
                        "a": float(np.mean(a_vals)),
                        "b": float(np.mean(b_vals)),
                        "n_mean": float(np.mean(n_vals)),
                        "folds": len(a_vals),
                    }
                else:
                    coeffs[s][v][str(lead)] = {
                        "a": None, "b": None, "n_mean": 0.0, "folds": 0
                    }
    return coeffs


# --------------------------------------------------------------------------- #
# Score: apply each init's own fold coefficients                               #
# --------------------------------------------------------------------------- #
def build_init_json(
    init_dt: datetime,
    raw: dict,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
    fold_coef: dict[int, dict],
) -> dict:
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    coef = fold_coef[init_dt.month]  # model trained without this month
    locations: dict = {}

    for station_id, coords in STATIONS.items():
        var_metrics = {var: {m: [] for m in METRICS} for var in VARIABLES}
        for li, lead in enumerate(LEAD_TIMES):
            valid = init_dt + timedelta(hours=lead)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            obs_vals = obs[station_id].get(valid_iso)
            for var in VARIABLES:
                c = coef[station_id][var].get(str(lead))
                raw_fcst = raw[init_iso][station_id][var][li]
                obs_val = obs_vals.get(var) if obs_vals else None
                if (
                    c is None
                    or c["a"] is None
                    or raw_fcst is None
                    or not np.isfinite(raw_fcst)
                    or obs_val is None
                    or not np.isfinite(obs_val)
                ):
                    for m in METRICS:
                        var_metrics[var][m].append(None)
                    continue
                corrected = c["a"] + c["b"] * raw_fcst
                clim_val = climatology_lookup(clim[station_id], var, valid)
                pm = point_metrics(corrected, float(obs_val), clim_val)
                for m in METRICS:
                    var_metrics[var][m].append(pm[m])

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
    }


# --------------------------------------------------------------------------- #
# Output: metadata, index, NPZ                                                 #
# --------------------------------------------------------------------------- #
def write_metadata(out_dir: Path, year: int) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": "Linear regression bias correction: corrected = a + b * raw_gfs",
        "training": "leave-one-month-out CV on gfs_interpolated 2025 forecasts",
        "regressions_per_fold": "per (station, variable, lead_time) — 48",
        "folds": 12,
        "raw_forecast": "GFS interpolated 2025; recovered as observation + bias",
        "min_pairs": MIN_PAIRS,
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "year": year,
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, +/-15-day window",
        "coefficients_file": "coefficients.json (fold-averaged a, b per station/var/lead)",
        "npz_file": f"{METHOD_ID}_{year}.npz",
        "npz_schema": "see benchmarking-site/AGENTS.md",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def write_index(out_dir: Path, files: list[str]) -> None:
    (out_dir / "index.json").write_text(
        json.dumps({"files": sorted(files)}, indent=2) + "\n"
    )


def export_npz(out_dir: Path, year: int) -> Path | None:
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


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--cycles", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run folds + score (kept for compatibility; fitting is inline)",
    )
    parser.add_argument("--export-npz-only", action="store_true")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        return

    obs = {sid: load_observations(sid) for sid in STATIONS}
    clim = {sid: build_climatology(obs[sid]) for sid in STATIONS}

    # All inits drive the folds (train pairs come from the full year minus the
    # held-out month), regardless of any scoring date filter.
    all_inits = init_datetimes(args.year)
    print("Recovering raw GFS forecasts (obs + bias) …")
    raw = load_raw_gfs(all_inits, obs)
    print("Fitting 12 leave-one-month-out folds (48 regressions each) …")
    fold_coef = fit_folds(all_inits, raw, obs)

    coeffs = average_coefficients(fold_coef)
    COEFFS_PATH.write_text(
        json.dumps(
            {
                "method_id": METHOD_ID,
                "training": "leave-one-month-out CV on gfs_interpolated 2025",
                "note": "a, b are means across the 12 fold fits per station/var/lead",
                "min_pairs": MIN_PAIRS,
                "coefficients": coeffs,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {COEFFS_PATH}")

    # Scoring set (date filters only restrict which JSON files we write).
    cycle_hours = INIT_HOURS_UTC
    if args.cycles:
        cycle_hours = tuple(int(h.strip()) for h in args.cycles.split(","))
    inits = [d for d in all_inits if d.hour in cycle_hours]
    if args.start_date:
        start = parse_utc(f"{args.start_date}T00:00:00Z")
        inits = [d for d in inits if d >= start]
    if args.end_date:
        end = parse_utc(f"{args.end_date}T18:00:00Z")
        inits = [d for d in inits if d <= end]

    print(f"Scoring {len(inits)} initializations -> {out_dir}")
    written = skipped = 0
    for i, init_dt in enumerate(inits, 1):
        fname = init_filename(init_dt)
        if args.resume and (out_dir / fname).exists():
            skipped += 1
            continue
        payload = build_init_json(init_dt, raw, obs, clim, fold_coef)
        (out_dir / fname).write_text(json.dumps(payload, indent=2) + "\n")
        written += 1
        if written % 100 == 0 or i == len(inits):
            print(f"  [{i}/{len(inits)}] wrote {fname}")

    write_metadata(out_dir, args.year)
    existing = sorted(p.name for p in all_init_json_paths(out_dir))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(
        f"Done. {written} written, {skipped} skipped, "
        f"{len(existing)} total init files; index.json updated."
    )


if __name__ == "__main__":
    main()
