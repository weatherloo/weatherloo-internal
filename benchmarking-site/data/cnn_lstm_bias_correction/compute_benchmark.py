#!/usr/bin/env python3
"""CNN-LSTM HRRR bias-correction benchmark.

Scores the trained CNN-LSTM bias-correction model (``src/hrrr_bias_correction``)
on the 2025 test split and writes per-init JSON + a consolidated NPZ into this
directory, so the model shows up as a method on the benchmarking dashboard.

WHAT THE MODEL PREDICTS
    The network predicts the HRRR-minus-observation **bias** at the station,
    per lead hour, for t2m (degC) and 10 m wind_speed (km/h):

        b_hat = predicted (HRRR - obs)

    The bias-corrected forecast is therefore ``corrected = HRRR - b_hat`` and its
    error against the observation is

        r = corrected - obs = (HRRR - obs) - b_hat = y_true - b_hat

    where ``y_true`` is the bias target already stored in the Zarr store by
    ``make_data.py``. So the corrected-forecast residual is computed directly
    from the stored target and the model prediction -- no HRRR re-interpolation.

METRICS (same per-init convention as every other method here)
    rmse = mae = |r|,  bias = r,  acc = anomaly-sign agreement vs a DOY+hour
    station climatology. The dashboard averages these across inits.

COVERAGE (null by design, not missing data to backfill)
    * Station: the model is trained on **eric_d_soulis** only, so ``cyyz`` is
      emitted with all-null arrays.
    * Leads: the model horizon is f48, so leads 54/60/66/72 h are null -- the
      same limitation documented for ``hrrr_interpolated``.

ACC anomaly baseline: DOY + UTC-hour climatology from 2025 station obs with a
+/-15-day calendar window (same proxy as the other methods so scores stay
comparable).

Flags:
  --model-path   trained Keras model (default: artifacts/.../final_model.keras)
  --zarr-store   Zarr store with the 'test' split (default: config.default.json)
  --split        Zarr split group to score (default: test)
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
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"

# Defaults mirror src/hrrr_bias_correction/config.default.json.
DEFAULT_ZARR_STORE = Path(
    "/mnt/wato-drive/gguirgui/weatherloo-data/hrrr_bias_correction/hrrr"
)
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "hrrr_bias_correction" / "final_model.keras"

INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
# The CNN-LSTM horizon is f48; leads beyond are null by design.
MAX_LEAD_HOURS = 48
METHOD_ID = "cnn_lstm_bias_correction"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
# The model is trained on this station only; the other is null by design.
MODEL_STATION = "eric_d_soulis"
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

# Model target order (make_data.py TARGET_VARS) -> benchmark variable keys.
TARGET_TO_VARIABLE = {0: "t2m", 1: "wind_speed"}


# --------------------------------------------------------------------------- #
# Shared helpers (observations, climatology, metrics) -- same as peer methods  #
# --------------------------------------------------------------------------- #

def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_observations(station_id: str, year: int) -> dict[str, dict[str, float | None]]:
    path = OBS_ROOT / station_id / f"observations_6h_{year}.json"
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
    """DOY + UTC-hour climatology using a calendar-day window within the obs."""
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


def null_variable_block() -> dict:
    return {
        "lead_times_hours": LEAD_TIMES,
        **{m: [None] * len(LEAD_TIMES) for m in METRICS},
    }


# --------------------------------------------------------------------------- #
# Model + Zarr I/O                                                             #
# --------------------------------------------------------------------------- #

def load_test_split(zarr_store: Path, split: str):
    """Return (x, y_true, init_datetimes) for one Zarr split group."""
    import xarray as xr

    ds = xr.open_zarr(zarr_store, group=split, consolidated=False)
    try:
        x = np.nan_to_num(np.asarray(ds["x"].values, dtype=np.float32), nan=0.0)
        y_true = np.asarray(ds["y"].values, dtype=np.float32)  # keep NaN as missing
        init_unix = np.asarray(ds["init_time"].values, dtype=np.int64)
    finally:
        ds.close()

    inits = [
        datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(minute=0, second=0)
        for ts in init_unix
    ]
    return x, y_true, inits


def predict_bias(model_path: Path, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the trained model to get predicted bias, shape (n, n_leads, n_targets)."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(x, batch_size=batch_size, verbose=1)
    return np.asarray(preds, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Per-init JSON construction                                                   #
# --------------------------------------------------------------------------- #

def init_filename(init_dt: datetime) -> str:
    return f"{init_dt.strftime('%Y-%m-%dT%H')}Z.json"


def init_json_paths(out_dir: Path, year: int) -> list[Path]:
    return sorted(out_dir.glob(f"{year}-*T*Z.json"))


def build_variable_block(
    residual_by_lead: dict[int, float],
    obs_station: dict[str, dict[str, float | None]],
    clim_station: dict[str, dict[str, float]],
    init_dt: datetime,
    var: str,
) -> dict:
    """Assemble one variable's metric arrays for the model station."""
    block = {"lead_times_hours": LEAD_TIMES, **{m: [] for m in METRICS}}
    for lead in LEAD_TIMES:
        # Beyond the model horizon or no finite residual -> null everything.
        r = residual_by_lead.get(lead)
        if lead > MAX_LEAD_HOURS or r is None or not np.isfinite(r):
            for m in METRICS:
                block[m].append(None)
            continue

        valid = init_dt + timedelta(hours=lead)
        valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
        obs_vals = obs_station.get(valid_iso)
        obs_val = obs_vals.get(var) if obs_vals else None

        if obs_val is None:
            # No ground truth at this valid time: score residual, skip ACC.
            block["rmse"].append(abs(float(r)))
            block["mae"].append(abs(float(r)))
            block["bias"].append(float(r))
            block["acc"].append(None)
            continue

        # corrected = obs + residual, so point_metrics reproduces |r| / r and a
        # climatology-anomaly ACC consistent with every other method.
        corrected = float(obs_val) + float(r)
        clim_val = climatology_lookup(clim_station, var, valid)
        pm = point_metrics(corrected, float(obs_val), clim_val)
        for m in METRICS:
            block[m].append(pm[m])
    return block


def build_init_json(
    init_dt: datetime,
    residuals: dict[str, dict[int, float]],
    obs_station: dict[str, dict[str, float | None]],
    clim_station: dict[str, dict[str, float]],
) -> dict:
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    locations: dict = {}
    for station_id, coords in STATIONS.items():
        if station_id == MODEL_STATION:
            variables = {
                var: build_variable_block(
                    residuals[var], obs_station, clim_station, init_dt, var
                )
                for var in VARIABLES
            }
        else:
            # Model is not trained for this station.
            variables = {var: null_variable_block() for var in VARIABLES}
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
# NPZ / metadata / index                                                       #
# --------------------------------------------------------------------------- #

def write_index(out_dir: Path, files: list[str]) -> None:
    (out_dir / "index.json").write_text(
        json.dumps({"files": sorted(files)}, indent=2) + "\n"
    )


def write_metadata(out_dir: Path, year: int, model_path: Path, zarr_store: Path) -> None:
    meta = {
        "method_id": METHOD_ID,
        "model": "CNN-LSTM HRRR bias correction (src/hrrr_bias_correction)",
        "predicts": "HRRR-minus-observation bias per lead (t2m degC, wind_speed km/h)",
        "scored_quantity": "bias-corrected forecast = HRRR - predicted_bias",
        "model_path": str(model_path),
        "zarr_store": str(zarr_store),
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "year": year,
        "stations": {
            MODEL_STATION: "scored (model trained here)",
            "cyyz": "null by design (model not trained for this station)",
        },
        "lead_times_hours": LEAD_TIMES,
        "available_lead_times_hours": [h for h in LEAD_TIMES if h <= MAX_LEAD_HOURS],
        "unavailable_lead_times_hours": [h for h in LEAD_TIMES if h > MAX_LEAD_HOURS],
        "lead_time_note": (
            "CNN-LSTM horizon is f48. Leads 54/60/66/72 h are null by design, "
            "the same limitation as hrrr_interpolated."
        ),
        "acc_climatology": "DOY + UTC-hour mean from station obs, +/-15-day window",
        "npz_file": f"{METHOD_ID}_{year}.npz",
        "npz_schema": "see benchmarking-site/AGENTS.md",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


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
                            arrays[metric][init_idx, station_idx, var_idx, lead_idx] = value

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
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--split", type=str, default="test", help="Zarr split group to score")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--zarr-store", type=Path, default=DEFAULT_ZARR_STORE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Skip init files that already exist")
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Rebuild NPZ from existing JSON without running the model",
    )
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year, args.model_path, args.zarr_store)
        return

    if not args.model_path.exists():
        raise SystemExit(
            f"Model not found: {args.model_path}\n"
            "Train the model first (scripts/submit_training_slurm.sh) or pass --model-path."
        )

    obs = load_observations(MODEL_STATION, args.year)
    clim = build_climatology(obs)

    print(f"Loading split '{args.split}' from {args.zarr_store}")
    x, y_true, inits = load_test_split(args.zarr_store, args.split)
    print(f"  {x.shape[0]} samples, x{x.shape}, y{y_true.shape}")

    print(f"Loading model: {args.model_path}")
    b_hat = predict_bias(args.model_path, x, args.batch_size)
    if b_hat.shape != y_true.shape:
        raise SystemExit(
            f"Prediction shape {b_hat.shape} != target shape {y_true.shape}; "
            "check the model matches the data split."
        )

    # residual r = y_true - b_hat = corrected_forecast - obs, per (sample, lead, target)
    residual = y_true - b_hat

    files: list[str] = []
    for i, init_dt in enumerate(inits):
        if init_dt.year != args.year or init_dt.hour not in INIT_HOURS_UTC:
            continue
        fname = init_filename(init_dt)
        out_path = out_dir / fname
        if args.resume and out_path.exists():
            files.append(fname)
            continue

        # residuals[var][lead_hour] = r ; model lead index == forecast hour (0..48)
        residuals: dict[str, dict[int, float]] = {var: {} for var in VARIABLES}
        for target_idx, var in TARGET_TO_VARIABLE.items():
            for lead in LEAD_TIMES:
                if lead > MAX_LEAD_HOURS:
                    continue
                residuals[var][lead] = float(residual[i, lead, target_idx])

        payload = build_init_json(init_dt, residuals, obs, clim)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        files.append(fname)

    write_metadata(out_dir, args.year, args.model_path, args.zarr_store)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files under {out_dir}, index.json updated.")


if __name__ == "__main__":
    main()
