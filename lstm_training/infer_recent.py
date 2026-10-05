"""
Run a trained LSTM bias-correction model on real recent data to compare
predicted vs actual temperature for a date range (e.g. "the past week").

The model predicts the bias (forecast - obs) at a target init from a sliding
window of seq_len earlier biases + time-of-day/day-of-year features (same
feature engineering as train.py / retrain_best.py).

The window stops lead_time hours short of the target init: bias at init I is
only knowable at I + lead_time, so a correction issued at T can only use inits
up to T - lead_time. At a 48h lead that is 8 six-hourly steps of separation,
not the adjacent step. For each target init T:

    corrected_prediction(T) = raw_forecast(T) - predicted_bias(T)

where raw_forecast(T) is reconstructed from the already-known
obs(T's valid time) + actual_bias(T), exactly like linear_regression's trick.

Normalization (bias_mean/bias_std) is NOT persisted anywhere by
retrain_best.py, so it's recomputed here the same deterministic way, from the
same 2025-only NPZ used at training time -- this must match exactly or the
model's predictions are meaningless.

Windows that cross a real data gap (a method whose 2026 backfill only covers
part of the year) are skipped rather than silently fed bogus "history" --
detected by checking the window's time span against what seq_len consecutive
6-hourly steps should span, with slack for a few dropped NaNs.

Usage:
    python lstm_training/infer_recent.py --station eric_d_soulis --variable t2m \
        --method ecmwf_aifs --lead_time 24 \
        --start-date 2026-07-19 --end-date 2026-07-25
"""
import argparse
import glob
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from train import BiasLSTM, build_features, parse_timestamps

LSTM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LSTM_DIR)
DATA_ROOT = os.path.join(REPO_ROOT, "benchmarking-site", "data")

CYCLE_HOURS = 6  # nominal spacing between inits


def load_raw_npz_bias(method, station, variable, lead_time, year):
    path = os.path.join(DATA_ROOT, method, f"{method}_{year}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True)
    si = list(data["station_ids"]).index(station)
    vi = list(data["variables"]).index(variable)
    li = list(data["lead_times_hours"]).index(lead_time)
    bias = data["bias"][:, si, vi, li].astype(np.float32)
    inits = data["initializations"].copy()
    data.close()
    return bias, inits


def load_combined_bias_series(method, station, variable, lead_time):
    """2025+2026 bias series concatenated in time order, NaNs dropped."""
    all_bias, all_inits = [], []
    for year in (2025, 2026):
        bias, inits = load_raw_npz_bias(method, station, variable, lead_time, year)
        if bias is None:
            continue
        all_bias.append(bias)
        all_inits.append(inits)
    bias = np.concatenate(all_bias)
    inits = np.concatenate(all_inits)
    order = np.argsort(inits)
    bias, inits = bias[order], inits[order]
    valid = ~np.isnan(bias)
    return bias[valid], inits[valid]


def training_normalization(method, station, variable, lead_time):
    """Recompute the exact bias_mean/bias_std retrain_best.py used (2025-only)."""
    bias, _ = load_raw_npz_bias(method, station, variable, lead_time, 2025)
    bias = bias[~np.isnan(bias)]
    return float(np.mean(bias)), float(np.std(bias))


def window_is_contiguous(inits_window, seq_len, slack=3.0):
    """True if the window spans roughly seq_len consecutive 6h steps.

    slack allows for a handful of NaN-dropped steps without falsely
    rejecting an otherwise-real, mostly-continuous window; it will NOT
    tolerate a genuine multi-week/month gap (e.g. a method whose 2026
    backfill only covers the most recent week).
    """
    ts = [datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S") for s in inits_window]
    span_hours = (ts[-1] - ts[0]).total_seconds() / 3600.0
    expected_hours = (seq_len - 1) * CYCLE_HOURS
    return span_hours <= expected_hours * slack


def load_observed_and_raw(method, station, variable, lead_time, year):
    """valid_time-keyed obs for `year` and every earlier year on disk.

    The bias series spans 2025+2026, so restricting observations to a single
    year silently drops every init whose valid time falls outside it.
    """
    obs_by_time = {}
    obs_dir = os.path.join(DATA_ROOT, "observations", station)
    for path in sorted(glob.glob(os.path.join(obs_dir, "observations_6h_*.json"))):
        try:
            file_year = int(os.path.basename(path).split("_")[-1].split(".")[0])
        except ValueError:
            continue
        if file_year > year:
            continue
        obs_data = json.loads(open(path).read())
        for row in obs_data["observations"]:
            obs_by_time[row["valid_time"]] = row.get(variable)
    return obs_by_time


def raw_forecast_for_init(method, station, variable, lead_time, init_iso, obs_by_time):
    """Reconstruct raw forecast for one init from its per-init JSON's bias + obs."""
    fname = f"{init_iso[:13]}Z.json"  # e.g. 2026-07-19T00:00:00Z -> 2026-07-19T00Z.json
    path = os.path.join(DATA_ROOT, method, fname)
    if not os.path.exists(path):
        return None, None
    payload = json.load(open(path))
    v = payload["locations"][station]["variables"][variable]
    leads = v["lead_times_hours"]
    li = leads.index(lead_time)
    bias = v["bias"][li]
    if bias is None:
        return None, None
    init_dt = datetime.strptime(init_iso.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    valid_iso = (init_dt + timedelta(hours=lead_time)).strftime("%Y-%m-%dT%H:%M:%SZ")
    obs_val = obs_by_time.get(valid_iso)
    if obs_val is None:
        return None, None
    raw_forecast = obs_val + bias
    return raw_forecast, obs_val


def predict_range(
    station, variable, method, lead_time, start_date, end_date,
    verbose=True, retrain_root=None,
):
    """Corrected vs raw vs observed for every init in [start_date, end_date].

    Returns (results, combo_dir). Importable so the hindcast builder that feeds
    the comparison site does not have to shell out to this script.

    `retrain_root` overrides where trained combos are read from, for building
    against an alternate model set without disturbing retrain_output/.
    """
    args = argparse.Namespace(
        station=station, variable=variable, method=method, lead_time=lead_time,
        start_date=start_date, end_date=end_date, retrain_root=retrain_root,
    )
    return _run(args, verbose=verbose)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", required=True)
    p.add_argument("--variable", default="t2m")
    p.add_argument("--method", required=True)
    p.add_argument("--lead_time", type=int, required=True)
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--out", default=None, help="Output JSON path")
    p.add_argument("--retrain-dir", dest="retrain_root", default=None,
                   help="Alternate retrain_output root")
    args = p.parse_args()

    results, combo_dir = _run(args)

    out_path = args.out or os.path.join(
        combo_dir, f"recent_{args.start_date}_{args.end_date}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


def _run(args, verbose=True):
    retrain_root = getattr(args, "retrain_root", None) or os.path.join(
        LSTM_DIR, "retrain_output"
    )
    combo_dir = os.path.join(
        retrain_root, f"{args.station}_{args.variable}",
        f"{args.method}_{args.lead_time}h",
    )
    metrics = json.load(open(os.path.join(combo_dir, "metrics.json")))
    params = metrics["best_params"]
    seq_len = params["seq_len"]

    bias_mean, bias_std = training_normalization(
        args.method, args.station, args.variable, args.lead_time
    )
    if verbose:
        print(f"Training normalization: mean={bias_mean:.4f} std={bias_std:.4f}")

    bias_series, inits_series = load_combined_bias_series(
        args.method, args.station, args.variable, args.lead_time
    )
    bias_norm = ((bias_series - bias_mean) / bias_std).astype(np.float32)
    hours, doys = parse_timestamps(inits_series)
    features = build_features(bias_norm, hours, doys)

    device = torch.device("cpu")
    model = BiasLSTM(
        input_size=5,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(combo_dir, "best_model.pt"), map_location=device, weights_only=True)
    )
    model.eval()

    obs_by_time = load_observed_and_raw(
        args.method, args.station, args.variable, args.lead_time, 2026
    )

    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    inits_dt = [
        datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        for s in inits_series
    ]

    # bias at init I is only observable at I + lead_time, so the window feeding a
    # correction issued at target init T must end at the latest init <= T - lead_time.
    init_hours = np.array([dt.timestamp() / 3600.0 for dt in inits_dt], dtype=np.float64)

    results = []
    skipped_gap = 0
    skipped_short = 0
    for target_idx in range(len(inits_series)):
        target_dt = inits_dt[target_idx]
        if not (start <= target_dt <= end):
            continue

        end_idx = int(
            np.searchsorted(init_hours, init_hours[target_idx] - args.lead_time, side="right")
        ) - 1
        start_idx = end_idx - seq_len + 1
        if start_idx < 0:
            skipped_short += 1
            continue

        window_inits = inits_series[start_idx : end_idx + 1]
        if not window_is_contiguous(window_inits, seq_len):
            skipped_gap += 1
            continue

        window_feats = features[start_idx : end_idx + 1]
        x = torch.from_numpy(window_feats).unsqueeze(0)  # (1, seq_len, 5)
        with torch.no_grad():
            pred_bias_norm = model(x).item()
        pred_bias = pred_bias_norm * bias_std + bias_mean

        target_init_iso = str(inits_series[target_idx])
        raw_forecast, obs_val = raw_forecast_for_init(
            args.method, args.station, args.variable, args.lead_time,
            target_init_iso, obs_by_time,
        )
        if raw_forecast is None:
            continue

        corrected = raw_forecast - pred_bias
        valid_iso = (
            target_dt + timedelta(hours=args.lead_time)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        results.append({
            "init": target_init_iso,
            "valid_time": valid_iso,
            "actual": obs_val,
            "raw_forecast": raw_forecast,
            "lstm_corrected": corrected,
        })

    if verbose:
        print(
            f"{len(results)} predictions in range, {skipped_gap} skipped (data gap in window), "
            f"{skipped_short} skipped (insufficient history before the {args.lead_time}h cutoff)"
        )

        if results:
            errs_raw = [r["raw_forecast"] - r["actual"] for r in results]
            errs_lstm = [r["lstm_corrected"] - r["actual"] for r in results]
            rmse_raw = float(np.sqrt(np.mean(np.square(errs_raw))))
            rmse_lstm = float(np.sqrt(np.mean(np.square(errs_lstm))))
            print(f"raw RMSE: {rmse_raw:.4f}  LSTM-corrected RMSE: {rmse_lstm:.4f}")

    return results, combo_dir


if __name__ == "__main__":
    main()
