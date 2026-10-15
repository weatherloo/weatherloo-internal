"""
For each lead time, find which method's LSTM correction has the lowest RMSE
over a real recent date range, using every method that has both trained
weights and real data for that range. Writes one JSON with the winning
method + its raw/LSTM RMSE per lead, for charting.

Usage:
    python lstm_training/best_per_lead.py --station eric_d_soulis --variable t2m \
        --start-date 2026-07-19 --end-date 2026-07-25
"""
import argparse
import json
import os

import numpy as np
import torch

from infer_recent import (
    DATA_ROOT,
    LSTM_DIR,
    load_combined_bias_series,
    load_observed_and_raw,
    raw_forecast_for_init,
    training_normalization,
    window_is_contiguous,
)
from train import BiasLSTM, build_features, parse_timestamps

LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHODS = [
    "climatology", "persistence", "ecmwf_aifs", "gfs_analysis",
    "gefs_mean", "hrrr_interpolated", "gfs_interpolated",
]


def evaluate(station, variable, method, lead_time, start_date, end_date):
    combo_dir = os.path.join(
        LSTM_DIR, "retrain_output", f"{station}_{variable}", f"{method}_{lead_time}h"
    )
    metrics_path = os.path.join(combo_dir, "metrics.json")
    model_path = os.path.join(combo_dir, "best_model.pt")
    if not (os.path.exists(metrics_path) and os.path.exists(model_path)):
        return None

    metrics = json.load(open(metrics_path))
    params = metrics["best_params"]
    seq_len = params["seq_len"]

    bias_mean, bias_std = training_normalization(method, station, variable, lead_time)
    bias_series, inits_series = load_combined_bias_series(method, station, variable, lead_time)
    if len(bias_series) <= seq_len:
        return None
    bias_norm = ((bias_series - bias_mean) / bias_std).astype(np.float32)
    hours, doys = parse_timestamps(inits_series)
    features = build_features(bias_norm, hours, doys)

    device = torch.device("cpu")
    model = BiasLSTM(
        input_size=5, hidden_size=params["hidden_size"],
        num_layers=params["num_layers"], dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    obs_by_time = load_observed_and_raw(method, station, variable, lead_time, 2026)

    from datetime import datetime, timezone
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    inits_dt = [
        datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        for s in inits_series
    ]

    raws, actuals, corrected = [], [], []
    for target_idx in range(seq_len, len(inits_series)):
        target_dt = inits_dt[target_idx]
        if not (start <= target_dt <= end):
            continue
        window_inits = inits_series[target_idx - seq_len : target_idx]
        if not window_is_contiguous(window_inits, seq_len):
            continue
        window_feats = features[target_idx - seq_len : target_idx]
        x = torch.from_numpy(window_feats).unsqueeze(0)
        with torch.no_grad():
            pred_bias_norm = model(x).item()
        pred_bias = pred_bias_norm * bias_std + bias_mean

        target_init_iso = str(inits_series[target_idx])
        raw_forecast, obs_val = raw_forecast_for_init(
            method, station, variable, lead_time, target_init_iso, obs_by_time
        )
        if raw_forecast is None:
            continue
        raws.append(raw_forecast)
        actuals.append(obs_val)
        corrected.append(raw_forecast - pred_bias)

    if not raws:
        return None

    raws, actuals, corrected = np.array(raws), np.array(actuals), np.array(corrected)
    rmse_raw = float(np.sqrt(np.mean((raws - actuals) ** 2)))
    rmse_lstm = float(np.sqrt(np.mean((corrected - actuals) ** 2)))
    return {"n": len(raws), "rmse_raw": rmse_raw, "rmse_lstm": rmse_lstm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", required=True)
    p.add_argument("--variable", default="t2m")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--min-samples", type=int, default=10,
        help="Minimum valid predictions required for a method to be eligible "
             "at a given lead time (guards against a data-gap-starved method "
             "\"winning\" on 1-2 lucky points)",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results_by_lead = {}
    for lead in LEAD_TIMES:
        best = None
        for method in METHODS:
            r = evaluate(args.station, args.variable, method, lead, args.start_date, args.end_date)
            if r is None:
                continue
            eligible = r["n"] >= args.min_samples
            flag = "" if eligible else f"  SKIPPED (n < {args.min_samples})"
            print(f"  lead={lead:>2}h method={method:<18} n={r['n']:>2} "
                  f"raw={r['rmse_raw']:.4f} lstm={r['rmse_lstm']:.4f}{flag}")
            if not eligible:
                continue
            if best is None or r["rmse_lstm"] < best["rmse_lstm"]:
                best = {**r, "method": method, "lead_time": lead}
        if best:
            results_by_lead[lead] = best
            print(f"lead={lead:>2}h  WINNER: {best['method']}  "
                  f"raw={best['rmse_raw']:.4f}  lstm={best['rmse_lstm']:.4f}\n")
        else:
            print(f"lead={lead:>2}h  no usable method\n")

    out_path = args.out or os.path.join(
        LSTM_DIR, f"best_per_lead_{args.station}_{args.variable}_{args.start_date}_{args.end_date}.json"
    )
    with open(out_path, "w") as f:
        json.dump(list(results_by_lead.values()), f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
