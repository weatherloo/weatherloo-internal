"""
Combine all retrain_output/<station>_<variable>/<method>_<lead>h/predictions.npz
files into one CSV: method, lead_time, timestamp, predicted, actual.

timestamp is the actual init timestamp (ISO-8601) each test-set prediction
corresponds to, so predictions from different methods can be joined on
calendar time.

Usage:
    python lstm_training/combine_predictions.py --station cyyz --variable t2m
    python lstm_training/combine_predictions.py --station eric_d_soulis --variable t2m
"""
import argparse
import csv
import os
import re

import numpy as np

LSTM_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_RE = re.compile(r"^(?P<method>[a-z_]+)_(?P<lead>\d+)h$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="cyyz")
    p.add_argument("--variable", default="t2m")
    args = p.parse_args()

    combo_key = f"{args.station}_{args.variable}"
    retrain_dir = os.path.join(LSTM_DIR, "retrain_output", combo_key)
    out_csv = os.path.join(retrain_dir, "combined_predictions.csv")

    rows = []
    for combo_dir in sorted(os.listdir(retrain_dir)):
        m = DIR_RE.match(combo_dir)
        if not m:
            continue
        method = m.group("method")
        lead = int(m.group("lead"))

        npz_path = os.path.join(retrain_dir, combo_dir, "predictions.npz")
        if not os.path.exists(npz_path):
            print(f"WARNING: missing {npz_path}, skipping")
            continue

        data = np.load(npz_path, allow_pickle=True)
        preds = data["predictions"]
        targets = data["targets"]
        timestamps = data["timestamps"]
        assert len(preds) == len(targets) == len(timestamps), (
            f"{combo_dir}: predictions/targets/timestamps length mismatch"
        )

        for ts, pred, actual in zip(timestamps, preds, targets):
            rows.append((method, lead, str(ts), float(pred), float(actual)))

        print(f"{method}_{lead}h: {len(preds)} rows")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "lead_time", "timestamp", "predicted", "actual"])
        writer.writerows(rows)

    n_combos = len({(m, l) for m, l, *_ in rows})
    print(f"\nWrote {len(rows)} rows from {n_combos} combos to {out_csv}")


if __name__ == "__main__":
    main()
