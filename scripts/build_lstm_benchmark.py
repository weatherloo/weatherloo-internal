#!/usr/bin/env python3
"""Score the LSTM bias-corrected forecasts as a benchmark method.

The trained LSTMs in lstm_training/retrain_output/ correct one NWP source at
one lead time each. Nothing on the benchmarking site read them, so their skill
was never visible next to the raw models they exist to improve. This writes
them out in the same npz layout every other method uses, so the site can plot
"<base> corrected by the LSTM" beside plain "<base>".

The error identity
------------------
    corrected(T)   = raw_forecast(T) - predicted_bias(T)
    raw_forecast(T) = obs(T + L) + actual_bias(T)

    corrected(T) - obs(T + L) = actual_bias(T) - predicted_bias(T)

so the corrected method's per-init bias is exactly the residual left over after
the LSTM's prediction. Matching every other method, each entry is one
forecast/observation pair, so rmse = mae = |bias| per init and acc is left NaN.
Pooling happens downstream in build_site_aggregates.py.

Coverage is whatever has a trained model: t2m only, all 12 leads at
eric_d_soulis, 6h and 12h at cyyz. Everything else stays NaN and simply does
not plot.

    python scripts/build_lstm_benchmark.py --base ecmwf_aifs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarking-site" / "data"
RETRAIN = ROOT / "lstm_training" / "retrain_output"

sys.path.insert(0, str(ROOT / "lstm_training"))

STATIONS = ["cyyz", "eric_d_soulis"]
VARIABLES = ["t2m", "wind_speed"]
LEADS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]


def trained_combos(base: str) -> list[tuple[str, str, int]]:
    """(station, variable, lead) triples that actually have a trained model."""
    out = []
    for station in STATIONS:
        for variable in VARIABLES:
            combo_root = RETRAIN / f"{station}_{variable}"
            if not combo_root.is_dir():
                continue
            for lead in LEADS:
                d = combo_root / f"{base}_{lead}h"
                if (d / "best_model.pt").exists() and (d / "metrics.json").exists():
                    out.append((station, variable, lead))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="ecmwf_aifs", help="NWP source the LSTM corrects")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-12-31")
    args = p.parse_args()

    method_id = f"lstm_{args.base}"
    combos = trained_combos(args.base)
    if not combos:
        raise SystemExit(f"no trained models for base {args.base!r} under {RETRAIN}")
    print(f"{method_id}: {len(combos)} trained combos")

    from infer_recent import predict_range

    # init -> (station, variable, lead) -> corrected-minus-observed
    residual: dict[str, dict[tuple[str, str, int], float]] = defaultdict(dict)
    for station, variable, lead in combos:
        try:
            preds, _ = predict_range(
                station, variable, args.base, lead, args.start, args.end, verbose=False
            )
        except Exception as e:  # a combo with unusable history should not sink the run
            print(f"  [skip] {station}/{variable}/{lead}h: {type(e).__name__}: {e}")
            continue
        for r in preds:
            if r["actual"] is None or r["lstm_corrected"] is None:
                continue
            residual[r["init"]][(station, variable, lead)] = (
                r["lstm_corrected"] - r["actual"]
            )
        print(f"  {station}/{variable}/{lead:2d}h: {len(preds)} inits")

    if not residual:
        raise SystemExit("no predictions produced; nothing to write")

    inits_all = sorted(residual)
    by_year: dict[int, list[str]] = defaultdict(list)
    for iso in inits_all:
        by_year[int(iso[:4])].append(iso)

    out_dir = DATA / method_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for year, inits in sorted(by_year.items()):
        shape = (len(inits), len(STATIONS), len(VARIABLES), len(LEADS))
        bias = np.full(shape, np.nan)
        for ti, iso in enumerate(inits):
            for (station, variable, lead), val in residual[iso].items():
                bias[ti, STATIONS.index(station), VARIABLES.index(variable),
                     LEADS.index(lead)] = val

        np.savez_compressed(
            out_dir / f"{method_id}_{year}.npz",
            method_id=np.array(method_id),
            station_ids=np.array(STATIONS),
            variables=np.array(VARIABLES),
            metrics=np.array(["rmse", "mae", "bias", "acc"]),
            lead_times_hours=np.array(LEADS, dtype=np.int16),
            initializations=np.array(inits),
            bias=bias,
            rmse=np.abs(bias),
            mae=np.abs(bias),
            acc=np.full(shape, np.nan),
        )
        n = int(np.sum(~np.isnan(bias)))
        print(f"wrote {method_id}_{year}.npz  {len(inits)} inits, {n} scored cells")

    meta = {
        "method_id": method_id,
        "base_method": args.base,
        "description": (
            f"{args.base} after LSTM bias correction. Per-init value is "
            "actual_bias - predicted_bias, i.e. the error remaining after the "
            "correction."
        ),
        "coverage": "t2m only; leads limited to combos with a trained model",
        "generated_from": str(RETRAIN.relative_to(ROOT)),
    }
    with (out_dir / "metadata.json").open("w", newline="") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
