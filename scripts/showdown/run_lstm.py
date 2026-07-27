#!/usr/bin/env python3
"""LSTM bias-correction forecast for a single initialization.

Runs ``lstm_training.infer_recent.predict_range`` once per lead and keeps only
the record belonging to ``--init``, giving one forecast trajectory rather than
the time series across inits that ``infer_recent`` normally produces.

Each lead has its own trained model under ``lstm_training/retrain_output``, so
leads with no checkpoint are reported as missing instead of being interpolated
from their neighbours.

    python scripts/showdown/run_lstm.py --init 2026-07-20T00Z --method ecmwf_aifs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lstm_training"))

DEFAULT_LEADS = [6, 12, 18, 24, 30, 36, 42, 48]


def parse_init(text: str) -> datetime:
    """Accept 2026-07-20T00Z, 2026-07-20T00:00:00Z, or 2026-07-20."""
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            cleaned = text.replace("Z", "+0000") if "Z" in text else text
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognised init: {text!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="2026-07-20T00Z")
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--variable", default="t2m")
    p.add_argument("--method", default="ecmwf_aifs", help="NWP source the LSTM corrects")
    p.add_argument("--leads", default=",".join(str(x) for x in DEFAULT_LEADS))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    init = parse_init(args.init)
    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    day = init.strftime("%Y-%m-%d")

    from infer_recent import predict_range

    want = init.strftime("%Y-%m-%dT%H:%M:%SZ")
    steps: list[dict] = []

    for lead in leads:
        combo = ROOT / "lstm_training" / "retrain_output" / f"{args.station}_{args.variable}" / f"{args.method}_{lead}h"
        if not (combo / "best_model.pt").exists():
            print(f"  lead {lead:>3}h  no checkpoint -> skipped", file=sys.stderr)
            steps.append({"lead_hours": lead, "value": None, "reason": "no_checkpoint"})
            continue
        try:
            results, _ = predict_range(
                args.station, args.variable, args.method, lead, day, day, verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - one bad lead must not sink the run
            print(f"  lead {lead:>3}h  failed: {exc}", file=sys.stderr)
            steps.append({"lead_hours": lead, "value": None, "reason": "inference_failed"})
            continue

        rec = next((r for r in results if r.get("init") == want), None)
        if rec is None:
            print(f"  lead {lead:>3}h  init not in output -> skipped", file=sys.stderr)
            steps.append({"lead_hours": lead, "value": None, "reason": "init_missing"})
            continue

        steps.append({
            "lead_hours": lead,
            "valid_time": rec["valid_time"],
            "value": rec["lstm_corrected"],
            "raw_source_value": rec["raw_forecast"],
            "actual": rec["actual"],
        })
        print(f"  lead {lead:>3}h  {rec['lstm_corrected']:.2f} degC "
              f"(raw {rec['raw_forecast']:.2f}, actual {rec['actual']:.2f})", file=sys.stderr)

    doc = {
        "schema_version": "1",
        "model": "lstm",
        "model_label": "LSTM",
        "base_source": args.method,
        "station_id": args.station,
        "variable": args.variable,
        "units": "degC",
        "init": want,
        "cadence_hours": 6,
        "provenance": {
            "code": "lstm_training/infer_recent.py",
            "checkpoints": f"lstm_training/retrain_output/{args.station}_{args.variable}/{args.method}_<lead>h/best_model.pt",
            "note": "One trained model per lead; leads without a checkpoint are null.",
        },
        "steps": steps,
    }

    out = Path(args.out) if args.out else ROOT / "benchmarking-site" / "data" / "showdown" / f"lstm_{args.method}_{init:%Y%m%dT%HZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    got = sum(1 for s in steps if s.get("value") is not None)
    print(f"Wrote {out} ({got}/{len(steps)} leads)", file=sys.stderr)


if __name__ == "__main__":
    main()
