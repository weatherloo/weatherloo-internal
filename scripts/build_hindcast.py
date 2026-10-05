#!/usr/bin/env python3
"""Assemble the retrospective comparison dataset for the hindcast site.

For each initialization in a date range, joins four series on valid time:

    reference    what a public forecast said (Open-Meteo Previous Runs)
    raw_model    the in-repo NWP forecast for the same init/lead
    ours         that NWP forecast after LSTM bias correction
    actual       the station observation that verified it

Output is one JSON per station/variable/lead under
``benchmarking-site/data/hindcast/``, served to the frontend at
``/data/hindcast/...``.

The reference series is optional: if the Open-Meteo file for this
station/lead has not been fetched, records are still written with
``reference: null`` so the rest of the comparison remains usable.

    python scripts/fetch_openmeteo_previous.py --lead 48 --start ... --end ...
    python scripts/build_hindcast.py --lead 48 --start 2026-01-01 --end 2026-07-25

Note on provenance: ``raw_model`` is reconstructed as ``actual + bias``, since
the benchmark files store error metrics rather than forecast values. That
identity is exact, but it does mean an init whose verifying observation is
missing cannot contribute a row at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarking-site" / "data"
OUT_ROOT = DATA / "hindcast"

sys.path.insert(0, str(ROOT / "lstm_training"))

DEFAULT_METHOD = "ecmwf_aifs"


def load_reference(station: str, lead: int) -> dict[str, dict]:
    """valid_time -> record, from whichever Open-Meteo file covers this lead."""
    ref_dir = DATA / "openmeteo_previous" / station
    if not ref_dir.is_dir():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(ref_dir.glob(f"previous_{lead}h_*.json")):
        doc = json.loads(path.read_text())
        for rec in doc.get("records", []):
            out[rec["valid_time"]] = rec
    return out


def finite(x):
    """JSON-safe: NaN/Inf become null rather than invalid JSON literals."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--variable", default="t2m")
    p.add_argument("--method", default=DEFAULT_METHOD, help="NWP method to correct")
    p.add_argument("--lead", type=int, default=48)
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--retrain-dir", dest="retrain_root", default=None,
                   help="Alternate retrain_output root")
    args = p.parse_args()

    from infer_recent import predict_range  # imported late; needs torch

    preds, _ = predict_range(
        args.station, args.variable, args.method, args.lead,
        args.start, args.end, verbose=True, retrain_root=args.retrain_root,
    )

    reference = load_reference(args.station, args.lead)
    if not reference:
        print(
            f"[warn] no Open-Meteo reference for {args.station} at {args.lead}h -- "
            f"writing reference: null (run scripts/fetch_openmeteo_previous.py first)"
        )

    ref_key = "t2m" if args.variable == "t2m" else "wind_speed"
    records = []
    for r in preds:
        ref = reference.get(r["valid_time"])
        records.append(
            {
                "init": r["init"],
                "valid_time": r["valid_time"],
                "reference": finite(ref.get(ref_key)) if ref else None,
                "raw_model": finite(r["raw_forecast"]),
                "ours": finite(r["lstm_corrected"]),
                "actual": finite(r["actual"]),
            }
        )

    def rmse(key):
        errs = [
            (rec[key] - rec["actual"]) ** 2
            for rec in records
            if rec[key] is not None and rec["actual"] is not None
        ]
        return math.sqrt(sum(errs) / len(errs)) if errs else None

    doc = {
        "schema_version": "1",
        "station_id": args.station,
        "variable": args.variable,
        "method": args.method,
        "lead_hours": args.lead,
        "period": {"start": args.start, "end": args.end},
        "units": "degC" if args.variable == "t2m" else "km/h",
        "series": {
            "reference": "Open-Meteo Previous Runs (public forecast at this lead)",
            "raw_model": f"{args.method} forecast, reconstructed as actual + bias",
            "ours": f"{args.method} after LSTM bias correction",
            "actual": "station observation",
        },
        "summary_rmse": {
            "reference": rmse("reference"),
            "raw_model": rmse("raw_model"),
            "ours": rmse("ours"),
        },
        "n_records": len(records),
        "records": records,
    }

    out = OUT_ROOT / args.station / f"{args.variable}_{args.method}_{args.lead}h.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {out} ({len(records)} records)")
    for k, v in doc["summary_rmse"].items():
        print(f"  RMSE {k:10s} {'n/a' if v is None else f'{v:.4f}'}")


if __name__ == "__main__":
    main()
