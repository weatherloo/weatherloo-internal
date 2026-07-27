#!/usr/bin/env python3
"""Merge the per-model runs into the single document the site consumes.

Inputs are whatever ``scripts/showdown/`` has written under
``benchmarking-site/data/showdown/`` for one init: ground truth, the three
in-repo ML methods, and the raw NWP baselines.

Scoring is done on the **common grid** -- the leads where every scored method
has a value (the 6-hourly grid, since only the CNN-LSTM is hourly). Ranking
methods on their own native grids would reward the hourly model for being
graded on different points, so the leaderboard uses the shared subset and the
hourly detail is carried alongside for display.

    python scripts/showdown/combine.py --init 2026-07-20T00Z
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOWDOWN = ROOT / "benchmarking-site" / "data" / "showdown"

# (file stem, group, blurb). Order sets the default display order.
SOURCES = [
    ("unet",                  "ours",      "Residual U-Net post-processing a GFS 0.25 deg grid, bilinear to the station."),
    ("lstm_ecmwf_aifs",       "ours",      "Per-lead LSTM correcting ECMWF AIFS bias from a sliding window of past errors."),
    ("cnn_lstm",              "ours",      "CNN-LSTM reading a 30x30 HRRR crop and predicting per-lead station bias."),
    ("baseline_ecmwf_aifs",   "reference", "ECMWF AIFS as issued, no correction."),
    ("baseline_gfs_analysis", "reference", "NOAA GFS as issued, no correction."),
    ("baseline_hrrr_interpolated", "reference", "NOAA HRRR interpolated to the station, no correction."),
]


def parse_init(text: str) -> datetime:
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            cleaned = text.replace("Z", "+0000") if "Z" in text else text
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognised init: {text!r}")


def rmse(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return math.sqrt(sum((f - a) ** 2 for f, a in pairs) / len(pairs))


def mae(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return sum(abs(f - a) for f, a in pairs) / len(pairs)


def bias(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    return sum(f - a for f, a in pairs) / len(pairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="2026-07-20T00Z")
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    init = parse_init(args.init)
    stamp = f"{init:%Y%m%dT%HZ}"

    truth_doc = json.loads((SHOWDOWN / f"truth_{stamp}.json").read_text())
    truth = {s["lead_hours"]: s["value"] for s in truth_doc["steps"]}

    methods: list[dict] = []
    for stem, group, blurb in SOURCES:
        path = SHOWDOWN / f"{stem}_{stamp}.json"
        if not path.exists():
            print(f"  {stem}: absent -> skipped")
            continue
        doc = json.loads(path.read_text())
        steps = {s["lead_hours"]: s for s in doc["steps"]}
        methods.append({
            "id": doc["model"] if group == "ours" else doc["model"],
            "key": stem,
            "label": doc["model_label"],
            "group": group,
            "blurb": blurb,
            "base_source": doc.get("base_source"),
            "cadence_hours": doc.get("cadence_hours"),
            "provenance": doc.get("provenance", {}),
            "_steps": steps,
        })

    # Common grid: leads where truth and every scored method have a value.
    scored = [m for m in methods]
    candidate_leads = sorted(
        L for L in truth
        if truth[L] is not None and 0 < L <= args.horizon
    )
    common = [
        L for L in candidate_leads
        if all((m["_steps"].get(L) or {}).get("value") is not None for m in scored)
    ]

    # Per-method series + scores.
    for m in scored:
        series = []
        for L in range(0, args.horizon + 1):
            st = m["_steps"].get(L)
            value = (st or {}).get("value")
            actual = truth.get(L)
            delta = None if value is None or actual is None else round(value - actual, 3)
            series.append({
                "lead_hours": L,
                "valid_time": (init + timedelta(hours=L)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": None if value is None else round(float(value), 3),
                "actual": actual,
                "delta": delta,
                "raw_source_value": (st or {}).get("raw_source_value"),
            })
        m["series"] = series
        pairs = [(m["_steps"][L]["value"], truth[L]) for L in common]
        m["scores"] = {
            "grid": "common_6h",
            "n": len(pairs),
            "rmse": None if rmse(pairs) is None else round(rmse(pairs), 4),
            "mae": None if mae(pairs) is None else round(mae(pairs), 4),
            "bias": None if bias(pairs) is None else round(bias(pairs), 4),
        }
        native = [
            (s["value"], truth[s["lead_hours"]])
            for s in m["series"]
            if s["value"] is not None and truth.get(s["lead_hours"]) is not None
            and s["lead_hours"] > 0
        ]
        m["scores_native"] = {
            "grid": f"native_{m['cadence_hours']}h",
            "n": len(native),
            "rmse": None if rmse(native) is None else round(rmse(native), 4),
            "mae": None if mae(native) is None else round(mae(native), 4),
        }
        del m["_steps"]

    # Overall ranking on the common grid (lower RMSE first).
    ranked = sorted([m for m in scored if m["scores"]["rmse"] is not None],
                    key=lambda m: m["scores"]["rmse"])
    for i, m in enumerate(ranked, 1):
        m["rank"] = i
    for m in scored:
        m.setdefault("rank", None)

    # Per-lead ranking, so the drawer can rank methods for one hour.
    per_lead: dict[str, list[dict]] = {}
    for L in range(0, args.horizon + 1):
        entries = []
        for m in scored:
            s = next(x for x in m["series"] if x["lead_hours"] == L)
            if s["value"] is None or s["delta"] is None:
                continue
            entries.append({"method": m["key"], "value": s["value"],
                            "delta": s["delta"], "abs_error": round(abs(s["delta"]), 3)})
        entries.sort(key=lambda e: e["abs_error"])
        for i, e in enumerate(entries, 1):
            e["rank"] = i
        per_lead[str(L)] = entries

    doc = {
        "schema_version": "1",
        "title": "48-hour forecast showdown",
        "station": {
            "id": truth_doc["station_id"],
            "name": "Eric D. Soulis station",
            "lat": 43.4668,
            "lon": -80.5164,
            "place": "University of Waterloo, ON",
        },
        "variable": "t2m",
        "units": "degC",
        "init": truth_doc["init"],
        "horizon_hours": args.horizon,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scoring": {
            "common_grid_leads": common,
            "note": "RMSE/MAE for the leaderboard use the leads every method has, so "
                    "the hourly CNN-LSTM is not scored on different points than the "
                    "6-hourly models. Each method's native-grid score is also given.",
        },
        "truth": {
            "cadence_hours": 1,
            "source": truth_doc["source"],
            "time_note": truth_doc["time_note"],
            "steps": truth_doc["steps"],
        },
        "methods": scored,
        "ranking": [m["key"] for m in ranked],
        "per_lead_ranking": per_lead,
        "unavailable": [
            {
                "id": "open_meteo",
                "label": "Open-Meteo",
                "group": "reference",
                "reason": "api.open-meteo.com, previous-runs-api.open-meteo.com and "
                          "archive-api.open-meteo.com are all refused by this "
                          "environment's egress policy (403 on CONNECT), and no "
                          "Open-Meteo response is cached in the repo.",
                "unblock": "Allowlist api.open-meteo.com, then run "
                           "scripts/fetch_openmeteo_previous.py --lead 24/48 and re-run this combiner.",
            }
        ],
    }

    out = Path(args.out) if args.out else SHOWDOWN / f"showdown_{stamp}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {out}")
    print(f"  common grid: {common}")
    for m in ranked:
        print(f"  {m['rank']}. {m['label']:<12} RMSE {m['scores']['rmse']:.3f}  MAE {m['scores']['mae']:.3f}  ({m['group']})")


if __name__ == "__main__":
    main()
