#!/usr/bin/env python3
"""Ground truth and raw-NWP baselines for a single initialization.

Two outputs, both for the 48 h window following ``--init``:

``truth``
    Hourly station temperature, resampled from the raw 15-minute HOBO archive
    under ``data/observations/<station>/raw/``. The archive clock is local
    standard time (UTC-5, no DST) per the station metadata, so timestamps are
    shifted to UTC here. An hour is filled only from a sample within
    ``--tolerance-min`` of the exact instant; nothing is interpolated across
    gaps.

``baselines``
    Raw NWP forecasts at the 6-hourly lead grid. The per-init benchmark files
    store *error* against the verifying observation rather than the forecast
    value, so each forecast is reconstructed as ``obs_6h + bias`` -- the same
    identity ``scripts/build_hindcast.py`` uses. The 6-hourly observation file
    is the one the bias was computed against, so it, not the hourly resample,
    has to be the addend.

    python scripts/showdown/build_truth_and_baselines.py --init 2026-07-20T00Z
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarking-site" / "data"

# Station archive clock: local standard time, no daylight saving.
STATION_UTC_OFFSET_HOURS = 5
MISSING_SENTINEL = -999.0

BASELINES = {
    "ecmwf_aifs": "ECMWF AIFS",
    "gfs_analysis": "GFS",
    "hrrr_interpolated": "HRRR",
}
LEADS = [6, 12, 18, 24, 30, 36, 42, 48]


def parse_init(text: str) -> datetime:
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            cleaned = text.replace("Z", "+0000") if "Z" in text else text
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognised init: {text!r}")


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_raw_15min(station: str, years: set[int]) -> dict[datetime, float]:
    """UTC instant -> temperature, from every raw archive covering `years`."""
    out: dict[datetime, float] = {}
    raw_dir = DATA / "observations" / station / "raw"
    for year in sorted(years):
        for path in sorted(raw_dir.glob(f"*_15min_{year}.csv")):
            with path.open() as fh:
                reader = csv.reader(fh)
                header = [c.strip() for c in next(reader)]
                try:
                    ti = header.index("Temperature")
                except ValueError:
                    continue
                for row in reader:
                    if len(row) <= ti:
                        continue
                    try:
                        value = float(row[ti])
                        if value <= MISSING_SENTINEL:
                            continue
                        local = datetime(
                            int(row[0]), int(row[1]), int(row[2]),
                            int(row[3]), int(row[4]), tzinfo=timezone.utc,
                        )
                    except (ValueError, IndexError):
                        continue
                    # Archive clock is UTC-5; shift forward to get true UTC.
                    out[local + timedelta(hours=STATION_UTC_OFFSET_HOURS)] = value
    return out


def hourly_truth(samples: dict[datetime, float], start: datetime, hours: int,
                 tolerance_min: int) -> list[dict]:
    """One row per hour; value only when a sample lands close enough."""
    tol = timedelta(minutes=tolerance_min)
    rows = []
    for h in range(hours + 1):
        want = start + timedelta(hours=h)
        best, best_gap = None, None
        for offset in range(-tolerance_min, tolerance_min + 1, 15):
            cand = want + timedelta(minutes=offset)
            if cand in samples:
                gap = abs(timedelta(minutes=offset))
                if best_gap is None or gap < best_gap:
                    best, best_gap = samples[cand], gap
        rows.append({
            "lead_hours": h,
            "valid_time": iso(want),
            "value": round(best, 3) if best is not None else None,
        })
    return rows


def load_obs_6h(station: str, years: set[int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for year in sorted(years):
        path = DATA / "observations" / station / f"observations_6h_{year}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for rec in doc.get("observations", []):
            out[rec["valid_time"]] = rec.get("t2m")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="2026-07-20T00Z")
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--tolerance-min", type=int, default=30)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    init = parse_init(args.init)
    end = init + timedelta(hours=args.horizon)
    years = {init.year, end.year}
    outdir = Path(args.outdir) if args.outdir else DATA / "showdown"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = f"{init:%Y%m%dT%HZ}"

    # ---- ground truth -------------------------------------------------
    samples = load_raw_15min(args.station, years)
    rows = hourly_truth(samples, init, args.horizon, args.tolerance_min)
    filled = sum(1 for r in rows if r["value"] is not None)
    truth_doc = {
        "schema_version": "1",
        "kind": "ground_truth",
        "station_id": args.station,
        "variable": "t2m",
        "units": "degC",
        "init": iso(init),
        "cadence_hours": 1,
        "source": f"raw 15-min HOBO archive, resampled to the hour (+/-{args.tolerance_min} min, no interpolation)",
        "time_note": "Archive clock is local standard time (UTC-5, no DST); shifted to UTC.",
        "coverage": {"hours": len(rows), "filled": filled},
        "steps": rows,
    }
    (outdir / f"truth_{stamp}.json").write_text(json.dumps(truth_doc, indent=2) + "\n")
    print(f"truth: {filled}/{len(rows)} hours filled")

    # ---- raw NWP baselines --------------------------------------------
    obs6 = load_obs_6h(args.station, years)
    for method, label in BASELINES.items():
        path = DATA / method / f"{init:%Y-%m-%dT%HZ}.json"
        steps: list[dict] = []
        if not path.exists():
            print(f"{method}: no benchmark file for this init -> skipped")
            continue
        doc = json.loads(path.read_text())
        loc = doc.get("locations", {}).get(args.station)
        if loc is None:
            print(f"{method}: station absent from file -> skipped")
            continue
        var = loc["variables"]["t2m"]
        by_lead = dict(zip(var["lead_times_hours"], var["bias"]))
        for lead in LEADS:
            valid = init + timedelta(hours=lead)
            bias = by_lead.get(lead)
            actual = obs6.get(iso(valid))
            if bias is None or actual is None or not math.isfinite(float(bias)):
                steps.append({"lead_hours": lead, "valid_time": iso(valid),
                              "value": None, "actual": actual,
                              "reason": "missing_bias" if bias is None else "missing_obs"})
                continue
            steps.append({
                "lead_hours": lead,
                "valid_time": iso(valid),
                "value": round(float(actual) + float(bias), 4),
                "actual": actual,
                "bias": round(float(bias), 4),
            })
        got = sum(1 for s in steps if s["value"] is not None)
        out_doc = {
            "schema_version": "1",
            "kind": "raw_nwp_baseline",
            "model": method,
            "model_label": label,
            "station_id": args.station,
            "variable": "t2m",
            "units": "degC",
            "init": iso(init),
            "cadence_hours": 6,
            "provenance": {
                "source_file": str(path.relative_to(ROOT)),
                "reconstruction": "value = obs_6h(valid_time) + bias(lead); benchmark files store error, not forecast value",
            },
            "steps": steps,
        }
        (outdir / f"baseline_{method}_{stamp}.json").write_text(json.dumps(out_doc, indent=2) + "\n")
        print(f"{method}: {got}/{len(LEADS)} leads")


if __name__ == "__main__":
    main()
