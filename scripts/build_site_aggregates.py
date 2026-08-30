#!/usr/bin/env python3
"""Emit one compact JSON per method for the benchmarking site to aggregate from.

Why this exists
---------------
The site had two ways to get benchmark numbers, and neither works on static
hosting:

1. ``/api/benchmark/...``, served by ``server/npz_api.py``. That is a local dev
   process; on a static deploy (Vercel) the route does not exist.
2. A fallback that reads ``data/<method>/index.json`` and then fetches every
   per-init file it lists -- 2280 of them for some methods, sequentially. That
   is not a page load, it is an outage.

This script collapses each method into a single file the browser can fetch once
and aggregate locally, so the site works with no backend at all.

What is stored
--------------
Only ``bias``. Every per-init record in this project holds a single
forecast/observation pair, so per-init ``rmse``, ``mae`` and ``|bias|`` are
the same number by construction (verified across all npz files). ``bias`` is
the one with a sign, so everything else is recoverable from it:

    bias = mean(b)          mae = mean(|b|)          rmse = sqrt(mean(b^2))

Note that rmse must be pooled from squared errors -- averaging per-init rmse
values yields mae, which is the bug this replaces.

ACC is deliberately not carried. At one sample per record it degenerates to
+/-1 and averaging that is meaningless; a real ACC needs anomalies pooled
across inits, which the stored per-init files cannot reconstruct.

    python scripts/build_site_aggregates.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarking-site" / "data"
OUT_DIR = DATA / "aggregates"

DECIMALS = 3


def clean(x: float) -> float | None:
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, DECIMALS)


def npz_files_for(method_dir: Path) -> list[Path]:
    return sorted(method_dir.glob(f"{method_dir.name}_[0-9][0-9][0-9][0-9].npz"))


def build_method(method_dir: Path) -> dict | None:
    files = npz_files_for(method_dir)
    if not files:
        return None

    inits: list[str] = []
    per_year = []
    meta = None
    for path in files:
        d = np.load(path, allow_pickle=True)
        if "bias" not in d.files:
            d.close()
            continue
        if meta is None:
            meta = {
                "lead_times_hours": [int(x) for x in d["lead_times_hours"].tolist()],
                "station_ids": [str(x) for x in d["station_ids"]],
                "variables": [str(x) for x in d["variables"]],
            }
        per_year.append((d["initializations"].copy(), d["bias"].copy()))
        inits.extend(str(x) for x in d["initializations"])
        d.close()

    if meta is None or not per_year:
        return None

    bias = np.concatenate([b for _, b in per_year], axis=0)
    init_arr = np.array(inits)
    order = np.argsort(init_arr)
    init_arr, bias = init_arr[order], bias[order]

    out_bias: dict[str, dict[str, list]] = {}
    for si, station in enumerate(meta["station_ids"]):
        out_bias[station] = {}
        for vi, variable in enumerate(meta["variables"]):
            slab = bias[:, si, vi, :]
            out_bias[station][variable] = [
                [clean(v) for v in row] for row in slab.tolist()
            ]

    return {
        "schema_version": "1",
        "method_id": method_dir.name,
        "source_files": [p.name for p in files],
        "lead_times_hours": meta["lead_times_hours"],
        "station_ids": meta["station_ids"],
        "variables": meta["variables"],
        "initializations": [str(x) for x in init_arr],
        "metric": "bias",
        "note": (
            "Per-init bias (forecast - obs), one sample per entry. Aggregate as "
            "bias=mean(b), mae=mean(|b|), rmse=sqrt(mean(b^2)). Averaging per-init "
            "rmse gives mae, not rmse. ACC is not derivable from these values."
        ),
        "bias": out_bias,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for method_dir in sorted(d for d in DATA.iterdir() if d.is_dir()):
        if method_dir.name in {"observations", "aggregates", "hindcast", "openmeteo_previous"}:
            continue
        doc = build_method(method_dir)
        if doc is None:
            print(f"[skip] {method_dir.name}: no npz with bias")
            continue
        dest = out_dir / f"{method_dir.name}.json"
        with dest.open("w", newline="") as f:
            f.write(json.dumps(doc, separators=(",", ":")) + "\n")
        kb = dest.stat().st_size / 1024
        written.append((method_dir.name, len(doc["initializations"]), kb))
        print(f"[ok]   {method_dir.name}: {len(doc['initializations'])} inits -> {dest.name} ({kb:.0f} KB)")

    index = {
        "schema_version": "1",
        "methods": [name for name, _, _ in written],
    }
    with (out_dir / "index.json").open("w", newline="") as f:
        f.write(json.dumps(index, indent=2) + "\n")
    total = sum(kb for _, _, kb in written)
    print(f"\n{len(written)} methods, {total:.0f} KB total")


if __name__ == "__main__":
    main()
