#!/usr/bin/env python3
"""Publish the hrrr_cnn site export to the benchmarking site.

The region pipeline's `export` stage (models/hrrr_cnn/final.py) already writes
the site's bias-npz format on the Modal data volume:

    weatherloo-temporal-data:/processed/regions/<region>/site_export/
        hrrr_cnn_2025.npz, metadata.json

This only fetches those two files, checks the npz matches the site schema, and
copies them to benchmarking-site/data/hrrr_cnn/. Then rebuild the aggregates:

    cd benchmarking-site && npm run build:data

Dry run by default; --apply writes. Needs the Modal CLI (`pip install modal`,
`modal setup`). Reads from the volume only; spends no compute.

    python pipelines/benchmarks/methods/hrrr_cnn/compute_benchmark.py
    python pipelines/benchmarks/methods/hrrr_cnn/compute_benchmark.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "benchmarking-site" / "data" / "hrrr_cnn"
METHOD = "hrrr_cnn"
SITE_STATIONS = {"eric_d_soulis", "cyyz"}
KEYS = {"bias", "station_ids", "variables", "lead_times_hours", "initializations"}


def fetch(volume: str, remote: str, local: Path) -> None:
    modal = os.environ.get("MODAL_BIN") or shutil.which("modal")
    if not modal:
        raise SystemExit("Modal CLI not found: pip install modal && modal setup")
    subprocess.run([modal, "volume", "get", "--force", volume, remote, str(local)], check=True)


def check(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        missing = KEYS - set(data.files)
        if missing:
            raise SystemExit(f"{npz_path.name}: missing {sorted(missing)}")
        stations = [str(s) for s in data["station_ids"]]
        if not set(stations) <= SITE_STATIONS:
            raise SystemExit(f"{npz_path.name}: unexpected stations {stations}")
        bias = data["bias"]
        return {"stations": stations, "variables": [str(v) for v in data["variables"]],
                "leads": [int(x) for x in data["lead_times_hours"]], "inits": len(data["initializations"]),
                "finite_cells": int(np.isfinite(bias).sum()), "shape": list(bias.shape)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="waterloo", help="Region whose site_export to publish")
    p.add_argument("--volume", default="weatherloo-temporal-data")
    p.add_argument("--apply", action="store_true", help="Write to benchmarking-site/data/hrrr_cnn/")
    args = p.parse_args()

    remote = f"/processed/regions/{args.region}/site_export"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in (f"{METHOD}_2025.npz", "metadata.json"):
            fetch(args.volume, f"{remote}/{name}", tmp / name)
        summary = check(tmp / f"{METHOD}_2025.npz")
        meta = json.loads((tmp / "metadata.json").read_text())
        if meta.get("method") != METHOD:
            raise SystemExit(f"metadata method is {meta.get('method')!r}, expected {METHOD!r}")
        print(json.dumps({"source": f"{args.volume}:{remote}", **summary}, indent=2))
        if not args.apply:
            print(f"dry run; pass --apply to copy into {OUT.relative_to(ROOT)}")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        for name in (f"{METHOD}_2025.npz", "metadata.json"):
            shutil.copy2(tmp / name, OUT / name)
    print(f"wrote {OUT.relative_to(ROOT)}; now run `npm run build:data` in benchmarking-site/")


if __name__ == "__main__":
    main()
