#!/usr/bin/env python3
"""Precompute static aggregate.json files from consolidated NPZ benchmarks.

For every method under data/<method_id>/ with a consolidated NPZ (same
discovery rules as server/npz_api.py: metadata.json "npz_file" override,
default <method_id>_<year>.npz), write data/<method_id>/aggregate.json with:

- "aggregates": mean across ALL inits per station/variable/metric/lead —
  identical values to GET /api/benchmark/<id>/aggregate with no filters
  (np.nanmean over the init axis, NaN -> null).
- "partials": per (month, init-cycle) nansum + non-NaN count per
  station/variable/metric/lead. Any month-aligned date range x cycle subset
  (all the dashboard's quarter/month/cycle filters) recombines exactly as
  sum(sums)/sum(counts), matching the API's masked nanmean.

Runs as part of `npm run build` (see scripts/build_static_aggregates.sh).
Methods without an NPZ are skipped and any existing aggregate.json is left
untouched.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SITE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SITE_ROOT / "data"
DEFAULT_YEAR = 2025
METRICS = ("rmse", "mae", "bias", "acc")
SCHEMA = "weatherloo-static-aggregate/1"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def npz_path_for(method_dir: Path, year: int) -> Path | None:
    filename = f"{method_dir.name}_{year}.npz"
    meta_path = method_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        filename = meta.get("npz_file", filename)
    path = method_dir / filename
    return path if path.is_file() else None


def json_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(None)
        else:
            out.append(float(value))
    return out


def build_aggregate(data: np.lib.npyio.NpzFile, year: int) -> dict:
    station_ids = [str(x) for x in data["station_ids"]]
    variables = [str(x) for x in data["variables"]]
    lead_times = [int(x) for x in data["lead_times_hours"].tolist()]
    inits = [parse_utc(str(x)) for x in data["initializations"]]
    months = np.array([dt.month for dt in inits])
    cycles = np.array([dt.hour for dt in inits])
    n_leads = len(lead_times)

    metric_arrays = {metric: data[metric] for metric in METRICS}

    aggregates: dict = {}
    for si, station in enumerate(station_ids):
        aggregates[station] = {}
        for vi, variable in enumerate(variables):
            entry: dict = {}
            n_samples = np.zeros(n_leads, dtype=np.int64)
            for metric in METRICS:
                slab = metric_arrays[metric][:, si, vi, :]
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    # All-NaN leads (e.g. climatology ACC) mean "no data" -> null.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    means = np.nanmean(slab, axis=0)
                entry[metric] = json_list(means)
                n_samples = np.maximum(
                    n_samples, np.sum(~np.isnan(slab), axis=0).astype(np.int64)
                )
            entry["n_samples"] = n_samples.tolist()
            aggregates[station][variable] = entry

    partials: dict = {}
    for month in sorted(set(months.tolist())):
        by_cycle: dict = {}
        for cycle in sorted(set(cycles.tolist())):
            mask = (months == month) & (cycles == cycle)
            if not mask.any():
                continue
            cells: dict = {}
            for si, station in enumerate(station_ids):
                cells[station] = {}
                for vi, variable in enumerate(variables):
                    per_metric: dict = {}
                    for metric in METRICS:
                        slab = metric_arrays[metric][mask, si, vi, :]
                        per_metric[metric] = {
                            "sum": json_list(np.nansum(slab, axis=0)),
                            "count": np.sum(~np.isnan(slab), axis=0).tolist(),
                        }
                    cells[station][variable] = per_metric
            by_cycle[str(cycle)] = {"n_inits": int(mask.sum()), "cells": cells}
        partials[str(month)] = by_cycle

    return {
        "schema": SCHEMA,
        "method_id": str(data["method_id"].item()),
        "year": year,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_inits": len(inits),
        "lead_times_hours": lead_times,
        "station_ids": station_ids,
        "variables": variables,
        "metrics": list(METRICS),
        "aggregates": aggregates,
        "partials": partials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--methods", nargs="*", help="Only rebuild these method_ids (default: all)"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    built: list[tuple[str, int]] = []
    skipped: list[str] = []
    for method_dir in sorted(args.data_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "observations":
            continue
        if args.methods and method_dir.name not in args.methods:
            continue
        npz_path = npz_path_for(method_dir, args.year)
        if npz_path is None:
            skipped.append(method_dir.name)
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            doc = build_aggregate(data, args.year)
        out_path = method_dir / "aggregate.json"
        out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        built.append((method_dir.name, out_path.stat().st_size))

    for name, size in built:
        print(f"[aggregate] {name}: wrote aggregate.json ({size / 1024:.0f} KiB)")
    if skipped:
        print(f"[aggregate] skipped (no NPZ): {', '.join(skipped)}")
    if not built:
        print("[aggregate] WARNING: no aggregates built")


if __name__ == "__main__":
    main()
