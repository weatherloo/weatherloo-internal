#!/usr/bin/env python3
"""Warm the local grid cache for the full training window.

Fetches every GFS + ERA5 region grid over the date range and writes the
un-normalized ``.npz`` cache under ``.cache/unet_training/`` (via
``load_sample_grids``), so a subsequent ``train.py`` run does zero network IO
on *any* epoch — including the first.

Parallelized with threads (the work is network-bound: GFS byte-range HTTP +
ERA5 zarr reads). Already-cached samples are skipped, so it is safely resumable.

Usage::

    .venv/bin/python models/unet/data/precache.py --workers 8
    .venv/bin/python models/unet/data/precache.py --start 2021-06-01 --end 2021-06-07
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from fetch_era5 import load_config, open_era5
    from dataset import (build_samples, load_sample_grids, _cache_path,
                         era5_last_valid_time, DEFAULT_START, DEFAULT_END)
except ImportError:
    from .fetch_era5 import load_config, open_era5
    from .dataset import (build_samples, load_sample_grids, _cache_path,
                          era5_last_valid_time, DEFAULT_START, DEFAULT_END)


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-cache GFS+ERA5 grids for training.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    cfg = load_config()
    era5_ds = open_era5(cfg["data"]["era5"]["store"], cfg["data"]["era5"]["token"])
    samples = build_samples(cfg, args.start, args.end,
                            last_valid=era5_last_valid_time(era5_ds))

    todo = [s for s in samples if not _cache_path(s).exists()]
    cached = len(samples) - len(todo)
    print(f"range {args.start} .. {args.end}: {len(samples)} samples "
          f"({cached} already cached, {len(todo)} to fetch, {args.workers} workers)")
    if not todo:
        print("nothing to do — cache is warm.")
        return

    t0 = time.time()
    done = 0
    errors: list[tuple[dict, str]] = []

    def fetch(sample: dict):
        load_sample_grids(cfg, sample, era5_ds)  # shared ERA5 ds is read-only/thread-safe
        return sample

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, s): s for s in todo}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — keep going, report at end
                errors.append((s, f"{type(exc).__name__}: {exc}"))
            done += 1
            if done % 50 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)}  ({rate:.1f}/s, ETA {eta/60:.1f} min, "
                      f"{len(errors)} errors)")

    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min. cached {done - len(errors)}/{len(todo)} new grids.")
    if errors:
        print(f"{len(errors)} failures (rerun to retry — resumable):")
        for s, msg in errors[:20]:
            print(f"  {s['date']} {s['cycle']:02d}z f{s['fxx']:03d}: {msg}")


if __name__ == "__main__":
    main()
