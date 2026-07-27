#!/usr/bin/env python3
"""Single CLI entrypoint for the U-Net pipeline — portable data fetch + train.

Wraps the existing pipeline (data/fetch_gfs.py, data/fetch_era5.py,
data/dataset.py, train.py) so it can run on any machine (e.g. WATcloud) by
pointing it at a data directory. No hardcoded local paths: everything lives
under ``--data-dir`` / ``--output-dir``.

Step 1 — fetch and cache all GFS + ERA5 region grids (parallel, resumable)::

    python run_pipeline.py fetch --start 2022-01-01 --end 2025-12-31 \
        --data-dir /path/to/data --workers 6

Step 2 — train on whatever step 1 cached::

    python run_pipeline.py train --data-dir /path/to/data \
        --output-dir /path/to/checkpoints --epochs 50

Both data sources are public and need **no credentials**:
  * GFS  — AWS Open Data, anonymous byte-range HTTP (with retry/backoff)
  * ERA5 — GCP ARCO zarr store, ``token="anon"``

See RUNBOOK.md in this directory for the full hand-off documentation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def _bootstrap(data_dir: Path) -> None:
    """Point every pipeline module's cache at ``data_dir``.

    Must run BEFORE the pipeline modules are imported (they resolve their
    cache dirs from UNET_DATA_DIR at import time). Using an env var — not a
    function argument — means DataLoader worker subprocesses inherit it too.
    """
    os.environ["UNET_DATA_DIR"] = str(data_dir)
    for sub in ("data", "model", "."):
        p = str((HERE / sub).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def _dir_size_bytes(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def cmd_fetch(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    _bootstrap(data_dir)

    # Import AFTER _bootstrap so cache dirs resolve under --data-dir.
    from fetch_era5 import load_config, open_era5
    from dataset import (build_samples, load_sample_grids, _cache_path,
                         era5_last_valid_time)
    import fetch_gfs

    cfg = load_config()
    print("=== U-Net pipeline: fetch ===")
    print(f"  data dir : {data_dir}")
    print(f"  GFS      : {cfg['data']['gfs']['aws_base']} (anonymous, byte-range GRIB)")
    print(f"  ERA5     : {cfg['data']['era5']['store']} (anonymous)")

    print("\nopening ERA5 store (reads metadata only) ...")
    era5_ds = open_era5(cfg["data"]["era5"]["store"], cfg["data"]["era5"]["token"])
    last_valid = era5_last_valid_time(era5_ds)
    print(f"  ERA5 final-data boundary: {last_valid.isoformat()} "
          f"(--end clamps to this; time axis is padded past it)")

    samples = build_samples(cfg, args.start, args.end, last_valid=last_valid)
    todo = [s for s in samples if not _cache_path(s).exists()]
    cached = len(samples) - len(todo)
    print(f"\nrange {args.start} .. {args.end}: {len(samples)} samples")
    print(f"  already cached: {cached}   to fetch: {len(todo)}   workers: {args.workers}")
    print(f"  expected footprint: ~1.5-2 MB/sample GRIB + ~21 KB/sample .npz "
          f"(~{_fmt_size(int(len(samples) * 1.8e6))} total"
          f"{', GRIB pruned after caching' if args.prune_grib else ''})")

    if not samples:
        print("\nno samples in range — nothing to do.")
        return 1

    t0 = time.time()
    done = 0
    errors: list[tuple[dict, str]] = []

    def fetch_one(sample: dict) -> dict:
        load_sample_grids(cfg, sample, era5_ds)  # shared ERA5 ds is read-only
        if args.prune_grib:
            key = (f"{sample['date'].replace('-', '')}_"
                   f"{sample['cycle']:02d}z_f{sample['fxx']:03d}")
            for f in fetch_gfs.CACHE_DIR.glob(f"{key}*"):
                f.unlink(missing_ok=True)
        return sample

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(fetch_one, s): s for s in todo}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 — collect, report at end
                    errors.append((s, f"{type(exc).__name__}: {exc}"))
                done += 1
                if done % 25 == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(todo)}  ({rate:.2f} samples/s, "
                          f"ETA {_fmt_dur(eta)}, {len(errors)} errors)")

    elapsed = time.time() - t0
    n_ok = sum(1 for s in samples if _cache_path(s).exists())
    cache_size = _dir_size_bytes(data_dir)

    print("\n=== fetch summary ===")
    print(f"  samples in range     : {len(samples)}")
    print(f"  cached and ready     : {n_ok}  ({cached} pre-existing, "
          f"{done - len(errors)} new)")
    print(f"  failures             : {len(errors)}")
    print(f"  cache size on disk   : {_fmt_size(cache_size)}  ({data_dir})")
    print(f"  elapsed              : {_fmt_dur(elapsed)}")
    if errors:
        print(f"\n{len(errors)} samples failed (first 20 below). Transient errors: "
              f"re-run the same command — it is resumable and skips cached samples.")
        print("HTTP 404s are usually permanent archive gaps; `train` skips "
              "uncached samples automatically.")
        for s, msg in errors[:20]:
            print(f"  {s['date']} {s['cycle']:02d}z f{s['fxx']:03d}: {msg}")
        return 1

    print(f"\nOK — cache is complete for {args.start} .. {args.end}.")
    print(f"next: python {Path(__file__).name} train --data-dir {data_dir} "
          f"--output-dir <checkpoints>")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _cached_date_range(cache_dir: Path) -> tuple[str, str, int]:
    """(min_date, max_date, count) over cached ``YYYY-MM-DD_CCz_fFFF.npz`` files."""
    dates = sorted(p.name[:10] for p in cache_dir.glob("*.npz")
                   if len(p.name) >= 10 and p.name[4] == "-")
    if not dates:
        raise SystemExit(f"error: no cached samples found under {cache_dir}\n"
                         f"       run `run_pipeline.py fetch --data-dir ...` first.")
    return dates[0], dates[-1], len(dates)


def cmd_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"error: --data-dir {data_dir} does not exist "
                         f"(run fetch first)")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _bootstrap(data_dir)

    from dataset import CACHE_DIR  # resolves under --data-dir via _bootstrap
    import train as train_mod

    # Default the date range to exactly what is cached.
    lo, hi, n_files = _cached_date_range(CACHE_DIR)
    start = args.start or lo
    end = args.end or hi

    print("=== U-Net pipeline: train ===")
    print(f"  data dir   : {data_dir}  ({n_files} cached samples, {lo} .. {hi})")
    print(f"  output dir : {output_dir}")
    print()

    # Redirect train.py's checkpoint dir, then reuse its training loop verbatim.
    train_mod.CKPT_DIR = output_dir
    ns = argparse.Namespace(
        start=start, end=end, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience, workers=args.workers,
        stats_path=str(output_dir / "stats.json"), recompute_stats=False,
        log_path=str(output_dir / "training_log.csv"),
        split_mode=args.split_mode, train_days=args.train_days,
        ckpt_name="best_model.pt",
        cached_only=True,  # train on exactly what fetch cached — no network IO
    )

    t0 = time.time()
    train_mod.train(ns)
    elapsed = time.time() - t0

    print("\n=== train summary ===")
    print(f"  elapsed : {_fmt_dur(elapsed)}")
    print(f"  outputs under {output_dir}:")
    for name in ("best_model.pt", "training_log.csv", "stats.json"):
        p = output_dir / name
        status = _fmt_size(p.stat().st_size) if p.exists() else "MISSING"
        print(f"    {name:<18} {status}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="U-Net GFS-correction pipeline: fetch data, then train. "
                    "All paths come from --data-dir/--output-dir (no local setup "
                    "needed; both data sources are public/anonymous).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser(
        "fetch", help="fetch + cache GFS and ERA5 region grids (parallel, resumable)",
        description="Download every GFS forecast + ERA5 truth region grid in the "
                    "date range into --data-dir. Parallel, retry-hardened "
                    "(exponential backoff on 429/5xx), and resumable: re-running "
                    "skips already-cached samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    f.add_argument("--start", required=True, metavar="YYYY-MM-DD",
                   help="first GFS init date (archive starts ~2021-03-23)")
    f.add_argument("--end", required=True, metavar="YYYY-MM-DD",
                   help="last GFS init date (clamped to ERA5 store coverage)")
    f.add_argument("--data-dir", required=True,
                   help="cache root; created if missing (~1.5-2 MB per sample)")
    f.add_argument("--workers", type=int, default=6,
                   help="parallel download threads")
    f.add_argument("--prune-grib", action="store_true",
                   help="delete intermediate GRIB files once a sample's .npz is "
                        "cached (cuts disk use ~99%%; only .npz is needed to train)")
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser(
        "train", help="train the U-Net on whatever fetch cached in --data-dir",
        description="Train the residual-correction U-Net using only the samples "
                    "cached in --data-dir (zero network IO). Writes best_model.pt, "
                    "training_log.csv and stats.json to --output-dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("--data-dir", required=True,
                   help="cache root produced by the fetch step")
    t.add_argument("--output-dir", required=True,
                   help="where best_model.pt / training_log.csv / stats.json go")
    t.add_argument("--epochs", type=int, default=50, help="max training epochs")
    t.add_argument("--batch-size", type=int, default=16, help="training batch size")
    t.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    t.add_argument("--patience", type=int, default=10,
                   help="early-stopping patience (epochs)")
    t.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    t.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                   help="restrict to init dates >= this (default: all cached)")
    t.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                   help="restrict to init dates <= this (default: all cached)")
    t.add_argument("--split-mode", default="chronological",
                   choices=["chronological", "interleaved_month"],
                   help="train/val split strategy")
    t.add_argument("--train-days", type=int, default=24,
                   help="interleaved_month: day-of-month <= this -> train")
    t.set_defaults(func=cmd_train)

    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    # Basic date sanity before any network work.
    for attr in ("start", "end"):
        v = getattr(args, attr, None)
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise SystemExit(f"error: --{attr} must be YYYY-MM-DD, got {v!r}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
