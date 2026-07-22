#!/usr/bin/env python3
"""Train the residual-correction U-Net (GFS -> GFS-ERA5 residual).

Trains ``ResidualUNet`` to predict the normalized ``GFS - ERA5`` residual, then
reports **denormalized** (real-unit) skill: does ``corrected = GFS - predicted``
beat the raw GFS forecast? See ``models/unet/README.md``.

Usage
-----
Quick sanity run (5 epochs on the June test week, separate stats file)::

    .venv/bin/python models/unet/train.py --sanity

Full run (whole 2021 usable window, writes canonical data/stats.json)::

    .venv/bin/python models/unet/train.py

Key metrics printed after training (validation set, real units):
  * residual RMSE/MAE per channel (t2m in degC; u10/v10 in m/s)
  * **corrected forecast RMSE vs raw GFS RMSE** for t2m — the bottom line.
    (t2m benchmark: GFS/LR at cyyz sit ~1.4-1.5 degC RMSE.)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))

from dataset import (  # noqa: E402
    GFSResidualDataset, CACHE_DIR, STATS_PATH, CHANNELS,
    DEFAULT_START, DEFAULT_END, denormalize_residual, load_config,
)
from unet import ResidualUNet  # noqa: E402

CKPT_DIR = HERE / "checkpoints"
LOG_PATH = HERE / "training_log.csv"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # CPU fallback. PyTorch's oneDNN (mkldnn) conv backend raises
    # "RuntimeError: could not create a primitive" on some constrained hosts
    # (notably cluster login nodes with tight cpu/memory cgroups). This model is
    # tiny (~117k params on a 21x41 grid), so the mkldnn speedup is negligible;
    # disable it by default to run reliably anywhere, with an opt-back-in escape
    # hatch (UNET_ENABLE_MKLDNN=1) for beefy CPU-only compute nodes.
    if os.environ.get("UNET_ENABLE_MKLDNN", "0") != "1":
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    """One pass; trains if optimizer given, else eval. Returns mean sample loss."""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, val_ds, device, stats: dict) -> None:
    """Denormalized skill on the validation set (real degC / m/s)."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    gfs_mean = np.asarray(stats["gfs"]["mean"], np.float32).reshape(1, -1, 1, 1)
    gfs_std = np.asarray(stats["gfs"]["std"], np.float32).reshape(1, -1, 1, 1)

    true_res_all, pred_res_all, raw_err_all = [], [], []
    for xb, yb in loader:
        pred = model(xb.to(device)).cpu()
        # Denormalized residuals (real units).
        true_res = denormalize_residual(yb, stats).numpy()
        pred_res = denormalize_residual(pred, stats).numpy()
        # Raw GFS error vs truth == the true residual, since (GFS - ERA5) is the
        # target: GFS - truth = true_residual. (gfs recovered for clarity/checks.)
        true_res_all.append(true_res)
        pred_res_all.append(pred_res)
        raw_err_all.append(true_res)  # raw GFS - truth

    true_res = np.concatenate(true_res_all)   # (N, 3, H, W)
    pred_res = np.concatenate(pred_res_all)
    raw_err = np.concatenate(raw_err_all)

    # corrected - truth = (GFS - pred_res) - (GFS - true_res) = true_res - pred_res
    corrected_err = true_res - pred_res
    # residual prediction error == corrected forecast error (same quantity).
    resid_err = pred_res - true_res

    print("\n=== Validation skill (denormalized, real units) ===")
    print(f"  {'channel':<8} {'resid_RMSE':>11} {'resid_MAE':>10} "
          f"{'rawGFS_RMSE':>12} {'corrected_RMSE':>15} {'improve':>9}")
    for c, name in enumerate(CHANNELS):
        unit = "degC" if name == "t2m" else "m/s"
        resid_rmse = float(np.sqrt(np.mean(resid_err[:, c] ** 2)))
        resid_mae = float(np.mean(np.abs(resid_err[:, c])))
        raw_rmse = float(np.sqrt(np.mean(raw_err[:, c] ** 2)))
        corr_rmse = float(np.sqrt(np.mean(corrected_err[:, c] ** 2)))
        improve = raw_rmse - corr_rmse
        print(f"  {name:<8} {resid_rmse:>10.3f} {resid_mae:>10.3f} "
              f"{raw_rmse:>11.3f} {corr_rmse:>14.3f} {improve:>+8.3f}  ({unit})")

    # Headline t2m comparison against the persistence/LR benchmarks.
    c = CHANNELS.index("t2m")
    raw_rmse = float(np.sqrt(np.mean(raw_err[:, c] ** 2)))
    corr_rmse = float(np.sqrt(np.mean(corrected_err[:, c] ** 2)))
    print("\n=== t2m headline (compare vs benchmarks ~1.4-1.5 degC at cyyz) ===")
    print(f"  raw GFS t2m RMSE (no correction): {raw_rmse:.3f} degC")
    print(f"  corrected t2m RMSE (GFS - pred):  {corr_rmse:.3f} degC")
    if corr_rmse < raw_rmse:
        print(f"  -> U-Net REDUCES t2m RMSE by {raw_rmse - corr_rmse:.3f} degC "
              f"({100 * (raw_rmse - corr_rmse) / raw_rmse:.1f}%). Model is helping.")
    else:
        print(f"  -> U-Net does NOT help t2m (corrected >= raw). "
              f"Likely underfit/needs more data or epochs.")


def train(args) -> None:
    device = pick_device()
    ckpt_path = CKPT_DIR / args.ckpt_name
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"device: {device}")
    print(f"range: {args.start} .. {args.end}   epochs: {args.epochs}   "
          f"batch: {args.batch_size}   workers: {args.workers}")
    print(f"split: {args.split_mode}"
          + (f" (train_days<={args.train_days})" if args.split_mode == "interleaved_month" else ""))

    cfg = load_config()
    stats_path = Path(args.stats_path)

    # cached_only is set by run_pipeline.py (train on exactly what fetch cached);
    # plain train.py runs keep the old fetch-on-demand behavior.
    cached_only = getattr(args, "cached_only", False)

    # Train split first (this computes + caches stats if missing); val reuses stats.
    train_ds = GFSResidualDataset("train", args.start, args.end, cfg=cfg,
                                  stats_path=stats_path, recompute_stats=args.recompute_stats,
                                  split_mode=args.split_mode, train_days=args.train_days,
                                  cached_only=cached_only)
    val_ds = GFSResidualDataset("val", args.start, args.end, cfg=cfg,
                                stats=train_ds.stats,
                                split_mode=args.split_mode, train_days=args.train_days,
                                cached_only=cached_only)
    print(f"samples: train {len(train_ds)}  /  val {len(val_ds)}")
    print(f"stats file: {stats_path}   checkpoint: {ckpt_path.name}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    model = ResidualUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_rows = []

    best_val = float("inf")
    epochs_no_improve = 0
    print("\nepoch  train_loss   val_loss        lr")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        flag = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "stats": train_ds.stats,
                        "epoch": epoch, "val_loss": val_loss,
                        "channels": list(CHANNELS),
                        "split_mode": args.split_mode, "train_days": args.train_days},
                       ckpt_path)
            flag = "  *best"
        else:
            epochs_no_improve += 1

        print(f"{epoch:>5}  {train_loss:>9.5f}  {val_loss:>9.5f}  {lr:>8.2e}{flag}")
        log_rows.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "lr": lr})

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: val loss has not improved for {args.patience} epochs.")
            break

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\ntraining log -> {log_path}")
    print(f"best val loss (normalized MSE): {best_val:.5f}   checkpoint -> {ckpt_path}")

    # Reload best checkpoint and report denormalized skill.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nloaded best checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")
    evaluate(model, val_ds, device, ckpt["stats"])


def parse_args():
    p = argparse.ArgumentParser(description="Train residual-correction U-Net.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--stats-path", default=str(STATS_PATH))
    p.add_argument("--recompute-stats", action="store_true")
    p.add_argument("--log-path", default=str(LOG_PATH))
    p.add_argument("--split-mode", default="chronological",
                   choices=["chronological", "interleaved_month"],
                   help="chronological (default) or per-month interleaved train/val split")
    p.add_argument("--train-days", type=int, default=24,
                   help="interleaved_month: day-of-month <= this -> train, else val")
    p.add_argument("--sanity", action="store_true",
                   help="5 epochs on the June test week; uses a separate stats file")
    args = p.parse_args()

    # Mode-specific artifact names so chronological & interleaved runs coexist.
    suffix = "" if args.split_mode == "chronological" else "_interleaved"
    args.ckpt_name = f"best_model{suffix}.pt"
    if args.stats_path == str(STATS_PATH):
        args.stats_path = str(STATS_PATH.parent / f"stats{suffix}.json")
    if args.log_path == str(LOG_PATH):
        args.log_path = str(HERE / f"training_log{suffix}.csv")
    # Force recompute when the mode-specific stats file doesn't exist yet.
    if suffix and not Path(args.stats_path).exists():
        args.recompute_stats = True

    if args.sanity:
        args.start, args.end = "2021-06-01", "2021-06-07"
        args.epochs = 5
        args.batch_size = min(args.batch_size, 8)
        args.stats_path = str(CACHE_DIR / "stats_selftest.json")
        args.recompute_stats = True
        args.log_path = str(HERE / "training_log_sanity.csv")
        args.ckpt_name = "best_model.pt"
    return args


if __name__ == "__main__":
    train(parse_args())
