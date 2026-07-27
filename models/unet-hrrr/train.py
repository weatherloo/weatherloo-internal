#!/usr/bin/env python3
"""Train a prototype U-Net on HRRR forecast inputs against ERA5 targets with a station-loss term."""

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

from hrrr_dataset import HRRRDataset  # noqa: E402
from unet import ResidualUNet  # noqa: E402

CKPT_DIR = HERE / "checkpoints"
LOG_PATH = HERE / "training_log.csv"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None, station_y_idx=None, station_x_idx=None,
              station_loss_weight=0.0):
    model.train(optimizer is not None)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for xb, yb, station_target, station_weight in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            station_target = station_target.to(device)
            station_weight = station_weight.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            if station_loss_weight > 0.0 and station_y_idx is not None and station_x_idx is not None:
                station_pred = pred[:, :, station_y_idx, station_x_idx]
                station_err = (station_pred - station_target).pow(2).mean(dim=1)
                station_mask = station_weight.clamp(min=0.0)
                if station_mask.sum() > 0:
                    station_loss = (station_err * station_mask).sum() / station_mask.sum().clamp_min(1e-8)
                    loss = loss + station_loss_weight * station_loss

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, station_y_idx=None, station_x_idx=None):
    model.eval()
    losses, station_losses = [], []
    for xb, yb, station_target, station_weight in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        station_target = station_target.to(device)
        station_weight = station_weight.to(device)
        pred = model(xb)
        losses.append(nn.functional.mse_loss(pred, yb).item())
        if station_y_idx is not None and station_x_idx is not None:
            station_pred = pred[:, :, station_y_idx, station_x_idx]
            station_err = (station_pred - station_target).pow(2).mean(dim=1)
            station_mask = station_weight.clamp(min=0.0)
            if station_mask.sum() > 0:
                station_losses.append((station_err * station_mask).sum().item() / station_mask.sum().clamp_min(1e-8).item())
    print("\n=== validation metrics ===")
    print(f"grid MSE: {np.mean(losses):.5f}")
    if station_losses:
        print(f"station MSE: {np.mean(station_losses):.5f}")


def train(args):
    device = pick_device()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.ckpt_name

    print(f"device: {device}")
    print(f"data root: {args.data_dir}")
    print(f"val year: {args.val_year}")
    print(f"input vars: {args.input_vars}")
    print(f"target vars: {args.target_vars}")

    train_ds = HRRRDataset(
        split="train",
        data_root=args.data_dir,
        val_year=args.val_year,
        variables=args.input_vars.split(","),
        target_vars=args.target_vars.split(","),
        station_ids=args.station_ids.split(","),
        max_samples=args.max_samples,
    )
    val_ds = HRRRDataset(
        split="val",
        data_root=args.data_dir,
        val_year=args.val_year,
        variables=args.input_vars.split(","),
        target_vars=args.target_vars.split(","),
        station_ids=args.station_ids.split(","),
        max_samples=args.max_samples,
        stats=train_ds.stats,
    )
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResidualUNet(
        in_channels=len(train_ds.input_vars),
        out_channels=len(train_ds.target_vars),
        base_features=args.base_features,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    epochs_no_improve = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, device, optimizer,
            station_y_idx=train_ds.station_y_idx,
            station_x_idx=train_ds.station_x_idx,
            station_loss_weight=args.station_loss_weight,
        )
        val_loss = run_epoch(
            model, val_loader, device, None,
            station_y_idx=val_ds.station_y_idx,
            station_x_idx=val_ds.station_x_idx,
            station_loss_weight=args.station_loss_weight,
        )
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "input_vars": train_ds.input_vars,
                "target_vars": train_ds.target_vars,
                "station_ids": train_ds.station_ids,
            }, ckpt_path)
            flag = " *best"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:02d}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}{flag}")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epochs_no_improve >= args.patience:
            print(f"early stopping after {epoch} epochs")
            break

    with open(output_dir / args.log_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"checkpoint: {ckpt_path}")
    print(f"log: {output_dir / args.log_name}")

    evaluate(model, val_loader, device, station_y_idx=val_ds.station_y_idx, station_x_idx=val_ds.station_x_idx)


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net on HRRR/ERA5 data")
    p.add_argument("--data-dir", default="/mnt/wato-drive/c52li/weatherloo-data")
    p.add_argument("--output-dir", default=str(HERE / "checkpoints"))
    p.add_argument("--ckpt-name", default="best_model.pt")
    p.add_argument("--log-name", default="training_log.csv")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--base-features", type=int, default=16)
    p.add_argument("--station-loss-weight", type=float, default=0.2)
    p.add_argument("--input-vars", default="t2m,u10,v10,q2,psfc,tp")
    p.add_argument("--target-vars", default="t2m,u10,v10")
    p.add_argument("--station-ids", default="stn_51459_toronto_intl_a")
    p.add_argument("--val-year", type=int, default=2025)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
