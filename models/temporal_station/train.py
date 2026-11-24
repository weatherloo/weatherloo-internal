#!/usr/bin/env python3
"""Train direct or autoregressive HRRR station-bias models on make_data.py Zarr."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import AutoregressiveModel, DirectModel


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("WEATHERLOO_DATA_ROOT", REPO_ROOT / "data"))
TARGET_SCALE = torch.tensor([5.0, 20.0])  # °C and km/h; only loss scaling


class BiasDataset(Dataset):
    """Read one initialization at a time from the existing train/val Zarr groups."""

    def __init__(self, store: Path, split: str, max_samples: int | None = None):
        import zarr

        group = zarr.open_group(str(store), mode="r")[split]
        self.x = group["x"]
        self.y = group["y"]
        self.size = min(len(self.x), max_samples) if max_samples else len(self.x)
        if self.x.shape[1:] != (49, 30, 30, 3) or self.y.shape[1:] != (49, 2):
            raise ValueError(f"unexpected Zarr schema: x={self.x.shape}, y={self.y.shape}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        x = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)
        valid_frame = np.isfinite(x).all(axis=(1, 2, 3))
        mask = np.isfinite(y) & valid_frame[:, None]
        # The raw store uses K for temperature and m/s for wind components.
        x[..., 0] = (x[..., 0] - 273.15) / 20.0
        x[..., 1:] /= 15.0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def masked_loss(prediction, target, mask):
    scale = TARGET_SCALE.to(prediction.device)
    error = torch.nn.functional.smooth_l1_loss(
        prediction, target / scale, reduction="none"
    )
    return (error * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    absolute = torch.zeros((2, 2), dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.float64)
    lead_absolute = torch.zeros((2, 49, 2), dtype=torch.float64)
    lead_counts = torch.zeros((49, 2), dtype=torch.float64)
    scale = TARGET_SCALE.to(device)
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        prediction = model(x) * scale
        # The forecast question starts at +1 h; f00 is useful for training only.
        mask[:, 0] = False
        raw_error = y.abs() * mask
        corrected_error = (y - prediction).abs() * mask
        absolute[0] += raw_error.sum(dim=(0, 1)).cpu()
        absolute[1] += corrected_error.sum(dim=(0, 1)).cpu()
        counts += mask.sum(dim=(0, 1)).cpu()
        lead_absolute[0] += raw_error.sum(dim=0).cpu()
        lead_absolute[1] += corrected_error.sum(dim=0).cpu()
        lead_counts += mask.sum(dim=0).cpu()
    if counts.sum() == 0:
        raise ValueError("validation data has no paired forecast and station labels")
    return {
        "count": counts.int().tolist(),
        "raw_mae": (absolute[0] / counts.clamp_min(1)).tolist(),
        "corrected_mae": (absolute[1] / counts.clamp_min(1)).tolist(),
        "lead_mae": {
            str(hour): {
                "count": lead_counts[hour].int().tolist(),
                "raw_mae": (
                    lead_absolute[0, hour] / lead_counts[hour].clamp_min(1)
                ).tolist(),
                "corrected_mae": (
                    lead_absolute[1, hour] / lead_counts[hour].clamp_min(1)
                ).tolist(),
            }
            for hour in (1, 3, 6, 12, 24, 48)
        },
    }


def smoke_test() -> None:
    x = torch.randn(2, 49, 30, 30, 3)
    y = torch.randn(2, 49, 2)
    mask = torch.ones_like(y, dtype=torch.bool)
    for model_class in (DirectModel, AutoregressiveModel):
        model = model_class()
        output = model(x)
        assert output.shape == y.shape
        loss = masked_loss(output, y, mask)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        print(f"{model_class.__name__}: shape and backward pass OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DATA_ROOT / "processed" / "hrrr_bias_correction" / "hrrr")
    parser.add_argument("--model", choices=("direct", "autoregressive"), default="direct")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "checkpoints" / "temporal_station")
    parser.add_argument("--smoke", action="store_true", help="Shape/gradient check without data")
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.smoke:
        smoke_test()
        return

    train = BiasDataset(args.store, "train", args.max_samples)
    val = BiasDataset(args.store, "val", args.max_samples)
    if len(train) == 0 or len(val) == 0:
        parser.error("train and val groups must each have at least one sample")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (DirectModel() if args.model == "direct" else AutoregressiveModel()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.store / "manifest.json").read_text())
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("model.py", "train.py")
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_loss(model(x), y, mask)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: {json.dumps(metrics)}", flush=True)
        score = metrics["corrected_mae"][0]  # temperature is the first target
        if score < best:
            best = score
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "source_store": str(args.store),
                "validation": metrics,
                "target_scale": TARGET_SCALE.tolist(),
                "bias_definition": "HRRR_station - observation; corrected = HRRR_station - prediction",
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "max_samples": args.max_samples,
                    "seed": 42,
                    "optimizer": "Adam(lr=1e-3)",
                    "best_epoch": epoch,
                },
                "data_manifest": manifest,
                "source_sha256": source_hashes,
            }, args.out / f"{args.model}.pt")


if __name__ == "__main__":
    main()
