#!/usr/bin/env python3
"""Train direct or autoregressive temporal HyperDS on multi-station Zarr."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import TemporalHyperDS


SCALE = torch.tensor([5.0, 20.0])  # temperature °C; wind km/h


class StationDataset(Dataset):
    def __init__(self, store, split):
        import zarr

        group = zarr.open_group(str(store), mode="r")[split]
        self.x, self.y, self.query_xy = (group[name] for name in ("x", "y", "query_xy"))
        if self.x.shape[1:] != (49, 64, 64, 3) or self.y.shape[1:] != (49, 4, 2):
            raise ValueError(f"unexpected multi-station schema: x={self.x.shape}, y={self.y.shape}")
        if self.query_xy.shape != (len(self.x), 4, 2):
            raise ValueError(f"unexpected query schema: {self.query_xy.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)
        query = np.asarray(self.query_xy[index], dtype=np.float32)
        mask = np.isfinite(y) & np.isfinite(x).all(axis=(1, 2, 3))[:, None, None]
        x[..., 0] = (x[..., 0] - 273.15) / 20.0
        x[..., 1:] /= 15.0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), torch.from_numpy(query), torch.from_numpy(y), torch.from_numpy(mask)


def loss_for(prediction, target, mask):
    error = torch.nn.functional.smooth_l1_loss(prediction, target / SCALE.to(target.device), reduction="none")
    return (error * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(2, 4, 2, dtype=torch.float64)
    counts = torch.zeros(4, 2, dtype=torch.float64)
    for x, query, y, mask in loader:
        x, query, y, mask = (item.to(device) for item in (x, query, y, mask))
        mask[:, 0] = False  # f00 is not an issued forecast lead
        prediction = model(x, query) * SCALE.to(device)
        totals[0] += (y.abs() * mask).sum(dim=(0, 1)).cpu()
        totals[1] += ((y - prediction).abs() * mask).sum(dim=(0, 1)).cpu()
        counts += mask.sum(dim=(0, 1)).cpu()
    return {"count": counts.int().tolist(),
            "raw_mae": (totals[0] / counts.clamp_min(1)).tolist(),
            "corrected_mae": (totals[1] / counts.clamp_min(1)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("direct", "autoregressive"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(42)
    train = StationDataset(args.store, "train")
    val = StationDataset(args.store, "val")
    manifest = json.loads((args.store / "manifest.json").read_text())
    if len(train) < 2 or len(val) < 2:
        raise ValueError("need multiple initialization cycles in each split")
    train_loader = DataLoader(train, batch_size=2, shuffle=True)
    val_loader = DataLoader(val, batch_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalHyperDS(autoregressive=args.model == "autoregressive").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, query, y, mask in train_loader:
            x, query, y, mask = (item.to(device) for item in (x, query, y, mask))
            mask[:, 0] = False
            mask[:, :, -1] = False  # Guelph is a spatially held-out station.
            optimizer.zero_grad()
            loss = loss_for(model(x, query), y, mask)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: {json.dumps(metrics)}", flush=True)
        counts = np.asarray(metrics["count"][:3], dtype=np.float64)[:, 0]
        errors = np.asarray(metrics["corrected_mae"][:3], dtype=np.float64)[:, 0]
        score = float((counts * errors).sum() / max(counts.sum(), 1))
        if score < best:
            best = score
            torch.save({
                "model": args.model, "state_dict": model.state_dict(),
                "validation": metrics, "best_epoch": epoch,
                "selection_score_seen_station_temperature_mae": score,
                "data_manifest": manifest, "scale": SCALE.tolist(), "seed": 42,
                "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("model.py", "train.py")},
            }, args.out / f"{args.model}.pt")


if __name__ == "__main__":
    main()
