#!/usr/bin/env python3
"""The paper's super-resolution baselines (UNet, EDSR) on the faithful HRRR setup.

As in arXiv:2401.11960 V-B2: satellite frames pass through one convolution and
are average-pooled to the input-field grid, concatenated with the 1-degree input
field, mapped to the 0.25-degree grid; station values are bilinear interpolations
of the output grid; loss = grid MSE + 0.05 x station MSE. Time dimension: each
hour is an independent sample with lead-hour and valid hour-of-day/day-of-year
channels. Same data, station split, selection and metrics as faithful_train.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import faithful_train as ft
from nwp_train import commit_checkpoints
from train import physical_station, station_loss

TIME_CHANNELS = 5


class SatelliteStem(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv = nn.Conv2d(8, channels, 3, padding=1)

    def forward(self, satellite, shape):
        return F.adaptive_avg_pool2d(self.conv(satellite.flatten(1, 2)), shape)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + .1 * self.body(x)


class EDSR(nn.Module):
    def __init__(self, inputs, width=64, blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(inputs, width, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)], nn.Conv2d(width, width, 3, padding=1))
        steps = {1: 0, 2: 1, 4: 2}[scale]
        self.up = nn.Sequential(*[layer for _ in range(steps)
                                  for layer in (nn.Conv2d(width, width * 4, 3, padding=1), nn.PixelShuffle(2))])
        self.tail = nn.Conv2d(width, 5, 3, padding=1)

    def forward(self, x):
        head = self.head(x)
        return self.tail(self.up(head + self.body(head)))


def conv_block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, inputs, width=32):
        super().__init__()
        self.e1, self.e2, self.e3 = conv_block(inputs, width), conv_block(width, 2 * width), conv_block(2 * width, 4 * width)
        self.d2, self.d1 = conv_block(6 * width, 2 * width), conv_block(3 * width, width)
        self.out = nn.Conv2d(width, 5, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=ft.POOL, mode="bilinear", align_corners=False)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat((F.interpolate(e3, scale_factor=2), e2), 1))
        d1 = self.d1(torch.cat((F.interpolate(d2, scale_factor=2), e1), 1))
        return self.out(d1)


class SRBaseline(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.satellite = SatelliteStem()
        inputs = 5 + 16 + TIME_CHANNELS
        self.net = UNet(inputs) if kind == "unet" else EDSR(inputs, scale=ft.POOL)

    def forward(self, field, satellite, hour, clock):
        """field [b,5,8,16] normalized 1-degree input -> [b,5,32,64] normalized 0.25-degree output."""
        b, _, h, w = field.shape
        time_channels = torch.cat((field.new_full((b, 1), hour / 48), clock), 1)[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat((field, self.satellite(satellite, (h, w)), time_channels), 1)
        base = F.interpolate(field, scale_factor=ft.POOL, mode="bilinear", align_corners=False)
        return base + self.net(x)


def sample_stations(grid, query):
    return F.grid_sample(grid, query[:, None], mode="bilinear", padding_mode="border",
                         align_corners=False).squeeze(2).transpose(1, 2)


def train_epoch(model, loader, optimizer, device, mean, std, train_stations, coordinates, hours_per_step=4, beta=.05):
    model.train()
    totals, steps = {"grid_mse": 0., "station_mse": 0.}, 0
    for batch in loader:
        target, satellite, station, forecast, clock = (value.to(device, non_blocking=True) for value in batch)
        fields = ft.coarse(forecast)
        query = coordinates[train_stations].to(device).expand(target.shape[0], -1, -1)
        optimizer.zero_grad()
        losses = []
        for hour in sorted(torch.randperm(49)[:hours_per_step].tolist()):
            grid = model(fields[:, hour], satellite, hour, clock[:, hour])
            grid_loss = F.mse_loss(grid, target[:, hour])
            stn_loss = station_loss(sample_stations(grid, query), station[:, hour][:, train_stations], mean, std,
                                    list(range(query.shape[1])))
            losses.append(grid_loss + beta * stn_loss)
            totals["grid_mse"] += grid_loss.item()
            totals["station_mse"] += stn_loss.item()
            steps += 1
        torch.stack(losses).mean().backward()
        optimizer.step()
    return {name: value / max(steps, 1) for name, value in totals.items()} | {"steps": steps}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, coordinates, groups, selection_group, prediction_output=None,
             station_ids=None):
    model.eval()
    n = coordinates.shape[0]
    sums = {k: torch.zeros(49, n, 4, dtype=torch.float64) for k in ("sq", "abs", "count")}
    selection, saved = [], {"forecast": [], "observation": []}
    selected = torch.tensor(groups[selection_group])
    for batch in loader:
        target, satellite, station, forecast, clock = (value.to(device, non_blocking=True) for value in batch)
        fields = ft.coarse(forecast)
        query = coordinates.to(device).expand(target.shape[0], -1, -1)
        out = torch.empty(target.shape[0], 49, n, 4, device=device)
        for hour in range(49):
            normalized = sample_stations(model(fields[:, hour], satellite, hour, clock[:, hour]), query)
            out[:, hour] = physical_station(normalized, mean, std)
            if hour:
                selection.append(station_loss(normalized[:, selected], station[:, hour][:, selected], mean, std,
                                              list(range(len(selected)))).item())
        mask = torch.isfinite(station)
        error = (out - torch.nan_to_num(station)) * mask
        sums["sq"] += error.square().sum(dim=0).double().cpu()
        sums["abs"] += error.abs().sum(dim=0).double().cpu()
        sums["count"] += mask.sum(dim=0).double().cpu()
        if prediction_output:
            saved["forecast"].append(out.cpu().numpy())
            saved["observation"].append(station.cpu().numpy())
    report = {"validation_station_loss": float(np.mean(selection)), "groups": {}}
    for group, indices in groups.items():
        indices = torch.tensor(indices)
        c = sums["count"][1:, indices].sum(dim=(0, 1)).clamp_min(1)
        report["groups"][group] = {"stations": len(indices), "count": c.int().tolist(),
                                   "model": {"mse": (sums["sq"][1:, indices].sum(dim=(0, 1)) / c).tolist(),
                                             "mae": (sums["abs"][1:, indices].sum(dim=(0, 1)) / c).tolist()}}
    if prediction_output:
        np.savez_compressed(prediction_output, forecast=np.concatenate(saved["forecast"]),
                            observation=np.concatenate(saved["observation"]), initializations=loader.dataset.init_times,
                            station_ids=np.array([ft.SITE_ALIASES.get(s, s) for s in station_ids]),
                            variables=np.array(["wind_speed", "t2m", "sp", "tp1h"]),
                            units=np.array(["m/s", "degC", "hPa", "mm"]), lead_times_hours=np.arange(49))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("unet", "edsr"), required=True)
    parser.add_argument("--era5-store", type=Path, required=True)
    parser.add_argument("--nwp-store", type=Path, required=True)
    parser.add_argument("--station-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-pool", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args(argv)
    ft.set_input_pool(args.input_pool)
    torch.manual_seed(42)
    device = torch.device("cuda")
    era5_manifest = json.loads((args.era5_store / "manifest.json").read_text())
    nwp_manifest = json.loads((args.nwp_store / "manifest.json").read_text())
    split_ids = json.loads((args.station_store / "split.json").read_text())
    datasets = {s: ft.FaithfulDataset(args.era5_store, args.nwp_store, args.station_store, s, era5_manifest, nwp_manifest)
                for s in ("train", "val", "test")}
    station_ids = [s["id"] for s in datasets["train"].stations]
    groups = {name: [station_ids.index(s) for s in ids] for name, ids in split_ids.items()}
    groups.update({sid: [station_ids.index(sid)] for sid in ("eric_d_soulis", "eccc_51459") if sid in station_ids})
    coordinates = datasets["train"].coordinates
    loaders = {s: DataLoader(d, batch_size=args.batch_size if s == "train" else 8, shuffle=s == "train",
                             drop_last=s == "train", num_workers=4, persistent_workers=True, pin_memory=True)
               for s, d in datasets.items()}
    stats = era5_manifest["normalization"]
    mean, std = (torch.tensor(stats[k], device=device) for k in ("field_mean", "field_std"))
    model = SRBaseline(args.kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    args.out.mkdir(parents=True, exist_ok=True)
    source = {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
              for n in ("faithful_baselines.py", "faithful_train.py", "stations.py")}
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, mean, std, groups["train"], coordinates)
        val = evaluate(model, loaders["val"], device, mean, std, coordinates, groups, "val")
        scheduler.step()
        improved = val["validation_station_loss"] < best
        best = min(best, val["validation_station_loss"])
        best_epoch = epoch if improved else best_epoch
        if improved:
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "kind": args.kind, "source_sha256": source,
                        "station_split": split_ids, "input_pool": ft.POOL}, args.out / "best.pt")
        record = {"epoch": epoch, "kind": args.kind, "training": train_metrics, "validation": val,
                  "seconds": time.monotonic() - started, "best": improved}
        print(json.dumps(record), flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        commit_checkpoints()
        if epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stopped": args.kind, "best_epoch": best_epoch}), flush=True)
            break
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=False)["state_dict"])
    final = evaluate(model, loaders["test"], device, mean, std, coordinates, groups, "test",
                     prediction_output=args.out / "predictions.npz", station_ids=station_ids)
    (args.out / "evaluation.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"completed": args.kind, "evaluation": final["groups"]}), flush=True)
    commit_checkpoints()


if __name__ == "__main__":
    main()
