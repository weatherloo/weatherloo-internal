#!/usr/bin/env python3
"""Paper-faithful HyperDS (arXiv:2401.11960) on HRRR, plus the time dimension.

Kept from the paper / pinned author code: ResNet-18 dual encoders, satellite
self-attention + field cross-attention fusion, MULTI_VAR generated decoders, the
input field interpolated at each query as the decoder's auxiliary value, 10
subgrid samples per 0.25-degree cell averaged against the grid label (station
coordinates inserted into their cells), loss = grid MSE + 0.05 x station MSE on
mean/variance-normalized variables, Adam 1e-4 with 5-epoch cosine cycles, 50
epochs, 4 samples x 4 accumulation steps (the paper's 4 per GPU x 4 GPUs),
random initialization, checkpoint = lowest station loss on validation stations,
and evaluation on unseen test stations in an unseen period.

Coarse input: HRRR regridded to 0.25 degree and average-pooled 4x to 1 degree, the
paper's input/target resolution ratio. Labels: ERA5 0.25 degree at the valid time.

Time dimension (the only intended extension): encoder contexts at 6-hourly
anchors, any hour 0..48 decoded from the blend of its bracketing anchor contexts,
lead-hour encoding, and valid hour-of-day/day-of-year. Forced by forecasting: the
satellite frames precede the initialization rather than the valid time.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import UPSTREAM_REVISION
from nwp_model import ANCHORS, MODES, NWPHyperDS, at_hour
from nwp_train import LABEL_END_2025, clock_features, commit_checkpoints
from train import physical_station, station_loss

SOURCE_FILES = ("model.py", "train.py", "nwp_model.py", "nwp_train.py", "faithful_train.py", "stations.py")
POOL = 4  # paper: ERA5 0.25 deg average-pooled to 1 deg; set_input_pool(1) feeds the full 0.25 deg field


def set_input_pool(pool):
    """Input coarsening factor for HyperDS and the SR baselines (4 = paper, 1 = full 0.25 deg HRRR)."""
    global POOL
    if HEIGHT % pool or WIDTH % pool:
        raise ValueError(pool)
    POOL = pool
HEIGHT, WIDTH = 32, 64
SITE_ALIASES = {"eccc_51459": "cyyz"}


def station_query(stations):
    return torch.tensor([[2 * (s["lon"] - (-88 - .125)) / (WIDTH * .25) - 1,
                          2 * (s["lat"] - (40 - .125)) / (HEIGHT * .25) - 1] for s in stations], dtype=torch.float32)


class FaithfulDataset(Dataset):
    """One HRRR initialization: ERA5 grid labels, GOES frames, HRRR fields, all-station hourly labels."""

    def __init__(self, era5_root, nwp_root, station_root, split, era5_manifest, nwp_manifest):
        import zarr

        era5_root, nwp_root, station_root = Path(era5_root), Path(nwp_root), Path(station_root)
        self.leads = np.array(nwp_manifest["leads"])
        self.rows, init_times = [], []
        for entry in nwp_manifest["months"]:
            if not entry["samples"]:
                continue
            path = nwp_root / entry["path"]
            splits = np.load(path / "split.npy")
            selected = np.flatnonzero(splits == split)
            if not len(selected):
                continue
            arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ("fields", "init_time", "era5_row")}
            group = zarr.open_group(str(era5_root / entry["era5_shard"]), mode="r")[entry["era5_split"]]
            for index in selected:
                self.rows.append((group, arrays, int(index)))
                init_times.append(int(arrays["init_time"][index]))
        self.init_times = np.array(init_times, dtype=np.int64)
        meta = json.loads((station_root / "stations.json").read_text())
        self.stations = meta["stations"]
        self.coordinates = station_query(self.stations)
        with np.load(station_root / "labels.npz") as saved:
            self.label_times, self.labels = saved["times"], saved["labels"]
        stats = era5_manifest["normalization"]
        self.mean = np.array(stats["field_mean"], dtype=np.float32)
        self.std = np.array(stats["field_std"], dtype=np.float32)
        self.sat_mean = np.array(stats["satellite_mean"], dtype=np.float32)
        self.sat_std = np.array(stats["satellite_std"], dtype=np.float32)
        self.label_end = LABEL_END_2025 if split == "test" else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        group, arrays, row = self.rows[index]
        era5_row = int(arrays["era5_row"][row])
        init = int(arrays["init_time"][row])
        if int(group["init_time"][era5_row]) != init:
            raise ValueError("NWP/ERA5 row alignment failed")
        target = np.asarray(group["grid"][era5_row], dtype=np.float32)
        satellite = np.asarray(group["satellite"][era5_row], dtype=np.float32)
        forecast = np.asarray(arrays["fields"][row], dtype=np.float32)
        stamps = init + np.arange(49) * 3600
        indices = np.searchsorted(self.label_times, stamps)
        if (indices >= len(self.label_times)).any() or not np.array_equal(self.label_times[indices], stamps):
            raise ValueError("station label time mismatch")
        station = self.labels[indices].copy()
        if self.label_end is not None:
            station[stamps >= self.label_end] = np.nan
        if not all(np.isfinite(v).all() for v in (target, satellite, forecast)):
            raise ValueError("incomplete model input")
        field_norm = lambda x: (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return (torch.from_numpy(field_norm(target)),
                torch.from_numpy((satellite - self.sat_mean[None, :, None, None]) / self.sat_std[None, :, None, None]),
                torch.from_numpy(station), torch.from_numpy(field_norm(forecast)),
                torch.from_numpy(clock_features(stamps)))


def coarse(forecast):
    """[b, lead, 5, 32, 64] -> [b, lead, 5, 8, 16] average-pooled input fields."""
    b, n = forecast.shape[:2]
    return F.avg_pool2d(forecast.flatten(0, 1), POOL).view(b, n, *forecast.shape[2:3], HEIGHT // POOL, WIDTH // POOL)


def grid_queries(batch, device, stations=None, random=True):
    """Paper subgrid sampling over every 0.25-degree cell (10 points per cell), stations inserted, vectorized."""
    y, x = torch.meshgrid(torch.arange(HEIGHT, device=device), torch.arange(WIDTH, device=device), indexing="ij")
    cells = torch.stack((x, y), dim=-1).reshape(1, HEIGHT * WIDTH, 1, 2)
    if random:
        offsets = torch.rand(batch, HEIGHT * WIDTH, 10, 2, device=device)
    else:
        offsets = torch.tensor([[.1, .2], [.3, .2], [.5, .2], [.7, .2], [.9, .2],
                                [.1, .8], [.3, .8], [.5, .8], [.7, .8], [.9, .8]], device=device)
        offsets = offsets.expand(batch, HEIGHT * WIDTH, 10, 2)
    scale = torch.tensor([WIDTH, HEIGHT], device=device)
    samples = 2 * (cells + offsets) / scale - 1
    if stations is not None and stations.shape[1]:
        index = ((stations + 1) / 2 * scale).long()
        index[..., 0].clamp_(0, WIDTH - 1)
        index[..., 1].clamp_(0, HEIGHT - 1)
        flat = index[..., 1] * WIDTH + index[..., 0]
        slots = torch.arange(stations.shape[1], device=device) % 10
        samples = samples.clone()
        samples[torch.arange(batch, device=device)[:, None], flat, slots[None]] = stations
    return samples.reshape(batch, HEIGHT * WIDTH * 10, 2)


def predict_hour(model, contexts, fields, leads, hour, queries, clock, empty):
    return model.decode_hour(contexts, fields, None, leads, hour, queries, model.conditioning(clock[:, hour], empty))


def train_epoch(model, loader, optimizer, device, mean, std, leads, train_stations, coordinates, hours_per_step=4,
                accumulation=4, beta=.05):
    model.train()
    totals, steps = {"grid_mse": 0., "station_mse": 0.}, 0
    optimizer.zero_grad()
    for batch_index, batch in enumerate(loader):
        target, satellite, station, forecast, clock = (value.to(device, non_blocking=True) for value in batch)
        batch_size = target.shape[0]
        fields = coarse(forecast)
        empty = target.new_zeros(batch_size, 0)
        query = coordinates[train_stations].to(device).expand(batch_size, -1, -1)
        contexts = model.encode_anchors(model.encode_satellite(satellite), fields, leads, clock, empty)
        losses = []
        for hour in sorted(torch.randperm(49)[:hours_per_step].tolist()):
            grid = grid_queries(batch_size, device, query)
            prediction = predict_hour(model, contexts, fields, leads, hour, torch.cat((query, grid), dim=1), clock, empty)
            k = query.shape[1]
            grid_pred = prediction[:, k:].reshape(batch_size, HEIGHT * WIDTH, 10, 5).mean(dim=2)
            grid_loss = F.mse_loss(grid_pred, target[:, hour].flatten(2).transpose(1, 2))
            stn_loss = station_loss(prediction[:, :k], station[:, hour][:, train_stations], mean, std,
                                    list(range(k)))
            loss = grid_loss + beta * stn_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at hour {hour}")
            losses.append(loss)
            totals["grid_mse"] += grid_loss.item()
            totals["station_mse"] += stn_loss.item()
            steps += 1
        (torch.stack(losses).mean() / accumulation).backward()
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
        if batch_index % 100 == 0:
            print(json.dumps({"batch": batch_index, "batches": len(loader), "mode": model.mode,
                              "loss": torch.stack(losses).mean().item()}), flush=True)
    return {name: value / max(steps, 1) for name, value in totals.items()} | {"steps": steps}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, leads, coordinates, groups, selection_group,
             prediction_output=None, station_ids=None):
    """Station MSE/MAE by group (paper metrics) for the model and interpolation baselines.

    Baselines are the paper's: the input field interpolated to the station
    (HRRR at 1 degree, as fed to the model) and the 0.25-degree field (the
    "ERA5 0.25 degree" analogue). Hour 0 is excluded from forecast summaries.
    """
    model.eval()
    hours = list(range(49))
    n = coordinates.shape[0]
    methods = ("model", "interp_1deg", "interp_025deg")
    sums = {f"{m}_{k}": torch.zeros(len(hours), n, 4, dtype=torch.float64) for m in methods for k in ("sq", "abs")}
    count = torch.zeros(len(hours), n, 4, dtype=torch.float64)
    selection, grid_losses, saved = [], [], {"model": [], "interp_025deg": [], "observation": []}
    selected = torch.tensor(groups[selection_group])
    for batch in loader:
        target, satellite, station, forecast, clock = (value.to(device, non_blocking=True) for value in batch)
        batch_size = target.shape[0]
        fields = coarse(forecast)
        empty = target.new_zeros(batch_size, 0)
        query = coordinates.to(device).expand(batch_size, -1, -1)
        # Grid MSE is a diagnostic here; a fixed 128-cell subset keeps per-epoch validation affordable.
        cells = torch.linspace(0, HEIGHT * WIDTH - 1, 128, device=device).long()
        grid = grid_queries(batch_size, device, random=False).reshape(batch_size, HEIGHT * WIDTH, 10, 2)[:, cells]
        grid = grid.reshape(batch_size, -1, 2)
        contexts = model.encode_anchors(model.encode_satellite(satellite), fields, leads, clock, empty)
        out = {m: torch.empty(batch_size, len(hours), n, 4, device=device) for m in methods}
        for hour in hours:
            prediction = predict_hour(model, contexts, fields, leads, hour, torch.cat((query, grid), dim=1), clock, empty)
            out["model"][:, hour] = physical_station(prediction[:, :n], mean, std)
            for name, field in (("interp_1deg", at_hour(fields, leads, hour, 1)), ("interp_025deg", forecast[:, hour])):
                value = F.grid_sample(field, query[:, None], mode="bilinear", padding_mode="border",
                                      align_corners=False).squeeze(2).transpose(1, 2)
                out[name][:, hour] = physical_station(value, mean, std)
            if hour == 0:
                continue
            selection.append(station_loss(prediction[:, selected], station[:, hour][:, selected], mean, std,
                                          list(range(len(selected)))).item())
            grid_pred = prediction[:, n:].reshape(batch_size, len(cells), 10, 5).mean(dim=2)
            grid_losses.append(F.mse_loss(grid_pred, target[:, hour].flatten(2).transpose(1, 2)[:, cells]).item())
        mask = torch.isfinite(station)
        truth = torch.nan_to_num(station)
        for name in methods:
            error = (out[name] - truth) * mask
            sums[f"{name}_sq"] += error.square().sum(dim=0).double().cpu()
            sums[f"{name}_abs"] += error.abs().sum(dim=0).double().cpu()
        count += mask.sum(dim=0).double().cpu()
        if prediction_output:
            saved["model"].append(out["model"].cpu().numpy())
            saved["interp_025deg"].append(out["interp_025deg"].cpu().numpy())
            saved["observation"].append(station.cpu().numpy())
    forecast_hours = torch.arange(1, 49)
    report = {"validation_station_loss": float(np.mean(selection)), "grid_mse": float(np.mean(grid_losses)),
              "variables": ["wind_speed_m/s", "t2m_degC", "sp_hPa", "tp_mm"], "groups": {}}
    for group, indices in groups.items():
        indices = torch.tensor(indices)
        c = count[forecast_hours][:, indices].sum(dim=(0, 1)).clamp_min(1)
        entry = {"stations": len(indices), "count": c.int().tolist()}
        for name in methods:
            entry[name] = {"mse": (sums[f"{name}_sq"][forecast_hours][:, indices].sum(dim=(0, 1)) / c).tolist(),
                           "mae": (sums[f"{name}_abs"][forecast_hours][:, indices].sum(dim=(0, 1)) / c).tolist()}
        lead_count = count[:, indices].sum(dim=1).clamp_min(1)
        entry["by_lead_mse"] = {str(h): {name: (sums[f"{name}_sq"][h, indices].sum(dim=0) / lead_count[h]).tolist()
                                         for name in methods} for h in range(1, 49)}
        report["groups"][group] = entry
    if prediction_output:
        ids = [SITE_ALIASES.get(s, s) for s in station_ids]
        np.savez_compressed(prediction_output, forecast=np.concatenate(saved["model"]),
                            nwp_forecast=np.concatenate(saved["interp_025deg"]),
                            observation=np.concatenate(saved["observation"]), initializations=loader.dataset.init_times,
                            station_ids=np.array(ids), variables=np.array(["wind_speed", "t2m", "sp", "tp1h"]),
                            units=np.array(["m/s", "degC", "hPa", "mm"]), lead_times_hours=np.arange(49))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-store", type=Path, required=True)
    parser.add_argument("--nwp-store", type=Path, required=True)
    parser.add_argument("--station-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="nwp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--hours-per-step", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-batches", type=int, help="smoke test: stop each epoch early")
    parser.add_argument("--input-pool", type=int, default=4, help="4 = paper (1 deg input); 1 = full 0.25 deg HRRR")
    parser.add_argument("--patience", type=int, default=10,
                        help="stop once validation has not improved for this many epochs (two cosine cycles)")
    args = parser.parse_args(argv)
    set_input_pool(args.input_pool)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("real training requires a GPU")
    era5_manifest = json.loads((args.era5_store / "manifest.json").read_text())
    nwp_manifest = json.loads((args.nwp_store / "manifest.json").read_text())
    split_ids = json.loads((args.station_store / "split.json").read_text())
    leads = nwp_manifest["leads"]
    datasets = {split: FaithfulDataset(args.era5_store, args.nwp_store, args.station_store, split, era5_manifest,
                                       nwp_manifest) for split in ("train", "val", "test")}
    station_ids = [s["id"] for s in datasets["train"].stations]
    groups = {name: [station_ids.index(s) for s in ids] for name, ids in split_ids.items()}
    groups.update({sid: [station_ids.index(sid)] for sid in ("eric_d_soulis", "eccc_51459") if sid in station_ids})
    coordinates = datasets["train"].coordinates
    if args.max_batches:
        for split in datasets:
            datasets[split].rows = datasets[split].rows[:args.max_batches * 4]
            datasets[split].init_times = datasets[split].init_times[:args.max_batches * 4]
    # drop_last: a 1-sample batch cannot train BatchNorm on the 1-degree encoder's 1x1 features.
    loaders = {split: DataLoader(dataset, batch_size=4 if split == "train" else 2, shuffle=split == "train",
                                 drop_last=split == "train", num_workers=4, persistent_workers=True, pin_memory=True)
               for split, dataset in datasets.items()}
    stats = era5_manifest["normalization"]
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    model = NWPHyperDS(args.mode, n_stations=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    args.out.mkdir(parents=True, exist_ok=True)
    source = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    latest = args.out / "latest.pt"
    first, best, best_epoch = 1, float("inf"), 0
    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["source_sha256"] != source:
            raise RuntimeError("source changed; use a new run directory")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first, best = checkpoint["epoch"] + 1, checkpoint["best_validation_station_loss"]
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        np.random.set_state(checkpoint["numpy_rng_state"])
    config = {"paper": "arXiv:2401.11960", "decoder": "MULTI_VAR", "input_pool": POOL, "patience": args.patience, "beta": .05,
              "subgrid_samples": 10, "grid_cells": "all", "batch_size": 4, "accumulation": args.accumulation,
              "effective_batch": 4 * args.accumulation, "lr": 1e-4, "cosine_t_max": 5, "epochs": args.epochs,
              "hours_per_step": args.hours_per_step, "anchor_hours": list(ANCHORS), "seed": 42,
              "selection": "lowest station loss on validation stations, validation period",
              "station_groups": {k: len(v) for k, v in groups.items()}}
    print(json.dumps({"training_started": args.mode, "device": torch.cuda.get_device_name(),
                      "train_initializations": len(datasets["train"]), "stations": len(station_ids),
                      "config": config, "first_epoch": first, "source_sha256": source}), flush=True)
    stop = min(args.epochs, args.stop_after_epoch or args.epochs)
    early_stopped = first > 1 and first - 1 - best_epoch >= args.patience
    for epoch in range(first, stop + 1):
        if early_stopped:
            break
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, mean, std, leads, groups["train"],
                                    coordinates, args.hours_per_step, args.accumulation)
        val_metrics = evaluate(model, loaders["val"], device, mean, std, leads, coordinates, groups, "val")
        score = val_metrics["validation_station_loss"]
        if not np.isfinite(score):
            raise RuntimeError("non-finite validation loss")
        scheduler.step()
        improved = score < best
        best = min(best, score)
        best_epoch = epoch if improved else best_epoch
        early_stopped = epoch - best_epoch >= args.patience
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "mode": args.mode,
                      "best_validation_station_loss": best, "best_epoch": best_epoch, "validation": val_metrics,
                      "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                      "numpy_rng_state": np.random.get_state(), "source_sha256": source, "config": config,
                      "station_split": split_ids, "upstream_revision": UPSTREAM_REVISION}
        torch.save(checkpoint, latest)
        if improved:
            torch.save(checkpoint, args.out / "best.pt")
        record = {"epoch": epoch, "mode": args.mode, "training": train_metrics, "validation": val_metrics,
                  "seconds": time.monotonic() - started, "best": improved, "early_stopped": early_stopped,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        print(json.dumps(record), flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        commit_checkpoints()
    if early_stopped:
        print(json.dumps({"early_stopped": args.mode, "best_epoch": best_epoch, "patience": args.patience}), flush=True)
    elif stop < args.epochs:
        print(json.dumps({"training_stage_saved": args.mode, "epoch": stop, "epochs": args.epochs}), flush=True)
        return
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=False)["state_dict"])
    final = evaluate(model, loaders["test"], device, mean, std, leads, coordinates, groups, "test",
                     prediction_output=args.out / "predictions.npz", station_ids=station_ids)
    (args.out / "evaluation.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"completed": args.mode, "evaluation": final["groups"]}), flush=True)
    commit_checkpoints()


if __name__ == "__main__":
    main()
