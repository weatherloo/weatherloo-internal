#!/usr/bin/env python3
"""Train HyperDS on issued HRRR/AIFS forecast trajectories (see nwp_model.py)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import UPSTREAM_REVISION, subgrid_queries
from nwp_model import ANCHORS, MODES, NWPHyperDS, at_hour
from train import physical_station

LABEL_END_2025 = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
SOURCE_FILES = ("model.py", "train.py", "nwp_model.py", "nwp_train.py", "nwp_data.py")


def clock_features(stamps):
    """Valid-time hour-of-day and day-of-year as sin/cos pairs, shape [..., 4]."""
    stamps = np.asarray(stamps, dtype=np.float64)
    hour = (stamps % 86400) / 86400
    year = (stamps / 86400 % 365.2425) / 365.2425
    return np.stack([np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
                     np.sin(2 * np.pi * year), np.cos(2 * np.pi * year)], -1).astype(np.float32)


def wind_scale(std):
    return float(math.sqrt(std[0] ** 2 + std[1] ** 2))


def observation_features(observed, forecast, std):
    """Issue-time observation minus lead-0 forecast at each station.

    observed [S,4] (wind m/s, t2m, sp, tp) at the initialization time; forecast
    [S,5] physical (u, v, t2m, sp, tp). Returns [4*S]: wind bias, t2m bias, masks.
    """
    speed = np.sqrt(forecast[:, 0] ** 2 + forecast[:, 1] ** 2)
    bias = np.stack((observed[:, 0] - speed, observed[:, 1] - forecast[:, 2]), -1)
    bias /= np.array([wind_scale(std), std[2]], dtype=np.float32)
    mask = np.isfinite(bias)
    return np.concatenate((np.where(mask, bias, 0), mask.astype(np.float32)), -1).reshape(-1).astype(np.float32)


class NWPDataset(Dataset):
    def __init__(self, era5_root, nwp_root, split, era5_manifest, nwp_manifest):
        import zarr

        era5_root, nwp_root = Path(era5_root), Path(nwp_root)
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
            arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r")
                      for name in ("fields", "station", "init_time", "era5_row")}
            group = zarr.open_group(str(era5_root / entry["era5_shard"]), mode="r")[entry["era5_split"]]
            for index in selected:
                self.rows.append((group, arrays, int(index)))
                init_times.append(int(arrays["init_time"][index]))
        self.init_times = np.array(init_times, dtype=np.int64)
        stations = era5_manifest["stations"]
        if [s["id"] for s in stations] != nwp_manifest["stations"]:
            raise ValueError("ERA5 and NWP station order differ")
        query = [[2 * (s["lon"] - (-88 - .125)) / (64 * .25) - 1, 2 * (s["lat"] - (40 - .125)) / (32 * .25) - 1]
                 for s in stations]
        self.coordinates = torch.tensor(query, dtype=torch.float32)
        stats = era5_manifest["normalization"]
        self.mean = np.array(stats["field_mean"], dtype=np.float32)
        self.std = np.array(stats["field_std"], dtype=np.float32)
        self.sat_mean = np.array(stats["satellite_mean"], dtype=np.float32)
        self.sat_std = np.array(stats["satellite_std"], dtype=np.float32)
        with np.load(era5_root / era5_manifest["station_overlay"]["path"]) as overlay:
            self.overlay = (overlay["times"], overlay["values"])
        self.label_end = LABEL_END_2025 if split == "test" else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        group, arrays, row = self.rows[index]
        era5_row = int(arrays["era5_row"][row])
        init = int(arrays["init_time"][row])
        if int(group["init_time"][era5_row]) != init:
            raise ValueError("NWP/ERA5 row alignment failed")
        target = np.asarray(group["grid"][era5_row], dtype=np.float32)  # hourly 0..48 labels
        satellite = np.asarray(group["satellite"][era5_row], dtype=np.float32)
        station = np.asarray(group["station"][era5_row], dtype=np.float32)
        times, values = self.overlay
        stamps = init + np.arange(station.shape[0]) * 3600
        indices = np.searchsorted(times, stamps)
        if (indices >= len(times)).any() or not np.array_equal(times[indices], stamps):
            raise ValueError("station overlay time mismatch")
        station[:, 0, :2] = values[indices]
        if self.label_end is not None:
            station[stamps >= self.label_end] = np.nan
        forecast = np.asarray(arrays["fields"][row], dtype=np.float32)
        forecast_station = np.asarray(arrays["station"][row], dtype=np.float32)
        observations = observation_features(station[0], forecast_station[0], self.std)
        if not all(np.isfinite(v).all() for v in (target, satellite, forecast, forecast_station)):
            raise ValueError("incomplete model input")
        normalize = lambda x, axis: (x - np.expand_dims(self.mean, axis)) / np.expand_dims(self.std, axis)
        return (torch.from_numpy(normalize(target, (0, 2, 3))),
                torch.from_numpy((satellite - self.sat_mean[None, :, None, None]) / self.sat_std[None, :, None, None]),
                self.coordinates, torch.from_numpy(station),
                torch.from_numpy(normalize(forecast, (0, 2, 3))),
                torch.from_numpy((forecast_station - self.mean) / self.std),
                torch.from_numpy(observations),
                torch.from_numpy(clock_features(init + np.arange(49) * 3600)))


def forecast_error_scales(dataset, max_rows=1000, seed=0):
    """RMSE of raw NWP vs station observations per target (wind speed, t2m, sp, tp).

    Station losses are divided by these, so a typical forecast error weighs the
    same for every variable. Climatological spread would make a 1 degC miss count
    about 25x less than a 2 m/s wind miss.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(dataset.rows), size=min(max_rows, len(dataset.rows)), replace=False)
    times, values = dataset.overlay
    squared, counts = np.zeros(4), np.zeros(4)
    for index in rows:
        group, arrays, row = dataset.rows[index]
        init = int(arrays["init_time"][row])
        station = np.asarray(group["station"][int(arrays["era5_row"][row])], dtype=np.float64)
        stamps = init + np.arange(station.shape[0]) * 3600
        station[:, 0, :2] = values[np.searchsorted(times, stamps)]
        forecast = np.asarray(arrays["station"][row], dtype=np.float64)  # [lead, station, 5]
        leads = dataset.leads[dataset.leads > 0]
        lead_index = [list(dataset.leads).index(h) for h in leads]
        raw = np.concatenate((np.hypot(forecast[lead_index, :, 0], forecast[lead_index, :, 1])[..., None],
                              forecast[lead_index, :, 2:]), -1)
        error = raw - station[leads]
        mask = np.isfinite(error)
        squared += np.where(mask, error, 0).__pow__(2).sum(axis=(0, 1))
        counts += mask.sum(axis=(0, 1))
    return np.maximum(np.sqrt(squared / np.maximum(counts, 1)), 1e-3).astype(np.float32).tolist()


def scaled_station_loss(prediction, target, mean, std, station_indices, scales):
    """Masked MSE of physical station targets divided by per-variable forecast error scales."""
    prediction = physical_station(prediction, mean, std)[:, station_indices]
    target = target[:, station_indices]
    mask = torch.isfinite(target)
    error = ((prediction - torch.nan_to_num(target)) / scales).square()
    return (error * mask).sum() / mask.sum().clamp_min(1)


def batch_to(batch, device):
    return [value.to(device, non_blocking=True) for value in batch]


def zero_residual(model):
    """Zero each decoder's output layer so the model starts exactly at the raw NWP values."""
    for decoder in model.decoders:
        torch.nn.init.zeros_(decoder.out_fc.weight)
        torch.nn.init.zeros_(decoder.out_fc.bias)


def grid_sample_cells(batch_size, height, width, device, stations=None, random=True, cells=128):
    grid = subgrid_queries(batch_size, height, width, device, stations=stations, random=random)
    chosen = (torch.randperm(height * width, device=device)[:cells] if random
              else torch.linspace(0, height * width - 1, cells, device=device).long())
    return grid.reshape(batch_size, height * width, 10, 2)[:, chosen].reshape(batch_size, -1, 2), chosen


def train_epoch(model, loader, optimizer, device, mean, std, leads, training_indices, scales, hours_per_step=12,
                grid_cells=128, station_beta=1.):
    """Per batch: encode the 6-hourly anchors once, then score randomly sampled hours.

    Like the paper's random subgrid cells, a random subset of the 48 forecast hours
    is decoded each step; every hour is supervised over an epoch.
    """
    model.train()
    totals = {"grid_mse": 0., "station_mse": 0.}
    steps = 0
    for batch_index, batch in enumerate(loader):
        target, satellite, query, station, forecast, forecast_station, observations, clock = batch_to(batch, device)
        batch_size, _, _, height, width = target.shape
        optimizer.zero_grad()
        contexts = model.encode_anchors(model.encode_satellite(satellite), forecast, leads, clock, observations)
        hours = sorted(torch.randperm(48)[:hours_per_step].add(1).tolist())
        losses = []
        for hour in hours:
            grid, cells = grid_sample_cells(batch_size, height, width, device, query[:, training_indices], cells=grid_cells)
            prediction = model.decode_hour(contexts, forecast, forecast_station, leads, hour,
                                           torch.cat((query, grid), dim=1), model.conditioning(clock[:, hour], observations))
            k = query.shape[1]
            grid_pred = prediction[:, k:].reshape(batch_size, len(cells), 10, 5).mean(dim=2)
            grid_loss = F.mse_loss(grid_pred, target[:, hour].flatten(2).transpose(1, 2)[:, cells])
            stn_loss = scaled_station_loss(prediction[:, :k], station[:, hour], mean, std, training_indices, scales)
            loss = grid_loss + station_beta * stn_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at hour {hour}")
            losses.append(loss)
            totals["grid_mse"] += grid_loss.item()
            totals["station_mse"] += stn_loss.item()
            steps += 1
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if batch_index % 100 == 0:
            print(json.dumps({"batch": batch_index, "batches": len(loader), "mode": model.mode,
                              "loss": torch.stack(losses).mean().item()}), flush=True)
    return {name: value / max(steps, 1) for name, value in totals.items()} | {"steps": steps}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, leads, selection_indices, scales, prediction_output=None,
             station_ids=None):
    """Hourly model and raw-NWP station errors on identical finite pairs (hour 0 excluded).

    The NWP baseline at hours between issued leads is the same time interpolation
    the model uses as its residual base (linear; precipitation = interval rate).
    """
    model.eval()
    scale_tensor = torch.as_tensor(scales, dtype=torch.float32, device=device)
    n_stations = loader.dataset.coordinates.shape[0]
    hours = list(range(49))
    sums = {name: torch.zeros(len(hours), n_stations, 4, dtype=torch.float64) for name in
            ("model_abs", "model_sq", "nwp_abs", "nwp_sq", "count")}
    selection, grid_losses = [], []
    predictions, raw, observed_all = [], [], []
    for batch in loader:
        target, satellite, query, station, forecast, forecast_station, observations, clock = batch_to(batch, device)
        batch_size, _, _, height, width = target.shape
        contexts = model.encode_anchors(model.encode_satellite(satellite), forecast, leads, clock, observations)
        grid, cells = grid_sample_cells(batch_size, height, width, device, random=False)
        queries = torch.cat((query, grid), dim=1)
        k = query.shape[1]
        batch_prediction = torch.empty(batch_size, len(hours), n_stations, 4, device=device)
        baseline = torch.empty_like(batch_prediction)
        for hour in hours:
            prediction = model.decode_hour(contexts, forecast, forecast_station, leads, hour, queries,
                                           model.conditioning(clock[:, hour], observations))
            batch_prediction[:, hour] = physical_station(prediction[:, :k], mean, std)
            baseline[:, hour] = physical_station(at_hour(forecast_station, leads, hour, 2), mean, std)
            if hour == 0:
                continue
            selection.append(scaled_station_loss(prediction[:, :k], station[:, hour], mean, std, selection_indices,
                                                 scale_tensor).item())
            grid_pred = prediction[:, k:].reshape(batch_size, len(cells), 10, 5).mean(dim=2)
            grid_losses.append(F.mse_loss(grid_pred, target[:, hour].flatten(2).transpose(1, 2)[:, cells]).item())
        mask = torch.isfinite(station) & torch.isfinite(baseline)
        truth = torch.nan_to_num(station)
        for name, value in (("model", batch_prediction), ("nwp", baseline)):
            error = (value - truth) * mask
            sums[f"{name}_abs"] += error.abs().sum(dim=0).double().cpu()
            sums[f"{name}_sq"] += error.square().sum(dim=0).double().cpu()
        sums["count"] += mask.sum(dim=0).double().cpu()
        if prediction_output:
            predictions.append(batch_prediction.cpu().numpy())
            raw.append(baseline.cpu().numpy())
            observed_all.append(station.cpu().numpy())

    def summarize(selected):
        selected = torch.tensor(selected)
        count = sums["count"][selected].sum(dim=0).clamp_min(1)
        return {name: {"mae": (sums[f"{name}_abs"][selected].sum(dim=0) / count).tolist(),
                       "rmse": (sums[f"{name}_sq"][selected].sum(dim=0) / count).sqrt().tolist()}
                for name in ("model", "nwp")} | {"count": count.int().tolist()}

    issued = [h for h in hours if h > 0 and h in list(leads)]
    between = [h for h in hours if h > 0 and h not in list(leads)]
    by_lead = {}
    for hour in hours:
        n = sums["count"][hour].clamp_min(1)
        by_lead[str(hour)] = {"count": sums["count"][hour].int().tolist(),
                              **{f"{name}_rmse": (sums[f"{name}_sq"][hour] / n).sqrt().tolist() for name in ("model", "nwp")}}
    if prediction_output:
        np.savez_compressed(prediction_output, forecast=np.concatenate(predictions), nwp_forecast=np.concatenate(raw),
                            observation=np.concatenate(observed_all), initializations=loader.dataset.init_times,
                            station_ids=np.array(station_ids), variables=np.array(["wind_speed", "t2m", "sp", "tp1h"]),
                            units=np.array(["m/s", "degC", "hPa", "mm"]), lead_times_hours=np.asarray(hours),
                            issued_nwp_leads=np.asarray(leads))
    summary = summarize([h for h in hours if h > 0])
    # Selection: skill ratio (model MSE / raw-forecast MSE on identical pairs) at the
    # selection stations, averaged over wind and temperature. < 1 beats the forecast;
    # each variable counts equally regardless of its error magnitude.
    forecast_hours = torch.tensor([h for h in hours if h > 0])
    selected = torch.tensor(list(selection_indices))
    model_sq = sums["model_sq"][forecast_hours][:, selected].sum(dim=(0, 1))
    nwp_sq = sums["nwp_sq"][forecast_hours][:, selected].sum(dim=(0, 1))
    skill = {name: float(model_sq[i] / nwp_sq[i]) for i, name in enumerate(("wind_speed", "t2m")) if nwp_sq[i] > 0}
    return {"selection_score": float(np.mean(list(skill.values()))) if skill else float("nan"),
            "selection_by_variable": skill,
            "validation_station_mse": float(np.mean(selection)), "grid_mse": float(np.mean(grid_losses)),
            "station_metrics": summary, "count": summary["count"],
            "issued_lead_metrics": summarize(issued),
            "interpolated_hour_metrics": summarize(between) if between else None,
            "by_lead": by_lead, "variables": ["wind_speed_m/s", "t2m_degC", "sp_hPa", "tp_mm"]}


def commit_checkpoints():
    import modal
    if not modal.is_local():
        modal.Volume.from_name("weatherloo-temporal-checkpoints").commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-store", type=Path, required=True)
    parser.add_argument("--nwp-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="nwp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    # Station truth is the objective; ERA5 grid loss is a dense regularizer (paper used 0.05).
    parser.add_argument("--station-beta", type=float, default=1.)
    parser.add_argument("--hours-per-step", type=int, default=12)
    args = parser.parse_args(argv)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("real training requires a GPU")
    era5_manifest = json.loads((args.era5_store / "manifest.json").read_text())
    nwp_manifest = json.loads((args.nwp_store / "manifest.json").read_text())
    leads = nwp_manifest["leads"]
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        dataset = NWPDataset(args.era5_store, args.nwp_store, split, era5_manifest, nwp_manifest)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size if shuffle else 2, shuffle=shuffle,
                                    num_workers=4, persistent_workers=True, pin_memory=True)
    scales_list = forecast_error_scales(loaders["train"].dataset)
    scales = torch.tensor(scales_list, device=device)
    print(json.dumps({"forecast_error_scales": dict(zip(("wind_speed_m/s", "t2m_degC", "sp_hPa", "tp_mm"), scales_list))}),
          flush=True)
    n_stations = len(era5_manifest["stations"])
    training_indices = era5_manifest.get("training_station_indices", list(range(n_stations)))
    selection_indices = era5_manifest.get("selection_station_indices", [0])
    stats = era5_manifest["normalization"]
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    model = NWPHyperDS(args.mode, n_stations=n_stations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    args.out.mkdir(parents=True, exist_ok=True)
    source = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    latest = args.out / "latest.pt"
    first, best = 1, float("inf")
    best_by_variable = {"wind_speed": float("inf"), "t2m": float("inf")}
    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["source_sha256"] != source:
            raise RuntimeError("source changed; use a new run directory")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first, best = checkpoint["epoch"] + 1, checkpoint["best_validation_station_mse"]
        best_by_variable = checkpoint["best_by_variable"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        np.random.set_state(checkpoint["numpy_rng_state"])
    elif args.warm_start and args.warm_start.is_file():
        state = torch.load(args.warm_start, map_location=device, weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(("clock_projection", "observation_projection")) for key in missing):
            raise RuntimeError(f"warm start mismatch: missing={missing} unexpected={unexpected}")
        print(json.dumps({"initialized_from": str(args.warm_start)}), flush=True)
    if not latest.is_file():
        zero_residual(model)
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": 1e-4, "station_beta": args.station_beta,
              "zero_initialized_residual": True,
              "subgrid_samples": 10, "grid_cells_per_step": 128, "hours_per_step": args.hours_per_step,
              "anchor_hours": list(ANCHORS), "seed": 42, "forecast_error_scales": scales_list,
              "warm_start": str(args.warm_start) if args.warm_start else None, "leads": leads,
              "source": nwp_manifest["source"]}
    print(json.dumps({"training_started": args.mode, "source": nwp_manifest["source"],
                      "device": torch.cuda.get_device_name(), "train_initializations": len(loaders["train"].dataset),
                      "validation_initializations": len(loaders["val"].dataset), "first_epoch": first,
                      "upstream_revision": UPSTREAM_REVISION, "source_sha256": source}), flush=True)
    stop = min(args.epochs, args.stop_after_epoch or args.epochs)
    for epoch in range(first, stop + 1):
        started = time.monotonic()
        train_metrics = train_epoch(model, loaders["train"], optimizer, device, mean, std, leads,
                                    training_indices, scales, args.hours_per_step, station_beta=args.station_beta)
        val_metrics = evaluate(model, loaders["val"], device, mean, std, leads, selection_indices, scales_list)
        score = val_metrics["selection_score"]
        if not np.isfinite(score):
            raise RuntimeError("non-finite validation loss")
        scheduler.step()
        improved = score < best
        best = min(best, score)
        improved_variables = [name for name in best_by_variable
                              if val_metrics["selection_by_variable"][name] < best_by_variable[name]]
        for name in improved_variables:
            best_by_variable[name] = val_metrics["selection_by_variable"][name]
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "mode": args.mode,
                      "best_validation_station_mse": best, "best_selection_score": best, "best_by_variable": best_by_variable,
                      "validation": val_metrics,
                      "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                      "numpy_rng_state": np.random.get_state(), "source_sha256": source,
                      "era5_manifest": era5_manifest, "nwp_manifest": nwp_manifest,
                      "upstream_revision": UPSTREAM_REVISION, "config": config}
        torch.save(checkpoint, latest)
        if improved:
            torch.save(checkpoint, args.out / "best.pt")
        for name in improved_variables:
            torch.save(checkpoint, args.out / f"best_{name}.pt")
        record = {"epoch": epoch, "mode": args.mode, "source": nwp_manifest["source"], "training": train_metrics,
                  "validation": val_metrics, "seconds": time.monotonic() - started, "best": improved,
                  "best_variables": improved_variables}
        print(json.dumps(record), flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        commit_checkpoints()
    if stop < args.epochs:
        print(json.dumps({"training_stage_saved": args.mode, "epoch": stop, "epochs": args.epochs}), flush=True)
        return
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device, weights_only=False)["state_dict"])
    test = NWPDataset(args.era5_store, args.nwp_store, "test", era5_manifest, nwp_manifest)
    final = evaluate(model, DataLoader(test, batch_size=2, num_workers=4), device, mean, std, leads,
                     selection_indices, scales_list, prediction_output=args.out / "predictions.npz",
                     station_ids=[s["id"] for s in era5_manifest["stations"]])
    (args.out / "evaluation.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"completed": args.mode, "evaluation": final}), flush=True)
    commit_checkpoints()


if __name__ == "__main__":
    main()
