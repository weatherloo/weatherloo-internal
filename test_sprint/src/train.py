from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import DEFAULT_V1_LEAD_HOURS, build_v1_datasets
from src.data.hrrr import HRRRPatchLoader, WaterlooPatchConfig
from src.data.stations import DEFAULT_V1_STATIONS, StationConfig, load_station_panel
from src.evaluation import PooledBiasCorrector, summarize_metrics_by_lead, summarize_station_metrics
from src.models.hrrr_cnn import (
    DenseTemperatureResidualUNet,
    V1LossConfig,
    compute_v1_loss,
    sample_station_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Waterloo HRRR post-processing v1.")
    parser.add_argument("--uw-csv", type=str, default=None, help="Path or URL to the UW 2026 CSV.")
    parser.add_argument("--hrrr-path", type=str, required=True, help="Path to the local HRRR NetCDF/Zarr dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for predictions and metrics.")
    parser.add_argument("--station-ids", type=str, default="", help="Comma-separated station ids to keep.")
    parser.add_argument("--weatherstats-limit-rows", type=int, default=4000, help="Hourly rows to request from Weatherstats.")
    parser.add_argument("--min-station-count", type=int, default=1, help="Minimum available stations for a valid sample.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count.")
    return parser.parse_args()


def build_station_configs(uw_csv_override: str | None) -> tuple[StationConfig, ...]:
    configs: list[StationConfig] = []
    for station in DEFAULT_V1_STATIONS:
        if station.station_id == "uwaterloo_eds" and uw_csv_override:
            configs.append(replace(station, csv_url=uw_csv_override))
        else:
            configs.append(station)
    return tuple(configs)


def parse_station_ids(raw_station_ids: str) -> list[str] | None:
    if not raw_station_ids.strip():
        return None
    return [item.strip() for item in raw_station_ids.split(",") if item.strip()]


def make_dataloaders(
    *,
    hrrr_loader: HRRRPatchLoader,
    station_panel: pd.DataFrame,
    station_ids: list[str] | None,
    batch_size: int,
    num_workers: int,
    min_station_count: int,
) -> dict[str, DataLoader]:
    datasets = build_v1_datasets(
        hrrr_loader=hrrr_loader,
        station_panel=station_panel,
        station_ids=station_ids,
        lead_hours=DEFAULT_V1_LEAD_HOURS,
        min_station_count=min_station_count,
    )
    return {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
        for split_name, dataset in datasets.items()
    }


def run_epoch(
    *,
    model: DenseTemperatureResidualUNet,
    dataloader: DataLoader,
    optimizer: Adam | None,
    loss_config: V1LossConfig,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_station = 0.0
    total_consistency = 0.0
    total_regularization = 0.0
    steps = 0

    for batch in dataloader:
        dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
        static = batch["static"].to(device=device, dtype=torch.float32)
        baseline = batch["baseline_t2m_c"].to(device=device, dtype=torch.float32)
        lead_hour = batch["lead_hour"].to(device=device, dtype=torch.long)
        station_xy = batch["station_xy"].to(device=device, dtype=torch.float32)
        station_values = batch["station_values_c"].to(device=device, dtype=torch.float32)
        station_mask = batch["station_mask"].to(device=device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        residual = model(dynamic, static=static, lead_hour=lead_hour)
        loss_dict = compute_v1_loss(
            residual_field=residual,
            baseline_field=baseline,
            station_xy=station_xy,
            station_values_c=station_values,
            station_mask=station_mask,
            config=loss_config,
        )
        if is_train:
            loss_dict["loss"].backward()
            optimizer.step()

        total_loss += float(loss_dict["loss"].detach().cpu())
        total_station += float(loss_dict["station_loss"].cpu())
        total_consistency += float(loss_dict["consistency_loss"].cpu())
        total_regularization += float(loss_dict["regularization_loss"].cpu())
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "station_loss": 0.0, "consistency_loss": 0.0, "regularization_loss": 0.0}
    return {
        "loss": total_loss / steps,
        "station_loss": total_station / steps,
        "consistency_loss": total_consistency / steps,
        "regularization_loss": total_regularization / steps,
    }


def sample_nearest_points(field: torch.Tensor, station_xy: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = field.shape
    x = ((station_xy[..., 0] + 1.0) * (width - 1) / 2.0).round().long().clamp(0, width - 1)
    y = ((station_xy[..., 1] + 1.0) * (height - 1) / 2.0).round().long().clamp(0, height - 1)
    batch_index = torch.arange(batch_size, device=field.device)[:, None]
    return field[batch_index, 0, y, x]


def collect_station_rows(
    *,
    model: DenseTemperatureResidualUNet,
    dataloader: DataLoader,
    station_ids: list[str],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            baseline = batch["baseline_t2m_c"].to(device=device, dtype=torch.float32)
            lead_hour = batch["lead_hour"].to(device=device, dtype=torch.long)
            station_xy = batch["station_xy"].to(device=device, dtype=torch.float32)
            station_values = batch["station_values_c"].to(device=device, dtype=torch.float32)
            station_mask = batch["station_mask"].to(device=device)

            residual = model(dynamic, static=static, lead_hour=lead_hour)
            corrected = baseline + residual
            model_sampled = sample_station_points(corrected, station_xy).cpu()
            bilinear_sampled = sample_station_points(baseline, station_xy).cpu()
            nearest_sampled = sample_nearest_points(baseline, station_xy).cpu()

            lead_cpu = batch["lead_hour"].cpu()
            station_values_cpu = station_values.cpu()
            station_mask_cpu = station_mask.cpu()

            batch_size = len(batch["valid_time_utc"])
            for batch_idx in range(batch_size):
                for station_idx, station_id in enumerate(station_ids):
                    if not bool(station_mask_cpu[batch_idx, station_idx]):
                        continue
                    rows.append(
                        {
                            "valid_time_utc": batch["valid_time_utc"][batch_idx],
                            "init_time_utc": batch["init_time_utc"][batch_idx],
                            "lead_hour": int(lead_cpu[batch_idx].item()),
                            "station_id": station_id,
                            "observed_c": float(station_values_cpu[batch_idx, station_idx].item()),
                            "model_predicted_c": float(model_sampled[batch_idx, station_idx].item()),
                            "hrrr_bilinear_c": float(bilinear_sampled[batch_idx, station_idx].item()),
                            "hrrr_nearest_c": float(nearest_sampled[batch_idx, station_idx].item()),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_all_methods(frame: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    summaries: dict[str, dict[str, float]] = {}
    by_lead: dict[str, pd.DataFrame] = {}
    method_columns = {
        "raw_nearest": "hrrr_nearest_c",
        "raw_bilinear": "hrrr_bilinear_c",
        "bias_baseline": "bias_baseline_c",
        "model": "model_predicted_c",
    }
    for method_name, column in method_columns.items():
        if column not in frame.columns:
            continue
        eval_frame = frame.rename(columns={column: "predicted_c"})
        summaries[method_name] = summarize_station_metrics(eval_frame[["observed_c", "predicted_c"]])
        by_lead[method_name] = summarize_metrics_by_lead(
            eval_frame[["observed_c", "predicted_c", "lead_hour"]]
        )
    return summaries, by_lead


def save_loss_curves(history_frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if history_frame.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axis_map = {
        "loss": axes[0, 0],
        "station_loss": axes[0, 1],
        "consistency_loss": axes[1, 0],
        "regularization_loss": axes[1, 1],
    }

    for metric_name, axis in axis_map.items():
        train_column = f"train_{metric_name}"
        val_column = f"val_{metric_name}"
        if train_column in history_frame:
            axis.plot(
                history_frame["epoch"],
                history_frame[train_column],
                marker="o",
                label="train",
            )
        if val_column in history_frame:
            axis.plot(
                history_frame["epoch"],
                history_frame[val_column],
                marker="o",
                label="validation",
            )
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_split_artifacts(
    *,
    output_dir: Path,
    split_name: str,
    frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split_name}_station_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    summaries, by_lead = summarize_all_methods(frame)
    metrics_path = output_dir / f"{split_name}_metrics.json"
    metrics_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    for method_name, method_frame in by_lead.items():
        method_frame.to_csv(output_dir / f"{split_name}_{method_name}_by_lead.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_ids = parse_station_ids(args.station_ids)
    station_configs = build_station_configs(args.uw_csv)
    station_panel = load_station_panel(
        station_configs,
        weatherstats_limit_rows=args.weatherstats_limit_rows,
    )

    patch_config = WaterlooPatchConfig()
    hrrr_loader = HRRRPatchLoader(args.hrrr_path, config=patch_config)
    dataloaders = make_dataloaders(
        hrrr_loader=hrrr_loader,
        station_panel=station_panel,
        station_ids=station_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_station_count=args.min_station_count,
    )

    train_dataset = dataloaders["train"].dataset
    effective_station_ids = train_dataset.station_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        (
            f"Using device={device.type} train_samples={len(dataloaders['train'].dataset)} "
            f"validation_samples={len(dataloaders['validation'].dataset)} "
            f"test_samples={len(dataloaders['test'].dataset)} "
            f"stations={len(effective_station_ids)}"
        ),
        flush=True,
    )
    model = DenseTemperatureResidualUNet(
        dynamic_channels=len(patch_config.dynamic_variables),
        static_channels=len(patch_config.static_variables),
        max_lead_hour=max(DEFAULT_V1_LEAD_HOURS),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_config = V1LossConfig()

    history: list[dict[str, float | int]] = []
    best_state = None
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            loss_config=loss_config,
            device=device,
        )
        val_stats = run_epoch(
            model=model,
            dataloader=dataloaders["validation"],
            optimizer=None,
            loss_config=loss_config,
            device=device,
        )
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})
        print(
            (
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"train_station={train_stats['station_loss']:.4f} "
                f"val_station={val_stats['station_loss']:.4f}"
            ),
            flush=True,
        )
        if val_stats["loss"] <= best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best_model.pt")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    save_loss_curves(history_frame, output_dir)
    (output_dir / "loss_config.json").write_text(json.dumps(asdict(loss_config), indent=2), encoding="utf-8")

    train_rows = collect_station_rows(
        model=model,
        dataloader=dataloaders["train"],
        station_ids=effective_station_ids,
        device=device,
    )
    bias_baseline = PooledBiasCorrector().fit(train_rows)

    for split_name in ("train", "validation", "test"):
        rows = collect_station_rows(
            model=model,
            dataloader=dataloaders[split_name],
            station_ids=effective_station_ids,
            device=device,
        )
        if rows.empty:
            continue
        rows["bias_baseline_c"] = bias_baseline.predict(rows)
        save_split_artifacts(output_dir=output_dir, split_name=split_name, frame=rows)

    print(f"Wrote artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

