from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
from typing import Iterable

import numpy as np
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

METHOD_COLUMNS: dict[str, tuple[str, str]] = {
    "raw_nearest": ("hrrr_nearest_c", "HRRR nearest"),
    "raw_bilinear": ("hrrr_bilinear_c", "HRRR bilinear"),
    "bias_baseline": ("bias_baseline_c", "Bias baseline"),
    "model": ("model_predicted_c", "CNN model"),
}
SPATIAL_VALIDATION_EXAMPLES_PER_EPOCH = 1
SPATIAL_TEST_EXAMPLES = 3


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
    progress_label: str | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_steps = len(dataloader)
    start_time = time.perf_counter()

    if progress_label:
        print(
            f"{progress_label} starting total_steps={total_steps}",
            flush=True,
        )

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

        if progress_label and (steps == 1 or steps % 25 == 0):
            elapsed_seconds = time.perf_counter() - start_time
            avg_seconds_per_step = elapsed_seconds / steps
            eta_seconds = avg_seconds_per_step * max(total_steps - steps, 0)
            print(
                (
                    f"{progress_label} step={steps}/{len(dataloader)} "
                    f"loss={float(loss_dict['loss'].detach().cpu()):.4f} "
                    f"station={float(loss_dict['station_loss'].cpu()):.4f} "
                    f"elapsed_s={elapsed_seconds:.1f} "
                    f"eta_s={eta_seconds:.1f}"
                ),
                flush=True,
            )

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
    for method_name, (column, _) in METHOD_COLUMNS.items():
        if column not in frame.columns:
            continue
        eval_frame = frame.rename(columns={column: "predicted_c"})
        summaries[method_name] = summarize_station_metrics(eval_frame[["observed_c", "predicted_c"]])
        by_lead[method_name] = summarize_metrics_by_lead(
            eval_frame[["observed_c", "predicted_c", "lead_hour"]]
        )
    return summaries, by_lead


def save_loss_curves(history_frame: pd.DataFrame, output_dir: Path) -> None:
    plt = import_module("matplotlib.pyplot")

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


def save_metrics_by_lead_plot(
    *,
    output_dir: Path,
    split_name: str,
    by_lead: dict[str, pd.DataFrame],
) -> None:
    plt = import_module("matplotlib.pyplot")

    if not by_lead:
        return

    metric_specs = (
        ("mae", "MAE (C)"),
        ("rmse", "RMSE (C)"),
        ("mean_bias", "Mean Bias (C)"),
        ("count", "Count"),
    )
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(10, 12), sharex=True)

    for axis, (metric_name, ylabel) in zip(axes, metric_specs, strict=True):
        for method_name, method_frame in by_lead.items():
            if metric_name not in method_frame.columns:
                continue
            axis.plot(
                method_frame["lead_hour"],
                method_frame[metric_name],
                marker="o",
                linewidth=1.8,
                label=METHOD_COLUMNS.get(method_name, ("", method_name))[1],
            )
        if metric_name == "mean_bias":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    axes[0].set_title(f"{split_name.title()} Metrics by Lead Hour")
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Lead hour")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_metrics_by_lead.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_scatter_plot(
    *,
    output_dir: Path,
    split_name: str,
    frame: pd.DataFrame,
) -> None:
    plt = import_module("matplotlib.pyplot")

    available_methods = [
        (method_name, column, label)
        for method_name, (column, label) in METHOD_COLUMNS.items()
        if column in frame.columns
    ]
    if frame.empty or not available_methods:
        return

    observed = frame["observed_c"].to_numpy(dtype=float)
    ncols = 2
    nrows = (len(available_methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, (_, column, label) in zip(flat_axes, available_methods, strict=False):
        predicted = frame[column].to_numpy(dtype=float)
        combined = np.concatenate([observed, predicted])
        min_value = float(np.nanmin(combined))
        max_value = float(np.nanmax(combined))
        metrics = summarize_station_metrics(
            frame[["observed_c"]].assign(predicted_c=frame[column])
        )

        axis.scatter(observed, predicted, s=10, alpha=0.25, edgecolors="none")
        axis.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1.0)
        axis.set_title(
            (
                f"{label}\n"
                f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
                f"Bias={metrics['mean_bias']:.3f}"
            )
        )
        axis.set_xlabel("Observed (C)")
        axis.set_ylabel("Predicted (C)")
        axis.grid(True, alpha=0.3)

    for axis in flat_axes[len(available_methods) :]:
        axis.set_visible(False)

    fig.suptitle(f"{split_name.title()} Prediction Scatter", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_prediction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_station_timeseries_plot(
    *,
    output_dir: Path,
    split_name: str,
    frame: pd.DataFrame,
) -> None:
    mdates = import_module("matplotlib.dates")
    plt = import_module("matplotlib.pyplot")

    if frame.empty or "lead_hour" not in frame.columns:
        return

    lead_hour = int(frame["lead_hour"].min())
    lead_frame = frame[frame["lead_hour"] == lead_hour].copy()
    if lead_frame.empty:
        return

    lead_frame["valid_time_utc"] = pd.to_datetime(lead_frame["valid_time_utc"], utc=True)
    station_ids = sorted(lead_frame["station_id"].unique().tolist())
    fig, axes = plt.subplots(
        len(station_ids),
        1,
        figsize=(14, max(3.5 * len(station_ids), 4)),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.flatten()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    series_columns = [
        ("observed_c", "Observed"),
        ("hrrr_bilinear_c", "HRRR bilinear"),
        ("bias_baseline_c", "Bias baseline"),
        ("model_predicted_c", "CNN model"),
    ]

    for axis, station_id in zip(flat_axes, station_ids, strict=True):
        station_frame = (
            lead_frame[lead_frame["station_id"] == station_id]
            .sort_values("valid_time_utc")
            .drop_duplicates(subset=["valid_time_utc"])
        )
        for column, label in series_columns:
            if column not in station_frame.columns:
                continue
            axis.plot(
                station_frame["valid_time_utc"],
                station_frame[column],
                linewidth=1.2 if column != "observed_c" else 1.5,
                alpha=0.9 if column in {"observed_c", "model_predicted_c"} else 0.75,
                label=label,
            )
        axis.set_title(station_id)
        axis.set_ylabel("Temp (C)")
        axis.grid(True, alpha=0.3)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)

    flat_axes[0].legend(ncol=4, fontsize=9)
    flat_axes[-1].set_xlabel("Valid time (UTC)")
    fig.suptitle(f"{split_name.title()} Station Time Series (Lead {lead_hour} h)", y=1.01)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{split_name}_station_timeseries_lead{lead_hour:02d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_spatial_example_plot(
    *,
    output_path: Path,
    split_name: str,
    baseline_field: np.ndarray,
    corrected_field: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    station_latitudes: np.ndarray,
    station_longitudes: np.ndarray,
    station_values: np.ndarray,
    station_mask: np.ndarray,
    valid_time_utc: str,
    init_time_utc: str,
    lead_hour: int,
    epoch: int | None = None,
) -> None:
    plt = import_module("matplotlib.pyplot")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    difference_field = corrected_field - baseline_field
    valid_station_mask = station_mask.astype(bool)
    station_values_valid = station_values[valid_station_mask]

    shared_fields = [baseline_field, corrected_field]
    if station_values_valid.size:
        shared_fields.append(station_values_valid)
    field_min = min(float(np.nanmin(values)) for values in shared_fields)
    field_max = max(float(np.nanmax(values)) for values in shared_fields)
    diff_abs_max = max(float(np.nanmax(np.abs(difference_field))), 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    baseline_axis, corrected_axis, difference_axis, overlay_axis = axes.flatten()

    baseline_mesh = baseline_axis.pcolormesh(
        lon_grid,
        lat_grid,
        baseline_field,
        shading="auto",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    baseline_axis.set_title("HRRR Input T2M")

    corrected_mesh = corrected_axis.pcolormesh(
        lon_grid,
        lat_grid,
        corrected_field,
        shading="auto",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    corrected_axis.set_title("Corrected Dense Field")

    difference_mesh = difference_axis.pcolormesh(
        lon_grid,
        lat_grid,
        difference_field,
        shading="auto",
        cmap="coolwarm",
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
    )
    difference_axis.set_title("Correction (Model - HRRR)")

    overlay_mesh = overlay_axis.pcolormesh(
        lon_grid,
        lat_grid,
        corrected_field,
        shading="auto",
        cmap="viridis",
        vmin=field_min,
        vmax=field_max,
    )
    overlay_axis.set_title("Corrected Field + Stations")
    if station_values_valid.size:
        overlay_axis.scatter(
            station_longitudes[valid_station_mask],
            station_latitudes[valid_station_mask],
            c=station_values_valid,
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
            s=55,
            edgecolors="black",
            linewidths=0.7,
        )

    for axis in (baseline_axis, corrected_axis, difference_axis, overlay_axis):
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(True, alpha=0.2)

    fig.colorbar(baseline_mesh, ax=baseline_axis, shrink=0.85, label="Temperature (C)")
    fig.colorbar(corrected_mesh, ax=corrected_axis, shrink=0.85, label="Temperature (C)")
    fig.colorbar(difference_mesh, ax=difference_axis, shrink=0.85, label="Delta (C)")
    fig.colorbar(overlay_mesh, ax=overlay_axis, shrink=0.85, label="Temperature (C)")

    title_prefix = split_name.title()
    if epoch is not None:
        title_prefix = f"{title_prefix} Epoch {epoch}"
    fig.suptitle(
        (
            f"{title_prefix} Spatial Example | lead={lead_hour}h\n"
            f"valid={valid_time_utc} | init={init_time_utc}"
        ),
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_spatial_diagnostic_plots(
    *,
    model: DenseTemperatureResidualUNet,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    max_examples: int,
    epoch: int | None = None,
) -> None:
    if max_examples <= 0:
        return

    station_latitudes = getattr(dataloader.dataset, "station_latitudes", None)
    station_longitudes = getattr(dataloader.dataset, "station_longitudes", None)
    if station_latitudes is None or station_longitudes is None:
        return

    was_training = model.training
    model.eval()
    saved_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            dynamic = batch["dynamic"].to(device=device, dtype=torch.float32)
            static = batch["static"].to(device=device, dtype=torch.float32)
            baseline = batch["baseline_t2m_c"].to(device=device, dtype=torch.float32)
            lead_hour = batch["lead_hour"].to(device=device, dtype=torch.long)

            residual = model(dynamic, static=static, lead_hour=lead_hour)
            corrected = baseline + residual

            baseline_cpu = baseline[:, 0].detach().cpu().numpy()
            corrected_cpu = corrected[:, 0].detach().cpu().numpy()
            lat_grid_cpu = batch["lat_grid"].cpu().numpy()
            lon_grid_cpu = batch["lon_grid"].cpu().numpy()
            lead_hour_cpu = lead_hour.cpu().numpy()
            station_values_cpu = batch["station_values_c"].cpu().numpy()
            station_mask_cpu = batch["station_mask"].cpu().numpy()

            batch_size = dynamic.shape[0]
            for batch_idx in range(batch_size):
                saved_examples += 1
                file_name_parts = [split_name, "spatial"]
                if epoch is not None:
                    file_name_parts.append(f"epoch{epoch:02d}")
                file_name_parts.extend(
                    [
                        f"example{saved_examples:02d}",
                        f"lead{int(lead_hour_cpu[batch_idx]):02d}",
                    ]
                )
                output_path = output_dir / f"{'_'.join(file_name_parts)}.png"
                save_spatial_example_plot(
                    output_path=output_path,
                    split_name=split_name,
                    baseline_field=baseline_cpu[batch_idx],
                    corrected_field=corrected_cpu[batch_idx],
                    lat_grid=lat_grid_cpu[batch_idx],
                    lon_grid=lon_grid_cpu[batch_idx],
                    station_latitudes=station_latitudes,
                    station_longitudes=station_longitudes,
                    station_values=station_values_cpu[batch_idx],
                    station_mask=station_mask_cpu[batch_idx],
                    valid_time_utc=batch["valid_time_utc"][batch_idx],
                    init_time_utc=batch["init_time_utc"][batch_idx],
                    lead_hour=int(lead_hour_cpu[batch_idx]),
                    epoch=epoch,
                )
                if saved_examples >= max_examples:
                    if was_training:
                        model.train()
                    return

    if was_training:
        model.train()


def save_by_lead_comparison_csv(
    *,
    output_dir: Path,
    split_name: str,
    by_lead: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    comparison_frame: pd.DataFrame | None = None

    for method_name, method_frame in by_lead.items():
        renamed = method_frame.rename(
            columns={
                column: f"{method_name}_{column}"
                for column in method_frame.columns
                if column != "lead_hour"
            }
        )
        if comparison_frame is None:
            comparison_frame = renamed
        else:
            comparison_frame = comparison_frame.merge(renamed, on="lead_hour", how="outer")

    if comparison_frame is None or comparison_frame.empty:
        return None

    comparison_frame = comparison_frame.sort_values("lead_hour").reset_index(drop=True)
    comparison_frame.to_csv(
        output_dir / f"{split_name}_model_by_lead_baseline_comparison.csv",
        index=False,
    )
    return comparison_frame


def save_by_lead_comparison_plot(
    *,
    output_dir: Path,
    split_name: str,
    comparison_frame: pd.DataFrame | None,
) -> None:
    plt = import_module("matplotlib.pyplot")

    if comparison_frame is None or comparison_frame.empty:
        return

    metric_specs = (
        ("mae", "MAE (C)"),
        ("rmse", "RMSE (C)"),
        ("mean_bias", "Mean Bias (C)"),
        ("count", "Count"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    flat_axes = axes.flatten()

    for axis, (metric_name, ylabel) in zip(flat_axes, metric_specs, strict=True):
        for method_name, (_, label) in METHOD_COLUMNS.items():
            column = f"{method_name}_{metric_name}"
            if column not in comparison_frame.columns:
                continue
            axis.plot(
                comparison_frame["lead_hour"],
                comparison_frame[column],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        if metric_name == "mean_bias":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_title(ylabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    flat_axes[0].legend(ncol=2)
    flat_axes[2].set_xlabel("Lead hour")
    flat_axes[3].set_xlabel("Lead hour")
    fig.suptitle(f"{split_name.title()} Model vs Baselines by Lead", y=1.01)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{split_name}_model_by_lead_baseline_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_metrics_summary_plot(
    *,
    output_dir: Path,
    split_name: str,
    summaries: dict[str, dict[str, float]],
) -> None:
    plt = import_module("matplotlib.pyplot")

    if not summaries:
        return

    metric_specs = (
        ("mae", "MAE (C)"),
        ("rmse", "RMSE (C)"),
        ("mean_bias", "Mean Bias (C)"),
        ("count", "Count"),
    )
    method_names = [name for name in METHOD_COLUMNS if name in summaries]
    if not method_names:
        return

    labels = [METHOD_COLUMNS[name][1] for name in method_names]
    positions = np.arange(len(method_names))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    flat_axes = axes.flatten()

    for axis, (metric_name, ylabel) in zip(flat_axes, metric_specs, strict=True):
        values = [summaries[name][metric_name] for name in method_names]
        axis.bar(positions, values)
        if metric_name == "mean_bias":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_title(ylabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{split_name.title()} Metrics Summary", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_metrics_summary.png", dpi=150, bbox_inches="tight")
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

    comparison_frame = save_by_lead_comparison_csv(output_dir=output_dir, split_name=split_name, by_lead=by_lead)
    save_metrics_by_lead_plot(output_dir=output_dir, split_name=split_name, by_lead=by_lead)
    save_by_lead_comparison_plot(output_dir=output_dir, split_name=split_name, comparison_frame=comparison_frame)
    save_metrics_summary_plot(output_dir=output_dir, split_name=split_name, summaries=summaries)
    save_prediction_scatter_plot(output_dir=output_dir, split_name=split_name, frame=frame)
    save_station_timeseries_plot(output_dir=output_dir, split_name=split_name, frame=frame)


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
            progress_label=f"train epoch={epoch}",
        )
        val_stats = run_epoch(
            model=model,
            dataloader=dataloaders["validation"],
            optimizer=None,
            loss_config=loss_config,
            device=device,
            progress_label=f"validation epoch={epoch}",
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
        save_spatial_diagnostic_plots(
            model=model,
            dataloader=dataloaders["validation"],
            device=device,
            output_dir=output_dir,
            split_name="validation",
            max_examples=SPATIAL_VALIDATION_EXAMPLES_PER_EPOCH,
            epoch=epoch,
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
        if split_name == "test":
            save_spatial_diagnostic_plots(
                model=model,
                dataloader=dataloaders[split_name],
                device=device,
                output_dir=output_dir,
                split_name=split_name,
                max_examples=SPATIAL_TEST_EXAMPLES,
            )

    print(f"Wrote artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

