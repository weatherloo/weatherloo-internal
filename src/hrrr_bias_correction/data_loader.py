"""Utilities for loading HRRR bias-correction tensors from a Zarr store.

Expected schema is produced by make_data.py:
  x: (sample, lead, row, col, channel)
  y: (sample, lead, target)

The loader keeps compatibility with model_arch.py by returning x tensors with
shape (N, 49, 30, 30, 3) and y tensors with shape (N, 49, 2).
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import xarray as xr

N_LEADS = 49
CROP = 30
N_CHANNELS = 3
N_TARGETS = 2
EXPECTED_X_DIMS = ("sample", "lead", "row", "col", "channel")
EXPECTED_Y_DIMS = ("sample", "lead", "target")


@dataclass(frozen=True)
class LoadedSplit:
    """In-memory split payload ready for Keras training."""

    x: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    init_time_unix: np.ndarray
    channel_names: tuple[str, ...]
    target_names: tuple[str, ...]


def _parse_names(attr_value: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(attr_value, str) and attr_value.strip():
        parts = tuple(p.strip() for p in attr_value.split(",") if p.strip())
        if parts:
            return parts
    return fallback


def open_split(store_path: Path | str, split: str) -> xr.Dataset:
    """Open one split group from the Zarr store."""
    return xr.open_zarr(Path(store_path), group=split, consolidated=False)


def validate_schema(ds: xr.Dataset) -> None:
    """Raise ValueError when required vars/dims are incompatible."""
    if "x" not in ds.data_vars or "y" not in ds.data_vars:
        raise ValueError("Dataset must contain data variables 'x' and 'y'.")

    if tuple(ds["x"].dims) != EXPECTED_X_DIMS:
        raise ValueError(f"Unexpected x dims {ds['x'].dims}; expected {EXPECTED_X_DIMS}.")
    if tuple(ds["y"].dims) != EXPECTED_Y_DIMS:
        raise ValueError(f"Unexpected y dims {ds['y'].dims}; expected {EXPECTED_Y_DIMS}.")

    if int(ds.sizes["lead"]) != N_LEADS:
        raise ValueError(f"Unexpected lead size {ds.sizes['lead']}; expected {N_LEADS}.")
    if int(ds.sizes["row"]) != CROP or int(ds.sizes["col"]) != CROP:
        raise ValueError(
            f"Unexpected crop {(ds.sizes['row'], ds.sizes['col'])}; expected {(CROP, CROP)}."
        )
    if int(ds.sizes["channel"]) != N_CHANNELS:
        raise ValueError(
            f"Unexpected channel size {ds.sizes['channel']}; expected {N_CHANNELS}."
        )
    if int(ds.sizes["target"]) != N_TARGETS:
        raise ValueError(f"Unexpected target size {ds.sizes['target']}; expected {N_TARGETS}.")


def load_split_arrays(
    store_path: Path | str,
    split: str,
    *,
    fill_x_nan: float = 0.0,
    fill_y_nan: float = 0.0,
) -> LoadedSplit:
    """Load a split as numpy arrays plus per-lead training mask.

    sample_weight marks lead times with complete targets:
      1.0 if all target components are finite, else 0.0.
    """
    with open_split(store_path, split) as ds:
        validate_schema(ds)

        x_raw = np.asarray(ds["x"].values, dtype=np.float32)
        y_raw = np.asarray(ds["y"].values, dtype=np.float32)
        init_time = np.asarray(ds["init_time"].values, dtype=np.int64)

        mask = np.isfinite(y_raw).all(axis=-1).astype(np.float32)
        x = np.nan_to_num(x_raw, nan=fill_x_nan)
        y = np.nan_to_num(y_raw, nan=fill_y_nan)

        channel_names = _parse_names(ds.attrs.get("channel_names"), ("t2m", "u10", "v10"))
        target_names = _parse_names(ds.attrs.get("target_names"), ("t2m", "wind_speed"))

    return LoadedSplit(
        x=x,
        y=y,
        sample_weight=mask,
        init_time_unix=init_time,
        channel_names=channel_names,
        target_names=target_names,
    )


def make_tf_dataset(
    loaded: LoadedSplit,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int | None = None,
):
    """Convert LoadedSplit to tf.data.Dataset yielding (x, y, sample_weight)."""
    try:
        tf = importlib.import_module("tensorflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for make_tf_dataset(); install tensorflow first."
        ) from exc

    ds = tf.data.Dataset.from_tensor_slices((loaded.x, loaded.y, loaded.sample_weight))
    if shuffle:
        ds = ds.shuffle(buffer_size=loaded.x.shape[0], seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

