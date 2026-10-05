#!/usr/bin/env python3
"""Training entrypoint for the HRRR bias-correction CNN-LSTM."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# Allow direct script execution via: python src/hrrr_bias_correction/train.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from .config import ExperimentConfig, load_config, save_default_config
    from .data_loader import load_split_arrays, make_tf_dataset
except ImportError:
    from config import ExperimentConfig, load_config, save_default_config
    from data_loader import load_split_arrays, make_tf_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file. If omitted, uses built-in defaults.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        help="Write a default JSON config template to this path and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and build model but skip model.fit.",
    )
    return parser.parse_args(argv)


def _import_tensorflow() -> Any:
    try:
        return importlib.import_module("tensorflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to run training. Install dependencies in a "
            "Python 3.10-3.12 environment (3.11 recommended)."
        ) from exc


def _set_global_seed(tf: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_optimizer(tf: Any, cfg: ExperimentConfig):
    adam_kwargs = {
        "learning_rate": cfg.training.learning_rate,
        "clipnorm": cfg.training.clipnorm,
    }
    if cfg.training.weight_decay > 0:
        adam_kwargs["weight_decay"] = cfg.training.weight_decay

    try:
        return tf.keras.optimizers.Adam(**adam_kwargs)
    except TypeError:
        # Older TensorFlow builds may not support Adam(weight_decay=...).
        adam_kwargs.pop("weight_decay", None)
        return tf.keras.optimizers.Adam(**adam_kwargs)


def _build_callbacks(tf: Any, cfg: ExperimentConfig) -> list[Any]:
    artifact_dir = cfg.paths.artifact_dir
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Any] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=cfg.training.reduce_lr_patience,
            factor=cfg.training.reduce_lr_factor,
            min_lr=cfg.training.min_learning_rate,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=cfg.runtime.save_best_only,
            save_weights_only=False,
        ),
    ]
    return callbacks


def run_training(cfg: ExperimentConfig, *, dry_run: bool = False) -> int:
    tf = _import_tensorflow()

    try:
        from .model_arch import build_model
    except ImportError:
        from model_arch import build_model

    if cfg.runtime.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    _set_global_seed(tf, cfg.runtime.seed)

    train_loaded = load_split_arrays(
        cfg.paths.zarr_store,
        cfg.data.train_split,
        fill_x_nan=cfg.data.fill_x_nan,
        fill_y_nan=cfg.data.fill_y_nan,
    )
    val_loaded = load_split_arrays(
        cfg.paths.zarr_store,
        cfg.data.val_split,
        fill_x_nan=cfg.data.fill_x_nan,
        fill_y_nan=cfg.data.fill_y_nan,
    )

    train_ds = make_tf_dataset(
        train_loaded,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.shuffle_train,
        seed=cfg.data.shuffle_seed,
    )
    val_ds = make_tf_dataset(
        val_loaded,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    model = build_model(
        input_shape=cfg.model.sample_input_shape,
        n_outputs=cfg.model.n_targets,
    )
    model.compile(
        optimizer=_build_optimizer(tf, cfg),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    artifact_dir = cfg.paths.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "resolved_config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    print(f"train samples: {train_loaded.x.shape[0]}")
    print(f"val samples:   {val_loaded.x.shape[0]}")
    print(f"input shape:   {cfg.model.sample_input_shape}")

    if dry_run:
        model.summary()
        print("Dry run complete. Skipped model.fit().")
        return 0

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.epochs,
        callbacks=_build_callbacks(tf, cfg),
        verbose=1,
    )

    history_path = artifact_dir / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2) + "\n")

    final_model_path = artifact_dir / "final_model.keras"
    model.save(str(final_model_path))

    print(f"Saved training history: {history_path}")
    print(f"Saved final model:      {final_model_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.write_default_config is not None:
        save_default_config(args.write_default_config)
        print(f"Wrote default config: {args.write_default_config}")
        return 0

    cfg = load_config(args.config)
    return run_training(cfg, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
