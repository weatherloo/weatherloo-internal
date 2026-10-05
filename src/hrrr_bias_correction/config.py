"""Central configuration for HRRR bias-correction experiments.

This module defines typed defaults for:
- Data locations and loader behavior
- Model architecture dimensions
- Training hyperparameters

It is designed to be imported by a training script, while also supporting
JSON-based overrides for reproducible runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    """Filesystem paths used across preprocessing, loading, and training."""

    hrrr_root: Path = Path("/mnt/wato-drive/c52li/weatherloo-data/hrrr")
    obs_root: Path = field(
        default_factory=lambda: _repo_root()
        / "benchmarking-site"
        / "data"
        / "observations"
        / "eric_d_soulis"
        / "raw"
    )
    zarr_store: Path = Path("/mnt/wato-drive/gguirgui/weatherloo-data/hrrr_bias_correction/hrrr")
    artifact_dir: Path = field(default_factory=lambda: _repo_root() / "artifacts" / "hrrr_bias_correction")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PathsConfig:
        return cls(
            hrrr_root=Path(payload.get("hrrr_root", cls().hrrr_root)),
            obs_root=Path(payload.get("obs_root", cls().obs_root)),
            zarr_store=Path(payload.get("zarr_store", cls().zarr_store)),
            artifact_dir=Path(payload.get("artifact_dir", cls().artifact_dir)),
        )


@dataclass
class DataConfig:
    """Settings for split selection and loader behavior."""

    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    fill_x_nan: float = 0.0
    fill_y_nan: float = 0.0
    shuffle_train: bool = True
    shuffle_seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Architecture settings aligned with make_data and data_loader outputs."""

    n_leads: int = 49
    crop: int = 30
    n_channels: int = 3
    n_targets: int = 2
    conv1_filters: int = 16
    conv2_filters: int = 32
    lstm_units: int = 64

    @property
    def sample_input_shape(self) -> tuple[int, int, int, int]:
        return (self.n_leads, self.crop, self.crop, self.n_channels)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Core optimization and fit-loop hyperparameters."""

    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    clipnorm: float = 1.0
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass
class RuntimeConfig:
    """Runtime behavior that is not part of model/data semantics."""

    seed: int = 42
    mixed_precision: bool = False
    save_best_only: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Top-level experiment config used by the training entrypoint."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        return cls(
            paths=PathsConfig.from_dict(payload.get("paths", {})),
            data=DataConfig.from_dict(payload.get("data", {})),
            model=ModelConfig.from_dict(payload.get("model", {})),
            training=TrainingConfig.from_dict(payload.get("training", {})),
            runtime=RuntimeConfig.from_dict(payload.get("runtime", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        return _stringify_paths(obj)


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    return obj


def load_config(config_path: Path | str | None = None) -> ExperimentConfig:
    """Load defaults, then merge optional JSON overrides from disk."""
    if config_path is None:
        return ExperimentConfig()

    path = Path(config_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must contain a top-level object.")
    return ExperimentConfig.from_dict(payload)


def save_default_config(path: Path | str) -> None:
    """Write a default config JSON template to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ExperimentConfig().to_dict(), indent=2) + "\n")
