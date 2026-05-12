from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import modal


APP_NAME = "weatherloo-train"
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_REMOTE_DIR = "/root/project"
DATA_MOUNT_DIR = PurePosixPath("/data")
ARTIFACT_MOUNT_DIR = PurePosixPath("/artifacts")
DEFAULT_DATA_VOLUME = os.environ.get("WEATHERLOO_MODAL_DATA_VOLUME", "weatherloo-hrrr-data")
DEFAULT_ARTIFACT_VOLUME = os.environ.get(
    "WEATHERLOO_MODAL_ARTIFACT_VOLUME",
    "weatherloo-train-artifacts",
)
DEFAULT_GPU = os.environ.get("WEATHERLOO_MODAL_GPU", "T4")
DEFAULT_TIMEOUT_SECONDS = int(
    os.environ.get("WEATHERLOO_MODAL_TIMEOUT_SECONDS", str(12 * 60 * 60))
)
# Set this once instead of passing an HRRR path on every `modal run`.
# This can be a local path to upload, a mounted volume path, or a public s3:// Zarr URI.
HARDCODED_HRRR_PATH = "s3://hrrrzarr"
HARDCODED_HRRR_VOLUME_SUBPATH = ""

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DEFAULT_DATA_VOLUME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(DEFAULT_ARTIFACT_VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "matplotlib",
        "netCDF4",
        "numpy",
        "pandas",
        "s3fs",
        "torch",
        "xarray",
        "zarr",
    )
    .add_local_dir(
        PROJECT_DIR,
        remote_path=PROJECT_REMOTE_DIR,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            "artifacts",
        ],
    )
)


def _is_remote_uri(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "s3"}


def _default_remote_subpath(local_path: Path, prefix: str) -> str:
    return PurePosixPath(prefix, local_path.name).as_posix()


def _upload_local_path(local_path: Path, remote_subpath: str, volume: modal.Volume) -> str:
    remote_subpath = PurePosixPath(remote_subpath).as_posix()
    with volume.batch_upload() as batch:
        if local_path.is_dir():
            batch.put_directory(str(local_path), remote_subpath)
        else:
            batch.put_file(str(local_path), remote_subpath)
    return remote_subpath


def _maybe_upload_uw_csv(uw_csv: str) -> str | None:
    if not uw_csv:
        return None
    if _is_remote_uri(uw_csv):
        return uw_csv

    local_path = Path(uw_csv).expanduser().resolve()
    if not local_path.exists():
        msg = f"UW CSV path does not exist: {local_path}"
        raise FileNotFoundError(msg)
    remote_subpath = _default_remote_subpath(local_path, "inputs/stations")
    uploaded_subpath = _upload_local_path(local_path, remote_subpath, data_volume)
    return str(DATA_MOUNT_DIR / uploaded_subpath)


def _prepare_hrrr_path() -> str:
    hrrr_path = HARDCODED_HRRR_PATH.strip()
    if not hrrr_path:
        msg = "Set HARDCODED_HRRR_PATH in modal_train.py before launching training."
        raise ValueError(msg)

    if _is_remote_uri(hrrr_path):
        return hrrr_path

    local_path = Path(hrrr_path).expanduser().resolve()
    if local_path.exists():
        target_subpath = HARDCODED_HRRR_VOLUME_SUBPATH or _default_remote_subpath(
            local_path,
            "inputs/hrrr",
        )
        uploaded_subpath = _upload_local_path(local_path, target_subpath, data_volume)
        return str(DATA_MOUNT_DIR / uploaded_subpath)

    if HARDCODED_HRRR_VOLUME_SUBPATH:
        return str(DATA_MOUNT_DIR / PurePosixPath(HARDCODED_HRRR_VOLUME_SUBPATH))

    if hrrr_path.startswith("/data/"):
        return hrrr_path

    msg = f"Hardcoded HRRR path does not exist locally and is not remote: {hrrr_path}"
    raise FileNotFoundError(msg)


def _make_output_subpath(run_name: str) -> str:
    if run_name:
        return PurePosixPath("runs", run_name).as_posix()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return PurePosixPath("runs", f"run-{timestamp}").as_posix()


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [PROJECT_REMOTE_DIR]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_DIR): data_volume,
        str(ARTIFACT_MOUNT_DIR): artifact_volume,
    },
)
def smoke_test() -> dict[str, object]:
    import torch

    completed = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        cwd=PROJECT_REMOTE_DIR,
        env=_pythonpath_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "gpu": DEFAULT_GPU,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_help_preview": completed.stdout.splitlines()[:10],
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        str(DATA_MOUNT_DIR): data_volume,
        str(ARTIFACT_MOUNT_DIR): artifact_volume,
    },
)
def run_training(
    *,
    hrrr_path: str,
    output_subpath: str,
    uw_csv: str | None = None,
    station_ids: str = "",
    weatherstats_limit_rows: int = 4000,
    min_station_count: int = 1,
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
) -> dict[str, object]:
    import torch

    output_dir = str(ARTIFACT_MOUNT_DIR / PurePosixPath(output_subpath))
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--hrrr-path",
        hrrr_path,
        "--output-dir",
        output_dir,
        "--weatherstats-limit-rows",
        str(weatherstats_limit_rows),
        "--min-station-count",
        str(min_station_count),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(learning_rate),
        "--num-workers",
        str(num_workers),
    ]
    if uw_csv:
        command.extend(["--uw-csv", uw_csv])
    if station_ids.strip():
        command.extend(["--station-ids", station_ids.strip()])

    completed = subprocess.run(
        command,
        cwd=PROJECT_REMOTE_DIR,
        env=_pythonpath_env(),
        capture_output=True,
        text=True,
    )
    artifact_volume.commit()

    stdout_tail = completed.stdout.splitlines()[-40:]
    stderr_tail = completed.stderr.splitlines()[-40:]
    if completed.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "returncode": completed.returncode,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                },
                indent=2,
            )
        )

    return {
        "gpu": DEFAULT_GPU,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "output_dir": output_dir,
        "artifact_volume": DEFAULT_ARTIFACT_VOLUME,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


@app.local_entrypoint()
def main(
    run_name: str = "",
    uw_csv: str = "",
    station_ids: str = "",
    weatherstats_limit_rows: int = 4000,
    min_station_count: int = 1,
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    smoke: bool = False,
) -> None:
    if smoke:
        result = smoke_test.remote()
        print(json.dumps(result, indent=2))
        return

    hrrr_path = _prepare_hrrr_path()
    resolved_uw_csv = _maybe_upload_uw_csv(uw_csv)
    output_subpath = _make_output_subpath(run_name)
    result = run_training.remote(
        hrrr_path=hrrr_path,
        output_subpath=output_subpath,
        uw_csv=resolved_uw_csv,
        station_ids=station_ids,
        weatherstats_limit_rows=weatherstats_limit_rows,
        min_station_count=min_station_count,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        num_workers=num_workers,
    )
    print(json.dumps(result, indent=2))
    print(
        (
            "Artifacts are stored in Modal Volume "
            f"'{DEFAULT_ARTIFACT_VOLUME}' under '{output_subpath}'."
        )
    )
