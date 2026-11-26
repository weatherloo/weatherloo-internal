"""Bounded remote ingest and training for the fixed-station temporal pilot."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import modal


HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    image = image.pip_install_from_requirements(str(HERE / "requirements.txt")).add_local_dir(
        str(HERE), remote_path="/app"
    )
app = modal.App("weatherloo-temporal-station", image=image)
data = modal.Volume.from_name("weatherloo-temporal-data")
checkpoints = modal.Volume.from_name("weatherloo-temporal-checkpoints")

ingest_image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    root = HERE.parents[1]
    ingest_image = ingest_image.pip_install(
        "numpy>=1.26,<2", "pandas>=2.1,<3", "scipy>=1.11", "xarray==2024.10.0", "cfgrib>=0.9.14",
        "eccodes>=1.6", "netCDF4>=1.6", "zarr>=2.16,<3",
    )
    ingest_image = ingest_image.add_local_file(
        str(root / "pipelines/download/download_hrrr.py"), "/app/ingest/download_hrrr.py"
    ).add_local_file(
        str(root / "pipelines/download/weather_download_common.py"), "/app/ingest/weather_download_common.py"
    ).add_local_file(
        str(root / "src/hrrr_bias_correction/make_data.py"), "/app/ingest/make_data.py"
    )


@app.function(image=ingest_image, cpu=4, memory=16384, timeout=2 * 60 * 60, volumes={"/data": data})
def ingest_pilot():
    """Download two 00Z cycles on remote scratch and retain only processed tensors."""
    import numpy as np
    import xarray as xr

    destination = Path("/data/processed/hrrr_bias_correction/hrrr")
    if destination.exists():
        raise FileExistsError(f"Pilot store already exists: {destination}")

    scratch = Path("/tmp/weatherloo-pilot")
    obs_root = scratch / "observations"
    obs_root.mkdir(parents=True)
    sources = []
    for year in (2021, 2024):
        url = f"https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
        path = obs_root / f"uw_hobo_15min_{year}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        sources.append({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    for day in ("2021-01-02", "2024-01-02"):
        subprocess.run([
            "python", "-u", "/app/ingest/download_hrrr.py",
            "--data-root", str(scratch / "raw"),
            "--cache-dir", str(scratch / "cache"),
            "--max-cache-gb", "2",
            "--start-date", day, "--end-date", day,
            "--cycles", "0", "--max-lead", "48", "--workers", "1",
        ], check=True)
        compact_day = day.replace("-", "")
        path = scratch / "raw/hrrr" / day[:4] / compact_day / f"hrrr_{compact_day}_t00z.nc"
        if not path.is_file():
            raise RuntimeError(f"HRRR downloader did not produce {path}")
        sources.append({"url": f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{compact_day}/conus/",
                        "cycle": f"{day}T00:00:00Z", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    output = scratch / "processed/hrrr"
    subprocess.run([
        "python", "-u", "/app/ingest/make_data.py",
        "--hrrr-root", str(scratch / "raw/hrrr"),
        "--obs-root", str(obs_root), "--out", str(output),
        "--splits", "train", "val", "--limit", "1",
    ], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    for split in ("train", "val"):
        if manifest["splits"][split]["kept"] != 1:
            raise RuntimeError(f"Expected one {split} sample: {manifest['splits'][split]}")
        with xr.open_zarr(output, group=split, consolidated=False) as ds:
            if ds.sizes.get("sample") != 1 or ds.sizes.get("lead") != 49:
                raise RuntimeError(f"Unexpected {split} tensor shape: {dict(ds.sizes)}")
            if not np.isfinite(ds["x"].isel(lead=0).values).all() or not np.isfinite(ds["y"].values).any():
                raise RuntimeError(f"Missing {split} inputs or labels")
    manifest["store"] = str(destination)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "pilot_provenance.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    data.commit()
    return {"volume_path": "/processed/hrrr_bias_correction/hrrr", "splits": {s: manifest["splits"][s]["kept"] for s in ("train", "val")}}


@app.function(image=ingest_image, cpu=4, memory=16384, timeout=4 * 60 * 60, volumes={"/data": data})
def ingest_training_set():
    """Build a small, multi-cycle winter dataset without retaining raw forecasts."""
    import xarray as xr

    dataset = "hrrr_jan_2021_2024_00_12"
    destination = Path("/data/processed/hrrr_bias_correction") / dataset
    if destination.exists():
        raise FileExistsError(f"Training store already exists: {destination}")
    scratch = Path("/tmp/weatherloo-training")
    obs_root = scratch / "observations"
    obs_root.mkdir(parents=True)
    sources = []
    for year in (2021, 2024):
        url = f"https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
        path = obs_root / f"uw_hobo_15min_{year}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        sources.append({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    windows = {
        "train": ("2021-01-02", "2021-01-15", 28),
        "val": ("2024-01-02", "2024-01-06", 10),
    }
    for split, (start, end, _) in windows.items():
        subprocess.run([
            "python", "-u", "/app/ingest/download_hrrr.py",
            "--data-root", str(scratch / "raw"),
            "--cache-dir", str(scratch / "cache"),
            "--max-cache-gb", "2",
            "--start-date", start, "--end-date", end,
            "--cycles", "0,12", "--max-lead", "48", "--workers", "2",
            "--resume",
        ], check=True)
        paths = sorted((scratch / "raw/hrrr" / start[:4]).glob("*/hrrr_*.nc"))
        print(f"{split}: downloaded {len(paths)} cycles", flush=True)
        sources.append({"split": split, "start": start, "end": end, "cycles_utc": [0, 12],
                        "hrrr_cycle_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})

    output = scratch / "processed/hrrr"
    subprocess.run([
        "python", "-u", "/app/ingest/make_data.py",
        "--hrrr-root", str(scratch / "raw/hrrr"),
        "--obs-root", str(obs_root), "--out", str(output),
        "--splits", "train", "val",
    ], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    for split, (_, _, expected) in windows.items():
        kept = manifest["splits"][split]["kept"]
        if kept < expected * 3 // 4:
            raise RuntimeError(f"Too few {split} cycles: {kept}/{expected}")
        with xr.open_zarr(output, group=split, consolidated=False) as ds:
            if ds.sizes.get("sample") != kept or ds.sizes.get("lead") != 49:
                raise RuntimeError(f"Unexpected {split} tensor shape: {dict(ds.sizes)}")
    manifest["store"] = str(destination)
    manifest["requested_windows"] = windows
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source_provenance.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    data.commit()
    return {split: manifest["splits"][split]["kept"] for split in windows}


@app.function(
    gpu=["L4", "A10", "T4"],
    timeout=2 * 60 * 60,
    volumes={"/data": data.with_mount_options(read_only=True), "/checkpoints": checkpoints},
)
def train(model: str, max_samples: int, epochs: int, dataset: str = "hrrr", run_name: str = ""):
    command = [
        "python", "/app/train.py",
        "--store", f"/data/processed/hrrr_bias_correction/{dataset}",
        "--out", f"/checkpoints/temporal_station/{run_name}" if run_name else "/checkpoints/temporal_station",
        "--model", model,
        "--epochs", str(epochs),
    ]
    if max_samples > 0:
        command.extend(("--max-samples", str(max_samples)))
    subprocess.run(command, check=True)
    checkpoints.commit()


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=6 * 60 * 60)
def train_from_new_data():
    counts = ingest_training_set.remote()
    print(f"Built training dataset: {counts}", flush=True)
    for model in ("direct", "autoregressive"):
        train.remote(model, 0, 20, "hrrr_jan_2021_2024_00_12", "jan_2021_2024_00_12_v1")
        print(f"Finished {model} training", flush=True)
    return counts


@app.local_entrypoint()
def main(model: str = "direct", max_samples: int = 100, epochs: int = 2,
         dataset: str = "hrrr", run_name: str = ""):
    if model not in {"direct", "autoregressive"}:
        raise ValueError("model must be direct or autoregressive")
    train.remote(model, max_samples, epochs, dataset, run_name)


@app.local_entrypoint()
def ingest():
    print(ingest_pilot.remote())


@app.local_entrypoint()
def build_training():
    print(ingest_training_set.remote())


@app.local_entrypoint()
def start_training():
    print(train_from_new_data.remote())
