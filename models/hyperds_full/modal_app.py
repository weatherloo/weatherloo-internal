"""Durable remote ingestion and training of full HyperDS plus temporal extensions."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import modal


UPSTREAM_REVISION = "1fe6d0b08e759548655009e1747428f9136a7839"
DATASET = "waterloo_era5_goes_v1"
data_volume = modal.Volume.from_name("weatherloo-temporal-data")
checkpoint_volume = modal.Volume.from_name("weatherloo-temporal-checkpoints")
ingest_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4", "xarray==2024.10.0", "zarr==2.18.7", "gcsfs==2024.10.0", "s3fs==2024.10.0",
    "h5netcdf==1.4.1", "scipy==1.14.1", "pyproj==3.7.0",
)
train_image = modal.Image.debian_slim(python_version="3.11").apt_install("git")
train_image = train_image.run_commands(
    "git clone https://github.com/ESDALab/HyperDS_tgrs.git /opt/hyperds",
    f"git -C /opt/hyperds checkout {UPSTREAM_REVISION}",
)
if modal.is_local():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    train_image = train_image.pip_install_from_requirements(str(here / "requirements.txt"))
    train_image = train_image.add_local_dir(str(here), "/app")
    ingest_image = ingest_image.add_local_dir(str(here), "/app")
    for source, target in (
        (root / "pipelines/download/fetch_station_observations.py", "/workspace/pipelines/download/fetch_station_observations.py"),
        (root / "src/hrrr_bias_correction/make_data.py", "/workspace/src/hrrr_bias_correction/make_data.py"),
    ):
        ingest_image = ingest_image.add_local_file(str(source), target)

app = modal.App("weatherloo-hyperds-full")


@app.function(image=ingest_image, cpu=8, memory=32768, timeout=4*60*60,
              volumes={"/data":data_volume})
def build_data():
    sys.path.insert(0, "/app")
    from data import ingest

    destination = Path(f"/data/processed/hyperds_full/{DATASET}")
    if (destination / "manifest.json").is_file():
        return json.loads((destination / "manifest.json").read_text())["splits"]
    staging = Path(tempfile.mkdtemp(prefix="hyperds-full-", dir="/tmp")) / "processed"
    manifest = ingest(staging)
    import shutil
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, destination)
    data_volume.commit()
    return manifest["splits"]


@app.function(image=train_image, cpu=4, memory=16384, timeout=10*60)
def smoke():
    subprocess.run(["python", "-u", "/app/train.py", "--smoke"], check=True)
    subprocess.run(["python", "-u", "/app/test_model.py"], check=True)


@app.function(image=train_image, gpu=["L4","A10","T4"], cpu=4, memory=16384, timeout=12*60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True), "/checkpoints":checkpoint_volume})
def train(mode: str):
    sys.path.insert(0,"/app")
    from train import main

    main([
        "--mode", mode,
        "--store", f"/data/processed/hyperds_full/{DATASET}",
        "--out", f"/checkpoints/hyperds_full/{DATASET}/{mode}",
    ])
    checkpoint_volume.commit()


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=24*60*60)
def run_experiment(data_call_id: str = ""):
    counts = modal.FunctionCall.from_id(data_call_id).get() if data_call_id else build_data.remote()
    print(f"Dataset build completed: {counts}", flush=True)
    smoke.remote()
    train.remote("paper")
    print("Completed paper model; initializing both temporal extensions",flush=True)
    modal.FunctionCall.gather(train.spawn("direct"),train.spawn("autoregressive"))
    print("Completed both temporal extensions",flush=True)
