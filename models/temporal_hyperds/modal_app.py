"""Remote multi-station data build and GPU training for temporal HyperDS."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

import modal


HERE = Path(__file__).resolve().parent
train_image = modal.Image.debian_slim(python_version="3.11")
ingest_image = modal.Image.debian_slim(python_version="3.11")
if modal.is_local():
    root = HERE.parents[1]
    train_image = train_image.pip_install_from_requirements(str(HERE / "requirements.txt"))
    train_image = train_image.add_local_dir(str(HERE), remote_path="/app")
    ingest_image = ingest_image.pip_install(
        "numpy>=1.26,<2", "pandas>=2.1,<3", "scipy>=1.11", "xarray==2024.10.0",
        "cfgrib>=0.9.14", "eccodes>=1.6", "netCDF4>=1.6", "zarr>=2.16,<3",
    )
    for source, destination in (
        (root / "pipelines/download/download_hrrr.py", "/workspace/pipelines/download/download_hrrr.py"),
        (root / "pipelines/download/weather_download_common.py", "/workspace/pipelines/download/weather_download_common.py"),
        (root / "pipelines/download/fetch_station_observations.py", "/workspace/pipelines/download/fetch_station_observations.py"),
        (root / "src/hrrr_bias_correction/make_data.py", "/workspace/src/hrrr_bias_correction/make_data.py"),
        (HERE / "make_data.py", "/workspace/models/temporal_hyperds/make_data.py"),
    ):
        ingest_image = ingest_image.add_local_file(str(source), destination)

app = modal.App("weatherloo-temporal-hyperds")
data = modal.Volume.from_name("weatherloo-temporal-data")
checkpoints = modal.Volume.from_name("weatherloo-temporal-checkpoints")
DATASET = "jan_2021_2024_multistation"


@app.function(image=ingest_image, cpu=4, memory=16384, timeout=4 * 60 * 60,
              volumes={"/data": data})
def build_data():
    destination = Path("/data/processed/temporal_hyperds") / DATASET
    if destination.exists():
        raise FileExistsError(f"processed data already exists: {destination}")
    scratch = Path("/tmp/weatherloo-hyperds")
    obs_root = scratch / "observations"
    obs_root.mkdir(parents=True, exist_ok=True)
    sources = []
    for year in (2021, 2024):
        url = f"https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
        path = obs_root / f"uw_hobo_15min_{year}.csv"
        request = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
        sources.append({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sys.path.insert(0, "/workspace/pipelines/download")
    from fetch_station_observations import fetch_eccc_hourly
    for station_id in (41983, 48569, 55299):
        for year in (2021, 2024):
            path = obs_root / f"eccc_{station_id}_{year}.json"
            rows = fetch_eccc_hourly(station_id, year)
            path.write_text(json.dumps({"features": rows}))
            sources.append({"station_id": station_id, "year": year,
                            "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for start, end in (("2021-01-02", "2021-01-15"), ("2024-01-02", "2024-01-06")):
        subprocess.run([
            "python", "-u", "/workspace/pipelines/download/download_hrrr.py",
            "--data-root", str(scratch / "raw"), "--cache-dir", str(scratch / "cache"),
            "--max-cache-gb", "2", "--start-date", start, "--end-date", end,
            "--cycles", "0,12", "--max-lead", "48", "--workers", "2", "--resume",
        ], check=True)
    output = scratch / "processed"
    subprocess.run([
        "python", "-u", "/workspace/models/temporal_hyperds/make_data.py",
        "--hrrr-root", str(scratch / "raw/hrrr"), "--obs-root", str(obs_root),
        "--out", str(output),
    ], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["splits"]["train"]["samples"] < 21 or manifest["splits"]["val"]["samples"] < 7:
        raise RuntimeError(f"too few paired cycles: {manifest['splits']}")
    for split, stations in (("train", range(3)), ("val", range(4))):
        for station in stations:
            if manifest["splits"][split]["valid_target_hours"][station][0] < 24:
                raise RuntimeError(f"too few {split} temperature labels at station {station}")
    manifest["sources"] = sources
    manifest["hrrr_files"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted((scratch / "raw/hrrr").glob("*/*/hrrr_*.nc"))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    data.commit()
    return manifest["splits"]


@app.function(image=train_image, gpu=["L4", "A10", "T4"], timeout=2 * 60 * 60,
              volumes={"/data": data.with_mount_options(read_only=True), "/checkpoints": checkpoints})
def train(model):
    subprocess.run([
        "python", "-u", "/app/train.py", "--model", model, "--epochs", "20",
        "--store", f"/data/processed/temporal_hyperds/{DATASET}",
        "--out", f"/checkpoints/temporal_hyperds/{DATASET}",
    ], check=True)
    checkpoints.commit()


@app.function(image=modal.Image.debian_slim(python_version="3.11"), timeout=6 * 60 * 60)
def run_experiment():
    counts = build_data.remote()
    print(f"Built multi-station data: {counts}", flush=True)
    for model in ("direct", "autoregressive"):
        train.remote(model)
        print(f"Finished {model}", flush=True)
    return counts
