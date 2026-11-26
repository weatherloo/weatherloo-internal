"""Remote NWP-input HyperDS: HRRR/AIFS shard ingestion and chained L4/A10 training.

    modal deploy models/hyperds_full/modal_nwp.py
    python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-nwp","run").spawn("hrrr").object_id)'
"""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/app")
from modal_app import train_image, data_volume, checkpoint_volume
from multiyear_plan import DATASET

app = modal.App("weatherloo-hyperds-nwp")
VERSION = "v1"
ERA5_ROOT = f"/data/processed/hyperds_full/{DATASET}"
WARM_START = "/checkpoints/hyperds_full/waterloo_2018_2025_stationqc_v3/paper/best.pt"
HRRR_SOULIS_2024 = "/checkpoints/hyperds_full/waterloo_2018_2025_stationqc_v3/diagnostics/validation_hrrr.npz"
EPOCHS = 30
STAGE = 5
nwp_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0", "scipy", "numpy")
if modal.is_local():
    nwp_image = nwp_image.add_local_dir(str(Path(__file__).resolve().parent), "/app")


def nwp_root(source):
    return f"/data/processed/hyperds_nwp/{source}_{VERSION}"


# r2: zero-initialized decoder output (start exactly at raw NWP) and station_beta=1.
# r3: 6-hourly anchor encoding with hourly decoder queries and hourly labels; L4/A10.
# r4: station loss and checkpoint selection scaled by raw-forecast error per variable.
# r5: checkpoint selection by Soulis skill ratio vs the raw forecast, wind and t2m equally.
RUN_TAG = "r5"


def run_root(source):
    return f"/checkpoints/hyperds_nwp/{source}_{VERSION}_{RUN_TAG}"


@app.function(image=nwp_image, cpu=8, memory=32768, timeout=3 * 60 * 60, max_containers=24, retries=1,
              volumes={"/data": data_volume})
def month(source: str, key: str):
    from nwp_data import build_month

    data_volume.reload()
    result = build_month(source, key, Path(ERA5_ROOT) / "shards" / key, Path(nwp_root(source)) / "shards" / key)
    data_volume.commit()
    return {key: result["samples"], "rejected": len(result["failures"])}


@app.function(image=nwp_image, cpu=2, memory=8192, timeout=30 * 60,
              volumes={"/data": data_volume, "/checkpoints": checkpoint_volume})
def check_reader():
    """Cross-check native station interpolation against the existing 2024 Soulis HRRR extraction."""
    from datetime import datetime, timezone
    import numpy as np
    from nwp_data import Reader

    with np.load(HRRR_SOULIS_2024) as saved:
        stamps, reference = saved["initializations"], saved["forecast"]
    report = {}
    for source in ("hrrr", "aifs"):
        reader = Reader(source)
        index = int(np.flatnonzero(stamps == int(datetime(2024, 7, 1, 12, tzinfo=timezone.utc).timestamp()))[0])
        grid, station = reader.read(datetime.fromtimestamp(int(stamps[index]), tz=timezone.utc))
        leads = reader.leads
        temperature = station[:, 0, 2]
        speed = np.hypot(station[:, 0, 0], station[:, 0, 1]) * 3.6
        report[source] = {"grid_shape": list(grid.shape), "station_shape": list(station.shape),
                          "soulis_t2m": temperature.round(2).tolist()[:10],
                          "grid_t2m_mean": float(grid[:, 2].mean()), "sp_mean": float(grid[:, 3].mean()),
                          "tp_max_mm_h": float(grid[:, 4].max()), "regrid": reader.regrid_note}
        if source == "hrrr":
            report[source]["max_abs_diff_vs_2024_extraction"] = {
                "t2m": float(np.abs(temperature - reference[index, leads, 1]).max()),
                "wind_kmh": float(np.abs(speed - reference[index, leads, 0]).max())}
        else:
            report[source]["mean_abs_diff_vs_hrrr_soulis_t2m"] = float(
                np.abs(temperature - reference[index, leads, 1]).mean())
    print(json.dumps(report, indent=1), flush=True)
    return report


@app.function(image=nwp_image, cpu=2, memory=4096, timeout=30 * 60, volumes={"/data": data_volume})
def finalize(source: str):
    from nwp_data import finalize as build

    data_volume.reload()
    manifest = build(source, ERA5_ROOT, nwp_root(source))
    data_volume.commit()
    print(json.dumps({"nwp_dataset_verified": source, "splits": manifest["splits"]}), flush=True)
    return manifest["splits"]


@app.function(image=nwp_image, cpu=.25, memory=1024, timeout=24 * 60 * 60)
def ingest(source: str):
    from nwp_data import months

    keys = months(source)
    for result in month.starmap([(source, key) for key in keys]):
        print(json.dumps({"committed_month": result, "source": source}), flush=True)
    return finalize.remote(source)


@app.function(image=train_image, cpu=4, memory=8192, timeout=15 * 60)
def smoke():
    import os
    import subprocess
    subprocess.run(["python", "-u", "-m", "unittest", "test_nwp", "-v"], check=True, cwd="/app",
                   env={**os.environ, "HYPERDS_UPSTREAM_ROOT": "/opt/hyperds"})


# The model is small (ResNet-18 encoders, batch 4); L4/A10 cost ~1/3 of an A100.
@app.function(image=train_image, gpu=["L4", "A10"], cpu=4, memory=16384, timeout=24 * 60 * 60,
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/checkpoints": checkpoint_volume})
def train_stage(source: str, mode: str, stop_epoch: int = STAGE):
    from nwp_train import main

    checkpoint_volume.reload()
    main(["--mode", mode, "--era5-store", ERA5_ROOT, "--nwp-store", nwp_root(source),
          "--out", f"{run_root(source)}/{mode}", "--epochs", str(EPOCHS),
          "--stop-after-epoch", str(stop_epoch), "--warm-start", WARM_START])
    checkpoint_volume.commit()
    if stop_epoch < EPOCHS:
        following = train_stage.spawn(source, mode, min(stop_epoch + STAGE, EPOCHS))
        print(json.dumps({"next_training_stage": [source, mode, min(stop_epoch + STAGE, EPOCHS)],
                          "call_id": following.object_id}), flush=True)
    else:
        print(json.dumps({"nwp_training_completed": [source, mode]}), flush=True)


@app.function(image=nwp_image, cpu=.25, memory=1024, timeout=24 * 60 * 60)
def run(source: str):
    """Ingest (resumable), audit, test, then train direct and autoregressive in parallel."""
    splits = ingest.remote(source)
    if not splits["train"] or not splits["val"]:
        raise RuntimeError(f"insufficient {source} data: {splits}")
    smoke.remote()
    calls = {mode: train_stage.spawn(source, mode, STAGE).object_id for mode in ("nwp", "nwp_autoregressive")}
    print(json.dumps({"nwp_training_launched": source, "calls": calls}), flush=True)
    return calls
