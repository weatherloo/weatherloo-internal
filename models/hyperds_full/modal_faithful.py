"""Paper-faithful HyperDS on HRRR (plus the time dimension), its baselines, and data.

    modal deploy models/hyperds_full/modal_faithful.py
    python -c 'import modal; modal.Function.from_name("weatherloo-hyperds-faithful","stations").spawn()'
"""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/app")
from modal_app import train_image, data_volume, checkpoint_volume, with_shared
from multiyear_plan import DATASET

app = modal.App("weatherloo-hyperds-faithful")
ERA5_ROOT = f"/data/processed/hyperds_full/{DATASET}"
NWP_ROOT = "/data/processed/hyperds_nwp/hrrr_v1"
STATION_SOURCE = "/data/processed/hyperds_nwp/stations_v1"
# v2: co-located ECCC IDs (< 2 km) merged before splitting; v1 put e.g. Wiarton A in train and validation.
STATION_ROOT = "/data/processed/hyperds_nwp/stations_v2"
SOULIS_OVERLAY = "/data/processed/observations/soulis_hourly_v3.npz"
RUN_ROOT = "/checkpoints/hyperds_faithful/hrrr_v2"
EPOCHS = 50
STAGE = 5
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
if modal.is_local():
    cpu_image = with_shared(cpu_image.add_local_dir(str(Path(__file__).resolve().parent), "/app"))


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=3 * 60 * 60, volumes={"/data": data_volume})
def stations():
    from station_labels import build

    data_volume.reload()
    manifest = build(STATION_SOURCE, SOULIS_OVERLAY)
    from station_labels import derive_deduplicated
    manifest, split = derive_deduplicated(STATION_SOURCE, STATION_ROOT)
    data_volume.commit()
    summary = {name: len(ids) for name, ids in split.items()}
    print(json.dumps({"station_split": summary, "kept": len(manifest["stations"]),
                      "duplicates_removed": len(manifest["duplicates_removed"])}), flush=True)
    return summary


@app.function(image=train_image, cpu=4, memory=8192, timeout=20 * 60)
def smoke():
    import os
    import subprocess
    subprocess.run(["python", "-u", "-m", "unittest", "test_faithful", "test_nwp", "-v"], check=True, cwd="/app",
                   env={**os.environ, "HYPERDS_UPSTREAM_ROOT": "/opt/hyperds"})


def faithful_args(mode, out, epochs, stop, pool=4):
    return ["--mode", mode, "--era5-store", ERA5_ROOT, "--nwp-store", NWP_ROOT, "--station-store", STATION_ROOT,
            "--out", out, "--epochs", str(epochs), "--stop-after-epoch", str(stop), "--input-pool", str(pool)]


def run_root(pool):
    # pool 4 = paper-faithful 1-degree input; pool 1 = full 0.25-degree HRRR input (ablation).
    return RUN_ROOT if pool == 4 else f"{RUN_ROOT}_pool{pool}"


@app.function(image=train_image, gpu=["A10", "L4"], cpu=4, memory=32768, timeout=2 * 60 * 60,
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/checkpoints": checkpoint_volume})
def probe(mode: str = "nwp"):
    """Two short epochs on 8 batches: GPU memory and per-batch time before a full run."""
    from faithful_train import main

    checkpoint_volume.reload()
    out = f"/checkpoints/hyperds_faithful/probe/{mode}"
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    main(faithful_args(mode, out, 2, 1) + ["--max-batches", "8"])
    record = json.loads((Path(out) / "metrics.jsonl").read_text().splitlines()[-1])
    return {"seconds_for_8_train_batches_plus_val": record["seconds"], "peak_gpu_gib": record["peak_gpu_gib"]}


# Retries: a preempted or externally stopped container resumes from latest.pt instead of ending the chain.
@app.function(image=train_image, gpu=["A10", "L4"], cpu=4, memory=32768, timeout=24 * 60 * 60,
              retries=modal.Retries(max_retries=3, initial_delay=30.0),
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/checkpoints": checkpoint_volume})
def train_stage(mode: str, stop_epoch: int = STAGE, pool: int = 4):
    from faithful_train import main

    checkpoint_volume.reload()
    out = f"{run_root(pool)}/{mode}"
    main(faithful_args(mode, out, EPOCHS, stop_epoch, pool))
    checkpoint_volume.commit()
    if (Path(out) / "evaluation.json").is_file():
        print(json.dumps({"training_finished": [mode, pool]}), flush=True)  # completed or early-stopped
    elif stop_epoch < EPOCHS:
        following = train_stage.spawn(mode, min(stop_epoch + STAGE, EPOCHS), pool)
        print(json.dumps({"next_training_stage": [mode, min(stop_epoch + STAGE, EPOCHS), pool],
                          "call_id": following.object_id}), flush=True)


@app.function(image=train_image, gpu=["A10", "L4"], cpu=4, memory=32768, timeout=24 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=30.0),
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/checkpoints": checkpoint_volume})
def baseline(kind: str, pool: int = 4):
    from faithful_baselines import main

    checkpoint_volume.reload()
    main(["--kind", kind, "--era5-store", ERA5_ROOT, "--nwp-store", NWP_ROOT, "--station-store", STATION_ROOT,
          "--out", f"{run_root(pool)}/baseline_{kind}", "--epochs", str(EPOCHS), "--input-pool", str(pool)])
    checkpoint_volume.commit()


@app.function(image=train_image, gpu=["A10", "L4"], cpu=4, memory=32768, timeout=4 * 60 * 60,
              volumes={"/data": data_volume.with_mount_options(read_only=True), "/checkpoints": checkpoint_volume})
def finalize(mode: str, pool: int = 4):
    """Stop training where it is and score the best checkpoint on the 2025 test stations."""
    import torch
    from faithful_train import main

    checkpoint_volume.reload()
    out = f"{run_root(pool)}/{mode}"
    epoch = torch.load(f"{out}/latest.pt", map_location="cpu", weights_only=False)["epoch"]
    # --epochs equal to the completed epoch: no further training, straight to the best-checkpoint test.
    main(faithful_args(mode, out, epoch, epoch, pool))
    checkpoint_volume.commit()
    return json.loads((Path(out) / "evaluation.json").read_text())["groups"]


@app.function(image=train_image, cpu=4, memory=32768, timeout=60 * 60,
              volumes={"/data": data_volume.with_mount_options(read_only=True),
                       "/checkpoints": checkpoint_volume.with_mount_options(read_only=True)})
def lead_metrics(pool: int = 1, runs: str = "nwp,baseline_unet,baseline_edsr"):
    """Hourly (+1..+48 h) RMSE/MAE/bias per station group from saved 2025 test predictions.

    Every method is scored on the identical finite pairs for each group, variable and hour.
    Interpolation baselines come from the HyperDS file (bilinear 0.25-degree HRRR at the station).
    """
    import numpy as np

    root = Path(run_root(pool))
    stations = json.loads((Path(STATION_ROOT) / "stations.json").read_text())["stations"]
    split = json.loads((Path(STATION_ROOT) / "split.json").read_text())
    ids = [s["id"] for s in stations]
    groups = {"test": [ids.index(i) for i in split["test"]], "eric_d_soulis": [ids.index("eric_d_soulis")],
              "cyyz": [ids.index("eccc_51459")]}
    forecasts, observation, inits = {}, None, None
    for run in runs.split(","):
        with np.load(root / run / "predictions.npz") as saved:
            forecasts[run] = saved["forecast"][..., :2].astype(np.float64)  # wind m/s, t2m degC
            if inits is None:
                inits, observation = saved["initializations"], saved["observation"][..., :2].astype(np.float64)
                if "nwp_forecast" in saved.files:
                    forecasts["interp_025deg"] = saved["nwp_forecast"][..., :2].astype(np.float64)
            elif not np.array_equal(saved["initializations"], inits):
                raise ValueError(f"{run}: initialization order differs")
    forecasts = {k: v.copy() for k, v in forecasts.items()}
    for v in forecasts.values():
        v[..., 0] *= 3.6  # km/h
    observation = observation.copy()
    observation[..., 0] *= 3.6
    report = {"units": {"wind": "km/h", "t2m": "degC"}, "initializations": int(len(inits)), "groups": {}}
    for group, index in groups.items():
        obs = observation[:, :, index]
        mask = np.isfinite(obs)
        for values in forecasts.values():
            mask &= np.isfinite(values[:, :, index])
        entry = {}
        for vi, variable in enumerate(("wind", "t2m")):
            m = mask[..., vi]
            per = {}
            for name, values in forecasts.items():
                error = np.where(m, values[:, :, index, vi] - obs[..., vi], np.nan)
                per[name] = {"rmse": [float(np.sqrt(np.nanmean(error[:, h] ** 2))) for h in range(1, 49)],
                             "mae": [float(np.nanmean(np.abs(error[:, h]))) for h in range(1, 49)],
                             "bias": [float(np.nanmean(error[:, h])) for h in range(1, 49)],
                             "rmse_all": float(np.sqrt(np.nanmean(error[:, 1:] ** 2)))}
            entry[variable] = {"count": [int(m[:, h].sum()) for h in range(1, 49)], "methods": per}
        report["groups"][group] = entry
    return report
