"""Remote multi-year ingestion and resumable full HyperDS forecast training."""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0,"/app")
from modal_app import ingest_image, train_image, data_volume, checkpoint_volume
from multiyear_plan import DATASET, plan


app = modal.App("weatherloo-hyperds-multiyear")
RUN = "waterloo_2018_2025_stationqc_v3"
DATA_ROOT = f"/data/processed/hyperds_full/{DATASET}"
OBS_ROOT = "/data/raw/observations/hyperds_2018_2026"
benchmark_image = ingest_image
if modal.is_local():
    site = Path(__file__).resolve().parents[2]/"benchmarking-site/data"
    for source in sorted(site.glob("*/*_2025.npz")):
        benchmark_image = benchmark_image.add_local_file(str(source),"/benchmark/"+str(source.relative_to(site)))
    for station in ("cyyz","eric_d_soulis"):
        source = site/"observations"/station/"observations_6h_2025.json"
        benchmark_image = benchmark_image.add_local_file(str(source),f"/benchmark/observations/{station}/{source.name}")


@app.function(image=ingest_image,cpu=4,memory=8192,timeout=2*60*60,volumes={"/data":data_volume})
def observations():
    sys.path.insert(0,"/app")
    from multiyear import prepare_observations

    data_volume.reload()
    result = prepare_observations(OBS_ROOT)
    data_volume.commit()
    return result


@app.function(image=ingest_image,cpu=4,memory=8192,timeout=4*60*60,max_containers=8,retries=1,
              volumes={"/data":data_volume})
def month(key: str):
    sys.path.insert(0,"/app")
    from multiyear import build_month

    data_volume.reload()
    result = build_month(Path(DATA_ROOT)/"shards"/key,key,Path(OBS_ROOT)/"hourly.npz")
    data_volume.commit()
    return {key:result["samples"]}


@app.function(image=ingest_image,cpu=4,memory=8192,timeout=60*60,volumes={"/data":data_volume})
def repair_observations():
    import subprocess
    from repair_soulis import build_overlay

    subprocess.run(["python","-u","/app/test_soulis.py"],check=True)
    data_volume.reload()
    report = build_overlay(OBS_ROOT,"/data/processed/observations/soulis_hourly_v3.npz")
    data_volume.commit()
    print(json.dumps({"soulis_overlay_saved":report}),flush=True)
    return report


@app.function(image=ingest_image,cpu=4,memory=8192,timeout=60*60,volumes={"/data":data_volume})
def audit():
    sys.path.insert(0,"/app")
    from multiyear import finalize

    data_volume.reload()
    report = finalize(DATA_ROOT,Path(OBS_ROOT)/"sources.json")
    data_volume.commit()
    print(json.dumps({"multiyear_dataset_verified":report}),flush=True)
    return report


@app.function(image=ingest_image,cpu=1,memory=2048,timeout=60,volumes={"/data":data_volume})
def status():
    data_volume.reload()
    root = Path(DATA_ROOT)
    done = [json.loads(path.read_text()) for path in sorted(root.glob("shards/*/manifest.json"))]
    result = {"dataset":DATASET,"run":RUN,"planned":plan()["planned_initializations"],"completed_months":len(done),
              "total_months":96,"completed_initializations":{split:sum(r["samples"] for r in done if r["split"]==split)
                                                                 for split in ("train","val","test")},
              "rejected_initializations":sum(len(r["failures"]) for r in done),
              "dataset_audited":(root/"audit.json").is_file(),
              "observations_cached":(Path(OBS_ROOT)/"hourly.npz").is_file()}
    if (root/"manifest.json").is_file():
        result["station_overlay_schema"] = json.loads((root/"manifest.json").read_text()).get("station_overlay",{}).get("schema")
    if done:
        result["months"] = [{"month":r["month"],"samples":r["samples"],"planned":r["planned"]} for r in done]
    return result


@app.function(image=ingest_image,cpu=.125,memory=512,timeout=24*60*60)
def ingest(observations_call_id: str=""):
    print(json.dumps({"multiyear_plan":plan()}),flush=True)
    if observations_call_id:
        modal.FunctionCall.from_id(observations_call_id).get()
    else:
        observations.remote()
    # Disjoint month directories allow independent writers and resumable builds.
    for result in month.map(plan()["months"]):
        print(json.dumps({"committed_month":result}),flush=True)
    return audit.remote()


@app.function(image=train_image,cpu=4,memory=8192,timeout=10*60)
def smoke():
    import subprocess
    subprocess.run(["python","-u","/app/test_multiyear.py"],check=True)
    subprocess.run(["python","-u","/app/test_model.py"],check=True)


@app.function(image=ingest_image,cpu=2,memory=4096,timeout=10*60)
def archive_samples():
    sys.path.insert(0,"/app")
    from datetime import datetime, timezone
    import numpy as np
    from data import fetch_satellite

    result = []
    for init,bucket in ((datetime(2018,1,1,12,tzinfo=timezone.utc),"noaa-goes16"),
                        (datetime(2025,7,1,18,tzinfo=timezone.utc),"noaa-goes19")):
        values,sources = fetch_satellite(init,bucket,production_cutoff=True,fill_small_gaps=True)
        assert values.shape==(2,4,128,256) and np.isfinite(values).all()
        result.append({"initialization":init.isoformat(),"bucket":bucket,"shape":list(values.shape),
                       "band_mean":values.mean(axis=(0,2,3)).tolist(),"sources":sources})
    print(json.dumps({"archive_samples_verified":result}),flush=True)
    return [{key:value for key,value in sample.items() if key!="sources"} for sample in result]


@app.function(image=benchmark_image,cpu=2,memory=4096,timeout=30*60,
              volumes={"/checkpoints":checkpoint_volume.with_mount_options(read_only=True),"/data":data_volume})
def benchmark(mode: str):
    import subprocess

    checkpoint_volume.reload()
    data_volume.reload()
    destination = f"/data/published/hyperds_full/{RUN}/hyperds_{mode}"
    subprocess.run(["python","-u","/app/benchmark.py","--predictions",
                    f"/checkpoints/hyperds_full/{RUN}/{mode}/predictions.npz",
                    "--benchmark-root","/benchmark","--out",destination,"--method",f"hyperds_{mode}"],check=True)
    data_volume.commit()
    return destination


@app.function(image=train_image,gpu="A100-80GB",cpu=4,memory=16384,timeout=24*60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def train_stage(mode: str,stop_epoch: int=5):
    sys.path.insert(0,"/app")
    from train import main

    checkpoint_volume.reload()
    main(["--mode",mode,"--store",DATA_ROOT,"--out",f"/checkpoints/hyperds_full/{RUN}/{mode}",
          "--epochs","50","--stop-after-epoch",str(stop_epoch)])
    checkpoint_volume.commit()
    if stop_epoch<50:
        following = train_stage.spawn(mode,min(stop_epoch+5,50))
        print(json.dumps({"next_training_stage":mode,"stop_epoch":min(stop_epoch+5,50),"call_id":following.object_id}),flush=True)
    elif mode=="paper":
        for extension in ("direct","autoregressive"):
            following = train_stage.spawn(extension,5)
            print(json.dumps({"temporal_training_launched":extension,"call_id":following.object_id}),flush=True)
    else:
        destination = benchmark.remote(mode)
        print(json.dumps({"benchmark_comparison_saved":mode,"path":destination}),flush=True)
        print(json.dumps({"multiyear_training_completed":mode}),flush=True)


@app.function(image=ingest_image,cpu=.125,memory=512,timeout=24*60*60)
def run_training(ingest_call_id: str):
    report = modal.FunctionCall.from_id(ingest_call_id).get()
    if not report["passed"]:
        raise RuntimeError("dataset audit failed")
    # Rebuild the root manifest after ingestion so the versioned Soulis repair
    # is included even when older ingestion workers built the immutable shards.
    repair_observations.remote()
    report = audit.remote()
    smoke.remote()
    call = train_stage.spawn("paper",5)
    print(json.dumps({"multiyear_training_launched":"paper","call_id":call.object_id}),flush=True)
    return call.object_id
