"""Independent GPU experiment on a frozen view of already-downloaded data."""

import json
from pathlib import Path
import subprocess
import sys

import modal

if not modal.is_local():
    sys.path.insert(0,"/app")
from modal_app import train_image, data_volume, checkpoint_volume
from multiyear_plan import DATASET


app = modal.App("weatherloo-hyperds-feasibility")
RUN = "feasibility_2018_2020_stationqc_v2"
SOURCE = f"/data/processed/hyperds_full/{DATASET}"
OUTPUT = f"/checkpoints/hyperds_full/{RUN}"
image = train_image
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for station in ("cyyz","eric_d_soulis"):
        for year in (2018,2020,2021):
            source = root/f"benchmarking-site/data/observations/{station}/observations_6h_{year}.json"
            image = image.add_local_file(str(source),f"/benchmark/observations/{station}/{source.name}")
    helper = "pipelines/benchmarks/methods/cnn_lstm_bias_correction/compute_benchmark.py"
    image = image.add_local_file(str(root/helper),f"/workspace/{helper}")


@app.function(image=image,cpu=4,memory=8192,timeout=2*60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def prepare():
    sys.path.insert(0,"/app")
    from feasibility import freeze_view, prepare_baselines, prepare_2021

    data_volume.reload()
    checkpoint_volume.reload()
    manifest = freeze_view(SOURCE,OUTPUT)
    print(json.dumps({"feasibility_view_frozen":manifest["splits"]}),flush=True)
    baselines = prepare_baselines(SOURCE,OUTPUT,manifest)
    holdout = prepare_2021(SOURCE,OUTPUT)
    checkpoint_volume.commit()
    return {"splits":manifest["splits"],"baselines":baselines,**holdout}


@app.function(image=image,cpu=4,memory=8192,timeout=10*60)
def smoke():
    subprocess.run(["python","-u","/app/test_feasibility.py"],check=True)
    subprocess.run(["python","-u","/app/test_multiyear.py"],check=True)
    subprocess.run(["python","-u","/app/test_model.py"],check=True)


@app.function(image=image,gpu=["L4","A10"],cpu=4,memory=16384,max_containers=2,timeout=12*60*60,retries=1,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def train(mode: str):
    sys.path.insert(0,"/app")
    from feasibility import train_mode

    data_volume.reload()
    checkpoint_volume.reload()
    train_mode(SOURCE,OUTPUT,mode)
    checkpoint_volume.commit()
    if mode=="paper":
        for extension in ("direct","autoregressive"):
            call = train.spawn(extension)
            print(json.dumps({"feasibility_temporal_launched":extension,"call_id":call.object_id}),flush=True)
    else:
        call = evaluate_holdout.spawn(mode)
        print(json.dumps({"feasibility_2021_evaluation_launched":mode,"call_id":call.object_id}),flush=True)


@app.function(image=image,cpu=.125,memory=512,timeout=4*60*60)
def run():
    smoke.remote()
    prepared = prepare.remote()
    call = train.spawn("paper")
    print(json.dumps({"feasibility_launched":call.object_id,"prepared":prepared}),flush=True)
    return {"training_call":call.object_id,**prepared}


@app.function(image=image,cpu=4,memory=8192,timeout=2*60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def rescore():
    from feasibility import corrected_evaluation

    data_volume.reload()
    checkpoint_volume.reload()
    result = corrected_evaluation(SOURCE,OUTPUT)
    checkpoint_volume.commit()
    calls = {mode:evaluate_holdout.spawn(mode).object_id for mode in ("direct","autoregressive")}
    print(json.dumps({"corrected_evaluation":result,"holdout_calls":calls}),flush=True)
    return {**result,"holdout_calls":calls}


@app.function(image=image,gpu=["L4","A10","T4"],cpu=4,memory=16384,max_containers=2,timeout=2*60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def evaluate_holdout(mode: str):
    from feasibility import evaluate_2021

    data_volume.reload()
    checkpoint_volume.reload()
    result = evaluate_2021(SOURCE,OUTPUT,mode)
    checkpoint_volume.commit()
    return result


@app.function(image=image,cpu=1,memory=2048,timeout=60,volumes={"/checkpoints":checkpoint_volume})
def status():
    checkpoint_volume.reload()
    root = Path(OUTPUT)
    result = {"run":RUN,"frozen":(root/"manifest.json").is_file(),"epochs_per_model":10,"modes":{}}
    if result["frozen"]:
        manifest = json.loads((root/"manifest.json").read_text())
        result["splits"] = manifest["splits"]
    for mode in ("paper","direct","autoregressive"):
        metrics = root/mode/"metrics.jsonl"
        item = {"epoch":0,"evaluation_saved":(root/mode/"evaluation.json").is_file()}
        if metrics.is_file():
            records = [json.loads(line) for line in metrics.read_text().splitlines() if line]
            last = records[-1]
            primary = last["validation"]["station_mae"][0]
            item.update(epoch=last["epoch"],training=last["training"],
                        validation_temperature_mae_c=primary[1],validation_wind_mae_kmh=primary[0]*3.6,
                        epoch_seconds=last["seconds"])
        summary = root/mode/"waterloo_summary.json"
        if summary.is_file():
            item["waterloo_matched_comparisons"] = json.loads(summary.read_text())
        item["corrected_evaluations"] = {}
        for year in (2020,2021):
            summary = root/f"evaluation_v2/{year}/{mode}/waterloo_summary.json"
            if summary.is_file():
                item["corrected_evaluations"][str(year)] = json.loads(summary.read_text())
        result["modes"][mode] = item
    return result
