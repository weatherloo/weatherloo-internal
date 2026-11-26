"""A frozen, smaller forecasting experiment using completed monthly shards."""

import repo_paths  # noqa: F401  (shared pipelines modules on sys.path)
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import zarr

from site_scoring import STATIONS as SITE_STATIONS, VARIABLES, LEADS, site_export, matched_comparison
from model import HyperDS, UPSTREAM_REVISION
from multiyear import combine_moments
from train import WeatherDataset, train_epoch, evaluate, physical_station, commit_checkpoints


RUN = "feasibility_2018_2020_stationqc_v2"
EPOCHS = 10
YEAR_SPLITS = {2018:"train",2019:"val",2020:"test"}
UTC = timezone.utc


def valid_rows(times,split):
    year = {value:key for key,value in YEAR_SPLITS.items()}[split]
    start = int(datetime(year,1,1,tzinfo=UTC).timestamp())
    end = int(datetime(year+1,1,1,tzinfo=UTC).timestamp())
    mask = (times>=start)&(times<end)
    if split in ("train","val"):
        mask &= times+48*3600<end
    return np.flatnonzero(mask)


def freeze_view(source,destination):
    source,destination = Path(source),Path(destination)
    if (destination/"manifest.json").is_file():
        manifest = json.loads((destination/"manifest.json").read_text())
        validate_view(source,manifest)
        return manifest
    # Preserve the original cohort and normalization: later completed months
    # must not expand this repair into a different training experiment.
    original = destination.parent/"feasibility_2018_2020_v1/manifest.json"
    manifest = json.loads(original.read_text())
    manifest.update(run=RUN,quality_controlled_at_utc=datetime.now(UTC).isoformat(),
                    original_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    station_overlay=json.loads((source/"manifest.json").read_text())["station_overlay"])
    if manifest["station_overlay"]["schema"]!="soulis_hourly_v3":
        raise ValueError("quality-controlled Soulis v3 overlay is required")
    report = validate_view(source,manifest)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (destination/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return manifest


def validate_view(source,manifest):
    if "station_overlay" in manifest:
        overlay = manifest["station_overlay"]
        if hashlib.sha256((Path(source)/overlay["path"]).read_bytes()).hexdigest()!=overlay["sha256"]:
            raise ValueError("frozen station overlay changed")
    stamps = {split:[] for split in ("train","val","test")}
    for shard in manifest["shards"]:
        path = Path(source)/shard["path"]
        raw = (path/"manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=shard["manifest_sha256"]:
            raise ValueError("frozen source shard changed")
        group = zarr.open_group(str(path),mode="r")[shard["source_split"]]
        times = np.asarray(group["init_time"])
        expected = valid_rows(times,shard["split"])
        if not np.array_equal(expected,shard["indices"]):
            raise ValueError("frozen row selection violates chronological split")
        stamps[shard["split"]].extend(times[expected].tolist())
    for split in stamps:
        if len(stamps[split])<100 or len(stamps[split])!=len(set(stamps[split])):
            raise ValueError(f"insufficient/duplicate {split} examples")
    for split,following in (("train","val"),("val","test")):
        if max(stamps[split])+48*3600>=min(stamps[following]):
            raise ValueError("training/selection targets overlap later split")
    return {"passed":True,"splits":{key:len(value) for key,value in stamps.items()},
            "checks":["verified_immutable_sources","frozen_row_indices","chronological_48h_guard",
                      "2018_only_normalization","Waterloo_primary_selection_and_test"]}


class ViewDataset(WeatherDataset):
    def __init__(self,source,manifest,split):
        shards = [s for s in manifest["shards"] if s["split"]==split]
        source_manifest = {"normalization":manifest["normalization"],
                           "shards":[{"path":s["path"],"split":s["source_split"]} for s in shards]}
        if "station_overlay" in manifest:
            source_manifest["station_overlay"] = manifest["station_overlay"]
        super().__init__(source,"train",source_manifest)
        self.rows = [(group,index) for group,shard in zip(self.groups,shards) for index in shard["indices"]]
        self.init_times = np.concatenate([np.asarray(group["init_time"])[s["indices"]]
                                          for group,s in zip(self.groups,shards)])


def _helpers():
    path = "/workspace/pipelines/benchmarks/methods/cnn_lstm_bias_correction/compute_benchmark.py"
    spec = importlib.util.spec_from_file_location("existing_benchmark_helpers",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OBS_ROOT = Path("/benchmark/observations")
    return module


def prepare_baselines(source,run,manifest,year=2020):
    """Fix persistence/climatology references before looking at model test scores."""
    run = Path(run)
    torch.set_num_threads(4)
    target = run/"baselines"
    target.mkdir(parents=True,exist_ok=True)
    helpers = _helpers()
    truth,climatology = {},{}
    for station in SITE_STATIONS:
        truth[station] = helpers.load_observations(station,year)
        climatology[station] = helpers.build_climatology(helpers.load_observations(station,2018))
    start = datetime(year,1,1,tzinfo=UTC)
    days = (datetime(year+1,1,1,tzinfo=UTC)-start).days
    inits = [start+timedelta(hours=h) for h in range(0,days*24,6)]
    errors = {name:np.full((len(inits),2,2,len(LEADS)),np.nan) for name in
              ("persistence","climatology_2018","era5_initial_persistence")}
    source_test = ViewDataset(source,manifest,"test")
    lookup = {int(stamp):index for index,stamp in enumerate(source_test.init_times)}
    query = source_test.coordinates[None]
    stats = manifest["normalization"]
    mean,std = torch.tensor(stats["field_mean"]),torch.tensor(stats["field_std"])
    for ii,init in enumerate(inits):
        raw_persistence = None
        source_index = lookup.get(int(init.timestamp()))
        if source_index is not None:
            group,row = source_test.rows[source_index]
            initial = torch.from_numpy(np.asarray(group["grid"][row,0],dtype=np.float32))[None]
            coarse = F.avg_pool2d((initial-mean[None,:,None,None])/std[None,:,None,None],4)
            interpolated = F.grid_sample(coarse,query[:,None],padding_mode="border",align_corners=False).squeeze(2).transpose(1,2)
            raw_persistence = physical_station(interpolated,mean,std)[0].numpy()
        for si,station in enumerate(SITE_STATIONS):
            ms = [s["id"] for s in manifest["stations"]].index(station)
            current = truth[station].get(init.strftime("%Y-%m-%dT%H:%M:%SZ"),{})
            for li,lead in enumerate(LEADS):
                if lead>48:
                    continue
                valid = init+timedelta(hours=lead)
                observed = truth[station].get(valid.strftime("%Y-%m-%dT%H:%M:%SZ"),{})
                for vi,variable in enumerate(VARIABLES):
                    value = observed.get(variable)
                    if value is None:
                        continue
                    predicted = {"persistence":current.get(variable),
                                 "climatology_2018":helpers.climatology_lookup(climatology[station],variable,valid),
                                 "era5_initial_persistence":None if raw_persistence is None else
                                    float(raw_persistence[ms,1] if variable=="t2m" else raw_persistence[ms,0]*3.6)}
                    for name,forecast in predicted.items():
                        if forecast is not None and np.isfinite(forecast):
                            errors[name][ii,si,vi,li] = forecast-float(value)
    for name,bias in errors.items():
        np.savez_compressed(target/f"{name}_{year}.npz",bias=bias,station_ids=np.array(SITE_STATIONS),
                            variables=np.array(VARIABLES),lead_times_hours=np.array(LEADS),
                            initializations=np.array([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in inits]))
    return list(errors)


def compare(mode,run,year=2020,predictions=None):
    run = Path(run)
    method = f"hyperds_{mode}_feasibility"
    output = run/mode/"benchmark"/method
    exported = site_export(predictions or run/mode/"predictions.npz","/benchmark/observations",output,method,year=year)
    comparisons = {path.stem:matched_comparison(exported,path) for path in sorted((run/"baselines").glob("*.npz"))}
    (output/"comparison.json").write_text(json.dumps(comparisons,indent=2)+"\n")
    primary = {name:{variable:value["eric_d_soulis"][variable]["pooled"] for variable in VARIABLES}
               for name,value in comparisons.items()}
    (run/mode/"waterloo_summary.json").write_text(json.dumps(primary,indent=2)+"\n")
    print(json.dumps({"feasibility_waterloo_results":mode,"matched_baselines":primary}),flush=True)


def corrected_evaluation(source,run):
    """Rescore saved 2020 forecasts; freeze 2021 as a wind-capable holdout."""
    run,source = Path(run),Path(source)
    original = json.loads((run/"manifest.json").read_text())
    corrected = run/"evaluation_v2/2020"
    prepare_baselines(source,corrected,original)
    for mode in ("direct","autoregressive"):
        compare(mode,corrected,predictions=run/mode/"predictions.npz")
    return prepare_2021(source,run)


def prepare_2021(source,run):
    run,source = Path(run),Path(source)
    original = json.loads((run/"manifest.json").read_text())
    # Year chosen from observation coverage, before reading any model errors.
    # The 2020 4.4 m wind sensor is missing, whereas 2021 has 1,441/1,460 slots.
    manifest = {**original,"shards":[],"evaluation_year":2021,
                "purpose":"Additional 2021 holdout selected for valid 4.4 m Soulis wind; no training or checkpoint selection changes"}
    for path in sorted((source/"shards").glob("2021-*/manifest.json")):
        raw = path.read_bytes()
        detail = json.loads(raw)
        if not detail["verified"] or detail["samples"]<.8*detail["planned"]:
            raise ValueError("2021 evaluation shard failed quality gate")
        group = zarr.open_group(str(path.parent),mode="r")[detail["split"]]
        manifest["shards"].append({"path":str(path.parent.relative_to(source)),"source_split":detail["split"],
            "split":"test","indices":list(range(len(group["init_time"]))),"month":detail["month"],
            "manifest_sha256":hashlib.sha256(raw).hexdigest()})
    if len(manifest["shards"])!=12:
        raise ValueError("2021 holdout requires all 12 completed months")
    destination = run/"evaluation_v2/2021"
    destination.mkdir(parents=True,exist_ok=True)
    frozen = destination/"manifest.json"
    if frozen.is_file() and json.loads(frozen.read_text())!=manifest:
        raise ValueError("2021 evaluation manifest changed")
    frozen.write_text(json.dumps(manifest,indent=2)+"\n")
    prepare_baselines(source,destination,manifest,year=2021)
    return {"test_2021_initializations":sum(len(s["indices"]) for s in manifest["shards"])}


def evaluate_2021(source,run,mode):
    """Evaluate the already-selected best weights; no optimizer steps."""
    run = Path(run)
    destination = run/"evaluation_v2/2021"
    manifest = json.loads((destination/"manifest.json").read_text())
    dataset = ViewDataset(source,manifest,"test")
    stats = manifest["normalization"]
    device = torch.device("cuda")
    torch.set_num_threads(4)
    mean,std = [torch.tensor(stats[key],device=device) for key in ("field_mean","field_std")]
    model = HyperDS(mode).to(device)
    saved = torch.load(run/mode/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(saved["state_dict"])
    out = destination/mode
    out.mkdir(parents=True,exist_ok=True)
    final = evaluate(model,DataLoader(dataset,batch_size=2,num_workers=2),device,mean,std,include_test=True,
                     selection_indices=[0],excluded_indices=[],prediction_output=out/"predictions.npz",
                     station_ids=[s["id"] for s in manifest["stations"]])
    final.update(checkpoint_epoch=saved["epoch"],optimizer_steps=0,evaluation_year=2021)
    (out/"evaluation.json").write_text(json.dumps(final,indent=2)+"\n")
    compare(mode,destination,year=2021)
    commit_checkpoints()
    return final


def train_mode(source,run,mode):
    run = Path(run)
    manifest = json.loads((run/"manifest.json").read_text())
    validate_view(source,manifest)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("feasibility training requires a GPU")
    device = torch.device("cuda")
    training,validation,test = [ViewDataset(source,manifest,split) for split in ("train","val","test")]
    train_loader = DataLoader(training,batch_size=4,shuffle=True,num_workers=2,persistent_workers=True)
    val_loader = DataLoader(validation,batch_size=2,num_workers=2,persistent_workers=True)
    stats = manifest["normalization"]
    mean,std = [torch.tensor(stats[key],device=device) for key in ("field_mean","field_std")]
    model = HyperDS(mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    out = run/mode
    out.mkdir(parents=True,exist_ok=True)
    source_sha = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                  for name in ("feasibility.py","model.py","train.py","data.py","multiyear.py","multiyear_plan.py","benchmark.py")}
    manifest_sha = hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest()
    latest,warm_start = out/"latest.pt",run/"paper/best.pt"
    first,best = 1,float("inf")
    if latest.is_file():
        saved = torch.load(latest,map_location=device,weights_only=False)
        if saved["source_sha256"]!=source_sha or saved["manifest_sha256"]!=manifest_sha:
            raise RuntimeError("frozen feasibility source/data changed")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
        np.random.set_state(saved["numpy_rng_state"])
        first,best = saved["epoch"]+1,saved["best_validation_station_mse"]
    elif mode!="paper":
        model.load_state_dict(torch.load(warm_start,map_location=device,weights_only=False)["state_dict"])
    print(json.dumps({"feasibility_training_started":mode,"device":torch.cuda.get_device_name(),"epochs":EPOCHS,
                      "training":len(training),"validation":len(validation),"test":len(test),
                      "primary_station":"eric_d_soulis","upstream_revision":UPSTREAM_REVISION}),flush=True)
    for epoch in range(first,EPOCHS+1):
        started = time.monotonic()
        fitted = train_epoch(model,train_loader,optimizer,device,mean,std,list(range(len(manifest["stations"]))))
        validated = evaluate(model,val_loader,device,mean,std,selection_indices=[0],excluded_indices=[])
        score = validated["validation_station_mse"]
        if not np.isfinite(score):
            raise RuntimeError("non-finite Waterloo validation loss")
        scheduler.step()
        improved = score<best
        best = min(best,score)
        saved = {"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                 "epoch":epoch,"mode":mode,"best_validation_station_mse":best,"validation":validated,
                 "rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),
                 "numpy_rng_state":np.random.get_state(),"source_sha256":source_sha,"manifest_sha256":manifest_sha,
                 "manifest":manifest,"upstream_revision":UPSTREAM_REVISION,"warm_start":str(warm_start) if mode!="paper" else None}
        torch.save(saved,latest)
        if improved:
            torch.save(saved,out/"best.pt")
        record = {"mode":mode,"epoch":epoch,"epochs":EPOCHS,"training":fitted,"validation":validated,
                  "best":improved,"seconds":time.monotonic()-started}
        with (out/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        commit_checkpoints()
        print(json.dumps(record),flush=True)
    model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=False)["state_dict"])
    final = evaluate(model,DataLoader(test,batch_size=2,num_workers=2),device,mean,std,include_test=True,
                     selection_indices=[0],excluded_indices=[],prediction_output=out/"predictions.npz",
                     station_ids=[s["id"] for s in manifest["stations"]])
    (out/"evaluation.json").write_text(json.dumps(final,indent=2)+"\n")
    if mode!="paper":
        compare(mode,run)
    commit_checkpoints()
    print(json.dumps({"feasibility_completed":mode}),flush=True)
