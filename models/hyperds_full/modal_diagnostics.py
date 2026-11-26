"""Read-only checkpoint evaluation; does not change the frozen training code."""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0,"/app")
from modal_app import train_image,data_volume,checkpoint_volume
from multiyear_plan import DATASET


app = modal.App("weatherloo-hyperds-diagnostics")
RUN = "waterloo_2018_2025_stationqc_v3"


@app.function(image=train_image,gpu=["L4","A10"],cpu=4,memory=16384,timeout=30*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def validation_rmse(mode: str="paper"):
    import hashlib
    import tempfile
    from datetime import datetime,timezone

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from model import HyperDS
    from train import WeatherDataset,evaluate

    if mode not in ("paper","direct","autoregressive"):
        raise ValueError(mode)
    torch.set_num_threads(4)
    checkpoint_volume.reload()
    data_volume.reload()
    run = Path("/checkpoints/hyperds_full")/RUN
    reports = []
    computed = {}
    for name in (("best","latest") if mode=="paper" else ("best",)):
        saved = torch.load(run/mode/f"{name}.pt",map_location="cpu",weights_only=False)
        if saved["mode"]!=mode:
            raise ValueError("checkpoint mode differs")
        epoch = saved["epoch"]
        for filename,expected in saved["source_sha256"].items():
            if hashlib.sha256((Path("/app")/filename).read_bytes()).hexdigest()!=expected:
                raise ValueError(f"checkpoint source differs: {filename}")
        manifest = saved["manifest"]
        store = Path("/data/processed/hyperds_full")/DATASET
        overlay = manifest["station_overlay"]
        if hashlib.sha256((store/overlay["path"]).read_bytes()).hexdigest()!=overlay["sha256"]:
            raise ValueError("frozen station overlay differs")
        diagnostics = run/"diagnostics"
        cache = diagnostics/f"validation_predictions_{mode}_{name}_epoch_{epoch}.npz"
        existing = diagnostics/f"validation_rmse_{mode}_{name}_epoch_{epoch}.json"
        if cache.is_file() and existing.is_file():
            reports.append(json.loads(existing.read_text()))
            print(json.dumps({"validation_rmse_cached":str(existing)}),flush=True)
            continue
        if epoch not in computed:
            validation = WeatherDataset(store,"val",manifest)
            stats = manifest["normalization"]
            mean,std = [torch.tensor(stats[key],device="cuda") for key in ("field_mean","field_std")]
            model = HyperDS(mode).cuda()
            model.load_state_dict(saved["state_dict"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"validation.npz"
                evaluate(model,DataLoader(validation,batch_size=2,num_workers=2),"cuda",mean,std,
                         selection_indices=[0],excluded_indices=[],prediction_output=path,
                         station_ids=[s["id"] for s in manifest["stations"]])
                with np.load(path) as values:
                    forecasts,truth = values["forecast"],values["observation"]
                    diagnostics.mkdir(parents=True,exist_ok=True)
                    # Keep only Soulis station predictions remotely for matched baseline scoring.
                    np.savez_compressed(cache,forecast=forecasts[:,:,0,:2],observation=truth[:,:,0,:2],
                                        initializations=values["initializations"],
                                        source_sha256=json.dumps(saved["source_sha256"]),
                                        overlay_sha256=overlay["sha256"])
                metrics,by_lead = {},{}
                for variable,column,scale,unit in (("temperature",1,1.,"degC"),("wind",0,3.6,"km/h")):
                    predicted,observed = forecasts[:,:,0,column],truth[:,:,0,column]
                    mask = np.isfinite(predicted)&np.isfinite(observed)
                    error = np.where(mask,(predicted.astype(np.float64)-observed.astype(np.float64))*scale,np.nan)
                    def summarize(values):
                        finite = values[np.isfinite(values)]
                        return {"rmse":float(np.sqrt(np.mean(finite**2))) if finite.size else None,
                                "mae":float(np.mean(np.abs(finite))) if finite.size else None,
                                "bias":float(finite.mean()) if finite.size else None,
                                "n":int(finite.size),"unit":unit}
                    # Forecast aggregates use +1..+48 h; lead 0 is reported separately.
                    metrics[variable] = summarize(error if mode=="paper" else error[:,1:])
                    for hour in range(error.shape[1]):
                        by_lead.setdefault(str(hour),{})[variable] = summarize(error[:,hour])
                computed[epoch] = {"metrics":metrics,"by_lead":by_lead}
            del model
        report = {"run":RUN,"mode":mode,"checkpoint":name,"epoch":epoch,
                  "split":"2024 validation","station":"eric_d_soulis",
                  "lead_hours":list(range(1 if mode=="paper" else 49)),
                  "optimizer_steps":0,**computed[epoch],
                  "evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
                  "aggregation":"sqrt(sum(squared physical-unit errors)/finite matched count)",
                  "note":("Initialization-time downscaling; does not measure future-forecast skill." if mode=="paper"
                          else "Hourly forecasting validation; retrospective ERA5 initialization. Lead 0 is separate from the +1..+48 h aggregate.")}
        out = run/"diagnostics"/f"validation_rmse_{mode}_{name}_epoch_{epoch}.json"
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+"\n")
        reports.append(report)
        print(json.dumps({"validation_rmse_saved":report}),flush=True)
    checkpoint_volume.commit()
    return reports


@app.function(image=train_image,cpu=2,memory=4096,timeout=30*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def paper_comparison():
    """Compare completed lead-zero predictions with both ERA5 resolutions."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    from datetime import datetime,timezone

    import numpy as np
    import torch
    from torch.nn import functional as F
    from train import WeatherDataset,physical_station

    torch.set_num_threads(2)
    checkpoint_volume.reload()
    data_volume.reload()
    run = Path("/checkpoints/hyperds_full")/RUN
    store = Path("/data/processed/hyperds_full")/DATASET
    manifest = json.loads((store/"manifest.json").read_text())
    overlay = manifest["station_overlay"]
    if hashlib.sha256((store/overlay["path"]).read_bytes()).hexdigest()!=overlay["sha256"]:
        raise ValueError("frozen station overlay differs")
    rows = [json.loads(line) for line in (run/"paper/metrics.jsonl").read_text().splitlines()]
    best = min(rows,key=lambda row:row["validation"]["validation_station_mse"])
    if rows[-1]["epoch"]!=50:
        raise ValueError("paper training is not complete")
    dataset = WeatherDataset(store,"test",manifest)
    with np.load(run/"paper/predictions.npz") as saved:
        if list(saved["lead_times_hours"])!=[0] or not np.array_equal(saved["initializations"],dataset.init_times):
            raise ValueError("saved prediction time mismatch")
        station = list(saved["station_ids"]).index("eric_d_soulis")
        if manifest["stations"][station]["id"]!="eric_d_soulis":
            raise ValueError("station order mismatch")
        predicted,observed = saved["forecast"][:,0,station],saved["observation"][:,0,station]
    with np.load(store/overlay["path"]) as labels:
        indices = np.searchsorted(labels["times"],dataset.init_times)
        if not np.array_equal(labels["times"][indices],dataset.init_times):
            raise ValueError("station label time mismatch")
        if not np.allclose(labels["values"][indices],observed[:,:2],equal_nan=True):
            raise ValueError("saved observations differ from the frozen overlay")
    baselines = {key:np.full_like(predicted,np.nan) for key in ("era5_1degree","era5_025degree")}
    query = dataset.coordinates[station:station+1][None,None]
    grids = {id(group):group["grid"] for group in dataset.groups}
    def read_initial(row):
        group,index = row
        return np.asarray(grids[id(group)][index,0],dtype=np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number,values in enumerate(pool.map(read_initial,dataset.rows)):
            fine = torch.from_numpy(values)[None]
            coarse = F.avg_pool2d(fine,4)
            for method,field in (("era5_1degree",coarse),("era5_025degree",fine)):
                value = F.grid_sample(field,query,padding_mode="border",align_corners=False).squeeze(2).transpose(1,2)
                baselines[method][number] = physical_station(value,torch.zeros(5),torch.ones(5))[0,0].numpy()
            if number%300==0:
                print(json.dumps({"paper_baseline_initializations_read":number+1,"total":len(dataset)}),flush=True)
    methods = {"hyperds_paper":predicted,**baselines}
    results = {}
    for variable,column,scale,unit in (("temperature",1,1.,"degC"),("wind",0,3.6,"km/h")):
        mask = np.isfinite(observed[:,column])
        for values in methods.values():
            mask &= np.isfinite(values[:,column])
        results[variable] = {}
        for method,values in methods.items():
            error = (values[mask,column].astype(np.float64)-observed[mask,column].astype(np.float64))*scale
            results[variable][method] = {"mse":float(np.mean(error**2)),"rmse":float(np.sqrt(np.mean(error**2))),
                                        "mae":float(np.mean(np.abs(error))),"bias":float(error.mean()),
                                        "n":int(len(error)),"unit":unit}
    report = {"run":RUN,"mode":"paper","epoch":best["epoch"],"checkpoint_selection":"2024 Soulis station loss",
              "split":"2025 test","station":"eric_d_soulis","lead_hours":[0],"optimizer_steps":0,
              "metrics":results,"evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
              "interpolation":"Bilinear at station coordinates; interpolate u/v before deriving wind speed; same finite mask per variable.",
              "note":"Retrospective ERA5 same-time downscaling. Soulis is seen in training years; test year is held out. Wind is scored against Soulis 4.4 m measurements, not a 10 m observation."}
    out = run/"diagnostics/paper_2025_downscaling_comparison.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    checkpoint_volume.commit()
    print(json.dumps({"paper_downscaling_comparison_saved":report}),flush=True)
    return report
