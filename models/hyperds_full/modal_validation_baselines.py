"""Train-only simple baselines and matched 2024 Soulis validation comparisons."""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0,"/app")
from modal_app import train_image,data_volume,checkpoint_volume,with_shared
from modal_diagnostics import RUN
from multiyear_plan import DATASET

app = modal.App("weatherloo-validation-baselines")
ROOT = Path("/checkpoints/hyperds_full")/RUN/"diagnostics"
hrrr_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0")
if modal.is_local():
    hrrr_image = with_shared(hrrr_image.add_local_dir(str(Path(__file__).resolve().parent), "/app"))


@app.function(image=train_image,cpu=8,memory=16384,timeout=60*60,
              volumes={"/data":data_volume.with_mount_options(read_only=True),"/checkpoints":checkpoint_volume})
def prepare():
    """Fit 49 separate univariate corrections using frozen 2018–2023 labels."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import numpy as np
    import torch
    from torch.nn import functional as F
    from train import WeatherDataset,physical_station

    torch.set_num_threads(1)
    checkpoint_volume.reload()
    data_volume.reload()
    ROOT.mkdir(parents=True,exist_ok=True)
    output = ROOT/"validation_baselines.npz"
    def save_lagged_persistence(fit):
        # Label matching allows a small positive time tolerance. Lag by an hour
        # so an initialization baseline cannot use a later station measurement.
        if fit.get("observed_persistence_lag_hours")==1:
            return fit
        store = Path("/data/processed/hyperds_full")/DATASET
        manifest = json.loads((store/"manifest.json").read_text())
        overlay = manifest["station_overlay"]
        if hashlib.sha256((store/overlay["path"]).read_bytes()).hexdigest()!=fit["overlay_sha256"]:
            raise ValueError("persistence station overlay differs")
        with np.load(output) as values:
            arrays = {name:values[name] for name in values.files}
        with np.load(store/overlay["path"]) as values:
            times,labels = values["times"],values["values"].astype(np.float64)
        labels[:,0] *= 3.6
        stamps = arrays["initializations"]-3600
        indices = np.searchsorted(times,stamps)
        if not np.array_equal(times[indices],stamps):
            raise ValueError("lagged persistence time mismatch")
        arrays["observed_persistence"] = np.broadcast_to(labels[indices,None,:],arrays["observation"].shape).copy()
        fit["observed_persistence_lag_hours"] = 1
        np.savez_compressed(output,**arrays)
        (ROOT/"validation_baselines_fit.json").write_text(json.dumps(fit,indent=2)+"\n")
        checkpoint_volume.commit()
        return fit
    if output.is_file():
        return save_lagged_persistence(json.loads((ROOT/"validation_baselines_fit.json").read_text()))
    store = Path("/data/processed/hyperds_full")/DATASET
    manifest = json.loads((store/"manifest.json").read_text())
    saved = torch.load(ROOT.parent/"paper/best.pt",map_location="cpu",weights_only=False)
    for filename,expected in saved["source_sha256"].items():
        if hashlib.sha256((Path("/app")/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"frozen source differs: {filename}")
    overlay = manifest["station_overlay"]
    if overlay!=saved["manifest"]["station_overlay"]:
        raise ValueError("checkpoint and dataset station overlays differ")
    if hashlib.sha256((store/overlay["path"]).read_bytes()).hexdigest()!=overlay["sha256"]:
        raise ValueError("station overlay checksum mismatch")
    with np.load(store/overlay["path"]) as values:
        times,labels = values["times"],values["values"].astype(np.float64)
    labels[:,0] *= 3.6
    values_by_split = {}
    for split in ("train","val"):
        dataset = WeatherDataset(store,split,manifest)
        stamps = dataset.init_times[:,None]+np.arange(49)*3600
        bounds = (np.datetime64("2018-01-01"),np.datetime64("2024-01-01")) if split=="train" else (
            np.datetime64("2024-01-01"),np.datetime64("2025-01-01"))
        if not ((stamps>=bounds[0].astype("datetime64[s]").astype(int))&
                (stamps<bounds[1].astype("datetime64[s]").astype(int))).all():
            raise ValueError("baseline split guard violated")
        indices = np.searchsorted(times,stamps)
        if not np.array_equal(times[indices],stamps):
            raise ValueError("overlay time mismatch")
        truth = labels[indices]
        initial = np.empty((len(dataset),2),dtype=np.float64)
        query = dataset.coordinates[0:1][None,None]
        grids = {id(group):group["grid"] for group in dataset.groups}
        def read_initial(row):
            group,index = row
            field = torch.from_numpy(np.asarray(grids[id(group)][index,0],dtype=np.float32))[None]
            field = F.avg_pool2d(field,4)
            value = F.grid_sample(field,query,padding_mode="border",align_corners=False).squeeze(2).transpose(1,2)
            result = physical_station(value,torch.zeros(5),torch.ones(5))[0,0,:2].numpy().astype(np.float64)
            result[0] *= 3.6
            return result
        with ThreadPoolExecutor(max_workers=16) as pool:
            for index,result in enumerate(pool.map(read_initial,dataset.rows)):
                initial[index] = result
                if index%1000==0:
                    print(json.dumps({"baseline_initial_fields":index+1,"split":split,"total":len(dataset)}),flush=True)
        values_by_split[split] = (dataset.init_times,initial,truth)
    _,train_initial,train_truth = values_by_split["train"]
    fit = {"training_years":list(range(2018,2024)),"validation_year":2024,"station":"eric_d_soulis",
           "dataset":DATASET,"overlay_sha256":overlay["sha256"],"source_sha256":saved["source_sha256"],
           "units":["km/h","degC"],"variable_order":["wind","temperature"],"coefficients":{},
           "correction_input":"1 degree ERA5 initialization persistence; not an HRRR correction",
           "wind_constraint":"corrected wind speed is floored at zero",
           "training_initializations":len(train_initial)}
    bias = np.zeros((49,2))
    intercept,slope = np.zeros((49,2)),np.zeros((49,2))
    for hour in range(49):
        fit["coefficients"][str(hour)] = {}
        for column,name in enumerate(("wind","temperature")):
            x,y = train_initial[:,column],train_truth[:,hour,column]
            mask = np.isfinite(x)&np.isfinite(y)
            x,y = x[mask],y[mask]
            if len(x)<2 or np.var(x)==0:
                raise ValueError("insufficient training data for regression")
            bias[hour,column] = np.mean(y-x)
            slope[hour,column] = np.mean((x-x.mean())*(y-y.mean()))/np.var(x)
            intercept[hour,column] = y.mean()-slope[hour,column]*x.mean()
            fit["coefficients"][str(hour)][name] = {"mean_correction":bias[hour,column],
                "intercept":intercept[hour,column],"slope":slope[hour,column],"n":len(x)}
    initializations,initial,truth = values_by_split["val"]
    raw = np.broadcast_to(initial[:,None,:],truth.shape).copy()
    corrected,linear = raw+bias,intercept+slope*raw
    corrected[:,:,0] = np.maximum(corrected[:,:,0],0)
    linear[:,:,0] = np.maximum(linear[:,:,0],0)
    persistence = np.broadcast_to(truth[:,0:1,:],truth.shape).copy()
    np.savez_compressed(output,initializations=initializations,observation=truth,
                        era5_persistence=raw,mean_bias=corrected,linear_regression=linear,
                        observed_persistence=persistence)
    (ROOT/"validation_baselines_fit.json").write_text(json.dumps(fit,indent=2)+"\n")
    fit = save_lagged_persistence(fit)
    print(json.dumps({"baseline_fit_saved":str(output),"training_initializations":len(train_initial),
                      "validation_initializations":len(initializations)}),flush=True)
    return fit


@app.function(image=hrrr_image,cpu=8,memory=16384,timeout=60*60,
              volumes={"/checkpoints":checkpoint_volume})
def hrrr():
    """Extract only Waterloo station forecasts; no archive fields are persisted."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime,timezone
    import time
    import dynamical_catalog
    import numpy as np
    from pyproj import CRS,Transformer

    checkpoint_volume.reload()
    with np.load(ROOT/"validation_baselines.npz") as values:
        stamps = values["initializations"]
    ds = dynamical_catalog.open("noaa-hrrr-forecast-48-hour",chunks=None)
    print(json.dumps({"hrrr_dimensions":dict(ds.sizes),"spatial_ref":ds.spatial_ref.attrs,
                      "variables":{name:ds[name].attrs for name in ("temperature_2m","wind_u_10m","wind_v_10m")}},default=str),flush=True)
    if ds.temperature_2m.attrs["units"]!="degree_Celsius":
        raise ValueError("unexpected HRRR temperature units")
    if any(ds[name].attrs["units"] not in ("m s-1","m/s") for name in ("wind_u_10m","wind_v_10m")):
        raise ValueError("unexpected HRRR wind units")
    crs = CRS.from_cf(ds.spatial_ref.attrs)
    px,py = Transformer.from_crs("EPSG:4326",crs,always_xy=True).transform(-80.5164,43.4668)
    # The catalog stores native HRRR north-to-south; interpolation uses ascending axes.
    for axis in ("x","y"):
        if ds[axis][0]>ds[axis][-1]:
            ds = ds.isel({axis:slice(None,None,-1)})
    xs,ys = ds.x.values,ds.y.values
    xi,yi = np.searchsorted(xs,px)-1,np.searchsorted(ys,py)-1
    if not (0<=xi<len(xs)-1 and 0<=yi<len(ys)-1):
        raise ValueError("station outside HRRR grid")
    wx,wy = (px-xs[xi])/(xs[xi+1]-xs[xi]),(py-ys[yi])/(ys[yi+1]-ys[yi])
    weights = np.array([[(1-wx)*(1-wy),wx*(1-wy)],[(1-wx)*wy,wx*wy]])
    if not ((weights>=0).all() and np.isclose(weights.sum(),1)):
        raise ValueError("invalid station interpolation weights")
    subset = ds[["temperature_2m","wind_u_10m","wind_v_10m"]].isel(x=slice(xi,xi+2),y=slice(yi,yi+2))
    months = sorted(set(str(np.datetime64(int(value),"s"))[:7] for value in stamps))
    directory = ROOT/"hrrr_validation_months"
    directory.mkdir(exist_ok=True)
    def read_forecast(stamp):
        for attempt in range(3):
            try:
                values = subset.sel(init_time=np.datetime64(int(stamp),"s"),
                                    lead_time=np.arange(49).astype("timedelta64[h]")).to_array().transpose(
                                        "variable","lead_time","y","x").values
                interpolated = (values*weights).sum(axis=(-2,-1))
                wind = np.sqrt(interpolated[1]**2+interpolated[2]**2)*3.6
                return np.stack((wind,interpolated[0]),axis=-1)
            except Exception:
                if attempt==2:
                    raise
                time.sleep(2**attempt)
    for month in months:
        output = directory/f"{month}.npz"
        indices = [i for i,value in enumerate(stamps) if str(np.datetime64(int(value),"s")).startswith(month)]
        selected = stamps[indices]
        if output.is_file():
            with np.load(output) as values:
                if not np.array_equal(values["initializations"],selected):
                    raise ValueError("HRRR cache cohort mismatch")
            continue
        forecasts = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i,forecast in enumerate(pool.map(read_forecast,selected)):
                forecasts.append(forecast)
                if i%20==0:
                    print(json.dumps({"hrrr_month":month,"station_forecasts":i+1,"total":len(selected)}),flush=True)
        np.savez_compressed(output,initializations=selected,forecast=np.asarray(forecasts))
        checkpoint_volume.commit()
        print(json.dumps({"hrrr_month_saved":month,"initializations":len(selected)}),flush=True)
    forecasts = np.full((len(stamps),49,2),np.nan)
    for month in months:
        with np.load(directory/f"{month}.npz") as values:
            indices = np.searchsorted(stamps,values["initializations"])
            if not np.array_equal(stamps[indices],values["initializations"]):
                raise ValueError("HRRR cache ordering mismatch")
            forecasts[indices] = values["forecast"]
    np.savez_compressed(ROOT/"validation_hrrr.npz",initializations=stamps,forecast=forecasts)
    provenance = {"source":"NOAA HRRR from dynamical.org","catalog_id":"noaa-hrrr-forecast-48-hour",
                  "url":"https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/",
                  "station":"eric_d_soulis","latitude":43.4668,"longitude":-80.5164,
                  "interpolation":"native Lambert conformal x/y bilinear; u/v before wind speed",
                  "grid_indices":[int(yi),int(xi)],"weights":weights.tolist(),
                  "units":["km/h","degC"],"initializations":len(stamps),
                  "finite_counts":np.isfinite(forecasts).sum(axis=(0,1)).tolist(),
                  "extracted_at_utc":datetime.now(timezone.utc).isoformat(),
                  "spatial_ref":ds.spatial_ref.attrs}
    (ROOT/"validation_hrrr_provenance.json").write_text(json.dumps(provenance,indent=2,default=str)+"\n")
    checkpoint_volume.commit()
    return provenance


@app.function(image=train_image,cpu=2,memory=4096,timeout=10*60,
              volumes={"/checkpoints":checkpoint_volume})
def compare():
    """All plotted methods use identical finite pairs for each variable/lead."""
    from datetime import datetime,timezone
    import numpy as np

    checkpoint_volume.reload()
    fit = json.loads((ROOT/"validation_baselines_fit.json").read_text())
    if fit.get("observed_persistence_lag_hours")!=1:
        raise ValueError("observed persistence must use the frozen one-hour lag")
    with np.load(ROOT/"validation_baselines.npz") as values:
        stamps,truth = values["initializations"],values["observation"]
        methods = {name:values[name] for name in (
            "era5_persistence","observed_persistence","mean_bias","linear_regression")}
    models = {}
    for mode in ("direct","autoregressive"):
        candidates = list(ROOT.glob(f"validation_predictions_{mode}_best_epoch_*.npz"))
        path = max(candidates,key=lambda path:int(path.stem.rsplit("_",1)[1]))
        epoch = int(path.stem.rsplit("_",1)[1])
        with np.load(path) as values:
            if (json.loads(str(values["source_sha256"]))!=fit["source_sha256"] or
                    str(values["overlay_sha256"])!=fit["overlay_sha256"]):
                raise ValueError("model/baseline source or overlay hashes differ")
            if not np.array_equal(values["initializations"],stamps):
                raise ValueError("model/baseline cohort mismatch")
            observed = values["observation"].astype(np.float64)
            observed[:,:,0] *= 3.6
            if not np.allclose(observed,truth,equal_nan=True,atol=1e-5,rtol=0):
                raise ValueError("model/baseline labels differ")
            forecast = values["forecast"].astype(np.float64)
            forecast[:,:,0] *= 3.6
            methods[mode] = forecast
        models[mode] = epoch
    with np.load(ROOT/"validation_hrrr.npz") as values:
        if not np.array_equal(values["initializations"],stamps):
            raise ValueError("HRRR cohort mismatch")
        methods["hrrr"] = values["forecast"]
    report = {"run":RUN,"split":"2024 validation","station":"eric_d_soulis","models":models,
              "baseline_fit_years":list(range(2018,2024)),"optimizer_steps":0,"by_lead":{},"metrics":{},
              "observed_persistence_lag_hours":1,
              "initializations":len(stamps),"evaluated_at_utc":datetime.now(timezone.utc).isoformat(),
              "mask":"Common finite pairs across all displayed methods, separately per variable and lead.",
              "note":"ERA5 inputs are retrospective; HRRR is a real forecast. Soulis wind truth is 4.4 m, HRRR/ERA5 are 10 m. Corrections fit ERA5-initialization persistence, not HRRR. Forecast aggregate excludes lead 0."}
    for column,variable,unit in ((0,"wind","km/h"),(1,"temperature","degC")):
        mask = np.isfinite(truth[:,:,column])
        for values in methods.values():
            mask &= np.isfinite(values[:,:,column])
        counts = mask.sum(axis=0)
        report["metrics"][variable] = {}
        def summarize(error):
            error = error[np.isfinite(error)]
            return {"n":len(error),"rmse":float(np.sqrt(np.mean(error**2))) if len(error) else None,
                    "mae":float(np.mean(np.abs(error))) if len(error) else None,
                    "bias":float(np.mean(error)) if len(error) else None,"unit":unit}
        for method,values in methods.items():
            errors = np.where(mask,values[:,:,column]-truth[:,:,column],np.nan)
            report["metrics"][variable][method] = summarize(errors[:,1:])
            for hour in range(49):
                report["by_lead"].setdefault(str(hour),{}).setdefault(variable,{})[method] = summarize(errors[:,hour])
        print(json.dumps({"matched_validation":variable,"per_lead_n_range":[int(counts.min()),int(counts.max())],
                          "results":report["metrics"][variable]}),flush=True)
    path = ROOT/f"validation_comparison_direct_{models['direct']}_autoregressive_{models['autoregressive']}.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    checkpoint_volume.commit()
    return str(path)
