"""Monthly remote shards for the Waterloo forecasting benchmark."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import xarray as xr

from data import ERA5_STORE, VARIABLES, LAT, LON, SAT_SHAPE, fetch_satellite, goes_time, utc_stamp
from multiyear_plan import DATASET, STATIONS, UTC, GOES19_START, month_records, plan


def retry(action):
    for attempt in range(3):
        try:
            return action()
        except Exception:
            if attempt==2:
                raise
            time.sleep(2**(attempt+1))


def prepare_observations(destination):
    """Cache immutable station source files once, then align labels on UTC hours."""
    sys.path.insert(0,"/workspace/pipelines/download")
    sys.path.insert(0,"/workspace/src")
    from fetch_station_observations import fetch_eccc_hourly
    from hrrr_bias_correction.make_data import SoulisObsIndex

    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    output = destination/"hourly.npz"
    if output.is_file() and (destination/"sources.json").is_file():
        return str(output)
    years = tuple(range(2018,2027))
    start = int(datetime(2018,1,1,tzinfo=UTC).timestamp())
    end = int(datetime(2026,1,3,tzinfo=UTC).timestamp())
    times = np.arange(start,end,3600,dtype=np.int64)
    labels = np.full((len(times),len(STATIONS),4),np.nan,dtype=np.float32)
    sources = []
    for year in years:
        path = destination/f"uw_hobo_15min_{year}.csv"
        url = f"https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
        if not path.is_file():
            def download():
                request = urllib.request.Request(url,headers={"User-Agent":"weatherloo-internal/1.0"})
                with urllib.request.urlopen(request,timeout=120) as response:
                    return response.read()
            path.write_bytes(retry(download))
        sources.append({"url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    soulis = SoulisObsIndex(destination,years)
    for index,stamp in enumerate(times):
        value = soulis.lookup(datetime.fromtimestamp(int(stamp),tz=UTC))
        if value is not None:
            labels[index,0,:2] = (value[1]/3.6,value[0])
    for station_index,station in enumerate(STATIONS[1:],start=1):
        for year in years:
            path = destination/f"eccc_{station['eccc']}_{year}.json"
            if not path.is_file():
                rows = retry(lambda:fetch_eccc_hourly(station["eccc"],year))
                path.write_text(json.dumps(rows))
            rows = json.loads(path.read_text())
            sources.append({"eccc_stn_id":station["eccc"],"year":year,"rows":len(rows),
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            for row in rows:
                stamp = int(utc_stamp(row["UTC_DATE"]+"Z").timestamp())
                index = (stamp-start)//3600
                if not 0<=index<len(times) or stamp%3600:
                    continue
                raw = [row.get(name) for name in ("WIND_SPEED","TEMP","STATION_PRESSURE","PRECIP_AMOUNT")]
                values = np.array([float(v) if v not in (None,"") and float(v)>-9000 else np.nan for v in raw],dtype=np.float32)
                values[0] /= 3.6
                values[2] *= 10
                labels[index,station_index] = values
            print(json.dumps({"station_sources_cached":station["id"],"year":year,"rows":len(rows)}),flush=True)
    # Basic measurement QC; missing labels remain missing and are never imputed.
    for variable,(low,high) in enumerate(((0,100),(-80,60),(500,1100),(0,500))):
        slab = labels[:,:,variable]
        slab[(slab<low)|(slab>high)] = np.nan
    np.savez_compressed(output,times=times,labels=labels)
    (destination/"sources.json").write_text(json.dumps(sources,indent=2)+"\n")
    print(json.dumps({"observations_ready":str(output),"hourly_label_counts":np.isfinite(labels).sum(axis=0).tolist()}),flush=True)
    return str(output)


def moments(values,axes):
    return {"sum":values.sum(axis=axes,dtype=np.float64).tolist(),
            "sum_sq":np.square(values,dtype=np.float64).sum(axis=axes,dtype=np.float64).tolist(),
            "count":int(np.prod([values.shape[axis] for axis in axes]))}


def combine_moments(items):
    count = sum(item["count"] for item in items)
    mean = sum(np.asarray(item["sum"]) for item in items)/count
    variance = sum(np.asarray(item["sum_sq"]) for item in items)/count-mean**2
    return mean.tolist(),np.maximum(np.sqrt(np.maximum(variance,0)),1e-3).tolist()


def build_month(destination,key,observations_path):
    """A month fits in memory; no full-year/global weather arrays are retained."""
    destination = Path(destination)
    if (destination/"manifest.json").is_file():
        return json.loads((destination/"manifest.json").read_text())
    records = month_records(key)
    split = records[0][0]
    times = sorted({(init+timedelta(hours=h)).replace(tzinfo=None) for _,init in records for h in range(49)})
    ds = xr.open_zarr(ERA5_STORE,chunks=None,storage_options={"token":"anon"},consolidated=True)
    subset = ds[list(VARIABLES)].sel(latitude=LAT,longitude=LON%360)

    def fetch_hour(valid):
        def read():
            values = subset.sel(time=np.datetime64(valid)).to_array().transpose("variable","latitude","longitude").values.astype(np.float32)
            values[2] -= 273.15
            values[3] /= 100
            values[4] *= 1000
            if not np.isfinite(values).all() or values[4].min()<-1e-4:
                raise ValueError(f"invalid ERA5 fields at {valid}")
            values[4] = np.maximum(values[4],0)
            return values
        return retry(read)

    fields = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,(valid,values) in enumerate(zip(times,pool.map(fetch_hour,times))):
            fields[valid] = values
            if index%120==0:
                print(json.dumps({"month":key,"era5_hours":index+1,"target_hours":len(times)}),flush=True)
    ds.close()
    with np.load(observations_path) as obs:
        obs_times,obs_labels = obs["times"],obs["labels"]

    def satellite_record(record):
        _,init = record
        bucket = "noaa-goes19" if init>=GOES19_START else "noaa-goes16"
        try:
            return retry(lambda:fetch_satellite(init,bucket,production_cutoff=True,fill_small_gaps=True)),None
        except Exception as error:
            return None,{"initialization":init.isoformat(),"error":f"{type(error).__name__}: {error}"}

    arrays,provenance,failures,accepted_times = [],[],[],set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (_,init),(result,failure) in zip(records,pool.map(satellite_record,records)):
            if failure:
                failures.append(failure)
                print(json.dumps({"rejected_initialization":failure,"month":key}),flush=True)
                continue
            satellite,satellite_sources = result
            valid_times = [(init+timedelta(hours=h)).replace(tzinfo=None) for h in range(49)]
            grid = np.stack([fields[t] for t in valid_times])
            stamps = int(init.timestamp())+np.arange(49)*3600
            indices = np.searchsorted(obs_times,stamps)
            if (indices>=len(obs_times)).any() or not np.array_equal(obs_times[indices],stamps):
                raise ValueError("station label time alignment failed")
            labels = obs_labels[indices].copy()
            if split=="test":
                # The site's 2025 truth files end at year-end; retain forecasts
                # crossing into 2026 but mask those unmatched station labels.
                labels[stamps>=int(datetime(2026,1,1,tzinfo=UTC).timestamp())] = np.nan
            arrays.append((int(init.timestamp()),grid,satellite,labels))
            accepted_times.update(t for t in valid_times if t.strftime("%Y-%m")==key)
            provenance.append({"initialization":init.isoformat(),"satellite":satellite_sources,
                               "grid_sha256":hashlib.sha256(grid.tobytes()).hexdigest(),
                               "satellite_sha256":hashlib.sha256(satellite.tobytes()).hexdigest()})
            if len(arrays)%12==0:
                print(json.dumps({"month":key,"accepted":len(arrays),"rejected":len(failures),"planned":len(records)}),flush=True)
    if not arrays:
        raise ValueError(f"no complete examples in {key}")
    destination.mkdir(parents=True,exist_ok=True)
    grid,satellite,labels = (np.stack([row[index] for row in arrays]) for index in (1,2,3))
    query = np.array([[2*(s['lon']-(LON[0]-.125))/(len(LON)*.25)-1,
                       2*(s['lat']-(LAT[0]-.125))/(len(LAT)*.25)-1] for s in STATIONS],dtype=np.float32)
    group = xr.Dataset({
        "grid":(("sample","lead","variable","latitude","longitude"),grid),
        "satellite":(("sample","frame","band","sat_y","sat_x"),satellite),
        "station":(("sample","lead","station_id","target"),labels),
        "query_xy":(("station_id","xy"),query),
        "init_time":("sample",np.array([row[0] for row in arrays],dtype=np.int64)),
    })
    group.to_zarr(destination,group=split,mode="w",consolidated=False,
                  encoding={"grid":{"chunks":(1,49,5,len(LAT),len(LON))},
                            "satellite":{"chunks":(1,2,4,*SAT_SHAPE)}})
    manifest = {"month":key,"split":split,"samples":len(arrays),"planned":len(records),
                "unique_era5_hours":len(times),"failures":failures,"records":provenance,
                "valid_station_labels":np.isfinite(labels).sum(axis=(0,1)).tolist()}
    if split=="train":
        manifest["field_moments"] = moments(np.stack([fields[t] for t in sorted(accepted_times)]),(0,2,3))
        manifest["satellite_moments"] = moments(satellite,(0,1,3,4))
    # Verify the bytes read back, not just the arrays used to write the store.
    import zarr
    stored = zarr.open_group(str(destination),mode="r")[split]
    for index,record in enumerate(provenance):
        for name in ("grid","satellite"):
            value = np.asarray(stored[name][index])
            if not np.isfinite(value).all() or hashlib.sha256(value.tobytes()).hexdigest()!=record[name+"_sha256"]:
                raise ValueError(f"{key} {name} verification failed")
        cutoff = utc_stamp(record["initialization"])
        for frame in record["satellite"]:
            for band in frame:
                if max(goes_time(band["key"],"e"),goes_time(band["key"],"c"))>cutoff-timedelta(minutes=15):
                    raise ValueError("satellite scan/production violates the 15-minute cutoff")
                if band["quality_control"]["invalid_fraction"]>.01:
                    raise ValueError("too many invalid satellite pixels")
    if not (-100<grid[:,:,2].min() and grid[:,:,2].max()<70 and 500<grid[:,:,3].min() and grid[:,:,3].max()<1100):
        raise ValueError("field temperature/pressure units invalid")
    manifest["verified"] = True
    manifest["stored_bytes"] = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"month_completed":key,"split":split,"samples":len(arrays),"rejected":len(failures)}),flush=True)
    return manifest


def finalize(root,observation_sources):
    root = Path(root)
    manifest = plan()
    manifest.update(schema="hyperds_full_sharded_v2",variables=["u10","v10","t2m","sp","tp1h"],
                    units=["m/s","m/s","degC","hPa","mm"],
                    station_targets=["wind_speed","t2m","sp","tp1h"],
                    latitude=LAT.tolist(),longitude=LON.tolist(),era5_store=ERA5_STORE,
                    station_sources=json.loads(Path(observation_sources).read_text()))
    shards,training = [],[]
    for key in manifest["months"]:
        path = root/"shards"/key/"manifest.json"
        raw = path.read_bytes()
        shard = json.loads(raw)
        if not shard["verified"] or shard["samples"]<.8*shard["planned"]:
            raise ValueError(f"{key}: fewer than 80% of planned cycles verified; inspect failures before training")
        shards.append({"path":f"shards/{key}","split":shard["split"],"samples":shard["samples"],
                       "manifest_sha256":hashlib.sha256(raw).hexdigest(),
                       "planned":shard["planned"],"rejected":len(shard["failures"]),
                       "valid_station_labels":shard["valid_station_labels"]})
        if shard["split"]=="train":
            training.append(shard)
    mean,std = combine_moments([s["field_moments"] for s in training])
    sat_mean,sat_std = combine_moments([s["satellite_moments"] for s in training])
    manifest["normalization"] = dict(field_mean=mean,field_std=std,satellite_mean=sat_mean,satellite_std=sat_std)
    manifest["shards"] = shards
    manifest["splits"] = {split:{"samples":sum(s["samples"] for s in shards if s["split"]==split),
                                      "valid_station_labels":sum(np.array(s["valid_station_labels"]) for s in shards if s["split"]==split).tolist()}
                          for split in ("train","val","test")}
    overlay_path = root.parents[1]/"observations/soulis_hourly_v3.npz"
    if not overlay_path.is_file():
        raise ValueError("quality-controlled Soulis v3 overlay is required before training")
    if overlay_path.is_file():
        import os
        import zarr
        provenance = json.loads(overlay_path.with_suffix(".json").read_text())
        manifest["station_overlay"] = {"path":os.path.relpath(overlay_path,root),**provenance}
        with np.load(overlay_path) as overlay:
            times,values = overlay["times"],overlay["values"]
        for split in ("train","val","test"):
            count = np.zeros(2,dtype=np.int64)
            for shard in (s for s in shards if s["split"]==split):
                group = zarr.open_group(str(root/shard["path"]),mode="r")[split]
                stamps = np.asarray(group["init_time"])[:,None]+np.arange(49)[None,:]*3600
                labels = values[np.searchsorted(times,stamps)].copy()
                if split=="test":
                    labels[stamps>=int(datetime(2026,1,1,tzinfo=UTC).timestamp())] = np.nan
                count += np.isfinite(labels).sum(axis=(0,1))
            manifest["splits"][split]["valid_station_labels"][0][:2] = count.tolist()
    previous = root/"manifest.json"
    if previous.is_file():
        old = json.loads(previous.read_text())
        schema = old.get("station_overlay",{}).get("schema","original")
        archived = root/"manifest_history"/f"{schema}.json"
        archived.parent.mkdir(exist_ok=True)
        if not archived.is_file():
            archived.write_text(previous.read_text())
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    report = validate(root)
    (root/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def validate(root):
    import zarr

    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    if "station_overlay" in manifest:
        overlay = manifest["station_overlay"]
        if hashlib.sha256((root/overlay["path"]).read_bytes()).hexdigest()!=overlay["sha256"]:
            raise ValueError("station overlay checksum mismatch")
        with np.load(root/overlay["path"]) as saved:
            values = saved["values"]
        for column,(low,high) in enumerate(((0,100),(-80,60))):
            if ((values[:,column]<low)|(values[:,column]>high)).any():
                raise ValueError("station overlay violates measurement QC")
    stamps = {split:[] for split in ("train","val","test")}
    for shard in manifest["shards"]:
        path = root/shard["path"]
        raw = (path/"manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=shard["manifest_sha256"]:
            raise ValueError("shard provenance changed")
        detail = json.loads(raw)
        group = zarr.open_group(str(path),mode="r")[shard["split"]]
        n = shard["samples"]
        if group["grid"].shape!=(n,49,5,32,64) or group["satellite"].shape!=(n,2,4,128,256):
            raise ValueError("shard geometry mismatch")
        if group["station"].shape!=(n,49,len(STATIONS),4) or group["query_xy"].shape!=(len(STATIONS),2):
            raise ValueError("station geometry mismatch")
        expected_query = np.array([[2*(s['lon']-(LON[0]-.125))/(len(LON)*.25)-1,
                                    2*(s['lat']-(LAT[0]-.125))/(len(LAT)*.25)-1] for s in STATIONS])
        if not np.allclose(np.asarray(group["query_xy"]),expected_query,atol=1e-6):
            raise ValueError("station coordinates differ from the benchmark contract")
        actual = np.asarray(group["init_time"])
        recorded = np.array([int(utc_stamp(r["initialization"]).timestamp()) for r in detail["records"]])
        if not np.array_equal(actual,recorded) or len(set(actual))!=n:
            raise ValueError("initialization provenance mismatch")
        stamps[shard["split"]].extend(actual.tolist())
    for split,following in (("train","val"),("val","test")):
        if max(stamps[split])+48*3600>=min(stamps[following]):
            raise ValueError("target hours cross the split boundary")
    for split in stamps:
        if len(stamps[split])!=len(set(stamps[split])):
            raise ValueError("duplicate initializations")
        if manifest["splits"][split]["valid_station_labels"][0][1]<1000:
            raise ValueError(f"insufficient Waterloo temperature labels in {split}")
    return {"passed":True,"shards":len(manifest["shards"]),"splits":manifest["splits"],
            "primary_station":"eric_d_soulis","selection_station":"eric_d_soulis",
            "checks":["monthly_full_season_coverage","stored_grid_and_satellite_sha256",
                      "satellite_scan_and_production_cutoff","canonical_station_geometry","units_and_missingness",
                      "training_only_normalization","chronological_isolation","Waterloo_label_coverage"]}


if __name__=="__main__":
    print(json.dumps(plan(),indent=2))
