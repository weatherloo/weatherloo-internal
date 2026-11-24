"""Score saved forecasts with the site's exact truth and compare matched errors."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np


STATIONS = ("cyyz","eric_d_soulis")
VARIABLES = ("t2m","wind_speed")
LEADS = tuple(range(6,73,6))


def site_export(predictions,truth_root,output,method,year=2025,stations=STATIONS,metadata=None):
    """Write the site's bias npz for `stations`; `metadata` overrides the default (HyperDS ERA5) notes."""
    stations = tuple(stations)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    start = datetime(year,1,1,tzinfo=timezone.utc)
    days = (datetime(year+1,1,1,tzinfo=timezone.utc)-start).days
    inits = [start+timedelta(hours=h) for h in range(0,days*24,6)]
    bias = np.full((len(inits),len(stations),2,len(LEADS)),np.nan,dtype=np.float64)
    truth = {}
    for station in stations:
        document = json.loads((Path(truth_root)/station/f"observations_6h_{year}.json").read_text())
        truth[station] = {row["valid_time"]:row for row in document["observations"]}
    with np.load(predictions) as data:
        forecasts = data["forecast"]
        model_times = {int(t):index for index,t in enumerate(data["initializations"])}
        model_stations = list(data["station_ids"])
        model_variables = list(data["variables"])
        model_leads = {int(h):index for index,h in enumerate(data["lead_times_hours"])}
        model_units = dict(zip(model_variables,data["units"]))
        if model_units["wind_speed"]!="m/s" or model_units["t2m"]!="degC":
            raise ValueError("unexpected forecast units")
        for ii,init in enumerate(inits):
            mi = model_times.get(int(init.timestamp()))
            if mi is None:
                continue
            for si,station in enumerate(stations):
                ms = model_stations.index(station)
                for li,lead in enumerate(LEADS):
                    if lead not in model_leads:
                        continue
                    stamp = (init+timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    observed = truth[station].get(stamp,{})
                    for vi,variable in enumerate(VARIABLES):
                        value = observed.get(variable)
                        if value is None or not np.isfinite(value):
                            continue
                        forecast = float(forecasts[mi,model_leads[lead],ms,model_variables.index(variable)])
                        if variable=="wind_speed":
                            forecast *= 3.6
                        if np.isfinite(forecast):
                            bias[ii,si,vi,li] = forecast-float(value)
    path = output/f"{method}_{year}.npz"
    np.savez_compressed(path,bias=bias,mae=np.abs(bias),rmse=np.abs(bias),acc=np.full_like(bias,np.nan),
                        station_ids=np.array(stations),variables=np.array(VARIABLES),lead_times_hours=np.array(LEADS),
                        initializations=np.array([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in inits]),
                        method_id=np.array(method),metrics=np.array(["rmse","mae","bias","acc"]))
    defaults = {"method":method,"year":year,"npz_file":path.name,"npz_schema":"see benchmarking-site/AGENTS.md",
                "primary_station":"eric_d_soulis","secondary_station":"cyyz",
                "forecast_horizon_hours":48,"units":{"t2m":"degC","wind_speed":"km/h"},
                "initialization_regime":"retrospective ERA5 initialization; GOES-East scan/production precedes initialization by at least 15 minutes; original dissemination latency unverified",
                "truth":f"canonical benchmarking-site observations_6h_{year}.json; raw model-training labels are not used for site scores",
                "missing":"null/NaN for unavailable observations, rejected satellite cycles, and leads beyond 48 h",
                "aggregation":"RMSE=sqrt(mean(bias**2)); MAE=mean(abs(bias)); bias=mean(bias). ACC unavailable."}
    metadata = {**defaults,**(metadata or {})}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return path


def error_metrics(error):
    return {"n":int(len(error)),"mae":float(np.abs(error).mean()),
            "rmse":float(np.sqrt(np.square(error).mean())),"bias":float(error.mean())}


def matched_comparison(ours,baseline):
    with np.load(ours) as a,np.load(baseline) as b:
        a_times,b_times = list(a["initializations"]),list(b["initializations"])
        b_index = {stamp:index for index,stamp in enumerate(b_times)}
        pairs = [(index,b_index[stamp]) for index,stamp in enumerate(a_times) if stamp in b_index]
        ai,bi = np.array(pairs,dtype=np.int64).reshape(-1,2).T
        result = {}
        shared = [s for s in a["station_ids"] if s in set(b["station_ids"])]
        for station in shared:
            result[station] = {}
            for variable in VARIABLES:
                entries,all_a,all_b = {},[],[]
                for lead in LEADS:
                    if lead not in a["lead_times_hours"] or lead not in b["lead_times_hours"]:
                        continue
                    av = a["bias"][ai,list(a["station_ids"]).index(station),list(a["variables"]).index(variable),list(a["lead_times_hours"]).index(lead)]
                    bv = b["bias"][bi,list(b["station_ids"]).index(station),list(b["variables"]).index(variable),list(b["lead_times_hours"]).index(lead)]
                    mask = np.isfinite(av)&np.isfinite(bv)
                    if not mask.any():
                        continue
                    av,bv = av[mask],bv[mask]
                    entries[str(lead)] = {"hyperds":error_metrics(av),"baseline":error_metrics(bv)}
                    all_a.append(av)
                    all_b.append(bv)
                result[station][variable] = {"by_lead":entries,
                    "pooled":({"hyperds":error_metrics(np.concatenate(all_a)),"baseline":error_metrics(np.concatenate(all_b))} if all_a else None)}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--benchmark-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--method",required=True)
    args = parser.parse_args()
    exported = site_export(args.predictions,args.benchmark_root/"observations",args.out,args.method)
    comparison = {path.parent.name:matched_comparison(exported,path)
                  for path in sorted(args.benchmark_root.glob("*/*_2025.npz"))}
    (args.out/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    print(json.dumps({"benchmark_export":str(exported),"baseline_methods":list(comparison)}),flush=True)


if __name__=="__main__":
    main()
