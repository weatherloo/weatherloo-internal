"""Versioned station-label correction; existing weather shards stay immutable."""

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def build_overlay(raw_root,output):
    sys.path.insert(0,"/workspace/src")
    from hrrr_bias_correction.make_data import SoulisObsIndex

    raw_root,output = Path(raw_root),Path(output)
    if output.is_file():
        provenance = json.loads(output.with_suffix(".json").read_text())
        if provenance["sha256"]!=hashlib.sha256(output.read_bytes()).hexdigest():
            raise ValueError("existing station overlay checksum mismatch")
        if provenance["parser_sha256"]!=hashlib.sha256(Path("/workspace/src/hrrr_bias_correction/make_data.py").read_bytes()).hexdigest():
            raise ValueError("station parser changed; use a new overlay version")
        return provenance
    source = SoulisObsIndex(raw_root,tuple(range(2018,2027)))
    with np.load(raw_root/"hourly.npz") as cached:
        times = cached["times"]
    labels = np.full((len(times),2),np.nan,dtype=np.float32)
    for index,stamp in enumerate(times):
        row = source.lookup(datetime.fromtimestamp(int(stamp),tz=timezone.utc))
        if row is not None:
            labels[index] = (row[1]/3.6,row[0])
    # Preserve the original hourly-cache QC when replacing station labels.
    for column,(low,high) in enumerate(((0,100),(-80,60))):
        values = labels[:,column]
        values[(values<low)|(values>high)] = np.nan
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,times=times,values=labels)
    temporary.replace(output)
    provenance = {"schema":"soulis_hourly_v3","sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                  "targets":["wind_speed","t2m"],"units":["m/s","degC"],
                  "parser_sha256":hashlib.sha256(Path("/workspace/src/hrrr_bias_correction/make_data.py").read_bytes()).hexdigest(),
                  "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw_root.glob("uw_hobo_15min_*.csv"))},
                  "quality_control":{"temperature_c":[-80,60],"wind_ms":[0,100],
                                     "average_wind":"masked if above valid same-height gust by more than 0.02 km/h"},
                  "note":"Date/Time 2020 temperature retained. Original range QC restored; average/gust inconsistency masked. Missing 4.4 m wind stays missing, never replaced with 2 m wind."}
    (output.with_suffix(".json")).write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance
