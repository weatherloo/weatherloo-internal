"""Writes a run's predictions into the exact contract benchmarking-site
expects (see benchmarking-site/AGENTS.md): one JSON per initialization,
index.json, metadata.json, all under benchmarking-site/data/<method_id>/.

A single run only covers one (station, variable, lead_time) combo, so each
exported file has that one slot filled and the rest null -- which the site's
own contract explicitly allows ("partial inits are fine").
"""
import json
import os
from pathlib import Path

import numpy as np

LEAD_TIMES_HOURS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
STATION_COORDS = {
    "cyyz": (43.6777, -79.6248),
    "eric_d_soulis": (43.4668, -80.5164),
}


def export_run(run_dir, resolved_cfg, benchmarking_data_root):
    data_cfg = resolved_cfg["data"]
    station = data_cfg["station"]
    variable = data_cfg["variable"]
    lead_time = data_cfg["lead_time"]
    method_id = f"lstm_{data_cfg.get('method', 'bias_correction')}"

    if station not in STATION_COORDS:
        raise ValueError(f"Unknown station '{station}', expected one of {list(STATION_COORDS)}")
    if lead_time not in LEAD_TIMES_HOURS:
        raise ValueError(f"lead_time {lead_time} not in {LEAD_TIMES_HOURS}")
    lead_idx = LEAD_TIMES_HOURS.index(lead_time)

    pred_path = Path(run_dir) / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.npz in {run_dir} -- run `train` (or `infer`) first")
    npz = np.load(pred_path, allow_pickle=True)
    predictions, targets, timestamps = npz["predictions"], npz["targets"], npz["timestamps"]

    out_dir = Path(benchmarking_data_root) / method_id
    out_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = STATION_COORDS[station]
    written = []
    for pred, target, ts in zip(predictions, targets, timestamps):
        ts = ts.item() if hasattr(ts, "item") else ts
        err = float(pred - target)
        rmse = mae = abs(err)
        bias = err

        rmse_arr = [None] * 12
        mae_arr = [None] * 12
        bias_arr = [None] * 12
        acc_arr = [None] * 12
        rmse_arr[lead_idx] = rmse
        mae_arr[lead_idx] = mae
        bias_arr[lead_idx] = bias

        # Filenames follow the site's <ISO date>T<HH>Z.json convention.
        hh = ts[11:13]
        date = ts[:10]
        filename = f"{date}T{hh}Z.json"

        payload = {
            "method": method_id,
            "initialization": ts,
            "locations": {
                station: {
                    "lat": lat,
                    "lon": lon,
                    "variables": {
                        variable: {
                            "lead_times_hours": LEAD_TIMES_HOURS,
                            "rmse": rmse_arr,
                            "mae": mae_arr,
                            "bias": bias_arr,
                            "acc": acc_arr,
                        }
                    },
                }
            },
        }
        with open(out_dir / filename, "w") as f:
            json.dump(payload, f)
        written.append(filename)

    written = sorted(set(written))
    with open(out_dir / "index.json", "w") as f:
        json.dump({"files": written}, f, indent=2)

    metadata_path = out_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "method_id": method_id,
        "model": f"LSTM bias correction (input: {data_cfg.get('method', 'unknown')})",
        "source_run_dir": str(run_dir),
        "covers": {"station": station, "variable": variable, "lead_time_hours": lead_time},
        "note": "Partial coverage: only the (station, variable, lead_time) this run was trained on is filled in.",
    })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"method_id": method_id, "out_dir": str(out_dir), "n_files": len(written)}
