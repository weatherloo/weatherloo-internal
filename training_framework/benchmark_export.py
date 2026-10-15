"""Writes a run's predictions into the exact contract benchmarking-site
expects (see benchmarking-site/AGENTS.md): one JSON per initialization,
index.json, metadata.json, all under benchmarking-site/data/<method_id>/.

predictions.npz rows are (prediction, target, timestamp). A run may also store
per-row `station`, `variable`, `lead_time` arrays (multi-lead / multi-station
models); otherwise every row takes those from the config's data block.

Export MERGES into existing per-init files: each row fills only its own
(station, variable, lead) slot, so several runs (e.g. one per lead time) can
publish into one method without clobbering each other. Unfilled slots stay
null, which the site contract allows ("partial inits are fine").
"""
import json
from pathlib import Path

import numpy as np

LEAD_TIMES_HOURS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
_REPO_STATIONS = Path(__file__).resolve().parents[1] / "benchmarking-site" / "data" / "observations" / "stations.json"


def method_id_for(resolved_cfg):
    """Config `export.method_id` wins; default is <adapter family>_<data method>
    (e.g. lstm_gfs_interpolated), which matches the original LSTM output."""
    override = resolved_cfg.get("export", {}).get("method_id")
    if override:
        return override
    family = resolved_cfg["model"]["adapter"].split("_")[0]
    return f"{family}_{resolved_cfg['data'].get('method', 'bias_correction')}"


def load_station_coords(benchmarking_data_root):
    """{station_id: (lat, lon)} from the site's observations/stations.json, so a
    new station only needs adding there."""
    path = Path(benchmarking_data_root) / "observations" / "stations.json"
    if not path.exists():
        path = _REPO_STATIONS
    stations = json.loads(path.read_text())["stations"]
    return {s["id"]: (s["coordinates"]["lat"], s["coordinates"]["lon"]) for s in stations}


def _column(npz, key, fallback, n):
    if key in npz.files:
        return [v.item() if hasattr(v, "item") else v for v in npz[key]]
    if fallback is None or isinstance(fallback, list) or fallback == "all":
        raise ValueError(
            f"predictions.npz has no per-row '{key}' and the config's data.{key} is "
            f"{fallback!r}; multi-{key} runs must save a '{key}' array in predictions.npz"
        )
    return [fallback] * n


def _empty_metrics():
    return {"lead_times_hours": LEAD_TIMES_HOURS, **{k: [None] * 12 for k in ("rmse", "mae", "bias", "acc")}}


def export_run(run_dir, resolved_cfg, benchmarking_data_root, adapter=None):
    data_cfg = resolved_cfg["data"]
    method_id = method_id_for(resolved_cfg)
    label = (resolved_cfg.get("export", {}).get("model_label")
             or getattr(adapter, "model_label", resolved_cfg["model"]["adapter"]))

    pred_path = Path(run_dir) / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.npz in {run_dir} -- run `train` (or `infer`) first")
    npz = np.load(pred_path, allow_pickle=True)
    predictions, targets, timestamps = npz["predictions"], npz["targets"], npz["timestamps"]
    n = len(predictions)
    stations = _column(npz, "station", data_cfg.get("station"), n)
    variables = _column(npz, "variable", data_cfg.get("variable"), n)
    leads = [int(x) for x in _column(npz, "lead_time", data_cfg.get("lead_time"), n)]

    coords = load_station_coords(benchmarking_data_root)
    unknown = sorted(set(stations) - set(coords))
    if unknown:
        raise ValueError(f"Unknown station(s) {unknown}; add them to observations/stations.json (known: {sorted(coords)})")
    bad_leads = sorted(set(leads) - set(LEAD_TIMES_HOURS))
    if bad_leads:
        raise ValueError(f"lead_time(s) {bad_leads} not in {LEAD_TIMES_HOURS}")

    out_dir = Path(benchmarking_data_root) / method_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {}  # filename -> payload, loaded from disk once so earlier exports are kept
    for pred, target, ts, station, variable, lead in zip(predictions, targets, timestamps, stations, variables, leads):
        ts = str(ts.item() if hasattr(ts, "item") else ts)
        filename = f"{ts[:10]}T{ts[11:13]}Z.json"  # site's <ISO date>T<HH>Z.json convention
        if filename not in payloads:
            existing = out_dir / filename
            payloads[filename] = (json.loads(existing.read_text()) if existing.exists()
                                  else {"method": method_id, "initialization": ts, "locations": {}})
        lat, lon = coords[station]
        loc = payloads[filename]["locations"].setdefault(station, {"lat": lat, "lon": lon, "variables": {}})
        slot = loc["variables"].setdefault(variable, _empty_metrics())

        err = float(pred - target)
        i = LEAD_TIMES_HOURS.index(lead)
        slot["rmse"][i] = slot["mae"][i] = abs(err)
        slot["bias"][i] = err

    for filename, payload in payloads.items():
        with open(out_dir / filename, "w") as f:
            json.dump(payload, f)

    index_path = out_dir / "index.json"
    files = set(json.loads(index_path.read_text())["files"]) if index_path.exists() else set()
    files.update(payloads)
    with open(index_path, "w") as f:
        json.dump({"files": sorted(files)}, f, indent=2)

    metadata_path = out_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    covers = {(c["station"], c["variable"], c["lead_time_hours"])
              for c in metadata.get("covers", []) if isinstance(c, dict)}
    covers.update(zip(stations, variables, leads))
    sources = metadata.get("source_run_dirs", [])
    if str(run_dir) not in sources:
        sources.append(str(run_dir))
    metadata.pop("source_run_dir", None)  # superseded by source_run_dirs
    metadata.update({
        "method_id": method_id,
        "model": f"{label} (input: {data_cfg.get('method', 'unknown')})",
        "source_run_dirs": sources,
        "covers": [{"station": s, "variable": v, "lead_time_hours": l} for s, v, l in sorted(covers)],
        "note": "Partial coverage: only the (station, variable, lead_time) slots listed in `covers` are filled in.",
    })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"method_id": method_id, "out_dir": str(out_dir), "n_files": len(payloads), "n_slots_filled": n}
