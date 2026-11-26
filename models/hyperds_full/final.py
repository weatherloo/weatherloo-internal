"""Each target's final forecast (the patch CNN) and the region's site export.

The CNN export from cnn_correction.run_cnn is the forecast; there is no second
model and nothing is averaged across model types. The report gives 2025 RMSE
overall and by lead against raw HRRR on the same pairs.
"""

import json
from pathlib import Path

import numpy as np

import mos

VARIABLES = ("t2m", "wind")
UNITS = {"t2m": 1.0, "wind": 3.6}  # wind is stored in m/s, scored in km/h


def load_cnn(path):
    with np.load(path) as saved:
        data = {k: saved[k] for k in saved.files}
    if "hrrr" not in data:
        raise ValueError(f"{path}: no raw HRRR in the CNN export; re-run the cnn stage")
    out = {"init": data["init"]}
    for j, v in enumerate(VARIABLES):
        out[v], out[f"{v}_obs"], out[f"{v}_raw"] = data["cnn"][..., j], data["obs"][..., j], data["hrrr"][..., j]
    return out


def score(cnn):
    """Returns (report, final) with final = {"init", "t2m", "wind"}."""
    final = {"init": cnn["init"]}
    report = {"model": "cnn", "variables": {}}
    for v in VARIABLES:
        unit = UNITS[v]
        forecast, obs, raw = cnn[v].astype(np.float64), cnn[f"{v}_obs"], cnn[f"{v}_raw"]
        mask = np.isfinite(obs) & np.isfinite(forecast) & np.isfinite(raw)
        error, raw_error = (forecast - obs) * unit, (raw - obs) * unit
        report["variables"][v] = {
            "count": int(mask.sum()), "rmse": mos.rmse(error[mask]), "raw_rmse": mos.rmse(raw_error[mask]),
            "missing_inits": int((~np.isfinite(forecast).all(axis=1)).sum()),
            "rmse_by_lead": [mos.rmse(error[:, h][mask[:, h]]) for h in range(error.shape[1])],
            "raw_rmse_by_lead": [mos.rmse(raw_error[:, h][mask[:, h]]) for h in range(error.shape[1])]}
        final[v] = forecast
    return report, final


def final_target(region, target_id, out=None):
    paths = region.outputs(target_id)
    report, final = score(load_cnn(paths["cnn"]))
    report = {"region": region.name, "target": target_id, **report}
    out = Path(out or paths["final"])
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out.with_name("final.npz"), **final)
    return report, final


SITE_METADATA = {
    "initialization_regime": "HRRRv4 operational 48 h forecasts at 00/06/12/18 UTC (dynamical.org catalog). "
                             "Corrections use only the issued run, earlier runs, and observations at or before initialization.",
    "training": "train initializations 2020-12-03 to 2023 (targets before 2024); 2024 for early stopping and "
                "input choices; one CNN per site (seed ensemble of three); 2025 scored once.",
    "missing": "lead 0 is not produced; leads beyond 48 h are missing; NaN where the observation or the HRRR run is unavailable",
}


def site_export(region, finals, truth_root, out, method):
    """Write the benchmarking-site npz for every target with a site_id, then return its path."""
    from benchmark import site_export as export

    scored = {t.site_id: finals[t.id] for t in region.targets if t.site_id and t.id in finals}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = mos.site_predictions(scored, out / "site_predictions.npz")
    sites = list(scored)
    metadata = {**SITE_METADATA, "method": method, "region": region.name, "model": "patch CNN with site trajectory",
                "primary_station": sites[0] if sites else None, "secondary_station": sites[1] if len(sites) > 1 else None}
    return export(predictions, truth_root, out, method, stations=tuple(scored), metadata=metadata)
