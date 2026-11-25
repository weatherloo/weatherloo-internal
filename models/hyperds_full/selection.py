"""Single-model selection for each target, and the region's site export.

Inputs are the per-target exports of mos.run_study_v2 (gradient boosting) and
cnn_correction.run_cnn. For each variable, the model with the lowest 2024
validation RMSE provides the whole 2025 forecast; models are never averaged.
2025 is scored only after the choice is fixed. A target without a CNN uses
gradient boosting. Where the chosen CNN has no prediction for a 2025
initialization (failed patch read), that initialization falls back to gradient
boosting, and the report counts such initializations.
"""

import json
from pathlib import Path

import numpy as np

import mos

VARIABLES = ("t2m", "wind")
UNITS = {"t2m": 1.0, "wind": 3.6}  # wind is stored in m/s, scored in km/h


def load_gbm(path):
    with np.load(path) as saved:
        data = {k: saved[k] for k in saved.files}
    if "val_init" not in data:
        raise ValueError(f"{path}: no 2024 validation predictions; re-run the gbm stage")
    return data


def load_cnn(path):
    with np.load(path) as saved:
        data = {k: saved[k] for k in saved.files}
    if "val_init" not in data:
        raise ValueError(f"{path}: no 2024 validation predictions; re-run the cnn stage")
    out = {"init": data["init"], "val_init": data["val_init"]}
    for j, v in enumerate(VARIABLES):
        out[v], out[f"{v}_obs"] = data["cnn"][..., j], data["obs"][..., j]
        out[f"val_{v}"], out[f"val_{v}_obs"] = data["val_cnn"][..., j], data["val_obs"][..., j]
    return out


def aligned(init_a, init_b):
    """Row indices into a and b for initializations present in both."""
    _, a, b = np.intersect1d(init_a, init_b, return_indices=True)
    return a, b


def fit_selection(gbm, cnn=None):
    """Returns (report, final) where final = {"init", "t2m", "wind"} over the gbm 2025 initializations."""
    final = {"init": gbm["init"]}
    report = {"selection": "lowest 2024 validation RMSE per variable; models are not averaged",
              "chosen": {}, "variables": {}}
    for v in VARIABLES:
        unit = UNITS[v]
        g_test, obs_test, raw_test = gbm[v].astype(np.float64), gbm[f"{v}_obs"], gbm[f"{v}_raw"]
        entry = {}
        if cnn is None:
            chosen, final[v] = "gbm", g_test
            entry["note"] = "no CNN configured; gradient boosting only"
        else:
            gv, cv = aligned(gbm["val_init"], cnn["val_init"])
            g_val, c_val, o_val = gbm[f"val_{v}"][gv], cnn[f"val_{v}"][cv], gbm[f"val_{v}_obs"][gv]
            mask = np.isfinite(g_val) & np.isfinite(c_val) & np.isfinite(o_val)
            entry["val_rmse"] = {"gbm": mos.rmse((g_val - o_val)[mask] * unit),
                                 "cnn": mos.rmse((c_val - o_val)[mask] * unit)}
            entry["val_count"] = int(mask.sum())
            chosen = min(entry["val_rmse"], key=entry["val_rmse"].get)
            gt, ct = aligned(gbm["init"], cnn["init"])
            c_test = np.full_like(g_test, np.nan)
            c_test[gt] = cnn[v][ct]
            missing = ~np.isfinite(c_test).all(axis=1) & np.isfinite(g_test).all(axis=1)
            final[v] = np.where(np.isfinite(c_test), c_test, g_test) if chosen == "cnn" else g_test
            entry["cnn_fallback_inits"] = int(missing.sum()) if chosen == "cnn" else 0
            mask = np.isfinite(obs_test) & np.isfinite(c_test) & np.isfinite(g_test)
            entry["test_matched"] = {"count": int(mask.sum()),
                                     **{name: mos.rmse((pred - obs_test)[mask] * unit)
                                        for name, pred in (("raw_hrrr", raw_test), ("gbm", g_test), ("cnn", c_test))}}
        report["chosen"][v] = chosen
        mask = np.isfinite(obs_test) & np.isfinite(final[v])
        error, raw_error = (final[v] - obs_test) * unit, (raw_test - obs_test) * unit
        entry["test_final"] = {"model": chosen, "count": int(mask.sum()), "rmse": mos.rmse(error[mask]),
                               "raw_rmse": mos.rmse(raw_error[mask]),
                               "rmse_by_lead": [mos.rmse(error[:, h][mask[:, h]]) for h in range(error.shape[1])],
                               "raw_rmse_by_lead": [mos.rmse(raw_error[:, h][mask[:, h]]) for h in range(error.shape[1])]}
        report["variables"][v] = entry
    return report, final


def select_target(region, target_id, out=None):
    target = region.target(target_id)
    paths = region.outputs(target_id)
    gbm = load_gbm(paths["gbm"])
    cnn = load_cnn(paths["cnn"]) if "cnn" in target.models else None
    report, final = fit_selection(gbm, cnn)
    report = {"region": region.name, "target": target_id, "models": list(target.models), **report}
    out = Path(out or paths["selection"])
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out.with_name("final.npz"), **final)
    return report, final


SITE_METADATA = {
    "initialization_regime": "HRRRv4 operational 48 h forecasts at 00/06/12/18 UTC (dynamical.org catalog). "
                             "Corrections use only the issued run, earlier runs, and observations at or before initialization.",
    "training": "train initializations 2020-12-03 to 2023 (targets before 2024); 2024 for model selection and early "
                "stopping; one model per variable, never averaged; 2025 scored once.",
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
    metadata = {**SITE_METADATA, "method": method, "region": region.name,
                "models": {t.site_id: list(t.models) for t in region.targets if t.site_id},
                "primary_station": sites[0] if sites else None, "secondary_station": sites[1] if len(sites) > 1 else None}
    return export(predictions, truth_root, out, method, stations=tuple(scored), metadata=metadata)
