"""Validation-weighted blend of a target's correction models, and the region's site export.

Inputs are the per-target exports of mos.run_study_v2 (gradient boosting) and
cnn_correction.run_cnn. One weight per variable is chosen on 2024 validation
RMSE from a 0.05 grid; 2025 is scored only after the weight is fixed. A target
without a CNN uses gradient boosting alone. Where the CNN has no prediction for
a 2025 initialization (failed patch read), the final forecast falls back to
gradient boosting.
"""

import json
from pathlib import Path

import numpy as np

import mos

VARIABLES = ("t2m", "wind")
UNITS = {"t2m": 1.0, "wind": 3.6}  # wind is stored in m/s, scored in km/h
WEIGHTS = np.round(np.linspace(0, 1, 21), 2)  # weight on gradient boosting


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


def choose_weight(gbm, cnn, obs, unit):
    mask = np.isfinite(gbm) & np.isfinite(cnn) & np.isfinite(obs)
    curve = {float(w): mos.rmse(((w * gbm + (1 - w) * cnn) - obs)[mask] * unit) for w in WEIGHTS}
    return min(curve, key=curve.get), curve, int(mask.sum())


def fit_blend(gbm, cnn=None):
    """Returns (report, final) where final = {"init", "t2m", "wind"} over the gbm 2025 initializations."""
    final = {"init": gbm["init"]}
    report = {"weights_on_gbm": {}, "selection": "2024 validation RMSE, 0.05 grid", "variables": {}}
    for v in VARIABLES:
        unit = UNITS[v]
        g_test, obs_test, raw_test = gbm[v].astype(np.float64), gbm[f"{v}_obs"], gbm[f"{v}_raw"]
        entry = {}
        if cnn is None:
            weight, blended = 1.0, g_test
            entry["note"] = "no CNN configured; gradient boosting only"
        else:
            gv, cv = aligned(gbm["val_init"], cnn["val_init"])
            weight, curve, count = choose_weight(gbm[f"val_{v}"][gv], cnn[f"val_{v}"][cv], gbm[f"val_{v}_obs"][gv], unit)
            entry["val_curve"], entry["val_count"] = curve, count
            gt, ct = aligned(gbm["init"], cnn["init"])
            c_test = np.full_like(g_test, np.nan)
            c_test[gt] = cnn[v][ct]
            blended = np.where(np.isfinite(c_test), weight * g_test + (1 - weight) * c_test, g_test)
            entry["cnn_fallback_inits"] = int((~np.isfinite(c_test).all(axis=1)).sum())
            mask = np.isfinite(obs_test) & np.isfinite(c_test) & np.isfinite(g_test)
            entry["test_matched"] = {
                "count": int(mask.sum()),
                **{name: mos.rmse((pred - obs_test)[mask] * unit) for name, pred in (
                    ("raw_hrrr", raw_test), ("gbm", g_test), ("cnn", c_test),
                    ("mean_50_50", (g_test + c_test) / 2), ("blend", blended))}}
        report["weights_on_gbm"][v] = weight
        mask = np.isfinite(obs_test) & np.isfinite(blended)
        error, raw_error = (blended - obs_test) * unit, (raw_test - obs_test) * unit
        entry["test_final"] = {"count": int(mask.sum()), "rmse": mos.rmse(error[mask]),
                               "raw_rmse": mos.rmse(raw_error[mask]),
                               "rmse_by_lead": [mos.rmse(error[:, h][mask[:, h]]) for h in range(error.shape[1])],
                               "raw_rmse_by_lead": [mos.rmse(raw_error[:, h][mask[:, h]]) for h in range(error.shape[1])]}
        report["variables"][v] = entry
        final[v] = blended
    return report, final


def blend_target(region, target_id, out=None):
    target = region.target(target_id)
    paths = region.outputs(target_id)
    gbm = load_gbm(paths["gbm"])
    cnn = load_cnn(paths["cnn"]) if "cnn" in target.models else None
    report, final = fit_blend(gbm, cnn)
    report = {"region": region.name, "target": target_id, "models": list(target.models), **report}
    out = Path(out or paths["blend"])
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out.with_name("final.npz"), **final)
    return report, final


SITE_METADATA = {
    "initialization_regime": "HRRRv4 operational 48 h forecasts at 00/06/12/18 UTC (dynamical.org catalog). "
                             "Corrections use only the issued run, earlier runs, and observations at or before initialization.",
    "training": "train initializations 2020-12-03 to 2023 (targets before 2024); 2024 for model selection, early stopping "
                "and blend weights; 2025 scored once.",
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
