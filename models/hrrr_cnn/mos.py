"""Station-level HRRR forecast correction (MOS) at a target site, with issue-time-safe predictors.

Rows are (initialization T0, lead h = 1..48, station). The target is the
observation minus the HRRR point forecast; predictions add the learned residual
back to HRRR. Only information available when the T0 run is issued is used:
the T0 run itself, earlier runs, and observations at or before T0.

Split by initialization: train 2020-12-03 .. targets before 2024, validation
2024 (targets before 2025), test 2025.
"""

import repo_paths  # noqa: F401  (shared pipelines modules on sys.path)
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

UTC = timezone.utc
TRAIN_END = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
VAL_END = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
TEST_END = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
TARGETS = ("t2m", "wind")


def load_points(root):
    """Concatenate monthly point files -> values [init, lead, station, var], init_time, stations, variables."""
    root = Path(root)
    values, inits, meta = [], [], None
    for path in sorted(root.glob("*/manifest.json")):
        detail = json.loads(path.read_text())
        if meta is None:
            meta = detail
        elif detail["stations"] != meta["stations"] or detail["variables"] != meta["variables"]:
            raise ValueError(f"{path.parent.name}: station/variable layout differs")
        with np.load(path.parent / "points.npz") as saved:
            if len(saved["init_time"]):
                values.append(saved["values"])
                inits.append(saved["init_time"])
    order = np.argsort(np.concatenate(inits))
    return (np.concatenate(values)[order], np.concatenate(inits)[order], meta["stations"], meta["variables"])


def nearest_stations(station_meta, ids, target, k):
    """Target plus its k nearest stations (by great-circle distance) among ids."""
    from station_labels import distance_km

    lookup = {s["id"]: s for s in station_meta}
    ranked = sorted((distance_km(lookup[target], lookup[i]), i) for i in ids if i in lookup)
    return [i for _, i in ranked[:k + 1]]


def split_of(init):
    if init + 48 * 3600 < TRAIN_END:
        return "train"
    if TRAIN_END <= init and init + 48 * 3600 < VAL_END:
        return "val"
    if VAL_END <= init < TEST_END:
        return "test"
    return None


def build(values, inits, point_ids, variables, station_meta, label_times, labels, label_ids, use):
    """Feature matrix for stations `use`. Returns X, names, y {t2m, wind}, hrrr {t2m, wind}, meta columns."""
    var = {name: i for i, name in enumerate(variables)}
    cols = [point_ids.index(s) for s in use]
    V = values[:, :, cols].astype(np.float64)  # [init, lead, station, var]
    n_init, n_lead, n_st, _ = V.shape
    lookup = {s["id"]: s for s in station_meta}
    lab = labels[:, [label_ids.index(s) for s in use]][..., :2].astype(np.float64)  # wind m/s, t2m

    def hrrr(name):
        return V[..., var[name]]

    t2m = hrrr("temperature_2m")
    ws10 = np.hypot(hrrr("wind_u_10m"), hrrr("wind_v_10m"))
    ws80 = np.hypot(hrrr("wind_u_80m"), hrrr("wind_v_80m"))
    # Observation at each valid time (NaN where missing or outside the label record).
    valid = inits[:, None] + np.arange(n_lead)[None] * 3600
    index = np.searchsorted(label_times, valid)
    ok = (index < len(label_times)) & (label_times[np.minimum(index, len(label_times) - 1)] == valid)
    obs = np.full((n_init, n_lead, n_st, 2), np.nan)
    obs[ok] = lab[index[ok]]
    obs_ws, obs_t = obs[..., 0], obs[..., 1]

    def obs_at(offset_hours):
        """Observation at T0 + offset (offset <= 0 only: issue-time safe)."""
        assert offset_hours <= 0
        stamp = inits + offset_hours * 3600
        i = np.searchsorted(label_times, stamp)
        hit = (i < len(label_times)) & (label_times[np.minimum(i, len(label_times) - 1)] == stamp)
        out = np.full((n_init, n_st, 2), np.nan)
        out[hit] = lab[i[hit]]
        return out

    position = {int(t): k for k, t in enumerate(inits)}

    def run_index(hours_back):
        return np.array([position.get(int(t) - hours_back * 3600, -1) for t in inits])

    def from_run(field, hours_back, lead):
        """field[run T0-hours_back, lead] for each T0 (NaN if that run is missing)."""
        idx = run_index(hours_back)
        out = np.full((n_init, n_st), np.nan)
        have = idx >= 0
        out[have] = field[idx[have], lead]
        return out

    obs0, obs3 = obs_at(0), obs_at(-3)
    # Error of the previous runs' 6-h forecasts verifying at T0, T0-6, T0-12, T0-18 (all observed by T0).
    err_t, err_w = [], []
    for k in range(4):
        verify = obs_at(-6 * k)
        err_t.append(verify[..., 1] - from_run(t2m, 6 * (k + 1), 6))
        err_w.append(verify[..., 0] - from_run(ws10, 6 * (k + 1), 6))
    with np.errstate(invalid="ignore"):
        recent_err_t = np.nanmean(np.stack(err_t), axis=0)
        recent_err_w = np.nanmean(np.stack(err_w), axis=0)
    names, columns = [], []

    def add(name, array):
        names.append(name)
        columns.append(np.broadcast_to(array, (n_init, n_lead, n_st)))

    for name in ("temperature_2m", "dew_point_temperature_2m", "relative_humidity_2m", "wind_gust_surface",
                 "total_cloud_cover_atmosphere", "downward_short_wave_radiation_flux_surface",
                 "downward_long_wave_radiation_flux_surface", "snow_thickness_surface", "snow_area_fraction_surface",
                 "precipitable_water_atmosphere", "composite_reflectivity"):
        add(name, hrrr(name))
    add("wind_u_10m", hrrr("wind_u_10m"))
    add("wind_v_10m", hrrr("wind_v_10m"))
    add("wind_speed_10m", ws10)
    add("wind_speed_80m", ws80)
    add("shear_80m_minus_10m", ws80 - ws10)
    add("pressure_surface_hpa", hrrr("pressure_surface") / 100)
    add("mslp_hpa", hrrr("pressure_reduced_to_mean_sea_level") / 100)
    add("precip_mm_h", np.nan_to_num(hrrr("precipitation_surface")) * 3600)
    ceiling = hrrr("geopotential_height_cloud_ceiling")
    add("log_ceiling_m", np.where(np.isfinite(ceiling), np.log1p(np.maximum(ceiling, 0)), np.log1p(20000)))
    add("t2m_minus_dewpoint", hrrr("temperature_2m") - hrrr("dew_point_temperature_2m"))
    add("t2m_3h_tendency", t2m - np.concatenate((np.repeat(t2m[:, :1], 3, axis=1), t2m[:, :-3]), axis=1))
    # Time.
    lead = np.arange(n_lead)[None, :, None].astype(float)
    hour = (valid % 86400) / 86400
    year = (valid / 86400 % 365.2425) / 365.2425
    add("lead_h", lead)
    add("valid_hour_sin", np.sin(2 * np.pi * hour)[..., None])
    add("valid_hour_cos", np.cos(2 * np.pi * hour)[..., None])
    add("doy_sin", np.sin(2 * np.pi * year)[..., None])
    add("doy_cos", np.cos(2 * np.pi * year)[..., None])
    add("init_hour", ((inits % 86400) / 3600)[:, None, None])
    # Station.
    add("station_lat", np.array([lookup[s]["lat"] for s in use])[None, None])
    add("station_lon", np.array([lookup[s]["lon"] for s in use])[None, None])
    add("station_elev", np.array([lookup[s].get("elevation") or np.nan for s in use], dtype=float)[None, None])
    add("station_index", np.arange(n_st, dtype=float)[None, None])
    # Issue-time observations and recent errors (all at or before T0).
    add("obs_t_T0", obs0[:, None, :, 1])
    add("obs_ws_T0", obs0[:, None, :, 0])
    add("obs_t_T0m3", obs3[:, None, :, 1])
    add("bias_t_T0", (obs0[..., 1] - t2m[:, 0])[:, None])
    add("bias_ws_T0", (obs0[..., 0] - ws10[:, 0])[:, None])
    add("recent_err_t_24h", recent_err_t[:, None])
    add("recent_err_ws_24h", recent_err_w[:, None])
    # Time-lagged: previous run's forecast for the same valid time (lead + 6 <= 48).
    prev_t = np.full((n_init, n_lead, n_st), np.nan)
    prev_w = np.full_like(prev_t, np.nan)
    idx = run_index(6)
    have = idx >= 0
    prev_t[have, :n_lead - 6] = t2m[idx[have], 6:]
    prev_w[have, :n_lead - 6] = ws10[idx[have], 6:]
    add("prev_run_t2m_minus_current", prev_t - t2m)
    add("prev_run_ws_minus_current", prev_w - ws10)
    # Time-lagged ensemble: runs T0-12 h and T0-18 h for the same valid time, and member spread.
    members_t, members_w = [t2m, prev_t], [ws10, prev_w]
    for back in (12, 18):
        lag_t = np.full((n_init, n_lead, n_st), np.nan)
        lag_w = np.full_like(lag_t, np.nan)
        idx = run_index(back)
        have = idx >= 0
        lag_t[have, :n_lead - back] = t2m[idx[have], back:]
        lag_w[have, :n_lead - back] = ws10[idx[have], back:]
        add(f"prev{back}_t2m_minus_current", lag_t - t2m)
        add(f"prev{back}_ws_minus_current", lag_w - ws10)
        members_t.append(lag_t)
        members_w.append(lag_w)
    with np.errstate(invalid="ignore"):
        add("lagged_spread_t2m", np.nanstd(np.stack(members_t), axis=0))
        add("lagged_spread_ws", np.nanstd(np.stack(members_w), axis=0))
    X = np.stack(columns, axis=-1)[:, 1:]  # leads 1..48
    y = {"t2m": (obs_t - t2m)[:, 1:], "wind": (obs_ws - ws10)[:, 1:]}
    base = {"t2m": t2m[:, 1:], "wind": ws10[:, 1:]}
    split = np.array([split_of(int(t)) for t in inits])
    return X.astype(np.float32), names, y, base, split


OBS_FEATURES = ("obs_t_T0", "obs_ws_T0", "obs_t_T0m3", "bias_t_T0", "bias_ws_T0", "recent_err_t_24h",
                "recent_err_ws_24h")
LAG_FEATURES = ("prev_run_t2m_minus_current", "prev_run_ws_minus_current")
ENSEMBLE_FEATURES = ("prev12_t2m_minus_current", "prev12_ws_minus_current", "prev18_t2m_minus_current",
                     "prev18_ws_minus_current", "lagged_spread_t2m", "lagged_spread_ws")
CORE_HRRR = ("temperature_2m", "wind_u_10m", "wind_v_10m", "wind_speed_10m")
TIME_STATION = ("lead_h", "valid_hour_sin", "valid_hour_cos", "doy_sin", "doy_cos", "init_hour", "station_lat",
                "station_lon", "station_elev", "station_index")


def bias_table(residual, lead, hour, month, train):
    """Mean residual by (6-h lead block, valid hour, month), falling back to (lead block, hour)."""
    fine, coarse = {}, {}
    keys_f = list(zip(lead // 6, hour, month))
    keys_c = list(zip(lead // 6, hour))
    for kf, kc, r, t in zip(keys_f, keys_c, residual, train):
        if t and np.isfinite(r):
            fine.setdefault(kf, []).append(r)
            coarse.setdefault(kc, []).append(r)
    fine = {k: float(np.mean(v)) for k, v in fine.items() if len(v) >= 20}
    coarse = {k: float(np.mean(v)) for k, v in coarse.items()}
    return np.array([fine.get(kf, coarse.get(kc, 0.)) for kf, kc in zip(keys_f, keys_c)])


def rmse(error):
    error = error[np.isfinite(error)]
    return float(np.sqrt(np.mean(error ** 2))) if len(error) else float("nan")


def run_study(point_root, station_root, target="eric_d_soulis", k=30, r5_predictions=None, seed=0, out=None):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    values, inits, point_ids, variables = load_points(point_root)
    station_root = Path(station_root)
    station_meta = json.loads((station_root / "stations.json").read_text())["stations"]
    with np.load(station_root / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    label_ids = [s["id"] for s in station_meta]
    use = nearest_stations(station_meta, [i for i in point_ids if i in label_ids], target, k)
    X, names, y, base, split = build(values, inits, point_ids, variables, station_meta, label_times, labels,
                                     label_ids, use)
    n_init, n_lead, n_st, n_feat = X.shape
    col = {n: i for i, n in enumerate(names)}
    flat = X.reshape(-1, n_feat)
    split_rows = np.repeat(split, n_lead * n_st)
    station_rows = np.tile(np.arange(n_st), n_init * n_lead)
    lead_rows = np.tile(np.repeat(np.arange(1, n_lead + 1), n_st), n_init)
    init_rows = np.repeat(inits, n_lead * n_st)
    valid_rows = init_rows + lead_rows * 3600
    hour_rows = (valid_rows % 86400) // 3600
    month_rows = np.array([datetime.fromtimestamp(int(t), tz=UTC).month for t in valid_rows[::n_st]]).repeat(n_st)
    is_target = station_rows == 0
    report = {"target": target, "stations": use, "features": names, "rows": int(len(flat)), "targets": {}}
    saved_predictions = {}
    for target_name in TARGETS:
        res = y[target_name].reshape(-1)
        hr = base[target_name].reshape(-1)
        finite = np.isfinite(res)
        train = (split_rows == "train") & finite
        val = (split_rows == "val") & finite
        test_target = (split_rows == "test") & is_target
        preds = {"raw_hrrr": np.zeros_like(res)}
        preds["bias_table"] = bias_table(res, lead_rows, hour_rows, month_rows, train & is_target)

        def feature_subset(keep):
            return [col[n] for n in names if keep(n)]

        all_features = list(range(n_feat))
        no_obs = feature_subset(lambda n: n not in OBS_FEATURES)
        core = feature_subset(lambda n: n in CORE_HRRR or n in TIME_STATION)
        # Ridge on the target station only (missing -> train median, plus the core inputs).
        ridge_cols = [c for c in all_features if names[c] != "station_index"]
        med = np.nanmedian(flat[train & is_target][:, ridge_cols], axis=0)
        def ridge_x(rows):
            data = flat[rows][:, ridge_cols].astype(np.float64)
            return np.where(np.isfinite(data), data, med)
        mu = ridge_x(train & is_target).mean(axis=0)
        sd = ridge_x(train & is_target).std(axis=0) + 1e-6
        best_alpha, best_score = None, np.inf
        for alpha in (0.1, 1, 10, 100, 1000):
            model = Ridge(alpha=alpha).fit((ridge_x(train & is_target) - mu) / sd, res[train & is_target])
            score = rmse(model.predict((ridge_x(val & is_target) - mu) / sd) - res[val & is_target])
            if score < best_score:
                best_alpha, best_score = alpha, score
        ridge = Ridge(alpha=best_alpha).fit((ridge_x(train & is_target) - mu) / sd, res[train & is_target])
        preds["ridge_target_only"] = ridge.predict((ridge_x(np.ones(len(res), bool)) - mu) / sd)

        def boost(rows_train, rows_val, features, label):
            params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                      "verbose": -1, "seed": seed, "num_threads": 16}
            categorical = [features.index(col["station_index"])] if col["station_index"] in features else []
            train_set = lgb.Dataset(flat[rows_train][:, features], res[rows_train], categorical_feature=categorical,
                                    free_raw_data=False)
            val_set = lgb.Dataset(flat[rows_val][:, features], res[rows_val], categorical_feature=categorical,
                                  reference=train_set)
            model = lgb.train(params, train_set, num_boost_round=3000, valid_sets=[val_set],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            report_entry = {"best_iteration": model.best_iteration,
                            "top_features": sorted(zip(model.feature_importance("gain"), [names[f] for f in features]),
                                                   reverse=True)[:12]}
            return model.predict(flat[:, features], num_iteration=model.best_iteration), report_entry

        details = {}
        preds["gbm_pooled"], details["gbm_pooled"] = boost(train, val & is_target, all_features, "pooled")
        preds["gbm_pooled_no_obs"], details["gbm_pooled_no_obs"] = boost(train, val & is_target, no_obs, "no_obs")
        preds["gbm_pooled_core_only"], details["gbm_pooled_core_only"] = boost(train, val & is_target, core, "core")
        preds["gbm_target_only"], details["gbm_target_only"] = boost(train & is_target, val & is_target,
                                                                     all_features, "target")
        rows = test_target & finite
        entry = {"count": int(rows.sum()), "methods": {}, "details": {k: {"best_iteration": v["best_iteration"],
                 "top_features": [(n, float(g)) for g, n in v["top_features"]]} for k, v in details.items()},
                 "ridge_alpha": best_alpha}
        unit = 3.6 if target_name == "wind" else 1.0
        for method, p in preds.items():
            error = (p - res)[rows] * unit
            by_lead = [rmse(error[lead_rows[rows] == h]) for h in range(1, n_lead + 1)]
            entry["methods"][method] = {"rmse": rmse(error), "mae": float(np.mean(np.abs(error))),
                                        "bias": float(np.mean(error)), "rmse_by_lead": by_lead}
        report["targets"][target_name] = entry
        saved_predictions[target_name] = {m: (hr + p)[rows] for m, p in preds.items()}
        saved_predictions[target_name]["observation"] = (hr + res)[rows]
        saved_predictions[target_name]["init"] = init_rows[rows]
        saved_predictions[target_name]["lead"] = lead_rows[rows]
    if r5_predictions:
        report["r5"] = compare_r5(r5_predictions, saved_predictions, target)
    report["bootstrap"] = bootstrap(saved_predictions, report)
    if out:
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(Path(out).with_suffix(".npz"), **{f"{t}_{k}": v for t, d in saved_predictions.items()
                                                             for k, v in d.items()})
    return report


def compare_r5(path, saved, target):
    """r5 HyperDS on the identical (init, lead) test pairs where it has a forecast."""
    with np.load(path) as r5:
        stations = list(r5["station_ids"])
        if target not in stations:
            return None
        s = stations.index(target)
        index = {int(t): i for i, t in enumerate(r5["initializations"])}
        forecast = r5["forecast"][:, :, s, :2].astype(np.float64)  # wind m/s, t2m
        leads = list(r5["lead_times_hours"])
    out = {}
    for target_name, column, unit in (("t2m", 1, 1.0), ("wind", 0, 3.6)):
        d = saved[target_name]
        r5_pred = np.array([forecast[index[int(i)], leads.index(int(h)), column] if int(i) in index and int(h) in leads
                            else np.nan for i, h in zip(d["init"], d["lead"])])
        mask = np.isfinite(r5_pred) & np.isfinite(d["observation"])
        out[target_name] = {"count": int(mask.sum()), "r5": rmse((r5_pred - d["observation"])[mask] * unit)}
        for method in [m for m in d if m not in ("observation", "init", "lead")]:
            out[target_name][method] = rmse((d[method] - d["observation"])[mask] * unit)
        saved[target_name]["r5"] = r5_pred
    return out


def bootstrap(saved, report, samples=1000, seed=0):
    """Day-block bootstrap of RMSE differences: best GBM vs raw HRRR (and vs r5 where present)."""
    rng = np.random.default_rng(seed)
    out = {}
    for target_name, unit in (("t2m", 1.0), ("wind", 3.6)):
        d = saved[target_name]
        days = (d["init"] // 86400).astype(np.int64)
        unique = np.unique(days)
        groups = {day: np.flatnonzero(days == day) for day in unique}
        best = min((m for m in d if m.startswith("gbm")), key=lambda m: report["targets"][target_name]["methods"][m]["rmse"])
        comparisons = {}
        for other in ("raw_hrrr", "bias_table", "r5"):
            if other not in d:
                continue
            mask = np.isfinite(d[other]) & np.isfinite(d["observation"])
            diffs = []
            for _ in range(samples):
                pick = np.concatenate([groups[day] for day in rng.choice(unique, len(unique))])
                pick = pick[mask[pick]]
                a = rmse((d[best][pick] - d["observation"][pick]) * unit)
                b = rmse((d[other][pick] - d["observation"][pick]) * unit)
                diffs.append(a - b)
            comparisons[other] = {"best": best, "rmse_difference_mean": float(np.mean(diffs)),
                                  "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        out[target_name] = comparisons
    return out


GBM_BASE = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
            "num_threads": 16}


def run_study_v2(point_root, station_root, target="eric_d_soulis", k=30, r5_predictions=None, out=None, seeds=5):
    """Round 2: every choice by 2024 validation RMSE at the target; 2025 test reported for all, winner flagged.

    Candidates: target-only GBM with round-1 features, plus the time-lagged ensemble features, a small
    hyperparameter grid, a pooled variant; the selected configuration is re-fit with several seeds and
    averaged. A 50/50 average with r5 (fixed, not fitted) is reported where r5 exists.
    """
    import lightgbm as lgb

    values, inits, point_ids, variables = load_points(point_root)
    station_root = Path(station_root)
    station_meta = json.loads((station_root / "stations.json").read_text())["stations"]
    with np.load(station_root / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    label_ids = [s["id"] for s in station_meta]
    use = nearest_stations(station_meta, [i for i in point_ids if i in label_ids], target, k)
    X, names, y, base, split = build(values, inits, point_ids, variables, station_meta, label_times, labels,
                                     label_ids, use)
    n_init, n_lead, n_st, n_feat = X.shape
    flat = X.reshape(-1, n_feat)
    split_rows = np.repeat(split, n_lead * n_st)
    station_rows = np.tile(np.arange(n_st), n_init * n_lead)
    lead_rows = np.tile(np.repeat(np.arange(1, n_lead + 1), n_st), n_init)
    init_rows = np.repeat(inits, n_lead * n_st)
    is_target = station_rows == 0
    col = {n: i for i, n in enumerate(names)}
    round1 = [col[n] for n in names if n not in ENSEMBLE_FEATURES]
    full = list(range(n_feat))
    grid = [{}, {"num_leaves": 31}, {"num_leaves": 127}, {"min_data_in_leaf": 50}, {"min_data_in_leaf": 600},
            {"learning_rate": 0.03, "feature_fraction": 0.6}]
    report = {"target": target, "stations": use, "features": names, "selection": "2024 validation RMSE at target",
              "targets": {}}
    export = {"init": inits[split == "test"], "lead": np.arange(1, n_lead + 1), "val_init": inits[split == "val"]}
    test_target_all = (split_rows == "test") & is_target
    val_target_all = (split_rows == "val") & is_target
    for target_name in TARGETS:
        unit = 3.6 if target_name == "wind" else 1.0
        res = y[target_name].reshape(-1)
        hr = base[target_name].reshape(-1)
        finite = np.isfinite(res)
        train, val = (split_rows == "train") & finite, (split_rows == "val") & finite
        val_t, test_t = val & is_target, (split_rows == "test") & is_target & finite

        def fit(rows_train, features, extra=None, seed=0):
            params = {**GBM_BASE, **(extra or {}), "seed": seed}
            categorical = [features.index(col["station_index"])] if col["station_index"] in features else []
            dtrain = lgb.Dataset(flat[rows_train][:, features], res[rows_train], categorical_feature=categorical)
            dval = lgb.Dataset(flat[val_t][:, features], res[val_t], categorical_feature=categorical, reference=dtrain)
            model = lgb.train(params, dtrain, num_boost_round=4000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            return model.predict(flat[:, features], num_iteration=model.best_iteration), model.best_iteration

        candidates = {}
        candidates["target_round1"] = (train & is_target, round1, {})
        candidates["target_lagged"] = (train & is_target, full, {})
        candidates["pooled_lagged"] = (train, full, {})
        for i, extra in enumerate(grid[1:], 1):
            candidates[f"target_lagged_grid{i}"] = (train & is_target, full, extra)
        preds, entry = {"raw_hrrr": np.zeros_like(res)}, {"methods": {}, "configs": {}}
        for name, (rows_train, features, extra) in candidates.items():
            preds[name], iterations = fit(rows_train, features, extra)
            entry["configs"][name] = {"params": extra, "features": "round1" if features is round1 else "full",
                                      "pooled": rows_train is train, "best_iteration": iterations}
        scored = {m: rmse((p - res)[val_t] * unit) for m, p in preds.items()}
        chosen = min((m for m in candidates), key=scored.get)
        rows_train, features, extra = candidates[chosen]
        members = [fit(rows_train, features, extra, seed)[0] for seed in range(seeds)]
        preds["final_seed_ensemble"] = np.mean(members, axis=0)
        scored["final_seed_ensemble"] = rmse((preds["final_seed_ensemble"] - res)[val_t] * unit)
        for method, p in preds.items():
            error = (p - res)[test_t] * unit
            entry["methods"][method] = {"val_rmse": scored.get(method), "test_rmse": rmse(error),
                                        "test_mae": float(np.mean(np.abs(error))), "test_bias": float(np.mean(error)),
                                        "test_rmse_by_lead": [rmse(error[lead_rows[test_t] == h]) for h in range(1, n_lead + 1)]}
        entry["chosen_by_validation"] = chosen
        entry["count_test"] = int(test_t.sum())
        report["targets"][target_name] = entry
        # Export the final model for every 2025 target row (observation present or not).
        export[target_name] = (hr + preds["final_seed_ensemble"])[test_target_all].reshape(-1, n_lead)
        export[f"{target_name}_raw"] = hr[test_target_all].reshape(-1, n_lead)
        export[f"{target_name}_obs"] = (hr + res)[test_target_all].reshape(-1, n_lead)
        # 2024 predictions, for fitting blend weights without touching 2025. Early stopping also used
        # 2024, so these are slightly optimistic, as are the CNN's.
        export[f"val_{target_name}"] = (hr + preds["final_seed_ensemble"])[val_target_all].reshape(-1, n_lead)
        export[f"val_{target_name}_obs"] = (hr + res)[val_target_all].reshape(-1, n_lead)
    if r5_predictions:
        report["r5_blend"] = blend_with_r5(r5_predictions, export, target)
    if out:
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(Path(out).with_suffix(".npz"), **export)
    return report


def blend_with_r5(path, export, target):
    """Fixed 50/50 average with r5 on matched 2025 pairs (no weights fitted)."""
    with np.load(path) as r5:
        stations = list(r5["station_ids"])
        target = {"eccc_51459": "cyyz"}.get(target, target)
        if target not in stations:
            return None
        s = stations.index(target)
        index = {int(t): i for i, t in enumerate(r5["initializations"])}
        leads = list(r5["lead_times_hours"])
        forecast = r5["forecast"][:, :, s, :2].astype(np.float64)
    out = {}
    for target_name, column, unit in (("t2m", 1, 1.0), ("wind", 0, 3.6)):
        r5_pred = np.full_like(export[target_name], np.nan)
        for i, t in enumerate(export["init"]):
            if int(t) in index:
                r5_pred[i] = forecast[index[int(t)], [leads.index(h) for h in export["lead"]], column]
        obs, final = export[f"{target_name}_obs"], export[target_name]
        mask = np.isfinite(obs) & np.isfinite(r5_pred)
        out[target_name] = {"count": int(mask.sum()),
                            "final": rmse((final - obs)[mask] * unit), "r5": rmse((r5_pred - obs)[mask] * unit),
                            "blend_50_50": rmse(((final + r5_pred) / 2 - obs)[mask] * unit),
                            "raw": rmse((export[f"{target_name}_raw"] - obs)[mask] * unit)}
    return out


def site_predictions(exports, path):
    """Combine per-target exports into the benchmark.py predictions schema (lead 0 left missing)."""
    inits = sorted({int(t) for e in exports.values() for t in e["init"]})
    ids = list(exports)
    forecast = np.full((len(inits), 49, len(ids), 2), np.nan, dtype=np.float32)
    position = {t: i for i, t in enumerate(inits)}
    for s, (site_id, e) in enumerate(exports.items()):
        rows = [position[int(t)] for t in e["init"]]
        forecast[rows, 1:, s, 0] = e["wind"]
        forecast[rows, 1:, s, 1] = e["t2m"]
    np.savez_compressed(path, forecast=forecast, initializations=np.array(inits), station_ids=np.array(ids),
                        variables=np.array(["wind_speed", "t2m"]), units=np.array(["m/s", "degC"]),
                        lead_times_hours=np.arange(49))
    return path
