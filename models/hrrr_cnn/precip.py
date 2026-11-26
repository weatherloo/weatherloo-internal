"""Precipitation forecasts at a site from HRRR: chance of precipitation (1 h and 6 h) and 6 h / 24 h amounts.

Truth. Soulis: the HOBO tipping-bucket gauge (15-min amounts, local standard
time UTC-5), summed to UTC hour-ending totals when all four samples exist;
the weighing gauge is not used (missing in 2024-2025). Neighbour stations:
the ECCC/ASOS 1-h precipitation already in the station labels (channel 3,
hour ending at the label time). HRRR ``precipitation_surface`` at lead h is
the mean rate over the hour ending at h, so hour h of a run verifies against
the label at T0 + h. A wet hour or block has at least 0.2 mm (one tip).

Products, all from information available at issue time (the run itself,
earlier runs, observations at or before T0):

* ``pop1``: P(>= 0.2 mm in hour h), h = 1..48
* ``pop6``: P(>= 0.2 mm in block b), b = hours 1-6, 7-12, ..., 43-48
* ``amt6``: expected amount in block b (mm); ``amt24`` = sum of four blocks

Model: a CNN on 6-h blocks of native patches with hourly and block heads, the
same model type as the temperature and wind forecasts. LightGBM (binary +
Tweedie) on point features is kept as an experiment (``with_gbm=True``).
"""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

import mos

UTC = timezone.utc
LOCAL_STANDARD = timezone(timedelta(hours=-5))
WET = 0.2  # mm
BLOCK = 6
N_BLOCKS = 48 // BLOCK


# ----------------------------------------------------------------------------- labels

def read_hobo_tipping(path):
    """(UTC epoch seconds of each 15-min sample, mm) from a HOBO year file with year,month,day,hour,minute columns."""
    stamps, values = [], []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [c.strip().lower() for c in next(reader)]
        if header[0] != "year":
            raise ValueError(f"{path}: unsupported HOBO layout {header[:2]}")
        column = next(i for i, c in enumerate(header) if "tipping" in c)
        for row in reader:
            try:
                y, m, d, h, mi = (int(row[i]) for i in range(5))
                value = float(row[column])
            except (ValueError, IndexError):
                continue
            stamp = datetime(y, m, d, h, mi, tzinfo=LOCAL_STANDARD).astimezone(UTC)
            stamps.append(int(stamp.timestamp()))
            values.append(value if 0 <= value < 100 else np.nan)  # -9999.99 = missing
    return np.array(stamps, dtype=np.int64), np.array(values, dtype=np.float64)


def hourly_totals(stamps, values, shift_minutes=0):
    """Hour-ending UTC totals; a sample at time s covers (s - 15 min, s] after shifting by shift_minutes.

    An hour is valid only with all four of its samples present and finite.
    """
    stamps = stamps + shift_minutes * 60
    hour_end = (stamps + 3599) // 3600 * 3600  # sample at :00 belongs to the hour ending then
    order = np.argsort(hour_end, kind="stable")
    hour_end, values = hour_end[order], values[order]
    hours, start, counts = np.unique(hour_end, return_index=True, return_counts=True)
    totals = np.add.reduceat(np.nan_to_num(values), start)
    finite = np.add.reduceat(np.isfinite(values).astype(int), start)
    totals = np.where((counts == 4) & (finite == 4), totals, np.nan)
    return hours, totals


def soulis_hourly(raw_dir, years=range(2021, 2026), shift_minutes=0):
    stamps, values = zip(*(read_hobo_tipping(Path(raw_dir) / f"uw_hobo_15min_{y}.csv") for y in years))
    return hourly_totals(np.concatenate(stamps), np.concatenate(values), shift_minutes)


def with_target_precip(labels, label_times, label_ids, target, times, mm):
    """Copy of the station labels with the target's precipitation channel taken from (times, mm)."""
    out = labels.copy()
    column = label_ids.index(target)
    out[:, column, 3] = np.nan
    index = np.searchsorted(times, label_times)
    hit = (index < len(times)) & (times[np.minimum(index, len(times) - 1)] == label_times)
    out[hit, column, 3] = mm[index[hit]]
    return out


def lag_check(label_times, series, reference, max_lag=3, before=None):
    """Correlation of wet/dry between series and reference shifted by lag hours (training period only)."""
    keep = np.ones(len(label_times), bool) if before is None else label_times < before
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        a = series[keep]
        b = np.roll(reference, -lag)[keep]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            out[lag] = float(np.corrcoef(a[ok] >= WET, b[ok] >= WET)[0, 1])
    return out


# ----------------------------------------------------------------------------- features

def neighbourhood(patch_precip, radius=None):
    """Per-lead neighbourhood stats from patch precipitation [init, lead, y, x] in mm/h."""
    if radius is not None:
        mid = patch_precip.shape[-1] // 2
        patch_precip = patch_precip[..., mid - radius:mid + radius + 1, mid - radius:mid + radius + 1]
    flat = patch_precip.reshape(patch_precip.shape[:2] + (-1,)).astype(np.float64)
    wet = np.where(np.isfinite(flat), flat >= WET, np.nan)  # missing is not dry
    with np.errstate(invalid="ignore", all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {"frac": np.nanmean(wet, axis=-1), "mean": np.nanmean(flat, axis=-1), "max": np.nanmax(flat, axis=-1)}


def build(values, inits, point_ids, variables, station_meta, label_times, labels, label_ids, use, patch_precip=None):
    """Hourly precipitation rows for stations `use` (target first).

    Returns X [init, 48, station, feature], names, y (mm, NaN if unobserved) [init, 48, station],
    hrrr (mm) [init, 48, station], split [init], block_nbhd (dict of [init, 8] for the target or None).
    patch_precip: [init, 49, y, x] mm/h at the target, aligned with inits (NaN rows where missing).
    """
    X, names, _, _, split = mos.build(values, inits, point_ids, variables, station_meta, label_times, labels,
                                      label_ids, use)
    var = {name: i for i, name in enumerate(variables)}
    cols = [point_ids.index(s) for s in use]
    pr = values[:, :, cols, var["precipitation_surface"]].astype(np.float64) * 3600  # [init, 49, st] mm in hour
    pr[:, 0] = np.nan  # no accumulation at lead 0
    n_init, n_lead, n_st = pr.shape
    lab = labels[:, [label_ids.index(s) for s in use], 3].astype(np.float64)

    def labels_at(stamps):
        i = np.searchsorted(label_times, stamps)
        hit = (i < len(label_times)) & (label_times[np.minimum(i, len(label_times) - 1)] == stamps)
        out = np.full(stamps.shape + (n_st,), np.nan)
        out[hit] = lab[i[hit]]
        return out

    valid = inits[:, None] + np.arange(n_lead)[None] * 3600
    obs = labels_at(valid)  # [init, 49, st], hour ending at valid time
    position = {int(t): k for k, t in enumerate(inits)}

    def lagged(hours_back):
        """Run T0 - hours_back at the same valid hour: lead h + hours_back."""
        out = np.full_like(pr, np.nan)
        idx = np.array([position.get(int(t) - hours_back * 3600, -1) for t in inits])
        have = idx >= 0
        out[have, :n_lead - hours_back] = pr[idx[have], hours_back:]
        return out

    def shifted(a, k):
        out = np.full_like(a, np.nan)
        if k > 0:
            out[:, :-k] = a[:, k:]
        elif k < 0:
            out[:, -k:] = a[:, :k]
        else:
            out[:] = a
        return out

    def window_sum(a, back, ahead):
        with np.errstate(invalid="ignore"):
            return np.nansum([shifted(a, k) for k in range(-back, ahead + 1)], axis=0)

    extra = {}
    for k in (-2, -1, 1, 2):
        extra[f"pr_lead_{'m' if k < 0 else 'p'}{abs(k)}"] = shifted(pr, k)
    extra["pr_3h_centred"] = window_sum(pr, 1, 1)
    extra["pr_6h_centred"] = window_sum(pr, 2, 3)
    members = [pr]
    for back in (6, 12, 18):
        member = lagged(back)
        extra[f"pr_prev{back}"] = member
        members.append(member)
    stack = np.stack(members)
    with np.errstate(invalid="ignore", all="ignore"):
        extra["pr_lagged_mean"] = np.nanmean(stack, axis=0)
        extra["pr_lagged_max"] = np.nanmax(stack, axis=0)
        extra["pr_lagged_wet_frac"] = np.nanmean(np.where(np.isfinite(stack), stack >= WET, np.nan), axis=0)
    # Issue-time observations (hours ending at or before T0).
    past = labels_at(inits[:, None] - np.arange(6)[None] * 3600)  # [init, 6, st]: T0, T0-1h, ...
    obs_1h, obs_3h, obs_6h = past[:, 0], past[:, :3].sum(axis=1), past.sum(axis=1)  # NaN unless all present
    previous = np.array([position.get(int(t) - 6 * 3600, -1) for t in inits])
    prev_0_6 = np.full((n_init, n_st), np.nan)
    have = previous >= 0
    prev_0_6[have] = pr[previous[have], 1:7].sum(axis=1)
    for name, array in (("obs_pr_1h_T0", obs_1h), ("obs_pr_3h_T0", obs_3h), ("obs_pr_6h_T0", obs_6h),
                        ("err_pr_6h_T0", obs_6h - prev_0_6)):
        extra[name] = np.broadcast_to(array[:, None], (n_init, n_lead, n_st))
    block_nbhd = None
    if patch_precip is not None:
        nb_all, nb_9 = neighbourhood(patch_precip), neighbourhood(patch_precip, radius=4)
        for prefix, stats in (("nb17", nb_all), ("nb9", nb_9)):
            for stat, array in stats.items():
                column = np.full((n_init, n_lead, n_st), np.nan)
                column[:, :, 0] = array
                extra[f"{prefix}_{stat}"] = column
        # Block neighbourhood: fraction of patch points whose 6-h total is wet; patch mean total.
        p = patch_precip[:, 1:].astype(np.float64)
        sums = p.reshape(n_init, N_BLOCKS, BLOCK, *p.shape[2:]).sum(axis=2)
        flat = sums.reshape(n_init, N_BLOCKS, -1)
        with np.errstate(invalid="ignore"):
            block_nbhd = {"frac": np.nanmean(np.where(np.isfinite(flat), flat >= WET, np.nan), axis=-1),
                          "mean": np.nanmean(flat, axis=-1)}
    extra_names = list(extra)
    extra_X = np.stack([extra[n][:, 1:] for n in extra_names], axis=-1).astype(np.float32)
    X = np.concatenate((X, extra_X), axis=-1)
    return X, names + extra_names, obs[:, 1:], pr[:, 1:], split, block_nbhd


def blocks(X, names, y, hrrr):
    """6-h block rows: hourly features averaged over the block plus precipitation sums and maxima."""
    n_init, _, n_st, n_feat = X.shape
    Xb = X.reshape(n_init, N_BLOCKS, BLOCK, n_st, n_feat)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Xb, axis=2)
        pr_index = [i for i, n in enumerate(names) if n == "precip_mm_h" or n.startswith(("pr_prev", "nb17_mean", "nb9_mean"))]
        sums = np.nansum(Xb[..., pr_index], axis=2)
        maxima = np.nanmax(Xb[..., pr_index], axis=2)
    names_b = [f"mean_{n}" for n in names] + [f"sum_{names[i]}" for i in pr_index] + [f"max_{names[i]}" for i in pr_index]
    Xb = np.concatenate((mean, sums, maxima), axis=-1).astype(np.float32)
    yb = y.reshape(n_init, N_BLOCKS, BLOCK, n_st).sum(axis=2)  # NaN unless all six hours observed
    hb = hrrr.reshape(n_init, N_BLOCKS, BLOCK, n_st).sum(axis=2)
    return Xb, names_b, yb, hb


# ----------------------------------------------------------------------------- scoring

def brier(p, event):
    ok = np.isfinite(p) & np.isfinite(event)
    return float(np.mean((p[ok] - event[ok]) ** 2)) if ok.any() else float("nan")


def reliability(p, event, bins=10):
    ok = np.isfinite(p) & np.isfinite(event)
    p, event = p[ok], event[ok]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "count": int((which == b).sum()),
             "forecast": float(p[which == b].mean()) if (which == b).any() else None,
             "observed": float(event[which == b].mean()) if (which == b).any() else None} for b in range(bins)]


def climatology(train_obs, train_month, train_hour, month, hour, statistic):
    """Training-period statistic of the target's observations by (month, valid hour), falling back to month."""
    table, coarse = {}, {}
    for o, m, h in zip(train_obs, train_month, train_hour):
        if np.isfinite(o):
            table.setdefault((m, h), []).append(o)
            coarse.setdefault(m, []).append(o)
    fine = {k: statistic(np.array(v)) for k, v in table.items() if len(v) >= 30}
    overall = statistic(np.concatenate([np.array(v) for v in coarse.values()])) if coarse else np.nan
    coarse = {k: statistic(np.array(v)) for k, v in coarse.items()}
    return np.array([fine.get((m, h), coarse.get(m, overall)) for m, h in zip(month, hour)])


def season_masks(valid_times):
    month = np.array([datetime.fromtimestamp(int(t), UTC).month for t in valid_times.ravel()]).reshape(valid_times.shape)
    warm = (month >= 4) & (month <= 10)
    return {"all": np.ones_like(warm), "apr_oct": warm, "nov_mar": ~warm}


def score_probability(forecasts, obs_mm, valid_times):
    """Brier, Brier skill vs climatology and reliability per season, on pairs where every forecast exists."""
    event = np.where(np.isfinite(obs_mm), (obs_mm >= WET).astype(float), np.nan)
    common = np.isfinite(event)
    for p in forecasts.values():
        common &= np.isfinite(p)
    out = {}
    for season, mask in season_masks(valid_times).items():
        m = common & mask
        entry = {"count": int(m.sum()), "event_rate": float(event[m].mean()) if m.any() else None}
        for name, p in forecasts.items():
            entry[name] = {"brier": brier(p[m], event[m])}
        if "climatology" in forecasts:
            ref = entry["climatology"]["brier"]
            for name in forecasts:
                entry[name]["bss_vs_climatology"] = 1 - entry[name]["brier"] / ref if ref else None
        for name, p in forecasts.items():
            entry[name]["reliability"] = reliability(p[m], event[m]) if season == "all" else None
        out[season] = entry
    return out


def score_amount(forecasts, obs_mm, valid_times):
    common = np.isfinite(obs_mm)
    for p in forecasts.values():
        common &= np.isfinite(p)
    out = {}
    for season, mask in season_masks(valid_times).items():
        m = common & mask
        entry = {"count": int(m.sum()), "obs_mean": float(obs_mm[m].mean()) if m.any() else None}
        for name, p in forecasts.items():
            error = p[m] - obs_mm[m]
            entry[name] = {"rmse": mos.rmse(error), "mae": float(np.abs(error).mean()) if m.any() else None,
                           "bias": float(error.mean()) if m.any() else None}
        out[season] = entry
    return out


# ----------------------------------------------------------------------------- data assembly

def load_patch_precip(patch_root, inits):
    """Target patch precipitation [init, 49, y, x] in mm/h aligned with inits (NaN where unavailable)."""
    from cnn_correction import load_patches

    patches, patch_inits, meta = load_patches(patch_root)
    if dict(zip(meta["variables"], [meta["units"][v] for v in meta["variables"]]))["precipitation_surface"] != "mm h-1":
        raise ValueError("patch precipitation must be stored in mm h-1 (context extraction)")
    channel = meta["variables"].index("precipitation_surface")
    out = np.full((len(inits), 49) + patches.shape[-2:], np.nan, dtype=np.float32)
    _, a, b = np.intersect1d(inits, patch_inits, return_indices=True)
    out[a] = patches[b][:, :, channel].astype(np.float32)
    return out


def assemble(point_root, station_root, target, k, soulis_raw=None, patch_root=None, min_coverage=0.3):
    values, inits, point_ids, variables = mos.load_points(point_root)
    station_root = Path(station_root)
    meta = json.loads((station_root / "stations.json").read_text())["stations"]
    with np.load(station_root / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    label_ids = [s["id"] for s in meta]
    if soulis_raw is not None:
        times, mm = soulis_hourly(soulis_raw)
        labels = with_target_precip(labels, label_times, label_ids, target, times, mm)
    # Neighbours with usable precipitation in the training period.
    train = label_times < mos.TRAIN_END
    coverage = np.isfinite(labels[train, :, 3]).mean(axis=0)
    candidates = [i for i in point_ids if i in label_ids and (i == target or coverage[label_ids.index(i)] >= min_coverage)]
    use = mos.nearest_stations(meta, candidates, target, k)
    patch = load_patch_precip(patch_root, inits) if patch_root else None
    X, names, y, hrrr, split, block_nbhd = build(values, inits, point_ids, variables, meta, label_times, labels,
                                                 label_ids, use, patch)
    return {"X": X, "names": names, "y": y, "hrrr": hrrr, "split": split, "inits": inits, "use": use,
            "block_nbhd": block_nbhd, "patch": patch,
            "coverage": {s: float(coverage[label_ids.index(s)]) for s in use}}


# ----------------------------------------------------------------------------- gradient boosting

GBM = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.8,
       "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 16}


def fit_gbm(X, y, split, kind, seeds=3, pooled_options=(False, True)):
    """X [init, row, station, feat], y [init, row, station] mm; kind 'pop' or 'amount'.

    Chooses target-only vs pooled training on 2024 validation at the target (station 0), then
    averages the chosen configuration over seeds. Returns predictions [init, row] at the target and a report.
    """
    import lightgbm as lgb

    n_init, n_row, n_st, n_feat = X.shape
    flat = X.reshape(-1, n_feat)
    label = y.reshape(-1)
    if kind == "pop":
        label = np.where(np.isfinite(label), (label >= WET).astype(float), np.nan)
    split_rows = np.repeat(split, n_row * n_st)
    station = np.tile(np.arange(n_st), n_init * n_row)
    finite = np.isfinite(label)
    train, val_t = (split_rows == "train") & finite, (split_rows == "val") & finite & (station == 0)
    target_rows = station == 0
    params = {**GBM, "objective": "binary" if kind == "pop" else "tweedie"}
    if kind == "amount":
        params["tweedie_variance_power"] = 1.5

    def fit(rows, seed):
        dtrain = lgb.Dataset(flat[rows], label[rows])
        dval = lgb.Dataset(flat[val_t], label[val_t], reference=dtrain)
        model = lgb.train({**params, "seed": seed}, dtrain, num_boost_round=3000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        return model.predict(flat[target_rows], num_iteration=model.best_iteration).reshape(n_init, n_row), model

    def val_score(pred):
        p = pred.reshape(-1)[np.flatnonzero(val_t[target_rows])]
        return brier(p, label[val_t]) if kind == "pop" else mos.rmse(p - label[val_t])

    report = {"kind": kind, "train_rows": {}, "val_score": {}}
    preds, models = {}, {}
    for pooled in pooled_options:
        name = "pooled" if pooled else "target"
        rows = train if pooled else train & target_rows
        if rows.sum() < 500:
            continue
        preds[name], models[name] = fit(rows, 0)
        report["train_rows"][name] = int(rows.sum())
        report["val_score"][name] = val_score(preds[name])
    chosen = min(report["val_score"], key=report["val_score"].get)
    rows = train if chosen == "pooled" else train & target_rows
    members = [preds[chosen]] + [fit(rows, seed)[0] for seed in range(1, seeds)]
    final = np.mean(members, axis=0)  # seed ensemble of one configuration, not a model blend
    report.update(chosen=chosen, val_score_final=val_score(final), best_iteration=models[chosen].best_iteration)
    importance = models[chosen].feature_importance("gain")
    report["top_features"] = [int(i) for i in np.argsort(importance)[::-1][:15]]
    return final, report


# ----------------------------------------------------------------------------- CNN

def train_cnn(data, patch_root, seeds=3, max_epochs=40, patience=5, batch_size=256, variables=None):
    """CNN on 6-h blocks at the target. Inputs: the native patch at the block's six leads, the site's
    0-48 h trajectory, and the block-mean scalar features. Heads: six hourly wet logits, a block wet
    logit, six hourly and one block log-mean amount (Tweedie deviance, power 1.5).
    Returns (val, test) dicts of pop1 [n, 48], pop6 [n, 8], amt6 [n, 8] and a report.
    """
    import torch
    from torch.nn import functional as F
    from cnn_correction import build_model, load_patches
    from points import PATCH_VARIABLES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = data["split"]
    patches, patch_inits, meta = load_patches(patch_root)
    common, a, b = np.intersect1d(data["inits"], patch_inits, return_indices=True)
    channels = [meta["variables"].index(v) for v in (variables or PATCH_VARIABLES)]
    rows = {s: a[split[a] == s] for s in ("train", "val", "test")}
    sample = b[split[a] == "train"][::7]
    stats = patches[sample][:, :, channels].astype(np.float32)
    mean = np.nanmean(stats, axis=(0, 1, 3, 4))[None, None, :, None, None]
    std = (np.nanstd(stats, axis=(0, 1, 3, 4)) + 1e-3)[None, None, :, None, None]
    P = torch.empty((len(b), 49, len(channels)) + patches.shape[-2:], dtype=torch.float16, device=device)
    for s in range(0, len(b), 256):
        block_ = (patches[b[s:s + 256]][:, :, channels].astype(np.float32) - mean) / std
        P[s:s + 256] = torch.from_numpy(np.nan_to_num(block_)).half()
    row_of = {int(i): k for k, i in enumerate(a)}  # data row -> patch tensor row
    mid = P.shape[-1] // 2
    traj = P[:, :, :, mid, mid].transpose(1, 2).contiguous()
    Xb, _, yb, _ = blocks(data["X"][:, :, :1], data["names"], data["y"][:, :, :1], data["hrrr"][:, :, :1])
    Xb, yb = Xb[:, :, 0], yb[:, :, 0]  # [init, 8, feat], [init, 8]
    yh = data["y"][:, :, 0].reshape(-1, N_BLOCKS, BLOCK)  # [init, 8, 6]
    s_train = Xb[rows["train"]].reshape(-1, Xb.shape[-1])
    s_mean, s_std = np.nanmean(s_train, axis=0), np.nanstd(s_train, axis=0) + 1e-3
    s = np.nan_to_num((Xb - s_mean) / s_std)
    scal = torch.tensor(np.concatenate((s, np.isfinite(Xb)), -1), dtype=torch.float32, device=device)
    target_h = torch.tensor(yh, dtype=torch.float32, device=device)
    target_b = torch.tensor(yb, dtype=torch.float32, device=device)
    offsets = torch.arange(-BLOCK + 1, 1, device=device)

    def batch(model, data_rows, block_index):
        prow = torch.tensor([row_of[int(r)] for r in data_rows.tolist()], device=device)
        end = (block_index + 1) * BLOCK  # lead at the block end
        idx = end[:, None] + offsets[None]
        patch = P[prow[:, None], idx].flatten(1, 2).float()
        return model(patch, scal[data_rows, block_index], traj=traj[prow].float(), lead=end)

    def tweedie(log_mu, y, power=1.5):
        mu = torch.exp(log_mu.clamp(-12, 6))
        return -y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power)

    def loss_fn(out, yh_, yb_):
        mh, mb = torch.isfinite(yh_), torch.isfinite(yb_)
        yh0, yb0 = torch.nan_to_num(yh_), torch.nan_to_num(yb_)
        bce_h = F.binary_cross_entropy_with_logits(out[:, :6], (yh0 >= WET).float(), reduction="none")
        bce_b = F.binary_cross_entropy_with_logits(out[:, 6], (yb0 >= WET).float(), reduction="none")
        tw_h, tw_b = tweedie(out[:, 7:13], yh0), tweedie(out[:, 13], yb0)
        def mean(x, m):
            return (x * m).sum() / m.sum().clamp_min(1)
        return mean(bce_h, mh) + mean(bce_b, mb) + 0.5 * (mean(tw_h, mh) + mean(tw_b, mb))

    def samples(data_rows):
        r = torch.tensor(data_rows, device=device)
        return r.repeat_interleave(N_BLOCKS), torch.arange(N_BLOCKS, device=device).repeat(len(r))

    def predict(model, data_rows):
        model.eval()
        r, k = samples(data_rows)
        with torch.no_grad():
            out = torch.cat([batch(model, r[i:i + 2048], k[i:i + 2048]) for i in range(0, len(r), 2048)])
        return out.cpu().numpy().reshape(len(data_rows), N_BLOCKS, 14)

    def evaluate(model, data_rows):
        r, k = samples(data_rows)
        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for i in range(0, len(r), 2048):
                out = batch(model, r[i:i + 2048], k[i:i + 2048])
                total += loss_fn(out, target_h[r[i:i + 2048], k[i:i + 2048]], target_b[r[i:i + 2048], k[i:i + 2048]]).item() * len(out)
                n += len(out)
        return total / max(n, 1)

    labelled = lambda rr: rr[np.isfinite(yb[rr]).any(axis=1) | np.isfinite(yh[rr]).any(axis=(1, 2))]
    train_rows, val_rows = labelled(rows["train"]), labelled(rows["val"])
    tr_r, tr_k = samples(train_rows)
    members, history = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = build_model(P.shape[2] * BLOCK, Xb.shape[-1], traj_channels=P.shape[2], n_out=14).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best, best_state, stale = np.inf, None, 0
        for epoch in range(max_epochs):
            model.train()
            order = torch.randperm(len(tr_r), device=device)
            for start in range(0, len(order), batch_size):
                pick = order[start:start + batch_size]
                r, k = tr_r[pick], tr_k[pick]
                loss = loss_fn(batch(model, r, k), target_h[r, k], target_b[r, k])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            score = evaluate(model, val_rows)
            history.append({"seed": seed, "epoch": epoch + 1, "val_loss": score})
            if score < best:
                best, stale = score, 0
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        members.append({part: predict(model, rows[part]) for part in ("val", "test")})

    def decode(raw):
        """Average member probabilities and amounts (seed ensemble of one model)."""
        prob = lambda x: 1 / (1 + np.exp(-x))
        pop1 = np.mean([prob(m[..., :6]) for m in raw], axis=0).reshape(len(raw[0]), -1)
        pop6 = np.mean([prob(m[..., 6]) for m in raw], axis=0)
        amt6 = np.mean([np.exp(np.clip(m[..., 13], -12, 6)) for m in raw], axis=0)
        return {"pop1": pop1, "pop6": pop6, "amt6": amt6}

    out = {}
    for part in ("val", "test"):
        full = {k_: np.full((len(data["inits"]),) + v.shape[1:], np.nan) for k_, v in
                decode([m[part] for m in members]).items()}
        decoded = decode([m[part] for m in members])
        for k_ in full:
            full[k_][rows[part]] = decoded[k_]
        out[part] = full
    report = {"history": history, "seed_val_loss": [min(h["val_loss"] for h in history if h["seed"] == s_)
                                                    for s_ in range(seeds)],
              "train_inits": int(len(train_rows)), "val_inits": int(len(val_rows)), "variables": [meta["variables"][c] for c in channels]}
    return out, report


# ----------------------------------------------------------------------------- study

def run(point_root, station_root, target, k=30, soulis_raw=None, patch_root=None, out=None, seeds=3, with_gbm=False):
    """Train the precipitation CNN and score 2025 against HRRR and climatology.

    with_gbm=True also fits the gradient-boosting experiment models and picks each product's model on 2024
    (the earlier study); the pipeline uses the CNN alone.
    """
    if patch_root is None:
        raise ValueError("the precipitation CNN needs the target's patches")
    data = assemble(point_root, station_root, target, k, soulis_raw, patch_root)
    X, names, y, hrrr, split, inits = data["X"], data["names"], data["y"], data["hrrr"], data["split"], data["inits"]
    Xb, names_b, yb, hb = blocks(X, names, y, hrrr)
    preds = {}
    report = {"target": target, "precip_coverage_train": {target: data["coverage"][target]}}
    if with_gbm:
        preds["gbm"], report["gbm"] = {}, {}
        report["stations"], report["precip_coverage_train"] = data["use"], data["coverage"]
        preds["gbm"]["pop1"], report["gbm"]["pop1"] = fit_gbm(X, y, split, "pop", seeds)
        preds["gbm"]["pop6"], report["gbm"]["pop6"] = fit_gbm(Xb, yb, split, "pop", seeds)
        preds["gbm"]["amt6"], report["gbm"]["amt6"] = fit_gbm(Xb, yb, split, "amount", seeds)
        for product in ("pop1", "pop6", "amt6"):
            report["gbm"][product]["top_features"] = [(names if product == "pop1" else names_b)[i]
                                                      for i in report["gbm"][product]["top_features"]]
    cnn_out, report["cnn"] = train_cnn(data, patch_root, seeds=seeds)
    preds["cnn"] = {p: np.where(np.isin(split, ["val", "test"])[:, None],
                                np.where(split[:, None] == "val", cnn_out["val"][p], cnn_out["test"][p]), np.nan)
                    for p in ("pop1", "pop6", "amt6")}
    # Target-only arrays.
    y_t, yb_t, h_t, hb_t = y[:, :, 0], yb[:, :, 0], hrrr[:, :, 0], hb[:, :, 0]
    valid_h = inits[:, None] + np.arange(1, 49)[None] * 3600
    valid_b = inits[:, None] + (np.arange(N_BLOCKS)[None] * BLOCK + BLOCK / 2) * 3600  # block middle
    month = lambda t: np.array([datetime.fromtimestamp(int(x), UTC).month for x in t.ravel()]).reshape(t.shape)
    hour = lambda t: ((t % 86400) // 3600).astype(int)
    tr = split == "train"
    clim_pop1 = climatology((y_t[tr] >= WET).astype(float).ravel() * np.where(np.isfinite(y_t[tr]), 1, np.nan).ravel(),
                            month(valid_h[tr]).ravel(), hour(valid_h[tr]).ravel(), month(valid_h).ravel(),
                            hour(valid_h).ravel(), np.mean).reshape(y_t.shape)
    bstart = lambda t: hour(t - 3 * 3600)
    clim_pop6 = climatology((yb_t[tr] >= WET).astype(float).ravel() * np.where(np.isfinite(yb_t[tr]), 1, np.nan).ravel(),
                            month(valid_b[tr]).ravel(), bstart(valid_b[tr]).ravel(), month(valid_b).ravel(),
                            bstart(valid_b).ravel(), np.mean).reshape(yb_t.shape)
    clim_amt6 = climatology(yb_t[tr].ravel(), month(valid_b[tr]).ravel(), bstart(valid_b[tr]).ravel(),
                            month(valid_b).ravel(), bstart(valid_b).ravel(), np.mean).reshape(yb_t.shape)
    nb_col = names.index("nb17_frac") if "nb17_frac" in names else None
    baselines = {
        "pop1": {"climatology": clim_pop1, "hrrr_point": np.where(np.isfinite(h_t), (h_t >= WET).astype(float), np.nan)},
        "pop6": {"climatology": clim_pop6, "hrrr_point": np.where(np.isfinite(hb_t), (hb_t >= WET).astype(float), np.nan)},
        "amt6": {"climatology": clim_amt6, "hrrr_point": hb_t},
    }
    if nb_col is not None:
        baselines["pop1"]["hrrr_neighbourhood_50km"] = X[:, :, 0, nb_col].astype(np.float64)
        baselines["pop6"]["hrrr_neighbourhood_50km"] = data["block_nbhd"]["frac"]
        baselines["amt6"]["hrrr_neighbourhood_50km"] = data["block_nbhd"]["mean"]
    obs_of = {"pop1": y_t, "pop6": yb_t, "amt6": yb_t}
    valid_of = {"pop1": valid_h, "pop6": valid_b, "amt6": valid_b}
    selection, scores, final = {}, {}, {}
    for product in ("pop1", "pop6", "amt6"):
        val, test = split == "val", split == "test"
        candidates = {m: preds[m][product] for m in preds}
        obs = obs_of[product]
        event = np.where(np.isfinite(obs), (obs >= WET).astype(float), np.nan)
        common = np.isfinite(obs[val])
        for p in candidates.values():
            common &= np.isfinite(p[val])
        if product.startswith("pop"):
            val_scores = {m: brier(p[val][common], event[val][common]) for m, p in candidates.items()}
        else:
            val_scores = {m: mos.rmse((p[val] - obs[val])[common]) for m, p in candidates.items()}
        chosen = min(val_scores, key=val_scores.get)
        selection[product] = {"chosen": chosen, "val_score": val_scores,
                              "metric": "Brier" if product.startswith("pop") else "RMSE mm"}
        forecasts = {**{f"model_{m}": p[test] for m, p in candidates.items()}, **{b_: v[test] for b_, v in baselines[product].items()}}
        scorer = score_probability if product.startswith("pop") else score_amount
        scores[product] = scorer(forecasts, obs[test], valid_of[product][test])
        final[product] = candidates[chosen]
    # 24-h amounts: sums of the chosen 6-h amount model over blocks 1-4 and 5-8.
    def day_sum(a):
        return a.reshape(a.shape[0], 2, 4).sum(axis=2)
    test = split == "test"
    obs24 = day_sum(yb_t)
    forecasts24 = {f"model_{selection['amt6']['chosen']}": day_sum(final["amt6"])[test],
                   "climatology": day_sum(clim_amt6)[test], "hrrr_point": day_sum(hb_t)[test]}
    if nb_col is not None:
        forecasts24["hrrr_neighbourhood_50km"] = day_sum(data["block_nbhd"]["mean"])[test]
    valid24 = inits[:, None] + np.array([12, 36])[None] * 3600
    scores["amt24"] = score_amount(forecasts24, obs24[test], valid24[test])
    report.update(selection=selection, scores=scores, wet_threshold_mm=WET,
                  truth="Soulis tipping bucket (UTC hour-ending, 4/4 samples); weighing gauge unused")
    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=float) + "\n")
        np.savez_compressed(Path(out).with_suffix(".npz"), init=inits[test], obs_1h=y_t[test], obs_6h=yb_t[test],
                            **{p: final[p][test] for p in final})
    return report
