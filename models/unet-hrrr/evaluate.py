#!/usr/bin/env python3
"""Station-level evaluation of the residual-correction U-Net (held-out 2021).

Loads the trained checkpoint, runs the U-Net over the **validation split**
(the chronological last 20% of the 2021 window — never seen in training),
forms the corrected forecast ``corrected = GFS - predicted_residual`` on the
full region grid, then bilinearly interpolates **both** the raw GFS grid and
the corrected grid to the two benchmark stations and scores each against the
**real ECCC / UW station observations** for 2021.

This is the honest, benchmark-comparable number: same stations, same bilinear
interpolation and wind convention as ``gfs_interpolated``, scored against actual
observations (not ERA5). The raw-GFS baseline is computed in the same run, so
``raw vs corrected`` is strictly apples-to-apples.

Note on comparability: the existing dashboard numbers (e.g. linear_regression
cyyz t2m RMSE 1.430, raw GFS 1.510) are for **2025**. Ours is the held-out
**2021** validation period, so compare *raw vs corrected here* for the U-Net's
effect; cross-year absolute values differ by season/sample.

Writes a summary aggregate JSON under ``models/unet/eval_results/`` (NOT into
``benchmarking-site/data/``) and prints the comparison table.

    .venv/bin/python models/unet/evaluate.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))

from dataset import (  # noqa: E402
    GFSResidualDataset, load_sample_grids, gfs_region_grid, _valid_time,
    CHANNELS, DEFAULT_START, DEFAULT_END, denormalize_residual, load_config,
)
from unet import ResidualUNet  # noqa: E402

CKPT_PATH = HERE / "checkpoints" / "best_model.pt"
OUT_DIR = HERE / "eval_results"
OBS_ROOT = HERE.parents[1] / "benchmarking-site" / "data" / "observations"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
VARIABLES = ["t2m", "wind_speed"]
OBS_YEAR = 2021


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_observations(station_id: str) -> dict[str, dict[str, float | None]]:
    path = OBS_ROOT / station_id / f"observations_6h_{OBS_YEAR}.json"
    data = json.loads(path.read_text())
    return {row["valid_time"]: {"t2m": row.get("t2m"),
                                "wind_speed": row.get("wind_speed")}
            for row in data["observations"]}


def _interp(lats: np.ndarray, lons: np.ndarray, field: np.ndarray,
            lat: float, lon: float) -> float:
    """Bilinear interp of a 2D (lat asc, lon 0..360) field to a point.

    Mirrors gfs_interpolated.interp_field (scipy RegularGridInterpolator).
    """
    lon_q = lon + 360.0 if lon < 0 else lon
    return float(RegularGridInterpolator((lats, lons), field, method="linear")
                 ((lat, lon_q)))


def station_values(lats, lons, grid3: np.ndarray) -> dict[str, dict[str, float]]:
    """Interp a (3,H,W) [t2m,u10,v10] grid to each station -> t2m degC, wind km/h.

    Wind: interpolate u/v separately, then sqrt(u^2+v^2)*3.6 (km/h) — per AGENTS.md.
    """
    out: dict[str, dict[str, float]] = {}
    for sid, c in STATIONS.items():
        t2m = _interp(lats, lons, grid3[0], c["lat"], c["lon"])
        u = _interp(lats, lons, grid3[1], c["lat"], c["lon"])
        v = _interp(lats, lons, grid3[2], c["lat"], c["lon"])
        out[sid] = {"t2m": t2m, "wind_speed": float(np.hypot(u, v) * 3.6)}
    return out


def _metrics(pairs: list[tuple[float, float]]) -> dict[str, float | None]:
    """RMSE/MAE/bias over (forecast, obs) pairs."""
    if not pairs:
        return {"rmse": None, "mae": None, "bias": None, "n": 0}
    err = np.array([f - o for f, o in pairs])
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)), "n": len(pairs)}


# --------------------------------------------------------------------------- #
# Per-season mean bias correction baseline                                      #
# --------------------------------------------------------------------------- #
# Meteorological season by month. Training (2021-03-23 .. ~Nov 5) has NO winter
# (DJF), so a val sample in an untrained season falls back to the nearest trained
# season by circular month distance (December -> SON/fall).
SEASON_OF = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_CENTER = {"MAM": 4, "JJA": 7, "SON": 10, "DJF": 1}


def _month_dist(m1: int, m2: int) -> int:
    d = abs(m1 - m2) % 12
    return min(d, 12 - d)


def nearest_trained_season(month: int, available) -> str:
    return min(available, key=lambda s: _month_dist(month, SEASON_CENTER[s]))


def seasonal_residual_fields(cfg, train_samples, era5_ds):
    """Mean (GFS-ERA5) residual field (3,H,W) per season present in training.

    This is the trivial baseline the U-Net must beat: a constant per-cell,
    per-channel seasonal correction with zero learned parameters.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for s in train_samples:
        gfs, era5 = load_sample_grids(cfg, s, era5_ds)
        res = gfs - era5
        seas = SEASON_OF[_valid_time(s).month]
        if seas not in sums:
            sums[seas] = np.zeros_like(res)
            counts[seas] = 0
        sums[seas] += res
        counts[seas] += 1
    return {k: sums[k] / counts[k] for k in sums}, counts


# Methods scored side by side (order = table row order).
METHODS = ["raw_gfs", "seasonal_bias", "unet", "era5_oracle"]
METHOD_LABEL = {
    "raw_gfs": "raw GFS (no correction)",
    "seasonal_bias": "seasonal mean bias corr.",
    "unet": "U-Net residual (learned)",
    "era5_oracle": "ERA5 oracle (ceiling)",
}


def main(args) -> None:
    device = pick_device()
    cfg = load_config()

    suffix = "" if args.split_mode == "chronological" else "_interleaved"
    ckpt_path = HERE / "checkpoints" / f"best_model{suffix}.pt"
    stats_path = HERE / "data" / f"stats{suffix}.json"

    # Rebuild the exact train/val splits + stats used in training.
    train_ds = GFSResidualDataset("train", DEFAULT_START, DEFAULT_END, cfg=cfg,
                                  stats_path=stats_path, split_mode=args.split_mode,
                                  train_days=args.train_days)
    val_ds = GFSResidualDataset("val", DEFAULT_START, DEFAULT_END, cfg=cfg,
                                stats=train_ds.stats, split_mode=args.split_mode,
                                train_days=args.train_days)
    stats = val_ds.stats
    samples = val_ds.samples
    print(f"device: {device}   split: {args.split_mode}"
          + (f" (train_days<={args.train_days})" if suffix else ""))
    print(f"train samples: {len(train_ds.samples)}   val samples: {len(samples)}  "
          f"({_valid_time(samples[0]).date()} .. {_valid_time(samples[-1]).date()})")

    # Load model.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResidualUNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")

    # Baseline: per-season mean residual fields from the TRAIN split only.
    season_fields, season_counts = seasonal_residual_fields(
        cfg, train_ds.samples, val_ds._era5_ds)
    available = set(season_fields)
    print("seasonal baseline fitted on train samples per season: "
          + ", ".join(f"{k}={season_counts[k]}" for k in sorted(season_counts))
          + f"  (available: {sorted(available)})")

    # Fixed region coordinates (identical for every GFS file in the box).
    coord_grid = gfs_region_grid(cfg, samples[0]["date"].replace("-", ""),
                                 samples[0]["cycle"], samples[0]["fxx"])
    lats = np.asarray(coord_grid.latitude.values, float)   # ascending
    lons = np.asarray(coord_grid.longitude.values, float)  # 0..360

    gfs_mean = np.asarray(stats["gfs"]["mean"], np.float32).reshape(-1, 1, 1)
    gfs_std = np.asarray(stats["gfs"]["std"], np.float32).reshape(-1, 1, 1)

    obs = {sid: load_observations(sid) for sid in STATIONS}

    # pairs[station][variable][method] = list of (forecast, obs).
    pairs = {sid: {v: {m: [] for m in METHODS} for v in VARIABLES}
             for sid in STATIONS}
    season_applied: dict[str, int] = {}  # "val_season->applied_season" tallies

    batch = 64
    for start in range(0, len(samples), batch):
        chunk = samples[start:start + batch]
        grids = [load_sample_grids(cfg, s, val_ds._era5_ds) for s in chunk]
        gfs_arrs = np.stack([g for g, _ in grids])          # (B,3,H,W) real units
        era5_arrs = np.stack([e for _, e in grids])         # ERA5 = perfect corrected
        x = (gfs_arrs - gfs_mean) / gfs_std
        with torch.no_grad():
            pred = model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy()
        pred_res = denormalize_residual(pred, stats)        # (B,3,H,W) real units
        unet_corr = gfs_arrs - pred_res                     # corrected = GFS - residual

        for i, s in enumerate(chunk):
            valid = _valid_time(s)
            valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Pick the season's mean-bias field (with fallback for untrained seasons).
            vseas = SEASON_OF[valid.month]
            aseas = vseas if vseas in available else nearest_trained_season(valid.month, available)
            season_applied[f"{vseas}->{aseas}"] = season_applied.get(f"{vseas}->{aseas}", 0) + 1
            seasonal_corr = gfs_arrs[i] - season_fields[aseas]

            method_grids = {
                "raw_gfs": gfs_arrs[i],
                "seasonal_bias": seasonal_corr,
                "unet": unet_corr[i],
                "era5_oracle": era5_arrs[i],
            }
            st = {m: station_values(lats, lons, g) for m, g in method_grids.items()}
            for sid in STATIONS:
                ov = obs[sid].get(valid_iso)
                if not ov:
                    continue
                for v in VARIABLES:
                    o = ov.get(v)
                    if o is None or not np.isfinite(o):
                        continue
                    for m in METHODS:
                        pairs[sid][v][m].append((st[m][sid][v], float(o)))

    # Build results + print table.
    results: dict = {}
    print("\n" + "=" * 82)
    print(f"{'station':<15} {'variable':<12} {'method':<26} "
          f"{'RMSE':>8} {'MAE':>8} {'vs_raw':>9}")
    print("-" * 82)
    for sid in STATIONS:
        results[sid] = {}
        for v in VARIABLES:
            unit = "degC" if v == "t2m" else "km/h"
            raw_rmse = _metrics(pairs[sid][v]["raw_gfs"])["rmse"]
            results[sid][v] = {}
            for m in METHODS:
                mm = _metrics(pairs[sid][v][m])
                results[sid][v][m] = mm
                if mm["rmse"] is None:
                    print(f"{sid:<15} {v:<12} {METHOD_LABEL[m]:<26} {'n/a':>8}")
                    continue
                if m == "raw_gfs":
                    vs = "  —  "
                else:
                    imp = raw_rmse - mm["rmse"]
                    vs = f"{100 * imp / raw_rmse:>+6.1f}%" if raw_rmse else "  —  "
                print(f"{sid:<15} {v + '(' + unit + ')':<12} {METHOD_LABEL[m]:<26} "
                      f"{mm['rmse']:>8.3f} {mm['mae']:>8.3f} {vs:>9}")
            print("-" * 82)
    print("vs_raw: + = lower RMSE than raw GFS. Held-out 2021 val period, scored vs real")
    print("ECCC/UW obs; same bilinear interp & wind convention as gfs_interpolated.")
    print("ERA5 oracle = corrected set exactly to ERA5 (the U-Net's training target) —")
    print("the best any GFS->ERA5 residual model could achieve at these stations.")
    print("season applied (val_season->trained_season): "
          + ", ".join(f"{k}:{n}" for k, n in sorted(season_applied.items())))

    # Write summary aggregate (NOT into benchmarking-site/data/).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "unet_postprocessing",
        "kind": "validation_period_summary",
        "note": "Held-out 2021 chronological val split; raw GFS vs seasonal-mean-bias "
                "baseline vs U-Net vs ERA5 oracle, scored against real 2021 station obs.",
        "methods": {m: METHOD_LABEL[m] for m in METHODS},
        "validation_period": {
            "start": _valid_time(samples[0]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": _valid_time(samples[-1]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_samples": len(samples),
        },
        "checkpoint": {"epoch": ckpt["epoch"], "val_loss_normalized_mse": ckpt["val_loss"]},
        "seasonal_baseline": {
            "train_samples_per_season": season_counts,
            "val_season_to_applied": season_applied,
            "note": "per-cell, per-channel mean(GFS-ERA5) by season; DJF absent in "
                    "training so December val falls back to nearest trained season.",
        },
        "interpolation": "bilinear (scipy RegularGridInterpolator) on region grid",
        "wind": "sqrt(u10^2 + v10^2) from corrected 10 m u/v components, m/s -> km/h",
        "observations_source": f"benchmarking-site/data/observations/*/observations_6h_{OBS_YEAR}.json",
        "stations": results,
        "generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary["split_mode"] = args.split_mode
    out_path = OUT_DIR / f"unet_postprocessing_val2021{suffix}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary -> {out_path}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Station-level U-Net evaluation.")
    p.add_argument("--split-mode", default="chronological",
                   choices=["chronological", "interleaved_month"],
                   help="which trained model/split to evaluate (default chronological)")
    p.add_argument("--train-days", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
