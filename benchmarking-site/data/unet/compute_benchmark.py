#!/usr/bin/env python3
"""Compute the U-Net post-processing benchmark JSON for the dashboard.

This is a thin **adapter**: all of the modelling lives in ``models/unet/`` and is
imported, never reimplemented, so the dashboard can never silently disagree with
``models/unet/evaluate.py`` about geometry, normalization, or the residual sign.

Per initialization and lead time it:

1. loads the GFS region grid — from the ``.npz`` cache written by
   ``models/unet/run_pipeline.py fetch`` when present (set ``UNET_DATA_DIR`` or
   pass ``--data-dir``), else fetching it via ``fetch_gfs.gfs_region_grid``;
2. runs ``ResidualUNet`` and forms ``corrected = GFS - predicted_residual``;
3. bilinearly interpolates the corrected grid to each station and scores it
   against the same station observations every other method on the site uses.

**Sign.** The network predicts ``GFS - ERA5``, so the correction is *subtracted*
(see ``models/unet/model/unet.py``). Adding it doubles the error instead of
removing it.

**Coverage.** The checkpoint only ever saw ``init_hours_utc`` and
``forecast_hours`` from ``models/unet/config.yaml`` — by default 00/12Z at
f006–f024. Cells outside that window are written as ``null`` rather than
silently extrapolated; pass ``--all-cells`` to run them anyway.

ACC anomaly baseline: DOY + UTC-hour climatology from station observations with
a ±15-day calendar window — identical to ``gfs_interpolated`` so ACC stays
comparable across methods.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.error
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
MODEL_DIR = REPO_ROOT / "models" / "unet"
DEFAULT_CHECKPOINT = MODEL_DIR / "checkpoints" / "best_model.pt"

METHOD_ID = "unet"
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
ALL_INIT_HOURS = (0, 6, 12, 18)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]
STATION_IDS = ["cyyz", "eric_d_soulis"]

DOWNLOAD_RETRIES = 6


def _bootstrap(data_dir: Path | None) -> None:
    """Put ``models/unet`` on the path, honouring the pipeline's data-dir env var.

    Must run before importing the pipeline modules: they resolve their cache
    directories from ``UNET_DATA_DIR`` at import time, which is what lets this
    script reuse whatever ``run_pipeline.py fetch`` already downloaded.
    """
    if data_dir is not None:
        d = Path(data_dir).expanduser().resolve()
        # dataset.py appends "unet_training" to $UNET_DATA_DIR. Pointing --data-dir
        # straight at the sample directory is the obvious thing to do and would
        # otherwise resolve to <dir>/unet_training, miss every cached sample, and
        # silently re-download the whole year — so accept either form.
        if d.name == "unet_training":
            d = d.parent
        os.environ["UNET_DATA_DIR"] = str(d)
    for sub in ("data", "model", "."):
        p = str((MODEL_DIR / sub).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_observations(station_id: str) -> dict[str, dict[str, float | None]]:
    path = OBS_ROOT / station_id / "observations_6h_2025.json"
    data = json.loads(path.read_text())
    return {
        row["valid_time"]: {"t2m": row.get("t2m"), "wind_speed": row.get("wind_speed")}
        for row in data["observations"]
    }


def build_climatology(
    obs_by_time: dict[str, dict[str, float | None]], window_days: int = 15
) -> dict[str, dict[str, float]]:
    """DOY + UTC-hour climatology using a calendar-day window within 2025 obs."""
    slots: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for valid_time, vals in obs_by_time.items():
        dt = parse_utc(valid_time)
        doy, hour = dt.timetuple().tm_yday, dt.hour
        for var in ("t2m", "wind_speed"):
            v = vals.get(var)
            if v is not None:
                slots.setdefault((var, hour), []).append((doy, float(v)))

    clim: dict[str, dict[str, float]] = {"t2m": {}, "wind_speed": {}}
    for var in ("t2m", "wind_speed"):
        for hour in ALL_INIT_HOURS:
            entries = slots.get((var, hour), [])
            if not entries:
                continue
            for target_doy in range(1, 367):
                window = [
                    v
                    for doy, v in entries
                    if min(
                        abs(doy - target_doy),
                        abs(doy - target_doy + 365),
                        abs(doy - target_doy - 365),
                    )
                    <= window_days
                ]
                if window:
                    clim[var][f"{target_doy:03d}-{hour:02d}"] = float(np.mean(window))
    return clim


def climatology_lookup(
    clim: dict[str, dict[str, float]], var: str, valid_time: datetime
) -> float | None:
    key = f"{valid_time.timetuple().tm_yday:03d}-{valid_time.hour:02d}"
    return clim.get(var, {}).get(key)


def point_metrics(forecast: float, obs: float, clim: float | None) -> dict:
    err = forecast - obs
    acc = None
    if clim is not None:
        f_anom, o_anom = forecast - clim, obs - clim
        denom = abs(f_anom) * abs(o_anom)
        if denom > 0:
            acc = float((f_anom * o_anom) / denom)
    return {"rmse": abs(err), "mae": abs(err), "bias": err, "acc": acc}


def init_datetimes(year: int) -> list[datetime]:
    start, end = datetime(year, 1, 1, tzinfo=timezone.utc), datetime(year, 12, 31, tzinfo=timezone.utc)
    inits, cur = [], start
    while cur.date() <= end.date():
        for hour in ALL_INIT_HOURS:
            inits.append(datetime(cur.year, cur.month, cur.day, hour, tzinfo=timezone.utc))
        cur += timedelta(days=1)
    return inits


def init_filename(init_dt: datetime) -> str:
    return f"{init_dt.strftime('%Y-%m-%dT%H')}Z.json"


def init_json_paths(out_dir: Path, year: int) -> list[Path]:
    return sorted(out_dir.glob(f"{year}-*T*Z.json"))


# ---------------------------------------------------------------------------
# Per-process state
# ---------------------------------------------------------------------------
_W: dict = {}


def init_worker(checkpoint: Path, data_dir: Path | None, all_cells: bool) -> None:
    # eccodes is not thread-safe, so workers are processes; each must then keep
    # BLAS to one thread or workers x cores threads thrash.
    torch.set_num_threads(1)
    _bootstrap(data_dir)

    from dataset import _cache_path, denormalize_residual  # noqa: E402
    from fetch_era5 import load_config  # noqa: E402
    from fetch_gfs import gfs_region_grid  # noqa: E402
    from unet import ResidualUNet  # noqa: E402

    cfg = load_config()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ResidualUNet(
        in_channels=len(ckpt.get("channels", ("t2m", "u10", "v10"))),
        out_channels=len(ckpt.get("channels", ("t2m", "u10", "v10"))),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    _W.update(
        cfg=cfg,
        model=model,
        stats=ckpt["stats"],
        denorm=denormalize_residual,
        cache_path=_cache_path,
        region_grid=gfs_region_grid,
        obs={sid: load_observations(sid) for sid in STATION_IDS},
        all_cells=all_cells,
    )
    _W["clim"] = {sid: build_climatology(_W["obs"][sid]) for sid in STATION_IDS}
    _W["stations"] = cfg["stations"]
    gfs_cfg = cfg["data"]["gfs"]
    _W["trained_cycles"] = set(gfs_cfg["init_hours_utc"])
    _W["trained_leads"] = set(gfs_cfg["forecast_hours"])


def load_gfs_grid(init_dt: datetime, lead: int):
    """``(lats, lons, (3,H,W) GFS grid in degC/m·s⁻¹)``, from cache when available."""
    sample = {
        "date": init_dt.strftime("%Y-%m-%d"),
        "cycle": init_dt.hour,
        "fxx": lead,
    }
    path = _W["cache_path"](sample)
    if path.exists():
        # Written by run_pipeline.py fetch — already region-cropped and in degC.
        with np.load(path) as z:
            gfs = z["gfs"].astype(np.float32)
        region = _W["cfg"]["region"]
        res = region["resolution_deg"]
        lats = np.arange(region["lat_min"], region["lat_max"] + res / 2, res)
        lons = np.arange(region["lon_min"], region["lon_max"] + res / 2, res) % 360.0
        return lats, lons, gfs

    ds = _W["region_grid"](_W["cfg"], init_dt.strftime("%Y%m%d"), init_dt.hour, lead)
    lats = np.asarray(ds.latitude.values, dtype=float)
    lons = np.asarray(ds.longitude.values, dtype=float)
    gfs = np.stack([ds[c].values for c in ("t2m", "u10", "v10")]).astype(np.float32)
    return lats, lons, gfs


def correct_grid(gfs: np.ndarray) -> np.ndarray:
    """``corrected = GFS - predicted_residual`` (see models/unet/model/unet.py)."""
    stats = _W["stats"]
    mean = np.asarray(stats["gfs"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["gfs"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    x = torch.from_numpy((gfs - mean) / std).unsqueeze(0)
    with torch.no_grad():
        pred = _W["model"](x).squeeze(0).numpy()
    return gfs - np.asarray(_W["denorm"](pred, stats), dtype=np.float32)


def _interp(lats: np.ndarray, lons: np.ndarray, field: np.ndarray, lat: float, lon: float) -> float:
    from scipy.interpolate import RegularGridInterpolator

    lon_q = lon + 360.0 if lon < 0 else lon
    if lats[0] > lats[-1]:
        lats, field = lats[::-1], field[::-1, :]
    return float(RegularGridInterpolator((lats, lons), field)((lat, lon_q)))


def station_values(lats, lons, grid3: np.ndarray, lat: float, lon: float) -> tuple[float, float]:
    """Corrected grid -> ``(t2m °C, wind km/h)``; wind from corrected u/v."""
    t2m = _interp(lats, lons, grid3[0], lat, lon)
    u = _interp(lats, lons, grid3[1], lat, lon)
    v = _interp(lats, lons, grid3[2], lat, lon)
    return t2m, float(np.hypot(u, v) * 3.6)


def build_init_json(init_dt: datetime) -> dict:
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")
    var_metrics = {
        sid: {var: {m: [] for m in METRICS} for var in VARIABLES} for sid in STATION_IDS
    }

    def fill_nulls() -> None:
        for sid in STATION_IDS:
            for var in VARIABLES:
                for m in METRICS:
                    var_metrics[sid][var][m].append(None)

    for lead in LEAD_TIMES:
        # Outside the checkpoint's training window these cells would be pure
        # extrapolation, so record them as gaps unless explicitly asked for.
        if not _W["all_cells"] and lead not in _W["trained_leads"]:
            fill_nulls()
            continue

        valid = init_dt + timedelta(hours=lead)
        valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            lats, lons, gfs = load_gfs_grid(init_dt, lead)
        except (urllib.error.HTTPError, RuntimeError, FileNotFoundError, KeyError) as exc:
            print(f"    {init_iso} f{lead:03d}: no model output ({exc}); nulls")
            fill_nulls()
            continue

        corrected = correct_grid(gfs)
        for sid in STATION_IDS:
            coords = _W["stations"][sid]
            fcst_t2m, fcst_wind = station_values(
                lats, lons, corrected, coords["lat"], coords["lon"]
            )
            obs_vals = _W["obs"][sid].get(valid_iso)
            for var, fcst in (("t2m", fcst_t2m), ("wind_speed", fcst_wind)):
                if not obs_vals or obs_vals.get(var) is None:
                    for m in METRICS:
                        var_metrics[sid][var][m].append(None)
                    continue
                pm = point_metrics(
                    fcst,
                    float(obs_vals[var]),
                    climatology_lookup(_W["clim"][sid], var, valid),
                )
                for m in METRICS:
                    var_metrics[sid][var][m].append(pm[m])

    locations = {
        sid: {
            "lat": _W["stations"][sid]["lat"],
            "lon": _W["stations"][sid]["lon"],
            "variables": {
                var: {"lead_times_hours": LEAD_TIMES, **var_metrics[sid][var]}
                for var in VARIABLES
            },
        }
        for sid in STATION_IDS
    }
    return {"method": METHOD_ID, "initialization": init_iso, "locations": locations}


def process_init(init_dt: datetime, out_dir: Path, resume: bool) -> tuple[str, bool]:
    fname = init_filename(init_dt)
    out_path = out_dir / fname
    if resume and out_path.exists():
        return fname, True
    payload = build_init_json(init_dt)
    # Write via a temp file so an interrupted run never leaves a truncated JSON
    # that --resume would then happily skip over.
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out_path)
    return fname, False


def format_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def write_metadata(out_dir: Path, year: int, checkpoint: Path, meta: dict, cfg: dict, all_cells: bool) -> None:
    gfs_cfg = cfg["data"]["gfs"]
    payload = {
        "method_id": METHOD_ID,
        "model": "U-Net residual post-processing of GFS (models/unet)",
        "objective": "network predicts (GFS - ERA5); corrected = GFS - predicted_residual",
        "baseline": "gfs_interpolated (same GFS input, no correction)",
        "checkpoint": str(Path(checkpoint).resolve().relative_to(REPO_ROOT)),
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_val_loss": meta.get("val_loss"),
        "split_mode": meta.get("split_mode"),
        "channels": list(meta.get("channels", ("t2m", "u10", "v10"))),
        "region": cfg["region"],
        "trained_cycles_utc": gfs_cfg["init_hours_utc"],
        "trained_forecast_hours": gfs_cfg["forecast_hours"],
        "coverage": (
            "all cycles/leads (extrapolated beyond training window)"
            if all_cells
            else "restricted to the checkpoint's trained cycles/leads; other cells null"
        ),
        "cycles": [f"{h:02d}Z" for h in gfs_cfg["init_hours_utc"]],
        "grid": "0p25",
        "year": year,
        "interpolation": "bilinear (scipy RegularGridInterpolator) on the corrected region grid",
        "wind": "sqrt(u10^2 + v10^2) after correcting u/v separately, m/s to km/h",
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, ±15-day window",
        "npz_file": f"{METHOD_ID}_{year}.npz",
        "npz_schema": "see benchmarking-site/AGENTS.md",
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_index(out_dir: Path, files: list[str]) -> None:
    (out_dir / "index.json").write_text(json.dumps({"files": sorted(files)}, indent=2) + "\n")


def export_npz(out_dir: Path, year: int) -> Path | None:
    json_paths = init_json_paths(out_dir, year)
    if not json_paths:
        print(f"No {year}-*T*Z.json files in {out_dir}; skipping NPZ export.")
        return None

    shape = (len(json_paths), len(STATION_IDS), len(VARIABLES), len(LEAD_TIMES))
    arrays = {metric: np.full(shape, np.nan, dtype=np.float64) for metric in METRICS}
    initializations: list[str] = []

    for init_idx, path in enumerate(json_paths):
        payload = json.loads(path.read_text())
        initializations.append(payload["initialization"])
        for s_idx, sid in enumerate(STATION_IDS):
            variables = payload["locations"][sid]["variables"]
            for v_idx, var_id in enumerate(VARIABLES):
                for metric in METRICS:
                    for l_idx, value in enumerate(variables[var_id][metric]):
                        if value is not None:
                            arrays[metric][init_idx, s_idx, v_idx, l_idx] = value

    npz_path = out_dir / f"{METHOD_ID}_{year}.npz"
    np.savez_compressed(
        npz_path,
        method_id=np.array(METHOD_ID),
        station_ids=np.array(STATION_IDS),
        variables=np.array(VARIABLES),
        metrics=np.array(METRICS),
        lead_times_hours=np.array(LEAD_TIMES, dtype=np.int16),
        initializations=np.array(initializations),
        **arrays,
    )
    print(f"Wrote {npz_path} ({shape[0]} inits, shape {shape} per metric).")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--start-date", type=str, default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Skip existing init files")
    parser.add_argument("--dry-run", action="store_true", help="Only 2025-01-15")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="run_pipeline.py fetch cache to reuse (else $UNET_DATA_DIR); "
        "cached samples need no download",
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Also score cycles/leads the checkpoint was never trained on "
        "(pure extrapolation; off by default)",
    )
    parser.add_argument("--shard", type=str, default=None, metavar="I/N")
    parser.add_argument("--export-npz-only", action="store_true")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip the index/NPZ write at the end. Use for concurrent shards, "
        "which would otherwise race and each publish a partial index; run one "
        "--export-npz-only pass afterwards.",
    )
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _bootstrap(args.data_dir)
    from fetch_era5 import load_config  # noqa: E402

    cfg = load_config()
    meta = {
        k: v
        for k, v in torch.load(args.checkpoint, map_location="cpu", weights_only=True).items()
        if k not in ("model_state", "stats")
    }

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year, args.checkpoint, meta, cfg, args.all_cells)
        return

    trained_cycles = set(cfg["data"]["gfs"]["init_hours_utc"])
    inits = init_datetimes(args.year)
    if not args.all_cells:
        inits = [d for d in inits if d.hour in trained_cycles]
    if args.dry_run:
        inits = [d for d in inits if d.date() == datetime(2025, 1, 15).date()]
    if args.start_date:
        inits = [d for d in inits if d >= parse_utc(f"{args.start_date}T00:00:00Z")]
    if args.end_date:
        inits = [d for d in inits if d <= parse_utc(f"{args.end_date}T18:00:00Z")]
    if args.shard:
        idx, total = (int(x) for x in args.shard.split("/"))
        if not 1 <= idx <= total:
            parser.error(f"--shard {args.shard}: expected 1/N..N/N")
        inits = inits[idx - 1 :: total]
        print(f"Shard {idx}/{total}: {len(inits)} initializations")

    print(
        f"Checkpoint {args.checkpoint} (epoch {meta.get('epoch')}, "
        f"val_loss {meta.get('val_loss'):.4f})"
    )
    print(f"Region {cfg['region']['lat_min']}..{cfg['region']['lat_max']}N, "
          f"{cfg['region']['lon_min']}..{cfg['region']['lon_max']}E")
    print(f"Trained cycles {sorted(trained_cycles)}Z, leads "
          f"{cfg['data']['gfs']['forecast_hours']}h"
          f"{' (IGNORED: --all-cells)' if args.all_cells else ''}")

    # A mis-pointed --data-dir is invisible otherwise: every lookup just misses
    # and the run quietly re-downloads the year from AWS at ~100x the runtime.
    from dataset import CACHE_DIR  # noqa: E402

    n_cached = sum(1 for _ in CACHE_DIR.glob(f"{args.year}-*.npz")) if CACHE_DIR.is_dir() else 0
    print(f"Sample cache {CACHE_DIR}: {n_cached} cached {args.year} sample(s)")
    if n_cached == 0:
        print("  WARNING: no cached samples for this year — every lead will be "
              "downloaded from AWS. Check --data-dir / $UNET_DATA_DIR.")

    print(f"Processing {len(inits)} initializations -> {out_dir}")

    started, done, skipped = time.monotonic(), 0, 0
    failures: list[dict[str, str]] = []

    def report(fname: str, was_skipped: bool) -> None:
        nonlocal done, skipped
        done += 1
        if was_skipped:
            skipped += 1
            return
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(inits) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(inits)}] {fname}  ({rate * 3600:.0f}/h, ETA {format_eta(eta)})",
              flush=True)

    if args.workers <= 1:
        init_worker(args.checkpoint, args.data_dir, args.all_cells)
        for init_dt in inits:
            try:
                report(*process_init(init_dt, out_dir, args.resume))
            except Exception as exc:  # noqa: BLE001 — one bad init must not end the run
                print(f"  FAILED {init_dt.isoformat()}: {exc}", flush=True)
                failures.append({"initialization": init_dt.isoformat(), "error": str(exc)})
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(args.checkpoint, args.data_dir, args.all_cells),
        ) as pool:
            futures = {pool.submit(process_init, d, out_dir, args.resume): d for d in inits}
            for fut in as_completed(futures):
                init_dt = futures[fut]
                try:
                    report(*fut.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED {init_dt.isoformat()}: {exc}", flush=True)
                    failures.append({"initialization": init_dt.isoformat(), "error": str(exc)})

    write_metadata(out_dir, args.year, args.checkpoint, meta, cfg, args.all_cells)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    if args.no_export:
        print(f"--no-export: wrote init JSON only ({len(existing)} present). "
              f"Finalize with --export-npz-only once all shards are done.")
    else:
        write_index(out_dir, existing)
        export_npz(out_dir, args.year)

    if skipped:
        print(f"Skipped {skipped} existing init(s) (--resume).")
    print(f"Done in {format_eta(time.monotonic() - started)}. {len(existing)} init files.")
    print("Next: bash benchmarking-site/scripts/build_static_aggregates.sh --methods unet")

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        print(f"{len(failures)} init(s) failed -> failures.json. Re-run with --resume to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
