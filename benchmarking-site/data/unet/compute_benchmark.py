#!/usr/bin/env python3
"""Compute the UNet bias-correction benchmark JSON.

Evaluates ``models/unet/checkpoints/best_model.pt`` as a correction layer on top
of GFS: for each initialization and lead time we fetch the GFS 0.25° fields,
crop them to the Kitchener-Waterloo bounding box, run the UNet to get a
corrected ``(t2m, u10, v10)`` patch, then bilinearly interpolate that corrected
patch to each station and score it against the same station observations every
other method on the dashboard uses.

Because the input is GFS on the same grid, cycles, and lead times as
``data/gfs_interpolated/``, this method is the directly corrected counterpart of
that baseline — a lower RMSE here is the model earning its keep.

Wind speed: sqrt(u10^2 + v10^2), converted m/s -> km/h (* 3.6). The UNet
corrects u10 and v10 separately, matching how it was trained; speed is derived
after correction, never corrected directly.

ACC anomaly baseline: DOY + UTC-hour climatology from station observations with
a ±15-day calendar window — identical to ``gfs_interpolated`` so ACC stays
comparable across methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

METHOD_DIR = Path(__file__).resolve().parent
BENCHMARKING_SITE = METHOD_DIR.parents[1]
REPO_ROOT = METHOD_DIR.parents[2]
OUT_DIR = METHOD_DIR
OBS_ROOT = BENCHMARKING_SITE / "data" / "observations"
CACHE_DIR = REPO_ROOT / ".cache" / "gfs_grib"
MODEL_DIR = REPO_ROOT / "models" / "unet"
DEFAULT_CHECKPOINT = MODEL_DIR / "checkpoints" / "best_model.pt"

sys.path.insert(0, str(MODEL_DIR))
from model import correct_fields, load_checkpoint  # noqa: E402

# Reuse the shared bbox/HTTP helpers the raw-data downloaders already use.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from weather_download_common import (  # noqa: E402
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PAD_DEG,
    download_bytes,
    parse_idx_ranges,
)

AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
INIT_HOURS_UTC = (0, 6, 12, 18)
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHOD_ID = "unet"

STATIONS = {
    "cyyz": {"lat": 43.6777, "lon": -79.6248},
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164},
}
STATION_IDS = list(STATIONS)
VARIABLES = ["t2m", "wind_speed"]
METRICS = ["rmse", "mae", "bias", "acc"]

# UNet channel order, fixed by the checkpoint.
GRIB_NEEDLES = {
    "t2m": ":TMP:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
}
CHANNEL_ORDER = ("t2m", "u10", "v10")

DOWNLOAD_RETRIES = 6


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_observations(station_id: str) -> dict[str, dict[str, float | None]]:
    path = OBS_ROOT / station_id / "observations_6h_2025.json"
    data = json.loads(path.read_text())
    return {
        row["valid_time"]: {
            "t2m": row.get("t2m"),
            "wind_speed": row.get("wind_speed"),
        }
        for row in data["observations"]
    }


def build_climatology(
    obs_by_time: dict[str, dict[str, float | None]], window_days: int = 15
) -> dict[str, dict[str, float]]:
    """DOY + UTC-hour climatology using a calendar-day window within 2025 obs."""
    slots: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for valid_time, vals in obs_by_time.items():
        dt = parse_utc(valid_time)
        doy = dt.timetuple().tm_yday
        hour = dt.hour
        for var in ("t2m", "wind_speed"):
            v = vals.get(var)
            if v is None:
                continue
            slots.setdefault((var, hour), []).append((doy, float(v)))

    clim: dict[str, dict[str, float]] = {"t2m": {}, "wind_speed": {}}
    for var in ("t2m", "wind_speed"):
        for hour in (0, 6, 12, 18):
            entries = slots.get((var, hour), [])
            if not entries:
                continue
            for target_doy in range(1, 367):
                vals_in_window = [
                    v
                    for doy, v in entries
                    if abs(doy - target_doy) <= window_days
                    or abs(doy - target_doy + 365) <= window_days
                    or abs(doy - target_doy - 365) <= window_days
                ]
                if vals_in_window:
                    clim[var][f"{target_doy:03d}-{hour:02d}"] = float(
                        np.mean(vals_in_window)
                    )
    return clim


def climatology_lookup(
    clim: dict[str, dict[str, float]], var: str, valid_time: datetime
) -> float | None:
    key = f"{valid_time.timetuple().tm_yday:03d}-{valid_time.hour:02d}"
    return clim.get(var, {}).get(key)


def _bbox_slices(lats: np.ndarray, lons: np.ndarray) -> tuple[slice, slice]:
    """Index ranges covering the padded KW bbox on an ascending lat/lon grid."""
    lat_lo, lat_hi = LAT_MIN - PAD_DEG, LAT_MAX + PAD_DEG
    # GFS longitudes are 0..360; the bbox is in -180..180.
    lon_lo, lon_hi = LON_MIN - PAD_DEG + 360.0, LON_MAX + PAD_DEG + 360.0

    lat_idx = np.where((lats >= lat_lo) & (lats <= lat_hi))[0]
    lon_idx = np.where((lons >= lon_lo) & (lons <= lon_hi))[0]
    if lat_idx.size < 2 or lon_idx.size < 2:
        raise RuntimeError(
            f"bbox crop degenerate: {lat_idx.size} lats x {lon_idx.size} lons"
        )
    return slice(lat_idx[0], lat_idx[-1] + 1), slice(lon_idx[0], lon_idx[-1] + 1)


def read_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lats ascending, lons, values)`` for a single-message GRIB2."""
    ds = cfgrib.open_dataset(path)
    var = list(ds.data_vars)[0]
    lats = np.asarray(ds.latitude.values, dtype=float)
    lons = np.asarray(ds.longitude.values, dtype=float)
    arr = np.asarray(ds[var].values, dtype=float)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]
    return lats, lons, arr


def fetch_gfs_patch(
    date_yyyymmdd: str, cycle_hour: int, fxx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lats, lons, patch)`` where patch is ``(3, H, W)``.

    Channels are ``(t2m °C, u10 m/s, v10 m/s)`` — the units the UNet was
    normalized against.
    """
    cycle = f"{cycle_hour:02d}"
    cycle_tag = f"{cycle_hour:02d}z"
    grib_url = (
        f"{AWS_BASE}/gfs.{date_yyyymmdd}/{cycle}/atmos/"
        f"gfs.t{cycle_tag}.pgrb2.0p25.f{fxx:03d}"
    )
    cache_key = f"{date_yyyymmdd}_{cycle_tag}_f{fxx:03d}"
    idx_path = CACHE_DIR / f"{cache_key}.idx"
    if idx_path.exists():
        idx_text = idx_path.read_text()
    else:
        idx_text = download_bytes(
            grib_url + ".idx", retries=DOWNLOAD_RETRIES
        ).decode("utf-8", errors="replace")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(idx_text)

    ranges = parse_idx_ranges(idx_text, GRIB_NEEDLES)
    missing = set(GRIB_NEEDLES) - set(ranges)
    if missing:
        raise RuntimeError(f"Missing GRIB fields in idx for {cache_key}: {missing}")

    channels: list[np.ndarray] = []
    lats = lons = None
    rows = cols = None
    for field in CHANNEL_ORDER:
        path = CACHE_DIR / f"{cache_key}_{field}.grib2"
        if not path.exists():
            start, end = ranges[field]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                download_bytes(grib_url, start, end, retries=DOWNLOAD_RETRIES)
            )
        f_lats, f_lons, arr = read_field(path)
        if rows is None:
            rows, cols = _bbox_slices(f_lats, f_lons)
            lats, lons = f_lats[rows], f_lons[cols]
        channels.append(arr[rows, cols])

    patch = np.stack(channels, axis=0)
    patch[0] -= 273.15  # GFS TMP is Kelvin; the UNet expects °C
    return lats, lons, patch


def interp_point(lats: np.ndarray, lons: np.ndarray, arr: np.ndarray, lat: float, lon: float) -> float:
    lon_q = lon + 360.0 if lon < 0 else lon
    return float(RegularGridInterpolator((lats, lons), arr)((lat, lon_q)))


def station_values(
    lats: np.ndarray, lons: np.ndarray, patch: np.ndarray, lat: float, lon: float
) -> tuple[float, float]:
    """Interpolate a corrected patch to a station -> ``(t2m °C, wind km/h)``."""
    t2m_c = interp_point(lats, lons, patch[0], lat, lon)
    u10 = interp_point(lats, lons, patch[1], lat, lon)
    v10 = interp_point(lats, lons, patch[2], lat, lon)
    return t2m_c, float(np.hypot(u10, v10) * 3.6)


def point_metrics(
    forecast: float, obs: float, clim: float | None
) -> dict[str, float | None]:
    err = forecast - obs
    acc = None
    if clim is not None:
        f_anom = forecast - clim
        o_anom = obs - clim
        denom = abs(f_anom) * abs(o_anom)
        if denom > 0:
            acc = float((f_anom * o_anom) / denom)
    return {"rmse": abs(err), "mae": abs(err), "bias": err, "acc": acc}


def init_datetimes(year: int) -> list[datetime]:
    """All benchmark inits: every day at 00, 06, 12, 18 UTC (1460 for 2025)."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, tzinfo=timezone.utc)
    inits: list[datetime] = []
    cur = start
    while cur.date() <= end.date():
        for hour in INIT_HOURS_UTC:
            inits.append(
                datetime(cur.year, cur.month, cur.day, hour, tzinfo=timezone.utc)
            )
        cur += timedelta(days=1)
    return inits


def init_filename(init_dt: datetime) -> str:
    return f"{init_dt.strftime('%Y-%m-%dT%H')}Z.json"


def init_json_paths(out_dir: Path, year: int) -> list[Path]:
    return sorted(out_dir.glob(f"{year}-*T*Z.json"))


def build_init_json(
    init_dt: datetime,
    obs: dict[str, dict[str, dict[str, float | None]]],
    clim: dict[str, dict[str, dict[str, float]]],
    model,
    stats: dict,
) -> dict:
    date_str = init_dt.strftime("%Y%m%d")
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00:00Z")

    var_metrics = {
        sid: {var: {m: [] for m in METRICS} for var in VARIABLES} for sid in STATIONS
    }

    for lead in LEAD_TIMES:
        valid = init_dt + timedelta(hours=lead)
        valid_iso = valid.strftime("%Y-%m-%dT%H:%M:%SZ")

        # One fetch + one forward pass per lead, shared by both stations.
        lats, lons, raw_patch = fetch_gfs_patch(date_str, init_dt.hour, lead)
        corrected = correct_fields(model, stats, raw_patch)

        for station_id, coords in STATIONS.items():
            fcst_t2m, fcst_wind = station_values(
                lats, lons, corrected, coords["lat"], coords["lon"]
            )
            obs_vals = obs[station_id].get(valid_iso)
            for var, fcst in (("t2m", fcst_t2m), ("wind_speed", fcst_wind)):
                if not obs_vals or obs_vals.get(var) is None:
                    for m in METRICS:
                        var_metrics[station_id][var][m].append(None)
                    continue
                obs_val = float(obs_vals[var])
                clim_val = climatology_lookup(clim[station_id], var, valid)
                pm = point_metrics(fcst, obs_val, clim_val)
                for m in METRICS:
                    var_metrics[station_id][var][m].append(pm[m])

    locations = {
        station_id: {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "variables": {
                var: {"lead_times_hours": LEAD_TIMES, **var_metrics[station_id][var]}
                for var in VARIABLES
            },
        }
        for station_id, coords in STATIONS.items()
    }

    return {"method": METHOD_ID, "initialization": init_iso, "locations": locations}


def process_init(
    init_dt: datetime,
    obs: dict,
    clim: dict,
    model,
    stats: dict,
    out_dir: Path,
    resume: bool,
) -> str:
    fname = init_filename(init_dt)
    out_path = out_dir / fname
    if resume and out_path.exists():
        return fname
    payload = build_init_json(init_dt, obs, clim, model, stats)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return fname


def write_metadata(out_dir: Path, year: int, checkpoint: Path, meta: dict) -> None:
    payload = {
        "method_id": METHOD_ID,
        "model": "UNet residual bias correction on GFS",
        "input": "NOAA GFS 0.25° (t2m, u10, v10) cropped to the KW bounding box",
        "baseline": "gfs_interpolated (same input, no correction)",
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_val_loss": meta.get("val_loss"),
        "train_days": meta.get("train_days"),
        "split_mode": meta.get("split_mode"),
        "channels": list(CHANNEL_ORDER),
        "bbox": {
            "lat_min": LAT_MIN - PAD_DEG,
            "lat_max": LAT_MAX + PAD_DEG,
            "lon_min": LON_MIN - PAD_DEG,
            "lon_max": LON_MAX + PAD_DEG,
        },
        "cycles": ["00Z", "06Z", "12Z", "18Z"],
        "grid": "0p25",
        "year": year,
        "interpolation": "bilinear, applied to the corrected grid",
        "wind": "sqrt(u10^2 + v10^2) after correcting u/v separately, m/s to km/h",
        "acc_climatology": "DOY + UTC-hour mean from 2025 station obs, ±15-day window",
        "source": "https://registry.opendata.aws/noaa-gfs-bdp-pds/",
        "npz_file": f"{METHOD_ID}_{year}.npz",
        "npz_schema": "see benchmarking-site/AGENTS.md",
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_index(out_dir: Path, files: list[str]) -> None:
    (out_dir / "index.json").write_text(
        json.dumps({"files": sorted(files)}, indent=2) + "\n"
    )


def export_npz(out_dir: Path, year: int) -> Path | None:
    """Consolidate per-init JSON files into one compressed NPZ for analysis."""
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
        locations = payload["locations"]
        for station_idx, station_id in enumerate(STATION_IDS):
            variables = locations[station_id]["variables"]
            for var_idx, var_id in enumerate(VARIABLES):
                var_data = variables[var_id]
                for metric in METRICS:
                    for lead_idx, value in enumerate(var_data[metric]):
                        if value is not None:
                            arrays[metric][init_idx, station_idx, var_idx, lead_idx] = value

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
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--workers", type=int, default=3, help="Parallel init workers")
    parser.add_argument("--download-retries", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="Skip existing init files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Process only 2025-01-15 (all four cycles)"
    )
    parser.add_argument("--cycles", type=str, default=None, help="Comma-separated init hours UTC")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="UNet checkpoint .pt"
    )
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Rebuild NPZ from existing JSON without fetching GFS or running the model",
    )
    args = parser.parse_args()

    global DOWNLOAD_RETRIES
    DOWNLOAD_RETRIES = args.download_retries

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model, stats, meta = load_checkpoint(args.checkpoint)

    if args.export_npz_only:
        export_npz(out_dir, args.year)
        write_metadata(out_dir, args.year, args.checkpoint, meta)
        return

    # Each init worker runs its own tiny forward pass; keep BLAS from oversubscribing.
    torch.set_num_threads(1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    obs = {sid: load_observations(sid) for sid in STATIONS}
    clim = {sid: build_climatology(obs[sid]) for sid in STATIONS}

    cycle_hours = INIT_HOURS_UTC
    if args.cycles:
        cycle_hours = tuple(int(h.strip()) for h in args.cycles.split(","))

    inits = [d for d in init_datetimes(args.year) if d.hour in cycle_hours]
    if args.dry_run:
        inits = [datetime(2025, 1, 15, hour, tzinfo=timezone.utc) for hour in cycle_hours]
    if args.start_date:
        start = parse_utc(f"{args.start_date}T00:00:00Z")
        inits = [d for d in inits if d >= start]
    if args.end_date:
        end = parse_utc(f"{args.end_date}T18:00:00Z")
        inits = [d for d in inits if d <= end]

    print(
        f"UNet checkpoint: {args.checkpoint} "
        f"(epoch {meta.get('epoch')}, val_loss {meta.get('val_loss'):.4f})"
    )
    print(f"Processing {len(inits)} initializations -> {out_dir}")

    if args.workers <= 1:
        for init_dt in inits:
            print(f"  wrote {process_init(init_dt, obs, clim, model, stats, out_dir, args.resume)}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_init, init_dt, obs, clim, model, stats, out_dir, args.resume
                ): init_dt
                for init_dt in inits
            }
            for fut in as_completed(futures):
                init_dt = futures[fut]
                try:
                    print(f"  wrote {fut.result()}")
                except Exception as exc:
                    print(f"  FAILED {init_dt.isoformat()}: {exc}")
                    raise

    write_metadata(out_dir, args.year, args.checkpoint, meta)
    existing = sorted(p.name for p in init_json_paths(out_dir, args.year))
    write_index(out_dir, existing)
    export_npz(out_dir, args.year)
    print(f"Done. {len(existing)} init files, index.json updated.")


if __name__ == "__main__":
    main()
