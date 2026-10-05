#!/usr/bin/env python3
"""Build CNN-LSTM training tensors from HRRR forecasts + Soulis station obs.

For each HRRR init cycle (00/06/12/18Z), produces one sample:

  X  (49, 30, 30, 3)  — 30×30 crop of HRRR ``t2m``, ``u10``, ``v10`` around
                        Eric D. Soulis, one frame per lead hour f00..f48
  y  (49, 2)          — station bias at each lead:
                        y_t = [HRRR_t2m_°C − Obs_t2m,
                               HRRR_wind_km/h − Obs_wind]
                        where HRRR_wind = 3.6 * sqrt(u10² + v10²), and both
                        HRRR point values are bilinearly interpolated to the
                        station lat/lon.

Soulis HOBO 15-min CSVs are matched to each valid time ``T0 + t`` with a
±15 min tolerance (exact instant preferred, else nearest). A run is dropped
only when T0 itself has no observation in that window. Missing leads / later
obs → NaN (partial samples allowed).

Default paths:
  HRRR:  /mnt/wato-drive/c52li/weatherloo-data/hrrr/{YYYY}/{YYYYMMDD}/hrrr_*.nc
  Obs:   <repo>/benchmarking-site/data/observations/eric_d_soulis/raw/
         uw_hobo_15min_{YYYY}.csv
  Out:   /mnt/wato-drive/gguirgui/weatherloo-data/hrrr_bias_correction/hrrr

Splits: train=2021–2023, val=2024, test=2025.
Single Zarr store with one group per split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HRRR_ROOT = Path("/mnt/wato-drive/c52li/weatherloo-data/hrrr")
DEFAULT_OBS_ROOT = (
    REPO_ROOT / "benchmarking-site" / "data" / "observations" / "eric_d_soulis" / "raw"
)
DEFAULT_OUT = Path("/mnt/wato-drive/gguirgui/weatherloo-data/hrrr_bias_correction/hrrr")

STATION = {"id": "eric_d_soulis", "lat": 43.4668, "lon": -80.5164}
# UW archive clock is local standard time (EST, no DST).
UW_ARCHIVE_TZ = timezone(timedelta(hours=-5))

CYCLES = (0, 6, 12, 18)
MAX_LEAD = 48
N_LEADS = MAX_LEAD + 1  # 49
CROP = 30
HALF = CROP // 2  # 15 → slice [c-15:c+15]
OBS_TOLERANCE = timedelta(minutes=15)

# HRRR input channels (native units in X: t2m K, u/v m/s).
INPUT_VARS = ("t2m", "u10", "v10")
N_CHANNELS = len(INPUT_VARS)

# Bias targets (station units: t2m °C, wind_speed km/h).
TARGET_VARS = ("t2m", "wind_speed")
N_TARGETS = len(TARGET_VARS)

SPLITS = {
    "train": (2021, 2022, 2023),
    "val": (2024,),
    "test": (2025,),
}

CHUNK_SAMPLE = 8
MS_TO_KMH = 3.6
K_TO_C = 273.15


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def nearest_grid_index(
    lat2d: np.ndarray, lon2d: np.ndarray, lat0: float, lon0: float
) -> tuple[int, int]:
    """Nearest (j, i) = (y, x) on a curvilinear lat/lon grid."""
    d2 = (lat2d - lat0) ** 2 + (lon2d - lon0) ** 2
    j, i = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(j), int(i)


def crop_window(
    j: int, i: int, ny: int, nx: int, half: int = HALF
) -> tuple[slice, slice] | None:
    """Return [j-half:j+half, i-half:i+half] or None if it would leave the grid."""
    j0, j1 = j - half, j + half
    i0, i1 = i - half, i + half
    if j0 < 0 or i0 < 0 or j1 > ny or i1 > nx:
        return None
    return slice(j0, j1), slice(i0, i1)


def bilinear_curvilinear(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    field: np.ndarray,
    lat0: float,
    lon0: float,
    window: int = 2,
) -> float:
    """Bilinear interp on a curvilinear lat/lon grid via local griddata."""
    j, i = nearest_grid_index(lat2d, lon2d, lat0, lon0)
    ny, nx = lat2d.shape
    j0, j1 = max(0, j - window), min(ny, j + window + 1)
    i0, i1 = max(0, i - window), min(nx, i + window + 1)
    la = lat2d[j0:j1, i0:i1].ravel()
    lo = lon2d[j0:j1, i0:i1].ravel()
    vals = np.asarray(field[j0:j1, i0:i1], dtype=np.float64).ravel()
    mask = np.isfinite(vals) & np.isfinite(la) & np.isfinite(lo)
    if int(mask.sum()) < 3:
        return float("nan")
    out = griddata(
        np.column_stack([la[mask], lo[mask]]),
        vals[mask],
        (lat0, lon0),
        method="linear",
    )
    out_f = float(np.asarray(out).reshape(()))
    return out_f if np.isfinite(out_f) else float("nan")


# ---------------------------------------------------------------------------
# Soulis observations
# ---------------------------------------------------------------------------

def _clean_metric(val: float, *, non_negative: bool = False) -> float | None:
    if not np.isfinite(val) or val <= -9000:
        return None
    if non_negative and val < 0:
        return None
    return float(val)


def load_soulis_hobo_csv(path: Path) -> list[tuple[datetime, dict[str, float | None]]]:
    """15-min HOBO CSV (2015+). Temp °C, wind km/h. Archive time is UTC−5."""
    samples: list[tuple[datetime, dict[str, float | None]]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 12:
                continue
            try:
                y, m, d, h, mi = (int(row[i].strip()) for i in range(5))
                temp = float(row[5].strip())
                wind = float(row[10].strip())
            except (ValueError, IndexError):
                continue
            dt = datetime(y, m, d, h, mi, tzinfo=UW_ARCHIVE_TZ).astimezone(timezone.utc)
            samples.append(
                (
                    dt,
                    {
                        "t2m": _clean_metric(round(temp, 2)),
                        "wind_speed": _clean_metric(round(wind, 2), non_negative=True),
                    },
                )
            )
    return samples


class SoulisObsIndex:
    """In-memory Soulis series with ±tolerance lookup at hourly valid times."""

    def __init__(self, obs_root: Path, years: tuple[int, ...], tolerance: timedelta = OBS_TOLERANCE):
        self.tolerance_s = tolerance.total_seconds()
        times: list[float] = []
        t2m: list[float] = []
        wind: list[float] = []
        for year in sorted(set(years)):
            path = obs_root / f"uw_hobo_15min_{year}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Soulis HOBO CSV: {path}")
            for dt, vals in load_soulis_hobo_csv(path):
                times.append(dt.timestamp())
                t2m.append(np.nan if vals["t2m"] is None else vals["t2m"])
                wind.append(np.nan if vals["wind_speed"] is None else vals["wind_speed"])
        if not times:
            raise RuntimeError(f"No Soulis samples loaded from {obs_root} for {years}")
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=np.float64)[order]
        self.t2m = np.asarray(t2m, dtype=np.float32)[order]
        self.wind = np.asarray(wind, dtype=np.float32)[order]

    def lookup(self, when: datetime) -> np.ndarray | None:
        """Return (t2m, wind_speed) or None if no sample within tolerance."""
        target = when.timestamp()
        idx = int(np.searchsorted(self.times, target))
        candidates: list[int] = []
        if 0 <= idx < self.times.size:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self.times[i] - target))
        if abs(self.times[best] - target) > self.tolerance_s:
            return None
        return np.asarray([self.t2m[best], self.wind[best]], dtype=np.float32)


# ---------------------------------------------------------------------------
# HRRR I/O
# ---------------------------------------------------------------------------

def open_hrrr(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_timedelta=False)


def lead_hours(ds: xr.Dataset) -> np.ndarray:
    vals = np.asarray(ds["lead"].values)
    if np.issubdtype(vals.dtype, np.timedelta64):
        return (vals / np.timedelta64(1, "h")).astype(np.int16)
    return vals.astype(np.int16)


def hrrr_path(hrrr_root: Path, init: datetime) -> Path:
    day = init.strftime("%Y%m%d")
    return hrrr_root / f"{init.year}" / day / f"hrrr_{day}_t{init.hour:02d}z.nc"


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def process_cycle(
    path: Path,
    init: datetime,
    obs: SoulisObsIndex,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (X, y) for one init, or None if crop invalid / T0 obs missing."""
    if obs.lookup(init) is None:
        return None

    with open_hrrr(path) as ds:
        missing = [v for v in INPUT_VARS if v not in ds.data_vars]
        if missing:
            return None
        lat2d = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon2d = np.asarray(ds["longitude"].values, dtype=np.float64)
        lon2d = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
        j, i = nearest_grid_index(lat2d, lon2d, lat0, lon0)
        win = crop_window(j, i, lat2d.shape[0], lat2d.shape[1])
        if win is None:
            return None
        jsl, isl = win

        leads = lead_hours(ds)
        lead_to_idx = {int(L): k for k, L in enumerate(leads)}

        x = np.full((N_LEADS, CROP, CROP, N_CHANNELS), np.nan, dtype=np.float32)
        y = np.full((N_LEADS, N_TARGETS), np.nan, dtype=np.float32)

        crops = {
            var: np.asarray(ds[var].values[:, jsl, isl], dtype=np.float32) for var in INPUT_VARS
        }
        lat_c = lat2d[jsl, isl]
        lon_c = lon2d[jsl, isl]

        for t in range(N_LEADS):
            if t not in lead_to_idx:
                continue
            k = lead_to_idx[t]
            t2m_frame = crops["t2m"][k]
            u_frame = crops["u10"][k]
            v_frame = crops["v10"][k]
            x[t, :, :, 0] = t2m_frame
            x[t, :, :, 1] = u_frame
            x[t, :, :, 2] = v_frame

            station_obs = obs.lookup(init + timedelta(hours=t))
            if station_obs is None:
                continue

            t2m_k = bilinear_curvilinear(lat_c, lon_c, t2m_frame, lat0, lon0)
            u = bilinear_curvilinear(lat_c, lon_c, u_frame, lat0, lon0)
            v = bilinear_curvilinear(lat_c, lon_c, v_frame, lat0, lon0)

            if np.isfinite(t2m_k) and np.isfinite(station_obs[0]):
                y[t, 0] = np.float32((t2m_k - K_TO_C) - float(station_obs[0]))
            if np.isfinite(u) and np.isfinite(v) and np.isfinite(station_obs[1]):
                wind_kmh = float(np.hypot(u, v) * MS_TO_KMH)
                y[t, 1] = np.float32(wind_kmh - float(station_obs[1]))

    return x, y


def iter_inits(years: tuple[int, ...]):
    for year in years:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, tzinfo=timezone.utc)
        cur = start
        while cur.date() <= end.date():
            for hour in CYCLES:
                yield datetime(cur.year, cur.month, cur.day, hour, tzinfo=timezone.utc)
            cur += timedelta(days=1)


def init_iso_to_unix(s: str) -> np.int64:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    return np.int64(int(dt.timestamp()))


def unix_to_init_iso(ts: int | np.integer) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def existing_inits(store_path: Path, split: str) -> set[str]:
    """ISO init strings already written under store/split (for --resume)."""
    if not (store_path / split).exists():
        return set()
    try:
        ds = xr.open_zarr(store_path, group=split, consolidated=False)
    except Exception:
        return set()
    try:
        if "init_time" not in ds.variables and "init_time" not in ds.coords:
            return set()
        vals = np.asarray(ds["init_time"].values)
    finally:
        ds.close()

    out: set[str] = set()
    for v in vals:
        if isinstance(v, (bytes, str)):
            s = v.decode() if isinstance(v, bytes) else v
            out.add(s if s.endswith("Z") else f"{s}Z")
        else:
            try:
                out.add(unix_to_init_iso(int(v)))
            except (TypeError, ValueError, OverflowError):
                out.add(np.datetime_as_string(np.datetime64(v), unit="s") + "Z")
    return out


def append_samples(
    store_path: Path,
    split: str,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    inits: list[str],
    *,
    create: bool,
) -> None:
    """Append a batch of samples to ``store_path/split``."""
    if not xs:
        return
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    init_arr = np.asarray([init_iso_to_unix(s) for s in inits], dtype=np.int64)

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "lead", "row", "col", "channel"), x),
            "y": (("sample", "lead", "target"), y),
        },
        coords={
            "lead": np.arange(N_LEADS, dtype=np.int16),
            "row": np.arange(CROP, dtype=np.int16),
            "col": np.arange(CROP, dtype=np.int16),
            "channel": np.arange(N_CHANNELS, dtype=np.int8),
            "target": np.arange(N_TARGETS, dtype=np.int8),
            "init_time": ("sample", init_arr),
        },
        attrs={
            "station_id": STATION["id"],
            "station_lat": STATION["lat"],
            "station_lon": STATION["lon"],
            "crop": CROP,
            "channel_names": ",".join(INPUT_VARS),
            "target_names": ",".join(TARGET_VARS),
            "init_time_units": "seconds since 1970-01-01T00:00:00Z",
            "bias_definition": "HRRR_bilinear(station) - Soulis_obs",
            "x_units": "t2m:K; u10:m/s; v10:m/s",
            "y_units": "t2m:degC; wind_speed:km/h",
            "obs_tolerance_minutes": int(OBS_TOLERANCE.total_seconds() // 60),
            "split": split,
        },
    )
    encoding = {
        "x": {"chunks": (CHUNK_SAMPLE, N_LEADS, CROP, CROP, N_CHANNELS)},
        "y": {"chunks": (CHUNK_SAMPLE, N_LEADS, N_TARGETS)},
    }
    store_path.mkdir(parents=True, exist_ok=True)
    if create:
        ds.to_zarr(store_path, group=split, mode="w", encoding=encoding, consolidated=False)
    else:
        ds.to_zarr(store_path, group=split, append_dim="sample", consolidated=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_split(
    split: str,
    years: tuple[int, ...],
    hrrr_root: Path,
    obs_root: Path,
    store_path: Path,
    *,
    resume: bool,
    limit: int | None,
) -> dict:
    # Load obs for split years plus a day of padding into the next year
    # (f48 from Dec 31 crosses into Jan 1–2 of year+1).
    obs_years = tuple(sorted({*years, max(years) + 1}))
    available = []
    for y in obs_years:
        if (obs_root / f"uw_hobo_15min_{y}.csv").is_file():
            available.append(y)
    obs = SoulisObsIndex(obs_root, tuple(available))

    lat0, lon0 = STATION["lat"], STATION["lon"]
    done = existing_inits(store_path, split) if resume else set()
    create = not (resume and bool(done))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    inits: list[str] = []
    stats = {
        "split": split,
        "years": list(years),
        "kept": 0,
        "skipped_missing_file": 0,
        "skipped_missing_t0_obs": 0,
        "skipped_existing": 0,
        "skipped_other": 0,
    }

    for init in iter_inits(years):
        if limit is not None and stats["kept"] >= limit:
            break
        init_iso = init.strftime("%Y-%m-%dT%H:00:00Z")
        if init_iso in done:
            stats["skipped_existing"] += 1
            continue
        path = hrrr_path(hrrr_root, init)
        if not path.is_file():
            stats["skipped_missing_file"] += 1
            continue
        if obs.lookup(init) is None:
            stats["skipped_missing_t0_obs"] += 1
            continue
        result = process_cycle(path, init, obs, lat0, lon0)
        if result is None:
            stats["skipped_other"] += 1
            continue
        x, y = result
        xs.append(x)
        ys.append(y)
        inits.append(init_iso)
        stats["kept"] += 1

        if len(xs) >= CHUNK_SAMPLE:
            append_samples(store_path, split, xs, ys, inits, create=create)
            create = False
            xs, ys, inits = [], [], []
            print(
                f"  [{split}] wrote batch (kept={stats['kept']}, last={init_iso})",
                flush=True,
            )

    if xs:
        append_samples(store_path, split, xs, ys, inits, create=create)

    return stats


def write_manifest(store_path: Path, all_stats: list[dict]) -> None:
    manifest = {
        "store": str(store_path),
        "station": STATION,
        "input_channels": list(INPUT_VARS),
        "targets": list(TARGET_VARS),
        "n_leads": N_LEADS,
        "crop": CROP,
        "obs_tolerance_minutes": int(OBS_TOLERANCE.total_seconds() // 60),
        "splits": {s["split"]: s for s in all_stats},
        "schema": {
            "x": "(sample, lead, row, col, channel)  # t2m[K], u10[m/s], v10[m/s]",
            "y": "(sample, lead, target)  # t2m bias[°C], wind_speed bias[km/h]",
            "init_time": "(sample,) unix seconds",
            "lead": "0..48 hours",
        },
    }
    (store_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hrrr-root", type=Path, default=DEFAULT_HRRR_ROOT)
    p.add_argument("--obs-root", type=Path, default=DEFAULT_OBS_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Zarr store directory")
    p.add_argument("--resume", action="store_true", help="Skip inits already in the store")
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Which splits to build",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on samples per split (smoke tests)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store_path = args.out
    store_path.mkdir(parents=True, exist_ok=True)
    print(f"HRRR root: {args.hrrr_root}")
    print(f"Obs root:  {args.obs_root}")
    print(f"Zarr out:  {store_path}")

    all_stats: list[dict] = []
    for split in args.splits:
        years = SPLITS[split]
        print(f"\n=== Building split={split} years={years} ===", flush=True)
        stats = build_split(
            split,
            years,
            args.hrrr_root,
            args.obs_root,
            store_path,
            resume=bool(args.resume),
            limit=args.limit,
        )
        all_stats.append(stats)
        print(f"=== Done {split}: {stats} ===", flush=True)

    write_manifest(store_path, all_stats)
    print(f"\nManifest -> {store_path / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
