#!/usr/bin/env python3
"""Fetch GFS forecast (U-Net input) for the southern Ontario region.

GFS 0.25° is pulled from AWS Open Data (``s3://noaa-gfs-bdp-pds``, public,
no credentials) using the same byte-range GRIB approach as the HRRR/GFS
benchmarks: download the ``.idx`` sidecar, find the byte offsets for the
fields we need, then HTTP Range-request just those messages. Downloaded
GRIB2 messages are cached under ``.cache/gfs_grib/`` (gitignored).

Run standalone to verify GFS loads and to print a GFS-vs-ERA5 comparison:

    .venv/bin/python models/unet/data/fetch_gfs.py

The GFS-minus-ERA5 gap is exactly the "error" the residual U-Net learns to
correct. If GFS and ERA5 agree to < 0.5 degC on average, that is suspicious
(e.g. accidentally comparing an analysis to itself) and is flagged.

Exposes helpers reused by ``dataset.py`` later:
    gfs_region_grid, gfs_point
"""

from __future__ import annotations

import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Reuse config + ERA5 helpers from the sibling module.
try:
    from fetch_era5 import (  # when run as a script from this directory
        load_config, open_era5, era5_point, era5_region, _to_era5_lon, _valid_time,
    )
except ImportError:  # when imported as models.unet.data.fetch_gfs
    from .fetch_era5 import (
        load_config, open_era5, era5_point, era5_region, _to_era5_lon, _valid_time,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
# Overridable via UNET_DATA_DIR (set by run_pipeline.py) — see dataset.py.
CACHE_DIR = Path(os.environ.get("UNET_DATA_DIR", str(REPO_ROOT / ".cache"))) / "gfs_grib"


# ---------------------------------------------------------------------------
# Byte-range GRIB download (mirrors benchmarking-site/data/gfs_interpolated)
# ---------------------------------------------------------------------------
def download_bytes(url: str, retries: int, start: int | None = None,
                   end: int | None = None) -> bytes:
    headers = {"User-Agent": "weatherloo-internal/1.0"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
            if attempt == retries - 1:
                raise
        delay = min(60.0, (2 ** attempt) + random.uniform(0, 1))
        print(f"  download retry {attempt + 1}/{retries - 1} in {delay:.1f}s: {url[:80]}...")
        time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"download failed: {url}")


def parse_idx_ranges(idx_text: str, needles: dict[str, str]) -> dict[str, tuple[int, int]]:
    lines = idx_text.strip().splitlines()
    ranges: dict[str, tuple[int, int]] = {}
    for i, line in enumerate(lines):
        for key, needle in needles.items():
            if key in ranges or needle not in line:
                continue
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
            ranges[key] = (start, end)
    missing = set(needles) - set(ranges)
    if missing:
        raise RuntimeError(f"Missing GRIB fields in idx: {missing}")
    return ranges


def _grib_url(cfg: dict, date_yyyymmdd: str, cycle_hour: int, fxx: int) -> str:
    base = cfg["data"]["gfs"]["aws_base"]
    cycle = f"{cycle_hour:02d}"
    return (f"{base}/gfs.{date_yyyymmdd}/{cycle}/atmos/"
            f"gfs.t{cycle}z.pgrb2.0p25.f{fxx:03d}")


def fetch_gfs_grib_fields(cfg: dict, date_yyyymmdd: str, cycle_hour: int,
                          fxx: int) -> dict[str, Path]:
    """Download (cached) the t2m/u10/v10 GRIB messages; return field -> path."""
    gfs = cfg["data"]["gfs"]
    needles = gfs["grib_needles"]
    retries = gfs["download_retries"]
    grib_url = _grib_url(cfg, date_yyyymmdd, cycle_hour, fxx)
    cache_key = f"{date_yyyymmdd}_{cycle_hour:02d}z_f{fxx:03d}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = CACHE_DIR / f"{cache_key}.idx"
    if idx_path.exists():
        idx_text = idx_path.read_text()
    else:
        idx_text = download_bytes(grib_url + ".idx", retries).decode("utf-8", "replace")
        idx_path.write_text(idx_text)
    ranges = parse_idx_ranges(idx_text, needles)

    paths: dict[str, Path] = {}
    for field in needles:
        path = CACHE_DIR / f"{cache_key}_{field}.grib2"
        paths[field] = path
        if path.exists():
            continue
        start, end = ranges[field]
        path.write_bytes(download_bytes(grib_url, retries, start, end))
    return paths


# ---------------------------------------------------------------------------
# Grid + point extraction
# ---------------------------------------------------------------------------
def _open_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lats ascending, lons 0..360, values) for a single-message GRIB."""
    ds = cfgrib.open_dataset(path)
    var = list(ds.data_vars)[0]
    lats = np.asarray(ds.latitude.values, dtype=float)
    lons = np.asarray(ds.longitude.values, dtype=float)
    arr = np.asarray(ds[var].values, dtype=float)
    if lats[0] > lats[-1]:
        lats, arr = lats[::-1], arr[::-1, :]
    return lats, lons, arr


def gfs_region_grid(cfg: dict, date_yyyymmdd: str, cycle_hour: int, fxx: int) -> xr.Dataset:
    """GFS t2m(degC)/u10/v10 clipped to the southern Ontario region box."""
    region = cfg["region"]
    paths = fetch_gfs_grib_fields(cfg, date_yyyymmdd, cycle_hour, fxx)
    lo, hi = _to_era5_lon(region["lon_min"]), _to_era5_lon(region["lon_max"])

    out: dict[str, xr.DataArray] = {}
    coords_ref = None
    for name, field in (("t2m", "t2m_k"), ("u10", "u10"), ("v10", "v10")):
        lats, lons, arr = _open_field(paths[field])
        lat_mask = (lats >= region["lat_min"]) & (lats <= region["lat_max"])
        lon_mask = (lons >= lo) & (lons <= hi)
        sub = arr[np.ix_(lat_mask, lon_mask)]
        if name == "t2m":
            sub = sub - 273.15
        clat, clon = lats[lat_mask], lons[lon_mask]
        out[name] = xr.DataArray(sub, dims=("latitude", "longitude"),
                                 coords={"latitude": clat, "longitude": clon})
        coords_ref = (clat, clon)
    return xr.Dataset(out)


def gfs_point(cfg: dict, date_yyyymmdd: str, cycle_hour: int, fxx: int,
              lat: float, lon: float) -> dict[str, float]:
    """Bilinearly interpolate GFS t2m(degC), u10, v10(m/s) to a lat/lon point."""
    paths = fetch_gfs_grib_fields(cfg, date_yyyymmdd, cycle_hour, fxx)
    lon_q = _to_era5_lon(lon)
    result: dict[str, float] = {}
    for name, field in (("t2m", "t2m_k"), ("u10", "u10"), ("v10", "v10")):
        lats, lons, arr = _open_field(paths[field])
        val = float(RegularGridInterpolator((lats, lons), arr)((lat, lon_q)))
        result[name] = val - 273.15 if name == "t2m" else val
    return result


# ---------------------------------------------------------------------------
# Verification / comparison
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    cyyz = cfg["stations"]["cyyz"]
    sample0 = cfg["sample_forecasts"][0]

    print("=== GFS source ===")
    print(f"  bucket: {cfg['data']['gfs']['aws_base']}  (0.25° archive ~2021-03-23 onward)")

    grid = gfs_region_grid(cfg, sample0["date"].replace("-", ""),
                           sample0["cycle"], sample0["fxx"])
    print("\n=== Region slice (southern Ontario) ===")
    print(f"  init: {sample0['date']} {sample0['cycle']:02d}Z f{sample0['fxx']:03d}")
    print(f"  variables: {list(grid.data_vars)}")
    print(f"  shape (lat x lon): {grid['t2m'].shape}")
    print(f"  lat range: {float(grid.latitude.min()):.2f} .. {float(grid.latitude.max()):.2f}")
    print(f"  lon range: {float(grid.longitude.min()):.2f} .. {float(grid.longitude.max()):.2f} (0..360)")

    era5_ds = open_era5(cfg["data"]["era5"]["store"], cfg["data"]["era5"]["token"])

    print("\n=== GFS vs ERA5 t2m at CYYZ (this is the error the U-Net corrects) ===")
    print(f"  {'valid_time':<22} {'GFS(degC)':>10} {'ERA5(degC)':>11} {'GFS-ERA5':>10}")
    diffs: list[float] = []
    for sample in cfg["sample_forecasts"]:
        vt = _valid_time(sample)
        g = gfs_point(cfg, sample["date"].replace("-", ""), sample["cycle"],
                      sample["fxx"], cyyz["lat"], cyyz["lon"])
        e = era5_point(cfg, vt, cyyz["lat"], cyyz["lon"], era5_ds)
        d = g["t2m"] - e["t2m"]
        diffs.append(d)
        print(f"  {vt.isoformat():<22} {g['t2m']:>10.2f} {e['t2m']:>11.2f} {d:>+10.2f}")

    mean_abs = float(np.mean(np.abs(diffs)))
    print(f"\n  mean |GFS - ERA5| over {len(diffs)} samples: {mean_abs:.2f} degC")
    if mean_abs < 0.5:
        print("  WARNING: GFS and ERA5 agree very closely (< 0.5 degC). "
              "Check that GFS forecast (not analysis) and correct valid times are used.")
    else:
        print("  OK: a non-trivial GFS->ERA5 error signal exists for the U-Net to learn.")


if __name__ == "__main__":
    main()
