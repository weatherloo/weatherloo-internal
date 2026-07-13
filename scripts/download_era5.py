#!/usr/bin/env python3
"""Download ERA5 ground-truth (bbox-subset) for bias-correction training.

Reads the public **ARCO-ERA5** analysis-ready Zarr on Google Cloud (hourly,
0.25 deg, no account required), subsets the four target variables to the
Kitchener-Waterloo bounding box, and writes one compact NetCDF per month:

    {data_root}/era5/{YYYY}/era5_{YYYYMM}.nc
        dims:  (time, latitude, longitude)
        vars:  t2m (K), u10 (m/s), v10 (m/s), tp (m accumulated)

Month granularity bounds memory and gives natural resume. ARCO-ERA5 lags
real-time by ~2-3 months; months with no data yet are skipped.

Precedent for zarr-via-xarray access: ``tpm/take_home/README.md`` and
``benchmarking-site/data/ecmwf_aifs/compute_benchmark.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from weather_download_common import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PAD_DEG,
    resolve_data_root,
)

DEFAULT_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_START = "2018-01"

# ARCO-ERA5 variable name -> compact output name (matches download_hrrr.py).
VAR_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
}


def months(start: str, end: str):
    """Yield (year, month) inclusive from 'YYYY-MM' start to 'YYYY-MM' end."""
    sy, sm = (int(x) for x in start.split("-")[:2])
    ey, em = (int(x) for x in end.split("-")[:2])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def open_store(store: str) -> xr.Dataset:
    ds = xr.open_zarr(store, storage_options={"token": "anon"}, chunks={})
    missing = [v for v in VAR_MAP if v not in ds.variables]
    if missing:
        raise RuntimeError(
            f"Variables not found in {store}: {missing}. "
            f"Check --store / ARCO variable names."
        )
    return ds


def subset_region(ds: xr.Dataset) -> xr.Dataset:
    # ARCO latitude is descending (90 -> -90); longitude is 0..360.
    lon_min = (LON_MIN - PAD_DEG) % 360
    lon_max = (LON_MAX + PAD_DEG) % 360
    return ds.sel(
        latitude=slice(LAT_MAX + PAD_DEG, LAT_MIN - PAD_DEG),
        longitude=slice(lon_min, lon_max),
    )


def process_month(ds: xr.Dataset, y: int, m: int, data_root: Path, resume: bool) -> str:
    out_path = data_root / "era5" / f"{y:04d}" / f"era5_{y:04d}{m:02d}.nc"
    if resume and out_path.exists():
        return f"skip {out_path.name}"

    month = ds[list(VAR_MAP)].sel(time=slice(f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}"))
    if month.sizes.get("time", 0) == 0:
        return f"MISSING {y:04d}-{m:02d} (no time steps in store)"

    month = subset_region(month)

    # Currency probe: the store's time axis is padded with fill values for
    # not-yet-reanalysed months. Load just the first hour of one variable (one
    # global chunk, ~4 MB) and skip the whole month if it's all-NaN, so we don't
    # pay the full ~12 GB month read for data that isn't published yet.
    probe = month[next(iter(VAR_MAP))].isel(time=0).load()
    if bool(np.isnan(probe.values).all()):
        return f"MISSING {y:04d}-{m:02d} (not published yet)"

    month = month.rename(VAR_MAP).load()
    if all(bool(np.isnan(month[v].values).all()) for v in month.data_vars):
        return f"MISSING {y:04d}-{m:02d} (all-NaN after load)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in month.data_vars}
    tmp = out_path.parent / (out_path.name + ".tmp")
    month.to_netcdf(tmp, encoding=encoding)
    tmp.replace(out_path)
    return f"wrote {out_path.name} ({month.sizes['time']} hrs)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Output root (else $WEATHERLOO_DATA_ROOT, else repo data/)")
    parser.add_argument("--start-date", default=DEFAULT_START, help="YYYY-MM inclusive (default 2018-01)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM inclusive (default: current month UTC)")
    parser.add_argument("--store", default=DEFAULT_STORE, help="Override the ARCO-ERA5 zarr path")
    parser.add_argument("--resume", action="store_true", help="Skip months whose NetCDF already exists")
    parser.add_argument("--dry-run", action="store_true", help="Process only the start month")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    now = datetime.now(timezone.utc)
    end = args.end_date or f"{now.year:04d}-{now.month:02d}"

    all_months = list(months(args.start_date, end))
    if args.dry_run:
        all_months = all_months[:1]

    print(f"ERA5 -> {data_root / 'era5'} | {len(all_months)} months | store {args.store}")
    ds = open_store(args.store)

    for y, m in all_months:
        try:
            print(f"  {process_month(ds, y, m, data_root, args.resume)}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED {y:04d}-{m:02d}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
