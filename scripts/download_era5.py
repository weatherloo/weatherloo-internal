#!/usr/bin/env python3
"""Download ERA5 ground-truth (bbox-subset) for bias-correction training.

Reads the public **ARCO-ERA5** analysis-ready Zarr on Google Cloud (hourly,
0.25 deg, no account required), subsets the four target variables to the
Kitchener-Waterloo bounding box, and writes one compact NetCDF per month:

    {data_root}/era5/{YYYY}/era5_{YYYYMM}.nc
        dims:  (time, latitude, longitude)
        vars:  t2m (K), u10 (m/s), v10 (m/s), tp (m accumulated),
               q2 (kg/kg), psfc (Pa), pblh (m), hgt (m), tsk (K), ust (m/s)

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
G = 9.80665

# ARCO-ERA5 variable name -> compact output name (matches download_hrrr.py).
VAR_MAP = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "surface_pressure": "psfc",
    "boundary_layer_height": "pblh",
    "geopotential_at_surface": "z_surface",
    "skin_temperature": "tsk",
    "friction_velocity": "ust",
}
OUTPUT_VARS = ("t2m", "u10", "v10", "tp", "q2", "psfc", "pblh", "hgt", "tsk", "ust")
# ARCO variables needed to produce each output variable (for incremental patches).
ARCO_FOR_OUTPUT: dict[str, tuple[str, ...]] = {
    "t2m": ("2m_temperature",),
    "u10": ("10m_u_component_of_wind",),
    "v10": ("10m_v_component_of_wind",),
    "tp": ("total_precipitation",),
    "psfc": ("surface_pressure",),
    "pblh": ("boundary_layer_height",),
    "hgt": ("geopotential_at_surface",),
    "tsk": ("skin_temperature",),
    "ust": ("friction_velocity",),
    "q2": ("2m_dewpoint_temperature", "surface_pressure"),
}
DIRECT_OUTPUT_VARS = tuple(v for v in OUTPUT_VARS if v not in ("q2", "hgt"))


def mixing_ratio_from_t_td_p(t_k: xr.DataArray, td_k: xr.DataArray, p_pa: xr.DataArray) -> xr.DataArray:
    """Mass mixing ratio at 2 m from temperature, dewpoint, and surface pressure."""
    td_c = td_k - 273.15
    es = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    q = 0.622 * es / (p_pa - 0.378 * es)
    return q / (1.0 - q)


def month_file_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return all(v in ds.data_vars for v in OUTPUT_VARS)
    except Exception:
        return False


def missing_month_vars(path: Path) -> tuple[str, ...]:
    if not path.is_file() or path.stat().st_size == 0:
        return OUTPUT_VARS
    try:
        with xr.open_dataset(path) as ds:
            return tuple(v for v in OUTPUT_VARS if v not in ds.data_vars)
    except Exception:
        return OUTPUT_VARS


def arco_vars_for_missing(missing: tuple[str, ...]) -> tuple[str, ...]:
    arco: list[str] = []
    for out_var in missing:
        for arco_var in ARCO_FOR_OUTPUT[out_var]:
            if arco_var not in arco:
                arco.append(arco_var)
    return tuple(arco)


def write_month_netcdf(ds: xr.Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    tmp = out_path.parent / (out_path.name + ".tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(out_path)


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
    if resume and month_file_complete(out_path):
        return f"skip {out_path.name}"

    missing = missing_month_vars(out_path) if resume and out_path.exists() else OUTPUT_VARS
    if resume and not missing:
        return f"skip {out_path.name}"

    arco_needed = arco_vars_for_missing(missing)
    month = ds[list(arco_needed)].sel(time=slice(f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}"))
    if month.sizes.get("time", 0) == 0:
        return f"MISSING {y:04d}-{m:02d} (no time steps in store)"

    month = subset_region(month)

    # Currency probe: the store's time axis is padded with fill values for
    # not-yet-reanalysed months. Load just the first hour of one variable (one
    # global chunk, ~4 MB) and skip the whole month if it's all-NaN, so we don't
    # pay the full ~12 GB month read for data that isn't published yet.
    probe = month[arco_needed[0]].isel(time=0).load()
    if bool(np.isnan(probe.values).all()):
        return f"MISSING {y:04d}-{m:02d} (not published yet)"

    month = month.rename({k: VAR_MAP[k] for k in arco_needed}).load()
    if all(bool(np.isnan(month[v].values).all()) for v in month.data_vars):
        return f"MISSING {y:04d}-{m:02d} (all-NaN after load)"

    if resume and out_path.exists() and missing != OUTPUT_VARS:
        with xr.open_dataset(out_path) as existing:
            merged = existing.load()
        # Merge direct fields first so derived q2/hgt can reuse them.
        for var in DIRECT_OUTPUT_VARS:
            if var in missing and var in month:
                merged[var] = month[var]
        if "hgt" in missing and "z_surface" in month:
            merged["hgt"] = month["z_surface"] / G
        if "q2" in missing:
            t2m = merged["t2m"] if "t2m" in merged else month["t2m"]
            psfc = merged["psfc"] if "psfc" in merged else month["psfc"]
            merged["q2"] = mixing_ratio_from_t_td_p(t2m, month["d2m"], psfc)
        merged = merged[list(v for v in OUTPUT_VARS if v in merged)]
        write_month_netcdf(merged, out_path)
        return f"patched {out_path.name} (+{','.join(missing)}, {merged.sizes['time']} hrs)"

    month = ds[list(VAR_MAP)].sel(time=slice(f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}"))
    month = subset_region(month)
    probe = month[next(iter(VAR_MAP))].isel(time=0).load()
    if bool(np.isnan(probe.values).all()):
        return f"MISSING {y:04d}-{m:02d} (not published yet)"

    month = month.rename(VAR_MAP).load()
    if all(bool(np.isnan(month[v].values).all()) for v in month.data_vars):
        return f"MISSING {y:04d}-{m:02d} (all-NaN after load)"

    month["hgt"] = month["z_surface"] / G
    month["q2"] = mixing_ratio_from_t_td_p(month["t2m"], month["d2m"], month["psfc"])
    month = month.drop_vars(["d2m", "z_surface"])
    month = month[list(OUTPUT_VARS)]

    write_month_netcdf(month, out_path)
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
