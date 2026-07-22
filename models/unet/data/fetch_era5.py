#!/usr/bin/env python3
"""Fetch ERA5 reanalysis (ground truth) for the southern Ontario region.

ERA5 is loaded from the GCP public ARCO zarr store
(``gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr``) with
``token="anon"`` — same approach as the take-home project.

Run standalone to verify the store loads and values are physically reasonable:

    .venv/bin/python models/unet/data/fetch_era5.py

This prints the region grid shape, variable names, and sample t2m/u10/v10
values at CYYZ for a few 2021 dates. (2021 rather than 2020 because the
GFS AWS archive, which the U-Net's input comes from, starts 2021-01-01 while
this ERA5 store ends 2021-12-31 — 2021 is the overlap window.)

Exposes helpers reused by ``fetch_gfs.py`` and (later) ``dataset.py``:
    load_config, open_era5, era5_region, era5_point
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import xarray as xr
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_era5_lon(lon: float) -> float:
    """Convert signed longitude (degrees East, W negative) to ERA5's 0..360."""
    return lon + 360.0 if lon < 0 else lon


@lru_cache(maxsize=1)
def open_era5(store: str, token: str = "anon") -> xr.Dataset:
    """Open the ERA5 ARCO zarr store (lazy; cached per process)."""
    return xr.open_zarr(store, chunks=None, storage_options={"token": token})


def era5_region(cfg: dict, valid_time: datetime, ds: xr.Dataset | None = None) -> xr.Dataset:
    """Return the t2m/u10/v10 region slice at one valid time.

    Latitude is returned ascending (south -> north) so it aligns with GFS.
    """
    data = cfg["data"]["era5"]
    region = cfg["region"]
    if ds is None:
        ds = open_era5(data["store"], data["token"])

    var_map = data["variables"]
    sub = ds[[var_map["t2m"], var_map["u10"], var_map["v10"]]].sel(
        time=_np_time(valid_time),
        latitude=slice(region["lat_max"], region["lat_min"]),  # ERA5 lat descends
        longitude=slice(_to_era5_lon(region["lon_min"]), _to_era5_lon(region["lon_max"])),
    )
    sub = sub.rename({var_map["t2m"]: "t2m", var_map["u10"]: "u10", var_map["v10"]: "v10"})
    # Flip to ascending latitude for a consistent grid orientation across sources.
    return sub.sortby("latitude")


def era5_point(cfg: dict, valid_time: datetime, lat: float, lon: float,
               ds: xr.Dataset | None = None) -> dict[str, float]:
    """Bilinearly interpolate ERA5 t2m (degC), u10, v10 (m/s) to a lat/lon point."""
    data = cfg["data"]["era5"]
    if ds is None:
        ds = open_era5(data["store"], data["token"])
    var_map = data["variables"]
    point = ds.sel(time=_np_time(valid_time)).interp(
        latitude=lat, longitude=_to_era5_lon(lon), method="linear"
    )
    return {
        "t2m": float(point[var_map["t2m"]].values) - 273.15,
        "u10": float(point[var_map["u10"]].values),
        "v10": float(point[var_map["v10"]].values),
    }


def _np_time(dt: datetime):
    """ERA5 times are timezone-naive UTC numpy datetime64."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _valid_time(sample: dict) -> datetime:
    d = datetime.strptime(sample["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d.replace(hour=sample["cycle"]) + _hours(sample["fxx"])


def _hours(h: int):
    from datetime import timedelta
    return timedelta(hours=h)


def main() -> None:
    cfg = load_config()
    data = cfg["data"]["era5"]
    ds = open_era5(data["store"], data["token"])

    print("=== ERA5 store ===")
    print(f"  store: {data['store']}")
    t0, t1 = str(ds.time.values[0])[:13], str(ds.time.values[-1])[:13]
    print(f"  time coverage: {t0} .. {t1} (6-hourly)")
    print(f"  full grid: lat={ds.latitude.size}, lon={ds.longitude.size}")

    # Region slice at the first sample valid time.
    sample0 = cfg["sample_forecasts"][0]
    vt0 = _valid_time(sample0)
    region = era5_region(cfg, vt0, ds)
    print("\n=== Region slice (southern Ontario) ===")
    print(f"  valid_time: {vt0.isoformat()}")
    print(f"  variables: {list(region.data_vars)}")
    print(f"  shape (lat x lon): {region['t2m'].shape}")
    print(f"  lat range: {float(region.latitude.min()):.2f} .. {float(region.latitude.max()):.2f}")
    print(f"  lon range: {float(region.longitude.min()):.2f} .. {float(region.longitude.max()):.2f} (0..360)")

    cyyz = cfg["stations"]["cyyz"]
    print("\n=== Sample ERA5 values at CYYZ (43.6777, -79.6248) ===")
    print(f"  {'valid_time':<22} {'t2m(degC)':>10} {'u10(m/s)':>9} {'v10(m/s)':>9}")
    for sample in cfg["sample_forecasts"]:
        vt = _valid_time(sample)
        p = era5_point(cfg, vt, cyyz["lat"], cyyz["lon"], ds)
        print(f"  {vt.isoformat():<22} {p['t2m']:>10.2f} {p['u10']:>9.2f} {p['v10']:>9.2f}")

    print("\nERA5 load OK. Values should look like seasonal southern-Ontario surface air temps.")


if __name__ == "__main__":
    main()
