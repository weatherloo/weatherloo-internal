"""Native HRRR point predictors at the stations around a target site (region-configurable).

For each HRRR initialization, read only the native storage chunk containing the
target station (all 49 leads) and bilinearly interpolate a set of surface fields
at every station inside that chunk. Output per month: values [init, lead, station,
variable] in catalog units, plus the station list. Designed so a region pipeline
can call it with any target site and station table.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from nwp_data import retry, station_weights, interpolate_stations

UTC = timezone.utc
CATALOG = "noaa-hrrr-forecast-48-hour"
FIRST_INIT = datetime(2020, 12, 3, tzinfo=UTC)  # HRRRv4: 48 h forecasts at 00/06/12/18 UTC
LAST_INIT = datetime(2025, 12, 31, 18, tzinfo=UTC)
VARIABLES = (
    "temperature_2m", "dew_point_temperature_2m", "relative_humidity_2m",
    "wind_u_10m", "wind_v_10m", "wind_u_80m", "wind_v_80m", "wind_gust_surface",
    "pressure_surface", "pressure_reduced_to_mean_sea_level", "precipitation_surface",
    "total_cloud_cover_atmosphere", "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface", "snow_thickness_surface", "snow_area_fraction_surface",
    "precipitable_water_atmosphere", "geopotential_height_cloud_ceiling", "composite_reflectivity",
)


def months():
    keys, year, month = [], FIRST_INIT.year, FIRST_INIT.month
    while (year, month) <= (LAST_INIT.year, LAST_INIT.month):
        keys.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return keys


def month_inits(key):
    year, month = map(int, key.split("-"))
    start = max(datetime(year, month, 1, tzinfo=UTC), FIRST_INIT)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    inits, current = [], start
    while current < end and current <= LAST_INIT:
        inits.append(current)
        current += timedelta(hours=6)
    return inits


class PointReader:
    """The native chunk around a target site, with bilinear weights for every station inside it."""

    def __init__(self, stations, target_id):
        import dynamical_catalog
        from pyproj import CRS, Transformer

        ds = dynamical_catalog.open(CATALOG, chunks=None)
        missing = [v for v in VARIABLES if v not in ds]
        if missing:
            raise ValueError(f"catalog lacks {missing}")
        chunk_y, chunk_x = ds["temperature_2m"].encoding["chunks"][-2:]
        transform = Transformer.from_crs("EPSG:4326", CRS.from_cf(ds.spatial_ref.attrs), always_xy=True)
        xs, ys = ds.x.values, ds.y.values
        target = next(s for s in stations if s["id"] == target_id)
        tx, ty = transform.transform(target["lon"], target["lat"])
        xi = int(np.argmin(np.abs(xs - tx)))
        yi = int(np.argmin(np.abs(ys - ty)))
        window = {"y": slice(yi // chunk_y * chunk_y, (yi // chunk_y + 1) * chunk_y),
                  "x": slice(xi // chunk_x * chunk_x, (xi // chunk_x + 1) * chunk_x)}
        wx, wy = xs[window["x"]], ys[window["y"]]
        points, kept = [], []
        for station in stations:
            px, py = transform.transform(station["lon"], station["lat"])
            inside_x = min(wx[0], wx[-1]) < px < max(wx[0], wx[-1])
            inside_y = min(wy[0], wy[-1]) < py < max(wy[0], wy[-1])
            if inside_x and inside_y:
                points.append((px, py))
                kept.append(station)
        self.stations = kept
        self.weights = station_weights(wx, wy, points)
        self.subset = ds[list(VARIABLES)].isel(window)
        self.window = {"y": [window["y"].start, window["y"].stop], "x": [window["x"].start, window["x"].stop]}
        self.units = {v: ds[v].attrs.get("units") for v in VARIABLES}

    def read(self, init):
        def load():
            return self.subset.sel(init_time=np.datetime64(init.replace(tzinfo=None))).to_array().values
        values = retry(load).astype(np.float32)  # [variable, lead, y, x]
        return interpolate_stations(values, self.weights).transpose(1, 2, 0)  # [lead, station, variable]


def build_month(key, stations, target_id, destination, reader=None):
    destination = Path(destination)
    if (destination / "manifest.json").is_file():
        return json.loads((destination / "manifest.json").read_text())
    reader = reader or PointReader(stations, target_id)
    inits = month_inits(key)

    def fetch(init):
        try:
            return reader.read(init), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    rows, kept, failures = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for init, (values, error) in zip(inits, pool.map(fetch, inits)):
            if error:
                failures.append({"initialization": init.isoformat(), "error": error})
            else:
                rows.append(values)
                kept.append(int(init.timestamp()))
    destination.mkdir(parents=True, exist_ok=True)
    values = np.stack(rows) if rows else np.zeros((0, 49, len(reader.stations), len(VARIABLES)), np.float32)
    np.savez_compressed(destination / "points.npz", values=values, init_time=np.array(kept, dtype=np.int64))
    manifest = {"month": key, "samples": len(kept), "planned": len(inits), "failures": failures,
                "stations": [s["id"] for s in reader.stations], "variables": list(VARIABLES), "units": reader.units,
                "target": target_id, "native_window": reader.window, "catalog": CATALOG,
                "sha256": hashlib.sha256((destination / "points.npz").read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


PATCH_VARIABLES = (
    "temperature_2m", "dew_point_temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m",
    "wind_u_80m", "wind_v_80m", "wind_gust_surface", "pressure_surface", "total_cloud_cover_atmosphere",
    "downward_short_wave_radiation_flux_surface", "downward_long_wave_radiation_flux_surface",
    "snow_area_fraction_surface", "precipitable_water_atmosphere", "precipitation_surface",
)


class PatchReader:
    """Native-grid square patch (2*half+1 points per side, ~3 km spacing) centred on a site, all leads."""

    def __init__(self, site, half=8):
        import dynamical_catalog
        from pyproj import CRS, Transformer

        ds = dynamical_catalog.open(CATALOG, chunks=None)
        transform = Transformer.from_crs("EPSG:4326", CRS.from_cf(ds.spatial_ref.attrs), always_xy=True)
        px, py = transform.transform(site["lon"], site["lat"])
        xi = int(np.argmin(np.abs(ds.x.values - px)))
        yi = int(np.argmin(np.abs(ds.y.values - py)))
        self.window = {"y": slice(yi - half, yi + half + 1), "x": slice(xi - half, xi + half + 1)}
        self.subset = ds[list(PATCH_VARIABLES)].isel(self.window)
        self.centre = {"x": float(ds.x.values[xi]), "y": float(ds.y.values[yi]), "dx_m": float(px - ds.x.values[xi]),
                       "dy_m": float(py - ds.y.values[yi])}
        self.units = {v: ds[v].attrs.get("units") for v in PATCH_VARIABLES}
        self.y_descending = bool(ds.y.values[0] > ds.y.values[-1])

    def read(self, init):
        def load():
            return self.subset.sel(init_time=np.datetime64(init.replace(tzinfo=None))).to_array().values
        values = retry(load).astype(np.float32)  # [variable, lead, y, x]
        if self.y_descending:
            values = values[:, :, ::-1]  # store north-up consistently: row 0 = south
        return values.transpose(1, 0, 2, 3).astype(np.float16)  # [lead, variable, y, x]


def build_patch_month(key, site, destination, half=8, reader=None):
    destination = Path(destination)
    if (destination / "manifest.json").is_file():
        return json.loads((destination / "manifest.json").read_text())
    reader = reader or PatchReader(site, half)
    inits = month_inits(key)

    def fetch(init):
        try:
            return reader.read(init), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    rows, kept, failures = [], [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for init, (values, error) in zip(inits, pool.map(fetch, inits)):
            if error:
                failures.append({"initialization": init.isoformat(), "error": error})
            else:
                rows.append(values)
                kept.append(int(init.timestamp()))
    destination.mkdir(parents=True, exist_ok=True)
    side = 2 * half + 1
    values = np.stack(rows) if rows else np.zeros((0, 49, len(PATCH_VARIABLES), side, side), np.float16)
    np.save(destination / "patches.npy", values)
    np.save(destination / "init_time.npy", np.array(kept, dtype=np.int64))
    manifest = {"month": key, "samples": len(kept), "planned": len(inits), "failures": failures,
                "variables": list(PATCH_VARIABLES), "units": reader.units, "half": half, "centre": reader.centre,
                "site": site["id"], "catalog": CATALOG}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
