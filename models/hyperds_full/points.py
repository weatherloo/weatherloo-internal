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


ALL_VARIABLES = PATCH_VARIABLES + (
    "pressure_reduced_to_mean_sea_level", "geopotential_height_cloud_ceiling", "composite_reflectivity",
    "visible_beam_downward_solar_flux_surface", "snow_thickness_surface", "snow_water_equivalent_surface",
    "snowfall_surface", "percent_frozen_precipitation_surface", "categorical_rain_surface",
    "categorical_snow_surface", "categorical_freezing_rain_surface", "categorical_ice_pellets_surface",
)
WIDE_VARIABLES = (
    "temperature_2m", "dew_point_temperature_2m", "wind_u_10m", "wind_v_10m", "pressure_reduced_to_mean_sea_level",
    "total_cloud_cover_atmosphere", "precipitation_surface", "precipitable_water_atmosphere",
)

# Catalog unit -> float16-safe storage unit (factor, unit).
STORAGE_UNITS = {
    "pressure_surface": (0.01, "hPa"), "pressure_reduced_to_mean_sea_level": (0.01, "hPa"),
    "precipitation_surface": (3600.0, "mm h-1"), "snowfall_surface": (3.6e6, "mm h-1"),
}


class ContextReader:
    """Every catalog field on the native patch, plus a block-averaged view of the whole native chunk.

    The anchor chunk is read once per initialization; the patch and the wide view are cut from it in
    memory, so the wide view costs no extra download. Both are stored south-up (row 0 = south).
    """

    def __init__(self, site, half=8, block=8):
        import dynamical_catalog
        from pyproj import CRS, Transformer

        ds = dynamical_catalog.open(CATALOG, chunks=None)
        chunk_y, chunk_x = ds["temperature_2m"].encoding["chunks"][-2:]
        transform = Transformer.from_crs("EPSG:4326", CRS.from_cf(ds.spatial_ref.attrs), always_xy=True)
        px, py = transform.transform(site["lon"], site["lat"])
        xi = int(np.argmin(np.abs(ds.x.values - px)))
        yi = int(np.argmin(np.abs(ds.y.values - py)))
        y0, x0 = yi // chunk_y * chunk_y, xi // chunk_x * chunk_x
        ny, nx = len(ds.y), len(ds.x)
        self.chunk = {"y": slice(y0, min(y0 + chunk_y, ny)), "x": slice(x0, min(x0 + chunk_x, nx))}
        self.local = (yi - y0, xi - x0)
        size_y, size_x = self.chunk["y"].stop - y0, self.chunk["x"].stop - x0
        if not (half <= self.local[0] < size_y - half and half <= self.local[1] < size_x - half):
            raise ValueError("patch crosses the native chunk edge; not supported by ContextReader")
        self.half, self.block = half, block
        self.subset = ds[list(ALL_VARIABLES)].isel(self.chunk)
        self.wide_index = [ALL_VARIABLES.index(v) for v in WIDE_VARIABLES]
        self.scale = [(ALL_VARIABLES.index(v), factor) for v, (factor, _) in STORAGE_UNITS.items()]
        self.units = {v: STORAGE_UNITS.get(v, (1, ds[v].attrs.get("units")))[1] for v in ALL_VARIABLES}
        self.y_descending = bool(ds.y.values[0] > ds.y.values[-1])
        self.window = {"y": [y0, self.chunk["y"].stop], "x": [x0, self.chunk["x"].stop], "site_local": list(self.local)}

    def read(self, init):
        def load():
            return self.subset.sel(init_time=np.datetime64(init.replace(tzinfo=None))).to_array().values
        values = retry(load).astype(np.float32)  # [variable, lead, y, x]
        # float16 storage: Pa overflows (max 65504) and kg m-2 s-1 / m s-1 rates underflow.
        for index, factor in self.scale:
            values[index] *= factor
        (ly, lx), h, b = self.local, self.half, self.block
        patch = values[:, :, ly - h:ly + h + 1, lx - h:lx + h + 1]
        wide = values[self.wide_index]
        wy, wx = wide.shape[2] // b * b, wide.shape[3] // b * b
        wide = np.nanmean(wide[:, :, :wy, :wx].reshape(len(self.wide_index), wide.shape[1], wy // b, b, wx // b, b),
                          axis=(3, 5))
        if self.y_descending:
            patch, wide = patch[:, :, ::-1], wide[:, :, ::-1]
        return (patch.transpose(1, 0, 2, 3).astype(np.float16), wide.transpose(1, 0, 2, 3).astype(np.float16))


def build_context_month(key, site, destination, half=8, block=8, reader=None):
    destination = Path(destination)
    if (destination / "manifest.json").is_file():
        return json.loads((destination / "manifest.json").read_text())
    reader = reader or ContextReader(site, half, block)
    inits = month_inits(key)

    def fetch(init):
        try:
            return reader.read(init), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"

    patches, wides, kept, failures = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:  # each read holds a whole chunk of 27 fields
        for init, (values, error) in zip(inits, pool.map(fetch, inits)):
            if error:
                failures.append({"initialization": init.isoformat(), "error": error})
            else:
                patches.append(values[0])
                wides.append(values[1])
                kept.append(int(init.timestamp()))
    destination.mkdir(parents=True, exist_ok=True)
    side = 2 * half + 1
    np.save(destination / "patches.npy", np.stack(patches) if patches else
            np.zeros((0, 49, len(ALL_VARIABLES), side, side), np.float16))
    np.save(destination / "wide.npy", np.stack(wides) if wides else np.zeros((0, 49, len(WIDE_VARIABLES), 1, 1), np.float16))
    np.save(destination / "init_time.npy", np.array(kept, dtype=np.int64))
    manifest = {"month": key, "samples": len(kept), "planned": len(inits), "failures": failures,
                "variables": list(ALL_VARIABLES), "wide_variables": list(WIDE_VARIABLES), "units": reader.units,
                "half": half, "block": block, "window": reader.window, "site": site["id"], "catalog": CATALOG}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
