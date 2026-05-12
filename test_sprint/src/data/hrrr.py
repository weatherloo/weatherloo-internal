from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class WaterlooPatchConfig:
    center_lat: float = 43.4725
    center_lon: float = -80.5449
    lat_half_span: float = 0.35
    lon_half_span: float = 0.45
    target_variable: str = "t2m"
    dynamic_variables: tuple[str, ...] = ("t2m", "d2m", "u10", "v10", "sp")
    static_variables: tuple[str, ...] = ()
    latitude_names: tuple[str, ...] = ("lat", "latitude")
    longitude_names: tuple[str, ...] = ("lon", "longitude")

    @property
    def lat_bounds(self) -> tuple[float, float]:
        return (self.center_lat - self.lat_half_span, self.center_lat + self.lat_half_span)

    @property
    def lon_bounds(self) -> tuple[float, float]:
        return (self.center_lon - self.lon_half_span, self.center_lon + self.lon_half_span)


class HRRRPatchLoader:
    def __init__(self, dataset_path: str | Path, config: WaterlooPatchConfig | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.config = config or WaterlooPatchConfig()
        self._dataset: xr.Dataset | None = None

    def open_dataset(self) -> xr.Dataset:
        if self._dataset is None:
            suffix = self.dataset_path.suffix.lower()
            if self.dataset_path.name.endswith(".zarr") or suffix == ".zarr":
                self._dataset = xr.open_zarr(self.dataset_path)
            else:
                self._dataset = xr.open_dataset(self.dataset_path)
        return self._dataset

    def load_patch(
        self,
        *,
        init_time_utc: str | pd.Timestamp,
        lead_hour: int,
    ) -> dict[str, Any]:
        ds = self.open_dataset()
        selected = self._select_init_and_lead(ds, init_time_utc=init_time_utc, lead_hour=lead_hour)
        patch = self._crop_to_waterloo(selected)
        lat_grid, lon_grid = self.extract_latlon_grids(patch)

        dynamic = np.stack(
            [self._as_float_array(patch[var], variable_name=var) for var in self.config.dynamic_variables],
            axis=0,
        )
        static = (
            np.stack(
                [self._as_float_array(patch[var], variable_name=var) for var in self.config.static_variables],
                axis=0,
            )
            if self.config.static_variables
            else np.zeros((0, *dynamic.shape[-2:]), dtype=np.float32)
        )
        baseline = self._as_float_array(
            patch[self.config.target_variable],
            variable_name=self.config.target_variable,
        )

        return {
            "dynamic": dynamic,
            "static": static,
            "baseline_t2m_c": baseline,
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
            "lead_hour": int(lead_hour),
            "init_time_utc": pd.Timestamp(init_time_utc, tz="UTC"),
        }

    def _select_init_and_lead(
        self,
        ds: xr.Dataset,
        *,
        init_time_utc: str | pd.Timestamp,
        lead_hour: int,
    ) -> xr.Dataset:
        subset = ds
        init_timestamp = pd.Timestamp(init_time_utc, tz="UTC").tz_convert(None)

        if "time" in subset.coords:
            subset = subset.sel(time=init_timestamp, method="nearest")

        if "step" in subset.coords:
            step_values = subset.coords["step"]
            if np.issubdtype(step_values.dtype, np.timedelta64):
                lead_key = np.timedelta64(int(lead_hour), "h")
            else:
                lead_key = lead_hour
            subset = subset.sel(step=lead_key, method="nearest")
        elif "lead_time" in subset.coords:
            subset = subset.sel(lead_time=lead_hour, method="nearest")

        return subset.squeeze(drop=True)

    def _crop_to_waterloo(self, ds: xr.Dataset) -> xr.Dataset:
        lat_name, lon_name = self._resolve_latlon_names(ds)
        lat_min, lat_max = self.config.lat_bounds
        lon_min, lon_max = self.config.lon_bounds

        lat = ds[lat_name]
        lon = self._normalize_longitudes(ds[lon_name])

        if lat.ndim == 1 and lon.ndim == 1:
            lat_slice = slice(lat_max, lat_min) if float(lat[0]) > float(lat[-1]) else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max) if float(lon[0]) <= float(lon[-1]) else slice(lon_max, lon_min)
            return ds.sel({lat_name: lat_slice, lon_name: lon_slice})

        mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        if not bool(mask.any()):
            msg = "No HRRR cells intersect the configured Waterloo patch."
            raise ValueError(msg)

        mask_values = np.asarray(mask)
        y_idx, x_idx = np.where(mask_values)
        y_dim, x_dim = mask.dims[-2:]
        return ds.isel(
            {
                y_dim: slice(int(y_idx.min()), int(y_idx.max()) + 1),
                x_dim: slice(int(x_idx.min()), int(x_idx.max()) + 1),
            }
        )

    def extract_latlon_grids(self, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        lat_name, lon_name = self._resolve_latlon_names(ds)
        lat = np.asarray(ds[lat_name])
        lon = np.asarray(self._normalize_longitudes(ds[lon_name]))
        if lat.ndim == 1 and lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            return lat_grid.astype(np.float32), lon_grid.astype(np.float32)
        return lat.astype(np.float32), lon.astype(np.float32)

    def station_coords_to_normalized_grid(
        self,
        *,
        station_latitudes: np.ndarray,
        station_longitudes: np.ndarray,
        patch_lat_grid: np.ndarray,
        patch_lon_grid: np.ndarray,
    ) -> np.ndarray:
        ny, nx = patch_lat_grid.shape
        center_y = ny // 2
        center_x = nx // 2

        lat_axis = patch_lat_grid[:, center_x].astype(np.float64)
        lon_axis = patch_lon_grid[center_y, :].astype(np.float64)

        y = self._interp_index(lat_axis, station_latitudes.astype(np.float64))
        x = self._interp_index(lon_axis, station_longitudes.astype(np.float64))

        x_norm = 2.0 * (x / max(nx - 1, 1)) - 1.0
        y_norm = 2.0 * (y / max(ny - 1, 1)) - 1.0
        return np.stack([x_norm, y_norm], axis=-1).astype(np.float32)

    @staticmethod
    def _interp_index(axis_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
        base_index = np.arange(len(axis_values), dtype=np.float64)
        if len(axis_values) < 2:
            return np.zeros_like(query_values, dtype=np.float64)
        if axis_values[0] > axis_values[-1]:
            axis_values = axis_values[::-1]
            base_index = base_index[::-1]
        return np.interp(query_values, axis_values, base_index, left=base_index.min(), right=base_index.max())

    def _resolve_latlon_names(self, ds: xr.Dataset) -> tuple[str, str]:
        lat_name = next((name for name in self.config.latitude_names if name in ds.coords or name in ds), None)
        lon_name = next((name for name in self.config.longitude_names if name in ds.coords or name in ds), None)
        if lat_name is None or lon_name is None:
            msg = "Unable to resolve latitude/longitude coordinate names from the HRRR dataset."
            raise ValueError(msg)
        return lat_name, lon_name

    @staticmethod
    def _normalize_longitudes(longitudes: xr.DataArray) -> xr.DataArray:
        return ((longitudes + 180.0) % 360.0) - 180.0

    @staticmethod
    def _as_float_array(data_array: xr.DataArray, *, variable_name: str) -> np.ndarray:
        values = np.asarray(data_array).astype(np.float32)
        if variable_name in {"t2m", "d2m"} and np.nanmean(values) > 150.0:
            values = values - 273.15
        if variable_name in {"sp", "msl"} and np.nanmean(values) > 2000.0:
            values = values / 100.0
        return values

