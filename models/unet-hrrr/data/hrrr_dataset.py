#!/usr/bin/env python3
"""HRRR -> ERA5/observations dataset for the U-Net prototype.

This version is deliberately lightweight and data-driven:
- model input: HRRR forecast fields at a given lead time
- target: ERA5 analysis at the matching valid time
- auxiliary loss: weather-station observations (Toronto area) interpolated
  from the model output grid to the station location

The data lives under /mnt/wato-drive/c52li/weatherloo-data/ and is assumed to
be available locally.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("UNET_DATA_ROOT", "/mnt/wato-drive/c52li/weatherloo-data"))

DEFAULT_VARIABLES = ("t2m", "u10", "v10", "q2", "psfc", "tp")
DEFAULT_STATIONS = ("stn_51459_toronto_intl_a",)


class HRRRDataset(Dataset):
    """Dataset over HRRR forecast files and matching ERA5/obs targets."""

    def __init__(self, split: str, data_root: str | Path | None = None,
                 years: Iterable[int] | None = None, val_year: int = 2025,
                 variables: Iterable[str] | None = None,
                 target_vars: Iterable[str] | None = None,
                 station_ids: Iterable[str] | None = None,
                 max_samples: int | None = None,
                 station_loss_weight: float = 0.2,
                 target_grid_shape: tuple[int, int] = (32, 32),
                 stats: dict | None = None):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        self.data_root = Path(data_root or DATA_ROOT)
        self.variables = list(variables or DEFAULT_VARIABLES)
        self.target_vars = list(target_vars or ("t2m", "u10", "v10"))
        self.station_ids = list(station_ids or DEFAULT_STATIONS)
        self.input_vars = self.variables
        self.station_loss_weight = station_loss_weight
        self.target_grid_shape = target_grid_shape
        self.stats = stats
        self._hrrr_cache: dict[Path, xr.Dataset] = {}
        self._era5_cache: dict[tuple[int, int], xr.Dataset] = {}
        self._station_cache: dict[tuple[str, int], list[dict]] = {}

        if years is None:
            years = list(range(2018, val_year)) if split == "train" else [val_year]
        self.years = list(years)

        self.samples = self._build_samples()
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.grid_lats, self.grid_lons = self._build_target_grid()
        self.station_lat, self.station_lon = self._station_coords(self.station_ids[0])
        self.station_y_idx, self.station_x_idx = self._station_grid_index(self.station_lat, self.station_lon)

    def _build_samples(self) -> list[dict]:
        hrrr_root = self.data_root / "hrrr"
        samples: list[dict] = []
        for year in self.years:
            year_dir = hrrr_root / str(year)
            if not year_dir.exists():
                continue
            for day_dir in sorted(year_dir.glob("*")):
                if not day_dir.is_dir():
                    continue
                for file in sorted(day_dir.glob("hrrr_*_t*z.nc")):
                    name = file.name
                    m = re.match(r"hrrr_(\d{8})_t(\d{2})z\.nc", name)
                    if not m:
                        continue
                    init_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
                    try:
                        ds = xr.open_dataset(file)
                    except Exception:
                        continue
                    try:
                        lead_vals = ds["lead"].values
                    except Exception:
                        lead_vals = np.arange(49)
                    for idx, lead in enumerate(lead_vals):
                        valid_dt = init_dt + timedelta(hours=int(lead))
                        # Keep the full historical range but use 2025 only for val.
                        if self.split == "train" and valid_dt.year >= 2025:
                            continue
                        if self.split == "val" and valid_dt.year != 2025:
                            continue
                        samples.append({
                            "file": file,
                            "init_dt": init_dt,
                            "lead_idx": int(idx),
                            "lead": int(lead),
                            "valid_dt": valid_dt,
                        })
                    ds.close()
        # Sort by valid time for deterministic ordering.
        samples.sort(key=lambda s: s["valid_dt"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        hrrr_ds = self._load_hrrr(sample["file"])
        lead_idx = sample["lead_idx"]
        valid_dt = sample["valid_dt"]

        hrrr_field = self._extract_hrrr_field(hrrr_ds, lead_idx)
        era5_field = self._extract_era5_field(valid_dt)

        x = self._regrid_to_target(hrrr_field, hrrr_ds, self.grid_lats, self.grid_lons)
        y = self._regrid_to_target(era5_field, None, self.grid_lats, self.grid_lons)

        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))

        station_target = self._station_target(valid_dt)
        if station_target is None:
            station_target = torch.zeros(3, dtype=torch.float32)
            station_weight = torch.zeros(1, dtype=torch.float32)
        else:
            station_weight = torch.ones(1, dtype=torch.float32)
        return x, y, torch.tensor(station_target, dtype=torch.float32), station_weight

    def _build_target_grid(self) -> tuple[np.ndarray, np.ndarray]:
        lat_min, lat_max = 42.0, 46.0
        lon_min, lon_max = -82.0, -78.0
        lat = np.linspace(lat_min, lat_max, self.target_grid_shape[0], dtype=np.float32)
        lon = np.linspace(lon_min, lon_max, self.target_grid_shape[1], dtype=np.float32)
        return lat, lon

    def _load_hrrr(self, path: Path) -> xr.Dataset:
        if path not in self._hrrr_cache:
            self._hrrr_cache[path] = xr.open_dataset(path)
        return self._hrrr_cache[path]

    def _load_era5(self, valid_dt: datetime) -> xr.Dataset:
        key = (valid_dt.year, valid_dt.month)
        if key not in self._era5_cache:
            file = self.data_root / "era5" / str(valid_dt.year) / f"era5_{valid_dt.year:04d}{valid_dt.month:02d}.nc"
            if not file.exists():
                raise FileNotFoundError(f"ERA5 file not found: {file}")
            self._era5_cache[key] = xr.open_dataset(file)
        return self._era5_cache[key]

    def _extract_hrrr_field(self, ds: xr.Dataset, lead_idx: int) -> np.ndarray:
        arrs = []
        for var in self.variables:
            da = ds[var].isel(lead=lead_idx)
            arrs.append(np.asarray(da.values, dtype=np.float32))
        return np.stack(arrs, axis=0)

    def _extract_era5_field(self, valid_dt: datetime) -> np.ndarray:
        ds = self._load_era5(valid_dt)
        arrs = []
        for var in self.variables:
            if var == "t2m":
                da = ds[var].sel(time=np.datetime64(valid_dt.replace(tzinfo=None)), method="nearest")
                arr = np.asarray(da.values, dtype=np.float32)
                arr = arr - 273.15
            elif var == "u10":
                da = ds[var].sel(time=np.datetime64(valid_dt.replace(tzinfo=None)), method="nearest")
                arr = np.asarray(da.values, dtype=np.float32)
            elif var == "v10":
                da = ds[var].sel(time=np.datetime64(valid_dt.replace(tzinfo=None)), method="nearest")
                arr = np.asarray(da.values, dtype=np.float32)
            else:
                da = ds[var].sel(time=np.datetime64(valid_dt.replace(tzinfo=None)), method="nearest")
                arr = np.asarray(da.values, dtype=np.float32)
            arrs.append(arr)
        return np.stack(arrs, axis=0)

    def _regrid_to_target(self, field: np.ndarray, ds: xr.Dataset | None,
                          lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        # field: (C, H, W) from HRRR or ERA5; the target grid is a simple regular lat/lon mesh.
        # We use the raw source arrays and interpolate in the source coordinate space.
        if field.ndim != 3:
            raise ValueError(f"unexpected field shape {field.shape}")
        if ds is None:
            # ERA5 field is already on a native lat/lon grid; assume it shares the same shape with the
            # source grid. We use a simple nearest-neighbor resampling to the target lat/lon grid.
            h, w = field.shape[1:]
            lat = np.linspace(42.0, 46.0, h, dtype=np.float32)
            lon = np.linspace(-82.0, -78.0, w, dtype=np.float32)
            out = np.empty((field.shape[0], len(lat_grid), len(lon_grid)), dtype=np.float32)
            for c in range(field.shape[0]):
                for i, la in enumerate(lat_grid):
                    jj = int(np.argmin(np.abs(lat - la)))
                    for j, lo in enumerate(lon_grid):
                        kk = int(np.argmin(np.abs(lon - lo)))
                        out[c, i, j] = field[c, jj, kk]
            return out

        # HRRR uses latitude/longitude coords on the cell centers; we use a lightweight bilinear
        # interpolation based on the coordinate arrays.
        lat = np.asarray(ds["latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["longitude"].values, dtype=np.float32)
        if lat.ndim == 2:
            lat_2d = lat
            lon_2d = lon
        else:
            raise ValueError("unexpected latitude/longitude layout")
        out = np.empty((field.shape[0], len(lat_grid), len(lon_grid)), dtype=np.float32)
        for c in range(field.shape[0]):
            src = field[c]
            for i, la in enumerate(lat_grid):
                for j, lo in enumerate(lon_grid):
                    # Nearest-neighbor fallback to keep it robust and cheap.
                    dist = np.abs(lat_2d - la) + np.abs(lon_2d - lo)
                    yx = np.unravel_index(np.argmin(dist), dist.shape)
                    out[c, i, j] = src[yx]
        return out

    def _station_coords(self, station_id: str) -> tuple[float, float]:
        meta_path = self.data_root / "observations" / "eccc" / station_id / "meta.json"
        if not meta_path.exists():
            return 44.0, -79.0
        with open(meta_path) as fh:
            meta = json.load(fh)
        return float(meta.get("latitude", 44.0)), float(meta.get("longitude", -79.0))

    def _station_grid_index(self, lat: float, lon: float) -> tuple[int, int]:
        lat_idx = int(np.argmin(np.abs(self.grid_lats - lat)))
        lon_idx = int(np.argmin(np.abs(self.grid_lons - lon)))
        return lat_idx, lon_idx

    def _station_target(self, valid_dt: datetime) -> np.ndarray | None:
        # Use Toronto Intl Airport as the primary station.
        station_id = self.station_ids[0]
        rows = self._load_station_rows(station_id, valid_dt.year)
        if not rows:
            return None
        dt = valid_dt.replace(tzinfo=timezone.utc)
        for row in rows:
            try:
                row_dt = datetime.fromisoformat(row["UTC_DATE"].replace("Z", "+00:00"))
            except Exception:
                continue
            if row_dt.replace(tzinfo=timezone.utc) == dt:
                temp = row.get("TEMP")
                wind_speed = row.get("WIND_SPEED")
                wind_dir = row.get("WIND_DIRECTION")
                if temp is None or wind_speed is None or wind_dir is None:
                    return None
                u10 = -float(wind_speed) * np.sin(np.deg2rad(float(wind_dir)))
                v10 = -float(wind_speed) * np.cos(np.deg2rad(float(wind_dir)))
                return np.array([float(temp), float(u10), float(v10)], dtype=np.float32)
        return None

    def _load_station_rows(self, station_id: str, year: int) -> list[dict]:
        key = (station_id, year)
        if key in self._station_cache:
            return self._station_cache[key]
        json_path = self.data_root / "observations" / "eccc" / station_id / "hourly" / f"eccc_climate_hourly_{year}.json"
        if not json_path.exists():
            self._station_cache[key] = []
            return []
        with open(json_path) as fh:
            payload = json.load(fh)
        rows = payload.get("rows", [])
        self._station_cache[key] = rows
        return rows
