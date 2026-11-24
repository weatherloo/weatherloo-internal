#!/usr/bin/env python3
"""Build a common-crop, multi-station January dataset for temporal HyperDS."""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from hrrr_bias_correction.make_data import (  # noqa: E402
    SoulisObsIndex, bilinear_curvilinear, crop_window, lead_hours,
    nearest_grid_index, open_hrrr,
)


STATIONS = (
    ("eric_d_soulis", 43.4737778, -80.5576389, None),
    ("elora_rcs", 43.65, -80.4166666667, 41983),
    ("kitchener_waterloo", 43.4608333333, -80.3786111111, 48569),
    ("guelph_turfgrass", 43.5418725, -80.2220811111, 55299),
)
WINDOWS = {"train": ("2021-01-02", "2021-01-15"), "val": ("2024-01-02", "2024-01-06")}
CROP = 64
LEADS = 49


def load_observations(obs_root):
    soulis = SoulisObsIndex(obs_root, (2021, 2024))
    observations = {"eric_d_soulis": soulis}
    for station_id, _, _, eccc_id in STATIONS[1:]:
        hourly = {}
        for year in (2021, 2024):
            path = obs_root / f"eccc_{eccc_id}_{year}.json"
            for row in json.loads(path.read_text())["features"]:
                stamp = datetime.fromisoformat(row["UTC_DATE"]).replace(tzinfo=timezone.utc)
                values = []
                for name in ("TEMP", "WIND_SPEED"):
                    value = row.get(name)
                    values.append(float(value) if value is not None and float(value) > -9000 else np.nan)
                hourly[stamp] = tuple(values)
        observations[station_id] = hourly
    return observations


def station_xy(lat, lon, station_lat, station_lon):
    """Map a latitude/longitude to the fractional HRRR grid position."""
    j, i = nearest_grid_index(lat, lon, station_lat, station_lon)
    if not (0 < j < lat.shape[0] - 1 and 0 < i < lat.shape[1] - 1):
        raise ValueError("station lies outside the common HRRR crop")
    jacobian = np.array([
        [(lat[j, i + 1] - lat[j, i - 1]) / 2, (lat[j + 1, i] - lat[j - 1, i]) / 2],
        [(lon[j, i + 1] - lon[j, i - 1]) / 2, (lon[j + 1, i] - lon[j - 1, i]) / 2],
    ])
    dx, dy = np.linalg.solve(jacobian, [station_lat - lat[j, i], station_lon - lon[j, i]])
    return np.array([2 * (i + dx) / (lat.shape[1] - 1) - 1,
                     2 * (j + dy) / (lat.shape[0] - 1) - 1], dtype=np.float32)


def build_cycle(path, init, observations):
    with open_hrrr(path) as ds:
        lat = np.asarray(ds.latitude.values, dtype=np.float64)
        lon = np.asarray(ds.longitude.values, dtype=np.float64)
        lon = np.where(lon > 180, lon - 360, lon)
        center_j, center_i = nearest_grid_index(lat, lon, STATIONS[0][1], STATIONS[0][2])
        window = crop_window(center_j, center_i, *lat.shape, half=CROP // 2)
        if window is None:
            raise ValueError("common crop is outside HRRR coverage")
        rows, cols = window
        lat, lon = lat[rows, cols], lon[rows, cols]
        queries = np.stack([station_xy(lat, lon, a, b) for _, a, b, _ in STATIONS])
        x = np.full((LEADS, CROP, CROP, 3), np.nan, dtype=np.float32)
        y = np.full((LEADS, len(STATIONS), 2), np.nan, dtype=np.float32)
        by_lead = {int(hour): index for index, hour in enumerate(lead_hours(ds))}
        frames = [np.asarray(ds[name].values[:, rows, cols], dtype=np.float32)
                  for name in ("t2m", "u10", "v10")]
        for hour in range(LEADS):
            if hour not in by_lead:
                continue
            index = by_lead[hour]
            x[hour] = np.stack([frame[index] for frame in frames], axis=-1)
            for station_index, (name, station_lat, station_lon, _) in enumerate(STATIONS):
                source = observations[name]
                valid_time = init + timedelta(hours=hour)
                observed = source.lookup(valid_time) if name == "eric_d_soulis" else source.get(valid_time)
                if observed is None:
                    continue
                t2m, u10, v10 = [bilinear_curvilinear(lat, lon, frame[index], station_lat, station_lon)
                                  for frame in frames]
                if np.isfinite(t2m) and np.isfinite(observed[0]):
                    y[hour, station_index, 0] = t2m - 273.15 - observed[0]
                if np.isfinite(u10) and np.isfinite(v10) and np.isfinite(observed[1]):
                    y[hour, station_index, 1] = 3.6 * np.hypot(u10, v10) - observed[1]
    return x, y, queries


def build(hrrr_root, obs_root, out):
    observations = load_observations(obs_root)
    manifest = {"stations": [{"id": name, "lat": lat, "lon": lon, "eccc_stn_id": eccc}
                             for name, lat, lon, eccc in STATIONS],
                "windows": WINDOWS, "crop": CROP, "leads": LEADS,
                "bias_definition": "HRRR_bilinear(station) - observation", "splits": {}}
    for split, (start, end) in WINDOWS.items():
        samples = []
        for path in sorted(hrrr_root.glob("*/*/hrrr_*_t??z.nc")):
            init = datetime.strptime(path.stem, "hrrr_%Y%m%d_t%Hz").replace(tzinfo=timezone.utc)
            if start <= init.strftime("%Y-%m-%d") <= end and init.hour in (0, 12):
                samples.append((init, *build_cycle(path, init, observations)))
        if not samples:
            raise ValueError(f"no HRRR samples for {split}")
        x = np.stack([item[1] for item in samples])
        y = np.stack([item[2] for item in samples])
        queries = np.stack([item[3] for item in samples])
        data = xr.Dataset({
            "x": (("sample", "lead", "row", "col", "channel"), x),
            "y": (("sample", "lead", "station", "target"), y),
            "query_xy": (("sample", "station", "xy"), queries),
            "init_time": ("sample", np.array([int(item[0].timestamp()) for item in samples], dtype=np.int64)),
        })
        data.to_zarr(out, group=split, mode="w" if split == "train" else "a", consolidated=False)
        manifest["splits"][split] = {"samples": len(samples),
                                     "valid_target_hours": np.isfinite(y).sum(axis=(0, 1)).tolist()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrrr-root", type=Path, required=True)
    parser.add_argument("--obs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.hrrr_root, args.obs_root, args.out), indent=2))
