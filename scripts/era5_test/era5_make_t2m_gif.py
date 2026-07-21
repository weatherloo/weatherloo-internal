#!/usr/bin/env python3
"""Create an animated GIF from a 24-hour window of ERA5 2m temperature."""
# python3 scripts/era5_test/era5_make_t2m_gif.py ~/../../mnt/wato-drive/c52li/weatherloo-data/era5/2025/era5_202512.nc -o /tmp/t2m.gif
# 
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image


def resolve_path(path_str: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_nc", help="Path to an ERA5 NetCDF file")
    parser.add_argument("-o", "--output", default="t2m_24h.gif", help="Output GIF path")
    parser.add_argument("--var", default="t2m", help="Variable to plot (default: t2m)")
    parser.add_argument("--window-hours", type=int, default=24, help="Length of the time window in hours")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first timestep to use")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second in the output GIF")
    parser.add_argument("--cmap", default="RdYlBu_r", help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None, help="Optional colorbar minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Optional colorbar maximum")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=(8, 6), help="Figure size (width height)")
    return parser.parse_args()


def resolve_data_and_time_name(ds: xr.Dataset, var: str) -> tuple[xr.DataArray, str, str, str]:
    if var in ds.data_vars:
        data = ds[var]
    elif var in ds.coords:
        data = ds[var]
    else:
        raise KeyError(f"Variable {var!r} not found in {ds}")

    time_name = None
    for candidate in ("time", "valid_time", "forecast_time"):
        if candidate in data.dims:
            time_name = candidate
            break
    if time_name is None:
        raise KeyError(f"Could not find a time dimension for {var!r} in {ds}")

    lat_name = None
    for candidate in ("latitude", "lat"):
        if candidate in data.dims:
            lat_name = candidate
            break
    lon_name = None
    for candidate in ("longitude", "lon"):
        if candidate in data.dims:
            lon_name = candidate
            break
    if lat_name is None or lon_name is None:
        raise KeyError(f"Expected latitude/longitude dimensions in {ds}")

    return data, time_name, lat_name, lon_name


def select_window(data: xr.DataArray, time_name: str, start_index: int, window_hours: int) -> xr.DataArray:
    total_steps = data.sizes[time_name]
    if total_steps <= start_index:
        raise IndexError(f"start_index={start_index} is beyond the available {total_steps} timesteps")

    window_length = max(1, window_hours)
    count = min(window_length, total_steps - start_index)
    window = data.isel({time_name: slice(start_index, start_index + count)})
    if window.sizes[time_name] == 0:
        raise ValueError(f"No timesteps found in the requested {window_hours}-hour window")
    return window


def build_frames(
    window: xr.DataArray,
    time_name: str,
    lat_name: str,
    lon_name: str,
    var: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
    figsize: tuple[float, float],
) -> list[Image.Image]:
    lat_vals = window[lat_name].values
    lon_vals = window[lon_name].values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    frames: list[Image.Image] = []
    for idx in range(window.sizes[time_name]):
        step = window.isel({time_name: idx})
        values = step.values
        if np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.float32)

        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D data for each time step, got shape {values.shape}")

        if var == "t2m" and np.nanmax(values) > 200:
            values = values - 273.15

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        mesh = ax.pcolormesh(lon_grid, lat_grid, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        time_label = step[time_name].values
        ax.set_title(f"{var} at {time_label}")
        ax.set_xlabel(lon_name)
        ax.set_ylabel(lat_name)
        fig.colorbar(mesh, ax=ax, shrink=0.9)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    return frames


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_nc)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with xr.open_dataset(input_path) as ds:
        data, time_name, lat_name, lon_name = resolve_data_and_time_name(ds, args.var)
        window = select_window(data, time_name, args.start_index, args.window_hours)
        frames = build_frames(
            window=window,
            time_name=time_name,
            lat_name=lat_name,
            lon_name=lon_name,
            var=args.var,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            dpi=args.dpi,
            figsize=tuple(args.figsize),
        )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(100, int(1000 / max(args.fps, 1))),
        loop=0,
    )
    print(f"Wrote GIF to {output_path}")


if __name__ == "__main__":
    main()
