#!/usr/bin/env python3
"""Make lead-time GIF sanity checks from raw HRRR NetCDFs.

Reads one init-cycle file written by ``download_hrrr.py`` and animates the KW
bbox fields so you can eyeball that grids, units, and lead progression look
sane before training on them.

Example:

    python scripts/sanity_gif_hrrr.py \\
      /mnt/wato-drive/$USER/weatherloo-data/hrrr/2018/20180713/hrrr_20180713_t12z.nc

    python scripts/sanity_gif_hrrr.py --data-root /mnt/wato-drive/$USER/weatherloo-data \\
      --init 2018-07-13T12:00:00Z
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from weather_download_common import resolve_data_root

VAR_SPECS = {
    "t2m": {
        "label": "2 m temperature (°C)",
        "cmap": "RdYlBu_r",
        "transform": lambda a: a - 273.15,
    },
    "wind": {
        "label": "10 m wind speed (m/s)",
        "cmap": "viridis",
        "transform": None,  # derived from u10/v10
    },
    "tp": {
        "label": "accumulated precip (kg/m²)",
        "cmap": "Blues",
        "transform": lambda a: a,
    },
}


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def resolve_nc_path(args: argparse.Namespace) -> Path:
    if args.nc_path:
        return Path(args.nc_path).expanduser().resolve()
    if not args.init:
        raise SystemExit("Provide a NetCDF path or --init YYYY-MM-DDTHH:00:00Z")
    init = parse_utc(args.init)
    data_root = resolve_data_root(args.data_root)
    return (
        data_root
        / "hrrr"
        / f"{init:%Y}"
        / f"{init:%Y%m%d}"
        / f"hrrr_{init:%Y%m%d}_t{init.hour:02d}z.nc"
    )


def lead_hours(ds: xr.Dataset) -> np.ndarray:
    lead = ds["lead"]
    values = lead.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return (values / np.timedelta64(1, "h")).astype(int)
    units = str(lead.attrs.get("units", "")).lower()
    if "hour" in units:
        return np.asarray(values, dtype=int)
    return np.asarray(values, dtype=int)


def load_field(ds: xr.Dataset, name: str) -> np.ndarray:
    """Return (lead, y, x) float32 field in plotting units."""
    if name == "wind":
        u = np.asarray(ds["u10"].values, dtype=np.float32)
        v = np.asarray(ds["v10"].values, dtype=np.float32)
        return np.hypot(u, v)
    arr = np.asarray(ds[name].values, dtype=np.float32)
    transform = VAR_SPECS[name]["transform"]
    return transform(arr) if transform is not None else arr


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_var_gif(
    ds: xr.Dataset,
    name: str,
    out_path: Path,
    *,
    stride: int,
    fps: float,
    dpi: int,
) -> Path:
    leads = lead_hours(ds)
    indices = list(range(0, len(leads), max(1, stride)))
    field = load_field(ds, name)[indices]
    lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    lat = np.asarray(ds["latitude"].values, dtype=np.float32)
    init = ds.attrs.get("initialization", out_path.stem)

    # Fixed color limits across leads so evolution is comparable.
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        raise RuntimeError(f"{name}: all values are NaN")
    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if name == "tp":
        vmin = 0.0
        vmax = max(vmax, 1e-3)
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0

    spec = VAR_SPECS[name]
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=dpi)
    mesh = ax.pcolormesh(
        lon,
        lat,
        field[0],
        cmap=spec["cmap"],
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(spec["label"])
    title = ax.set_title("")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    pil_frames: list[Image.Image] = []

    def draw(i: int) -> None:
        mesh.set_array(field[i].ravel())
        lh = int(leads[indices[i]])
        title.set_text(f"HRRR {init}  |  {name}  |  lead f{lh:02d}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(buf[:, :, :3].copy()))

    for i in range(len(indices)):
        draw(i)
    plt.close(fig)

    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    save_gif(pil_frames, out_path, duration_ms)
    return out_path


def default_out_dir(nc_path: Path, data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root / "viz" / "hrrr"
    # Prefer sibling viz/ under the hrrr tree's data root if recognizable.
    parts = nc_path.parts
    if "hrrr" in parts:
        i = parts.index("hrrr")
        return Path(*parts[:i]) / "viz" / "hrrr"
    return nc_path.parent / "viz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc_path", nargs="?", default=None, help="Path to one hrrr_*.nc cycle file")
    parser.add_argument("--data-root", default=None, help="Used with --init (else $WEATHERLOO_DATA_ROOT)")
    parser.add_argument("--init", default=None, help="Init time YYYY-MM-DDTHH:00:00Z")
    parser.add_argument("--vars", default="t2m,wind,tp", help="Comma-separated: t2m,wind,tp")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth lead hour (default 1)")
    parser.add_argument("--fps", type=float, default=4.0, help="GIF playback speed")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--out-dir", default=None, help="Directory for GIFs (default: {data_root}/viz/hrrr)")
    args = parser.parse_args()

    nc_path = resolve_nc_path(args)
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    data_root = resolve_data_root(args.data_root) if (args.data_root or args.init) else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir(nc_path, data_root)

    # Avoid silent timedelta reinterpretations of lead's units="hours".
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    stem = nc_path.stem  # hrrr_YYYYMMDD_tHHz
    if not re.match(r"hrrr_\d{8}_t\d{2}z$", stem):
        stem = nc_path.stem

    requested = [v.strip() for v in args.vars.split(",") if v.strip()]
    unknown = [v for v in requested if v not in VAR_SPECS]
    if unknown:
        raise SystemExit(f"Unknown vars {unknown}; choose from {sorted(VAR_SPECS)}")

    print(f"Reading {nc_path}")
    print(f"  leads={ds.sizes.get('lead')} grid=({ds.sizes.get('y')},{ds.sizes.get('x')}) -> {out_dir}")

    written: list[Path] = []
    for name in requested:
        out = out_dir / f"{stem}_{name}.gif"
        path = render_var_gif(
            ds, name, out, stride=args.stride, fps=args.fps, dpi=args.dpi
        )
        written.append(path)
        print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")

    ds.close()
    print("Done.")


if __name__ == "__main__":
    main()
