#!/usr/bin/env python3
"""Export one HRRR run as a small gridded map for the forecast app.

Reads NOAA HRRR (48-hour runs) from the dynamical.org catalog, cuts a square of
native 3 km cells around a site, and writes 2 m temperature, 10 m wind speed and
direction, and precipitation rate every few hours of the run. Values are quantized
to uint8 and base64-encoded so the file can ship inside the app bundle.

Only the storage chunks that overlap the square are read (each chunk holds all 49
leads of 265 x 300 cells), so one export reads one to four chunks per variable.

Example (from the repository root):

    python pipelines/download/export_hrrr_map.py --init 2026-11-27T06:00 \
        --out forecast-app/src/data/hrrr-map.json
"""

import argparse
import base64
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CATALOG = "noaa-hrrr-forecast-48-hour"
SOULIS = (43.4668, -80.5164)  # Eric D. Soulis station (benchmarking-site/AGENTS.md)
PLACES = [  # labels drawn on the map; dropped when outside the square
    ("Waterloo", 43.4643, -80.5204),
    ("Guelph", 43.5448, -80.2482),
    ("Hamilton", 43.2557, -79.8711),
    ("Toronto", 43.6532, -79.3832),
    ("London", 42.9849, -81.2453),
    ("Brantford", 43.1394, -80.2644),
    ("Stratford", 43.3700, -80.9822),
    ("Goderich", 43.7436, -81.7140),
    ("Owen Sound", 44.5690, -80.9406),
    ("Lake Huron", 43.85, -81.95),
    ("Lake Erie", 42.62, -80.75),
    ("Lake Ontario", 43.42, -79.45),
]


def quantize(values, lo, hi, transform=None):
    """Map values in [lo, hi] (after an optional sqrt) onto 0..254; 255 marks missing."""
    v = np.sqrt(np.clip(values, 0, None)) if transform == "sqrt" else values
    a, b = (math.sqrt(lo), math.sqrt(hi)) if transform == "sqrt" else (lo, hi)
    q = np.round((np.clip(v, a, b) - a) / (b - a) * 254)
    q = np.where(np.isfinite(values), q, 255).astype(np.uint8)
    return {"min": lo, "max": hi, "transform": transform, "data": base64.b64encode(q.tobytes()).decode()}


def block_centres(values, every):
    """Every Nth cell, offset to the middle of each N x N block: [lead, y, x] -> [lead, y / N, x / N]."""
    start = every // 2
    return values[:, start::every, start::every]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init", required=True, help="Run time in UTC, e.g. 2026-11-27T06:00")
    parser.add_argument("--lat", type=float, default=SOULIS[0])
    parser.add_argument("--lon", type=float, default=SOULIS[1])
    parser.add_argument("--cells", type=int, default=80, help="Square side in 3 km cells")
    parser.add_argument("--step", type=int, default=3, help="Hours between frames")
    parser.add_argument("--arrow-every", type=int, default=4, help="Keep wind direction every N cells (for arrows)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import dynamical_catalog
    from pyproj import CRS, Transformer

    init = datetime.fromisoformat(args.init).replace(tzinfo=timezone.utc)
    ds = dynamical_catalog.open(CATALOG, chunks=None)
    crs = CRS.from_cf(ds.spatial_ref.attrs)
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    xs, ys = ds.x.values, ds.y.values
    sx, sy = to_grid.transform(args.lon, args.lat)
    xi, yi = int(np.argmin(np.abs(xs - sx))), int(np.argmin(np.abs(ys - sy)))
    half = args.cells // 2
    window = {"x": slice(xi - half, xi + half), "y": slice(yi - half, yi + half)}
    if window["x"].start < 0 or window["y"].start < 0 or window["x"].stop > len(xs) or window["y"].stop > len(ys):
        raise SystemExit("The square runs off the HRRR grid; move the site or use fewer cells.")

    leads = list(range(args.step, 49, args.step))
    lead_index = [int(np.argmin(np.abs(ds.lead_time.values - np.timedelta64(h, "h")))) for h in leads]
    pick = {"init_time": np.datetime64(init.replace(tzinfo=None)), "lead_time": ds.lead_time.values[lead_index]}
    names = ["temperature_2m", "wind_u_10m", "wind_v_10m", "precipitation_surface"]
    sub = ds[names].isel(window).sel(pick)
    print(f"Reading {len(names)} fields, {len(leads)} leads, {args.cells}x{args.cells} cells for {init:%Y-%m-%d %H} UTC...")
    t2m, u, v, prate = (sub[n].values.astype(np.float32) for n in names)  # [lead, y, x]

    # Rows run north to south so row 0 is the top of the map.
    flip = ys[window["y"]][0] < ys[window["y"]][-1]
    if flip:
        t2m, u, v, prate = (a[:, ::-1, :] for a in (t2m, u, v, prate))
    wx, wy = xs[window["x"]], ys[window["y"]][::-1] if flip else ys[window["y"]]

    speed = np.hypot(u, v)  # m/s, grid-relative components; speed is unaffected
    # Grid-relative wind direction: HRRR's u/v follow the grid axes, which is how the map is drawn.
    toward = np.degrees(np.arctan2(u, v)) % 360  # direction the wind blows toward, clockwise from grid up
    mm_per_hour = prate * 3600.0  # kg m-2 s-1 == mm/s

    def position(lat, lon):
        px, py = to_grid.transform(lon, lat)
        col = (px - wx[0]) / (wx[-1] - wx[0]) * (len(wx) - 1)
        row = (py - wy[0]) / (wy[-1] - wy[0]) * (len(wy) - 1)
        return col, row

    # Angle between grid "up" and true north at the site, for the north arrow.
    c0, r0 = position(args.lat, args.lon)
    c1, r1 = position(args.lat + 0.5, args.lon)
    north_deg = math.degrees(math.atan2(c1 - c0, -(r1 - r0)))

    places = []
    for name, lat, lon in PLACES:
        col, row = position(lat, lon)
        if 0 <= col <= args.cells - 1 and 0 <= row <= args.cells - 1:
            places.append({"name": name, "col": round(col, 2), "row": round(row, 2), "water": name.startswith("Lake ")})

    out = {
        "schema_version": 1,
        "source": f"NOAA HRRR via dynamical.org catalog {CATALOG}",
        "init_time": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leads": leads,
        "grid": {"cells": args.cells, "spacing_km": 3, "north_deg": round(north_deg, 2)},
        "site": {"lat": args.lat, "lon": args.lon, "col": round(c0, 2), "row": round(r0, 2)},
        "places": places,
        "fields": {
            "t2m": {"unit": "degC", **quantize(t2m, math.floor(np.nanmin(t2m)), math.ceil(np.nanmax(t2m)))},
            "wind_speed": {"unit": "m/s", **quantize(speed, 0.0, math.ceil(np.nanmax(speed)))},
            # Sampled at cell centres (k + 0.5) * every - 0.5 along each axis.
            "wind_toward": {"unit": "deg_grid", "every": args.arrow_every,
                            **quantize(block_centres(toward, args.arrow_every), 0.0, 360.0)},
            "precip_rate": {"unit": "mm/h", **quantize(mm_per_hour, 0.0, max(1.0, math.ceil(np.nanmax(mm_per_hour))), "sqrt")},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    missing = {k: int(np.isnan(a).sum()) for k, a in {"t2m": t2m, "speed": speed, "precip": mm_per_hour}.items()}
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB). Missing values: {missing}")
    print(f"t2m {np.nanmin(t2m):.1f}..{np.nanmax(t2m):.1f} degC, wind {np.nanmax(speed):.1f} m/s max, "
          f"precip {np.nanmax(mm_per_hour):.2f} mm/h max, north {north_deg:.1f} deg from grid up")


if __name__ == "__main__":
    main()
