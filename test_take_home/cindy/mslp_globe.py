#!/usr/bin/env python3
"""
Build an interactive Plotly globe of mean sea level pressure (MSLP) from
ARCO ERA5 on GCS for the 24 h UTC window 2021-05-01 00Z–2021-05-01 18Z
(four 6-hourly snapshots; dataset cadence is 6 h).

Data: gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr
Variable: mean_sea_level_pressure (ERA5 short name msl), units Pa → hPa in the plot.

Plotly 6 removed Contourgeo; this uses Scattergeo on a subsampled lat–lon grid for a
continuous field look with a modest HTML size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gcsfs
import numpy as np
import plotly.graph_objects as go
import xarray as xr


ARCO_ZARR = "gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr"
VAR = "mean_sea_level_pressure"


def _center_longitude(lon: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll 0..360° longitudes to -180..180° with matching z[..., lon]."""
    half = len(lon) // 2
    lon_c = np.concatenate((lon[half:] - 360.0, lon[:half]))
    z_c = np.concatenate((z[..., half:], z[..., :half]), axis=-1)
    return lon_c, z_c


def load_mslp_hpa(
    stride: int,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    fs = gcsfs.GCSFileSystem(token="anon")
    store = fs.get_mapper(ARCO_ZARR)
    ds = xr.open_zarr(store, consolidated=True, chunks={})

    da = (
        ds[VAR]
        .sel(time=slice("2021-05-01", "2021-05-01T23:59:59"))
        .isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
        .load()
    )
    if da.sizes["time"] == 0:
        raise RuntimeError("No timesteps in selection; check date and dataset.")

    times = da.time.values
    labels = [np.datetime_as_string(t, unit="s") + " UTC" for t in times]

    lat = da.latitude.values.astype(float)
    lon = da.longitude.values.astype(float)
    # Pa → hPa; latitude ERA5 is 90 → -90; plot geo likes increasing lat
    zh = (da.values / 100.0).astype(np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        zh = zh[:, ::-1, :]

    lon_c, zh = _center_longitude(lon, zh)
    lon2d, lat2d = np.meshgrid(lon_c, lat)
    lon_r = lon2d.ravel()
    lat_r = lat2d.ravel()
    return zh, labels, lon_r, lat_r


def build_figure(
    lon_r: np.ndarray,
    lat_r: np.ndarray,
    zh: np.ndarray,
    labels: list[str],
) -> go.Figure:
    zmin = float(np.nanmin(zh))
    zmax = float(np.nanmax(zh))

    def scatter_k(k: int, *, showscale: bool) -> go.Scattergeo:
        vals = zh[k].ravel()
        return go.Scattergeo(
            lon=lon_r,
            lat=lat_r,
            mode="markers",
            marker=dict(
                size=4,
                color=vals,
                colorscale="RdYlBu_r",
                cmin=zmin,
                cmax=zmax,
                showscale=showscale,
                colorbar=dict(title="MSLP (hPa)"),
                opacity=0.92,
            ),
            customdata=vals,
            hovertemplate="MSLP: %{customdata:.1f} hPa<br>lat %{lat:.1f}, lon %{lon:.1f}<extra></extra>",
        )

    frames = [
        go.Frame(data=[scatter_k(k, showscale=(k == 0))], name=str(k), traces=[0])
        for k in range(len(labels))
    ]

    fig = go.Figure(data=[scatter_k(0, showscale=True)], frames=frames)

    slider_steps = [
        {
            "args": [
                [str(k)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": labels[k],
            "method": "animate",
        }
        for k in range(len(labels))
    ]

    fig.update_layout(
        title=dict(
            text="Mean sea level pressure (MSLP) — ERA5 ARCO, 24 h on 2021-05-01 UTC",
            x=0.5,
        ),
        height=820,
        margin=dict(l=0, r=0, t=60, b=0),
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "currentvalue": {"prefix": "Valid: "},
                "pad": {"t": 20},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 600, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        geo=dict(
            projection_type="orthographic",
            projection_rotation=dict(lon=0, lat=15),
            showland=True,
            landcolor="rgb(220, 220, 210)",
            showocean=True,
            oceancolor="rgb(200, 220, 240)",
            showlakes=True,
            lakecolor="rgb(200, 220, 240)",
            showcountries=True,
            countrywidth=0.3,
            lataxis=dict(showgrid=True, gridwidth=0.3),
            lonaxis=dict(showgrid=True, gridwidth=0.3),
        ),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="MSLP interactive globe (Plotly HTML).")
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Subsampling along lat/lon (4 ≈ 181×180 grid of markers). Smaller = sharper, larger file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mslp_globe_may1_2021.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    zh, labels, lon_r, lat_r = load_mslp_hpa(stride=max(1, args.stride))
    fig = build_figure(lon_r, lat_r, zh, labels)
    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out} ({len(labels)} frames, markers per frame {len(lon_r)})")


if __name__ == "__main__":
    main()
