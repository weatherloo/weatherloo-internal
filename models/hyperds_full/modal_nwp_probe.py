"""One-off read-only probe of the HRRR and AIFS catalogs used as forecast inputs."""

import json

import modal

app = modal.App("weatherloo-nwp-probe")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0", "numpy")


@app.function(image=image, cpu=2, memory=8192, timeout=20*60)
def probe(catalog_id: str):
    import dynamical_catalog
    import numpy as np

    ds = dynamical_catalog.open(catalog_id, chunks=None)
    report = {"catalog": catalog_id, "sizes": dict(ds.sizes),
              "init_first": str(ds.init_time.values[0]), "init_last": str(ds.init_time.values[-1]),
              "lead_first_last": [str(ds.lead_time.values[0]), str(ds.lead_time.values[-1])],
              "lead_values_head": [str(v) for v in ds.lead_time.values[:12]],
              "coords": list(ds.coords), "data_vars": {}}
    for name, var in ds.data_vars.items():
        report["data_vars"][name] = {"dims": var.dims, "units": var.attrs.get("units"),
                                     "long_name": var.attrs.get("long_name"),
                                     "chunks": str(var.encoding.get("chunks")),
                                     "step_type": var.attrs.get("step_type")}
    if "spatial_ref" in ds:
        report["spatial_ref"] = {k: v for k, v in ds.spatial_ref.attrs.items() if k in ("crs_wkt", "grid_mapping_name")}
    if "latitude" in ds.coords and ds.latitude.ndim == 2:
        lat, lon = ds.latitude.values, ds.longitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        inside = (lat >= 39.875) & (lat < 47.875) & (lon >= -88.125) & (lon < -72.125)
        cells = set(zip(((lat[inside] - 39.875) // .25).astype(int), ((lon[inside] + 88.125) // .25).astype(int)))
        ys, xs = np.nonzero(inside)
        report["domain_coverage"] = {"covered_cells": len(cells), "required_cells": 32 * 64,
                                     "y_range": [int(ys.min()), int(ys.max())], "x_range": [int(xs.min()), int(xs.max())],
                                     "native_points": int(inside.sum())}
        missing = [(r, c) for r in range(32) for c in range(64) if (r, c) not in cells]
        report["domain_coverage"]["missing_cells_sample"] = missing[:20]
        report["domain_coverage"]["missing_rows"] = sorted({r for r, _ in missing})
    else:
        report["lat_head"] = ds.latitude.values[:3].tolist()
        report["lon_head"] = ds.longitude.values[:3].tolist()
    # Year availability: count init_times per year with 00/06/12/18 hours.
    times = ds.init_time.values.astype("datetime64[h]")
    years = times.astype("datetime64[Y]").astype(int) + 1970
    report["inits_per_year"] = {int(y): int((years == y).sum()) for y in np.unique(years)}
    hours = (times.astype(int) % 24)
    report["init_hours"] = sorted({int(h) for h in np.unique(hours)})
    # One small read to confirm values and missing leads.
    sample = ds["temperature_2m"].sel(init_time=np.datetime64("2024-07-01T00")).isel(
        {d: slice(0, 4) for d in ds["temperature_2m"].dims if d not in ("init_time", "lead_time")}).values
    report["sample_t2m_finite_by_lead"] = np.isfinite(sample.reshape(sample.shape[0], -1)).all(axis=1).tolist()[:60]
    report["sample_t2m_mean"] = float(np.nanmean(sample))
    for name in ("precipitation_surface", "pressure_surface"):
        if name in ds:
            v = ds[name].sel(init_time=np.datetime64("2024-07-01T00")).isel(
                {d: slice(0, 4) for d in ds[name].dims if d not in ("init_time", "lead_time")}).values
            report[f"sample_{name}"] = {"finite_by_lead": np.isfinite(v.reshape(v.shape[0], -1)).all(axis=1).tolist()[:10],
                                        "mean": float(np.nanmean(v))}
    print(json.dumps(report, indent=1, default=str), flush=True)
    return report


@app.local_entrypoint()
def main():
    for result in probe.map(["noaa-hrrr-forecast-48-hour", "ecmwf-aifs-single-forecast"]):
        print(json.dumps(result, indent=1, default=str))


@app.function(image=image, cpu=2, memory=8192, timeout=20*60)
def hrrr_leads():
    """Finite leads over the Waterloo crop for sample HRRR cycles across versions."""
    import dynamical_catalog
    import numpy as np

    ds = dynamical_catalog.open("noaa-hrrr-forecast-48-hour", chunks=None)
    report = {}
    for stamp in ("2018-07-14T00", "2018-07-14T06", "2019-03-01T00", "2019-07-03T12", "2020-06-01T00",
                  "2020-11-30T00", "2020-12-03T00", "2021-06-01T00"):
        sub = ds.temperature_2m.sel(init_time=np.datetime64(stamp)).isel(y=slice(93, 465), x=slice(1136, 1613)).values
        finite = np.isfinite(sub.reshape(sub.shape[0], -1)).mean(axis=1)
        length = ds.ingested_forecast_length.sel(init_time=np.datetime64(stamp)).values
        report[stamp] = {"ingested_forecast_length_h": str(length.astype("timedelta64[h]")),
                         "full_leads": [int(i) for i in np.flatnonzero(finite == 1)][-3:],
                         "n_full": int((finite == 1).sum()), "partial": {int(i): float(f) for i, f in enumerate(finite) if 0 < f < 1}}
    print(json.dumps(report, indent=1), flush=True)
    return report
