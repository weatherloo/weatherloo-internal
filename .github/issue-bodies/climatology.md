# Climatology Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
A climatology forecast ignores today's weather and instead predicts what is *typical* for that calendar day and time at the station (e.g. "the average 2 pm temperature on January 15 at Toronto Pearson"). It is built from many years of historical observations or reanalysis, not from a live weather model run.

## Why We're Including This
This is a **lower bound baseline**: any useful forecast should beat "guess the long-term average." If a sophisticated model cannot outperform climatology, it is not adding skill for that variable and location.

## Data Sources & Access
- **Forecast / climatology field:** Derived from a long historical record (e.g. ECCC station archives and/or gridded reanalysis such as [ERA5 on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) via Earthmover-style access). Climatology is typically computed per station, day-of-year, and hour (00Z initialization aligns with 00 UTC climatology).
- **Ground truth (for scoring):** ECCC hourly station observations (same as epic).
- **Interpolation:** Not required if climatology is computed directly at station coordinates from station history; if built from a grid, interpolate to each station lat/lon.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/climatology/` (one file per initialization, 365 for 2025)
---
## Notes & Open Questions
- Climatology is often **lead-time invariant** (same value for 6h and 72h); RMSE/MAE may still vary slightly with verification timing — document the convention used.
- Wind climatology from u/v components: confirm whether station climatology uses observed wind speed or derived speed from components.
- Decide minimum years of record required for stable monthly/daily normals.
