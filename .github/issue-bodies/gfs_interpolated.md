# GFS Interpolated at Station Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
GFS (Global Forecast System) is NOAA's flagship **deterministic global weather model**. It produces forecasts on a latitude-longitude grid. "Interpolated at station" means we **estimate the model value at the exact station coordinates** from the surrounding grid points (bilinear or similar), then compare that to station observations — the standard way to score global models at point locations.

## Why We're Including This
GFS is the **primary operational NWP benchmark** in North America. Raw interpolated GFS (without MOS) shows baseline dynamical model skill and is the foundation for several post-processing methods in this epic.

## Data Sources & Access
- **Forecast:** NOAA GFS 00/06/12/18Z initializations for 2025 (e.g. [NOAA GFS on AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) or NOMADS); fields: 2m temperature, 10m u/v or wind speed.
- **Interpolation:** To station lat/lon for `cyyz` and `eric_d_soulis` (see epic JSON coordinates).
- **Ground truth:** ECCC hourly station observations.
- **Method id in JSON:** `gfs_interpolated` (per epic example).

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/gfs_interpolated/`
---
## Notes & Open Questions
- Confirm GFS cycle and grid (0.25° vs 0.5°) for the full year pipeline.
- Wind: use native 10m wind speed if available vs derive from u/v — document choice.
- Sanity-check RMSE increases and ACC decreases with lead time.

---
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and NPZ export.
