# GraphCast Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
GraphCast is Google DeepMind's **graph neural network weather model**, published in 2023. It predicts atmospheric variables on a global grid by learning from decades of reanalysis (e.g. ERA5). It produces fast deterministic forecasts that have been shown to rival or beat traditional NWP on many global metrics.

## Why We're Including This
GraphCast is a **landmark AI weather model** and a common reference in research comparisons. Including it shows how our station-level skill scores stack up against a widely cited ML baseline.

## Data Sources & Access
- **Forecast:** GraphCast open weights / WeatherLab / community inference on ERA5 or operational initial conditions; or pre-generated forecast archives if available for 2025.
- **Initial conditions:** May require ERA5 or similar — [ERA5 GCP zarr](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) via Earthmover-style tooling.
- **Interpolation:** GraphCast grid → Eric D. Soulis and CYYZ.
- **Ground truth:** ECCC hourly station observations.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/graphcast/`
---
## Notes & Open Questions
- **Compute:** running GraphCast for 1460×12 lead times is non-trivial — plan batch inference vs subset.
- Confirm which GraphCast version (operational vs research) and which output variables map to 2m T and 10m wind.
- Lead times in epic (6h steps to 72h) must match model output cadence (may need interpolation in time).

---
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and NPZ export.
