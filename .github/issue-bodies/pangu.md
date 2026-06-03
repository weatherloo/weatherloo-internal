# Pangu-Weather Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
Pangu-Weather is Huawei's **deep-learning global weather forecasting system** (3D neural networks), released as open research models that predict multiple atmospheric variables on a grid. Like GraphCast and AIFS, it is trained on historical reanalysis and run forward from an initial state to produce deterministic forecasts.

## Why We're Including This
Pangu is another **major published AI weather model**, often compared directly to GraphCast and GFS. It completes the picture of "leading ML global models" alongside AIFS and GraphCast in our internal benchmark.

## Data Sources & Access
- **Forecast:** Pangu-Weather open model weights and inference code (GitHub / community ports); initial conditions from ERA5 or operational analyses.
- **Reanalysis / ICs:** [ERA5 on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) or equivalent.
- **Interpolation:** Model grid → station coordinates for both locations.
- **Ground truth:** ECCC hourly station observations for 2025.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/pangu/`
---
## Notes & Open Questions
- GPU requirements and runtime for full-year 00Z benchmarks.
- Variable names / pressure levels for 2m T and 10m wind in Pangu outputs — mapping layer documented in pipeline.
- Confirm license for internal use and whether derived metrics JSON can be committed to the repo.
