# Model Output Statistics (MOS) Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
Model Output Statistics (MOS) is a **statistical post-processing** method that translates raw model output into a better forecast at a specific location. Traditionally it uses regression (sometimes multiple predictors and lead-time-specific equations) trained on past model-vs-observation errors. MOS is widely used in operational meteorology to correct systematic NWP biases at weather stations.

## Why We're Including This
MOS represents the **classical operational approach** to station-specific calibration. Comparing MOS to raw GFS and to AI models shows how much historical statistical correction buys relative to modern dynamical or ML forecasts.

## Data Sources & Access
- **Predictors:** Raw model fields (and possibly derived features: season, climatology, persistence) from GFS/GEFS at station locations via interpolation.
- **Training:** Multi-year record of paired forecasts and ECCC station observations; may use NOAA model archives or internal reforecast datasets.
- **Evaluation ground truth:** ECCC hourly observations, 2025, 00/06/12/18Z initializations.
- **Reference data:** [ERA5 on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) for climatology features or gap-filling if needed.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/mos/`
---
## Notes & Open Questions
- Scope v1: **univariate MOS per variable** vs multivariate system — document choice.
- MOS equations are often **season-stratified**; confirm training window and avoid 2025 leakage.
- Which base NWP and which archived cycle (00Z) are used for predictors?

---
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and optional NPZ export.
