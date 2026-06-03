# Mean Bias Correction Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
Mean bias correction adjusts a raw model forecast by removing a **fixed average error** learned from past data. For example, if a model consistently predicts 1°C too warm at CYYZ in winter, we subtract that bias from future forecasts. It is a simple post-processing step on top of an existing model, not a full weather model itself.

## Why We're Including This
This is a **classical, cheap post-processing baseline** that shows how much skill comes from fixing systematic errors alone — without a more complex statistical model. It helps separate "the model is biased" from "the model has no skill."

## Data Sources & Access
- **Base forecasts:** A parent gridded model (likely GFS or GEFS mean — confirm in implementation) for 2025, 00Z daily, all lead times; interpolate to station lat/lon before correction.
- **Training / bias estimates:** Historical paired samples of forecast vs ECCC station truth (e.g. prior season or rolling window) to compute mean bias per station, variable, and lead time.
- **Ground truth:** ECCC hourly station observations.
- **Reanalysis / archives:** [ERA5 (Earthmover/GCP)](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) optional if bias is computed against reanalysis instead of stations for development.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/mean_bias_correction/`
---
## Notes & Open Questions
- **Train/test leakage:** bias terms must be estimated only on data **before** the 2025 evaluation period (or use cross-validation).
- Apply correction in physical units; for wind derived from u/v, decide whether bias is applied to speed or components.
- Document which base model feed this correction uses.
