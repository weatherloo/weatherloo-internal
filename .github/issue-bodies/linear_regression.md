# Linear Regression Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
Linear regression post-processing learns a **straight-line relationship** between a model's forecast and what the station actually observed in the past — e.g. `corrected = a + b × raw_forecast`. It can correct both bias and scale errors. Like MOS, it is a statistical layer on top of a physical model, but with a simpler functional form.

## Why We're Including This
This is a **standard statistical baseline** between mean bias correction and full MOS. It shows whether a slightly richer correction (slope + intercept per lead time) materially improves scores over subtracting a constant bias.

## Data Sources & Access
- **Base forecasts:** Gridded NWP (likely GFS/GEFS) for 2025 initializations; interpolate to Eric D. Soulis and CYYZ.
- **Training data:** Historical pairs of (forecast, observation) per variable, station, and lead time from ECCC stations and archived model runs.
- **Ground truth:** ECCC hourly observations for 2025 scoring.
- **External:** NOAA GFS/GEFS archives or internal pipeline outputs; ERA5 via [GCP public ERA5 zarr](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) for development if model archives are not ready.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/linear_regression/`
---
## Notes & Open Questions
- Fit **separate regressions per lead time** (and likely per station); document feature set if multiple predictors are used.
- Wind speed from u/v: regression on speed vs on components — stay consistent with epic variable definition.
- Minimum sample size for stable coefficients; handle seasonality (month-stratified fits?).
