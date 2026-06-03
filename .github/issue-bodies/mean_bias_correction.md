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
---
## Implementation spec (for coding agents)

**Read first:** [`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md) in this repo.

### Repo paths
| What | Path |
|------|------|
| Ground truth | `benchmarking-site/data/observations/<station_id>/observations_6h_2025.json` |
| Your output | `benchmarking-site/data/<method_id>/` |
| Sample JSON | `benchmarking-site/data/gfs_interpolated/gfs_interpolated_sample.json` |
| Dashboard | `benchmarking-site/` → `python3 -m http.server 8080` |

### Station IDs (exact)
| `station_id` | Lat | Lon |
|--------------|-----|-----|
| `cyyz` | 43.6777 | -79.6248 |
| `eric_d_soulis` | 43.4668 | -80.5164 |

### Time (UTC)
- 365 inits: daily `2025-*-**T00:00:00Z`
- Lead times: `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]`
- Verify at `initialization + lead_hours` against `observations_6h_2025.json` `valid_time`

### Output
- `method_id`: **`mean_bias_correction`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `mean_bias_correction`
- **Base forecasts:** `gfs_interpolated` (run that method first or consume its pipeline output).
- Fit bias on data **before 2025** only (no leakage into evaluation year); per station, variable, and lead time.
- Apply bias correction in physical units before computing metrics vs 2025 truth.

