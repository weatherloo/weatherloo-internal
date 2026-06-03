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
- `method_id`: **`linear_regression`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `linear_regression`
- **Base forecasts:** `gfs_interpolated`
- Train regression on pre-2025 paired (forecast, observation) samples; separate fits per station, variable, and lead time unless documented otherwise.
- No 2025 data in training.

