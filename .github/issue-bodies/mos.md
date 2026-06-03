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
- **Evaluation ground truth:** ECCC hourly observations, 2025, 00Z initializations.
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
- `method_id`: **`mos`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `mos`
- **Base forecasts:** `gfs_interpolated` (plus optional predictors: season, persistence, climatology — document in Notes).
- Train MOS on pre-2025 data; apply to 2025 `gfs_interpolated` forecasts at stations.
- Document whether univariate or multivariate MOS.

