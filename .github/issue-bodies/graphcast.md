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
- **Compute:** running GraphCast for 365×12 lead times is non-trivial — plan batch inference vs subset.
- Confirm which GraphCast version (operational vs research) and which output variables map to 2m T and 10m wind.
- Lead times in epic (6h steps to 72h) must match model output cadence (may need interpolation in time).
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
- `method_id`: **`graphcast`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `graphcast`
- Run or obtain GraphCast deterministic output for 2025 inits; [ERA5 zarr on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr) for ICs if needed.
- Interpolate to stations; document GraphCast version and variable mapping to `t2m` / `wind_speed`.

