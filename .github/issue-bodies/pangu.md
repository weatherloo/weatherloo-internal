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
- `method_id`: **`pangu`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `pangu`
- Run or obtain Pangu-Weather output for 2025 00Z inits; document inference setup and variable mapping.
- Interpolate to stations; respect license for derived JSON in repo.

