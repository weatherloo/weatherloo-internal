# GEFS Ensemble Mean Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
GEFS (Global Ensemble Forecast System) is NOAA's **ensemble** weather model — it runs many slightly different forecasts to represent uncertainty. The **ensemble mean** is the average across those members at each grid point and lead time. For this epic we treat that mean as a **single deterministic forecast** (one value per variable), not a full probabilistic product.

## Why We're Including This
The GEFS mean is a **strong conventional NWP reference** that often outperforms a single deterministic GFS run because averaging reduces noise. It shows what a well-established operational ensemble approach achieves before AI models.

## Data Sources & Access
- **Forecast:** NOAA GEFS (AWS Open Data, NOMADS, or similar) — 00Z cycles for 2025; variables: 2m temperature and 10m wind (or u/v then `wind_speed = √(u² + v²)`).
- **Processing:** Compute member mean on the grid, then **interpolate** to Eric D. Soulis and CYYZ coordinates.
- **Ground truth:** ECCC hourly station observations.
- **Docs:** [NOAA GEFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast) — confirm exact archive path and latency for 2025.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/gefs_mean/`
---
## Notes & Open Questions
- How many ensemble members to include in the mean for v1?
- Storage/compute cost of full-year GEFS on AWS — subset vs on-demand extraction.
- Align GEFS valid times with ECCC observation times (UTC).
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
- `method_id`: **`gefs_mean`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `gefs_mean`
- Download GEFS ensemble members for 2025 **00Z** cycles; compute **member mean** on grid, then bilinear interpolate to station lat/lon.
- Document member count and handling of missing members.
- Wind: native speed or `sqrt(u² + v²)` — document choice.

