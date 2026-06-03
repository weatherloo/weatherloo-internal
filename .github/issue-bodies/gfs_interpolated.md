# GFS Interpolated at Station Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
GFS (Global Forecast System) is NOAA's flagship **deterministic global weather model**. It produces forecasts on a latitude-longitude grid. "Interpolated at station" means we **estimate the model value at the exact station coordinates** from the surrounding grid points (bilinear or similar), then compare that to station observations — the standard way to score global models at point locations.

## Why We're Including This
GFS is the **primary operational NWP benchmark** in North America. Raw interpolated GFS (without MOS) shows baseline dynamical model skill and is the foundation for several post-processing methods in this epic.

## Data Sources & Access
- **Forecast:** NOAA GFS 00Z initializations for 2025 (e.g. [NOAA GFS on AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) or NOMADS); fields: 2m temperature, 10m u/v or wind speed.
- **Interpolation:** To station lat/lon for `cyyz` and `eric_d_soulis` (see epic JSON coordinates).
- **Ground truth:** ECCC hourly station observations.
- **Method id in JSON:** `gfs_interpolated` (per epic example).

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/gfs_interpolated/`
---
## Notes & Open Questions
- Confirm GFS cycle and grid (0.25° vs 0.5°) for the full year pipeline.
- Wind: use native 10m wind speed if available vs derive from u/v — document choice.
- Sanity-check RMSE increases and ACC decreases with lead time.
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
- `method_id`: **`gfs_interpolated`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `gfs_interpolated`
- NOAA GFS **00Z** initializations, 2025; bilinear interpolation to station coordinates.
- **Copy structure from** `benchmarking-site/data/gfs_interpolated/gfs_interpolated_sample.json`.
- [AWS GFS Open Data](https://registry.opendata.aws/noaa-gfs-bdp-pds/) or NOMADS.

