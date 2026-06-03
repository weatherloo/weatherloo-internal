# ECMWF AIFS Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
AIFS (Artificial Intelligence Forecasting System) is ECMWF's **machine-learning global weather model** trained to emulate and extend traditional physics-based forecasting. Like other AI weather models, it ingests the current atmospheric state and steps forward in time on a grid, producing deterministic fields such as temperature and wind.

## Why We're Including This
AIFS represents **state-of-the-art operational AI forecasting** from a major global center. It is a top-tier comparison point for judging whether a custom in-house forecast needs to match or beat leading ML-based global models.

## Data Sources & Access
- **Forecast:** ECMWF open data / dissemination APIs (license and product names TBD — e.g. AIFS open charts or research datasets); 00Z (or 00 UTC) initializations for 2025 where available.
- **Variables:** 2m temperature; 10m wind speed or u/v components per epic note.
- **Interpolation:** Gridded AIFS output → station lat/lon for both locations.
- **Ground truth:** ECCC hourly station observations.
- **Fallback:** If 2025 AIFS is not fully archived, document partial year or nearest alternative product.

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/ecmwf_aifs/`
---
## Notes & Open Questions
- **Data access & licensing** — confirm what ECMWF allows for internal benchmarking and repo storage of derived JSON metrics (not necessarily raw GRIB).
- Native grid and time step may differ from GFS; temporal interpolation to hourly ECCC obs.
- AI model availability lag for full 2025 reforecast.
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
- `method_id`: **`ecmwf_aifs`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `ecmwf_aifs`
- Use ECMWF open/research AIFS products for 2025 00Z inits; document exact dataset URL and license.
- Interpolate gridded output to both stations; align native timestep to 6-hourly leads (interpolate in time if needed).

