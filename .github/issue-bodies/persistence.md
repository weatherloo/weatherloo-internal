# Persistence Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
A persistence forecast assumes **the weather will stay the same as the most recent observation** at the station. For a forecast initialized at 00Z, the prediction for lead time 6h is essentially "whatever the station observed around initialization time," carried forward. It is the simplest dynamical baseline and requires no weather model.

## Why We're Including This
This is a **strong short-lead baseline**, especially for temperature and wind when systems change slowly. Many operational models only beat persistence after the first several hours. It sets a realistic bar for near-term skill.

## Data Sources & Access
- **Forecast:** ECCC hourly station observations at Eric D. Soulis and CYYZ — use the observation valid at (or immediately before) each 00Z initialization and hold constant for all lead times (or apply a clear alternative rule and document it).
- **Ground truth (for scoring):** Same ECCC hourly observations at valid times for each lead time.
- **Interpolation:** Not required (station-native).

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/persistence/`
---
## Notes & Open Questions
- Define exactly which observation hour anchors persistence for 00Z initializations (e.g. 00Z obs vs previous hour).
- Persistence error usually **grows with lead time**; if not, check timestamp alignment between forecast and verification.
- Missing station obs: gap-fill strategy for 365 initializations in 2025.
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
- `method_id`: **`persistence`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `persistence`
- **Forecast:** for each init, set all lead times to the observation at **`valid_time == initialization`** from `observations_6h_2025.json` (exact 00:00 UTC row).
- No weather model download required; only ground-truth files + metric computation.

