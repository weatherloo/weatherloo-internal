# GFS Past-Hour Analysis Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
The GFS **analysis** is the model's best estimate of the current atmospheric state at initialization time — it blends observations with the model (data assimilation). "Past-hour analysis" here means using that **analysis field** (effectively "what GFS thinks is happening now") as the forecast, rather than a forward-propagated prediction. It is useful mainly at **very short lead times** and as a reference for how well the model state matches reality at cycle time.

## Why We're Including This
This method isolates **initial-condition / analysis error** vs propagation error in the full GFS forecast. It is a diagnostic benchmark: if analysis is already far from the station, longer-lead GFS errors will suffer.

## Data Sources & Access
- **Forecast:** GFS analysis products from the same 00Z cycles as other GFS tasks (NOAA open data / NOMADS); extract 2m temperature and 10m wind (or u/v) at valid times corresponding to each lead time's verification, or document if only analysis-at-init is used.
- **Interpolation:** To Eric D. Soulis and CYYZ station coordinates.
- **Ground truth:** ECCC hourly observations.
- **Clarify:** Exact definition of "past-hour analysis" in the pipeline (analysis at f00 vs analysis valid at lead-time valid time).

## Acceptance Criteria
(For coding agents)
- Benchmarking data for **2m temperature** and **10m wind speed** is displayed on the site for both locations
- Metrics are computed for all 12 lead times (6h–72h)
- Output JSON matches the epic schema under `data/gfs_analysis/`
---
## Notes & Open Questions
- **Operational definition must be written down** — epic name is ambiguous between analysis-at-initialization vs time-lagged analysis fields.
- Expect **best skill at short leads** if using analysis near valid time; long leads may not behave like a normal forecast.
- Same interpolation and wind-speed derivation rules as `gfs_interpolated`.
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
- `method_id`: **`gfs_analysis`**
- 365 files under `data/<method_id>/`; see AGENTS.md for schema and done checklist

### Method-specific

- **`method_id`:** `gfs_analysis`
- **Definition (pick one, document in PR):** (A) GFS analysis at initialization time (f00) held/applied per lead for scoring, or (B) analysis field valid at each lead time's verification instant.
- Same grid interpolation to stations as `gfs_interpolated`.

