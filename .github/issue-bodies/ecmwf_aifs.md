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
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and optional NPZ export.
