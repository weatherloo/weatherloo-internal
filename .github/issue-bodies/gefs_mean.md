# GEFS Ensemble Mean Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
GEFS (Global Ensemble Forecast System) is NOAA's **ensemble** weather model — it runs many slightly different forecasts to represent uncertainty. The **ensemble mean** is the average across those members at each grid point and lead time. For this epic we treat that mean as a **single deterministic forecast** (one value per variable), not a full probabilistic product.

## Why We're Including This
The GEFS mean is a **strong conventional NWP reference** that often outperforms a single deterministic GFS run because averaging reduces noise. It shows what a well-established operational ensemble approach achieves before AI models.

## Data Sources & Access
- **Forecast:** NOAA GEFS (AWS Open Data, NOMADS, or similar) — 00/06/12/18Z cycles for 2025; variables: 2m temperature and 10m wind (or u/v then `wind_speed = √(u² + v²)`).
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
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and optional NPZ export.
