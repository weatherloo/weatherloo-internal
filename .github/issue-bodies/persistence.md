# Persistence Benchmarking
**Parent epic:** #1  
**Type:** Benchmarking method  
---
## Background
A persistence forecast assumes **the weather will stay the same as the most recent observation** at the station. For a forecast initialized at 00/06/12/18Z, the prediction for lead time 6h is essentially "whatever the station observed around initialization time," carried forward. It is the simplest dynamical baseline and requires no weather model.

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
- Define exactly which observation hour anchors persistence for 00/06/12/18Z initializations (e.g. 00Z obs vs previous hour).
- Persistence error usually **grows with lead time**; if not, check timestamp alignment between forecast and verification.
- Missing station obs: gap-fill strategy for 1460 initializations in 2025.

---
## Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, init cycles (00/06/12/18Z), metrics, definition of done, and optional NPZ export.
