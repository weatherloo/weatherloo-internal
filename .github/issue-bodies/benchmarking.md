# Benchmarking Website (Epic)
**GitHub issue:** [#1 — Benchmarking Website](https://github.com/weatherloo/weatherloo-internal/issues/1)

## Epic: Interactive Weather Forecast Benchmarking Site

### Overview

We're building an internal website to compare how accurately different weather forecasting methods perform at two specific locations. The goal is to understand how well our custom forecast needs to perform to be competitive and we'll use this site as our reference as the project develops.

---

### Key Terms

| Term | Definition |
|---|---|
| **Lead time** | How far into the future a forecast is predicting. A 6h lead time means "what the weather will be 6 hours from now." |
| **Initialization** | The moment a forecast starts from. We initialize forecasts at **00, 06, 12, and 18 UTC** each day (00Z, 06Z, 12Z, 18Z — "Z" is Zulu/UTC). |
| **Deterministic forecast** | A forecast that gives one specific predicted value (e.g. "12°C"). As opposed to probabilistic/ensemble forecasts, which give a range of possible outcomes — those are out of scope for now. |
| **Gridded model output** | Weather models produce data across a geographic grid (like a mesh of squares covering a map), not at specific points. To compare model output to a station's exact location, we **bilinearly interpolate** — estimating the value at the station's coordinates from the four surrounding grid points. |
| **Analysis** | The model's best estimate of the atmosphere **at initialization** (after data assimilation), often GRIB `f000` — not a forward forecast. For **analysis** benchmark methods (`gfs_analysis`, `hrdps_analysis`, etc.) we **bilinearly interpolate** that **single init-time value** to the station and use it as the "forecast" at **every** lead time. Ground truth is still the observation at each lead's **valid time** (`initialization + lead_hours`). Error should **grow with lead time** (RMSE/MAE increase, ACC decrease) because the analysis only describes the state at init, not hours later. |
| **Skill score / accuracy metrics** | Numbers that describe how accurate a forecast is. We use four: RMSE, MAE, Bias, and ACC (defined in the Metrics section below). |

---

### The Website

An internal interactive site with a map showing two station locations. Clicking a location displays accuracy metric charts for each forecasting method, broken down by variable and lead time. See gfs-interpolated as an example.

This is version 1. More locations, variables, and methods will be added as the project grows.

---

### Scope

**Locations**
- Eric D. Soulis station
- Toronto Pearson International Airport (CYYZ)

**Variables**
- 2m temperature
- 10m wind speed

> **Note:** Some forecast sources give wind as two directional components (u = east-west, v = north-south) rather than a single wind speed value. If that's the case, wind speed can be derived as: `wind_speed = √(u² + v²)`

**Grid-to-station interpolation**
- All gridded methods use **bilinear interpolation** in latitude/longitude at each station's coordinates.
- **t2m:** bilinear interp on the 2 m temperature grid.
- **Wind:** bilinear interp on **u** and **v** separately at 10 m, then `wind_speed = √(u² + v²)` — do not interpolate speed directly.
- Reference: [`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md) and `data/gfs_interpolated/compute_benchmark.py`.

**Time period**
- Full year 2025, forecasts initialized at **00, 06, 12, 18 UTC** every day
- **1460** total initializations (365 days × 4 cycles)
- Save each run's output to the repo; the dashboard displays the average across all 1460 runs

**Lead times**

`6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h, 54h, 60h, 66h, 72h`

For example: the forecast initialized on Jan 1, 2025 **06Z** produces separate error calculations for the 6h, 12h, … 72h forecasts (valid at 12Z, 18Z, …).

**Ground truth**

**Incomplete coverage is OK:** In practice you may not have data for every init cycle (00/06/12/18Z), every lead time, or every ground-truth valid time — model archive gaps, download limits, station outages, and work-in-progress runs are expected. Store `null` in JSON metric arrays where a comparison is not possible; the site averages over whatever loads. Partial output is acceptable.

Actual hourly station observations from ECCC. Since model output is on a grid rather than at specific points, it must be **bilinearly interpolated** to each station's lat/lon coordinates before the error is calculated.

**Forecast type:** Deterministic only. Probabilistic and ensemble forecasts are out of scope for now.

---

### Data Format

Each forecasting method produces a JSON file in the format below. This is what the website reads to display charts.

```json
{
  "method": "gfs_interpolated",
  "initialization": "2025-01-15T06:00:00Z",
  "locations": {
    "cyyz": {
      "lat": 43.68,
      "lon": -79.63,
      "variables": {
        "t2m": {
          "lead_times_hours": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],
          "rmse": [0.8, 1.0, 1.2, "..."],
          "mae":  [0.6, 0.8, 1.0, "..."],
          "bias": [0.1, 0.1, 0.2, "..."],
          "acc":  [0.95, 0.91, 0.87, "..."]
        },
        "wind_speed": { "...": "..." }
      }
    },
    "eric_d_soulis": { "...": "..." }
  }
}
```

### NPZ export

Consolidated numpy archive for cross-init analysis — schema in [`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md).

---

### Metrics

All four metrics are for deterministic forecasts only.

| Metric | Full name | What it measures |
|---|---|---|
| **RMSE** | Root Mean Square Error | Overall error magnitude. Higher = less accurate. |
| **MAE** | Mean Absolute Error | Similar to RMSE, but less sensitive to large one-off errors. |
| **Bias** | — | Systematic over- or under-prediction. A well-calibrated model stays near zero. |
| **ACC** | Anomaly Correlation Coefficient | Whether the forecast captures day-to-day variability correctly. Higher = better. |

**Sanity checks** — if these aren't true, something is likely wrong with the data:
- [ ] RMSE and MAE **increase** as lead time increases
- [ ] ACC **decreases** as lead time increases
- [ ] Bias stays **consistent in sign** across lead times if a model has a systematic issue

---

### Forecasting Methods

Each method below is tracked in its own sub-issue. A method is considered done when its benchmarking output is correctly displayed on the site.

**Implementation effort** — tags on each method:

| Tag | Meaning |
|---|---|
| **training/fitting** | Fit correction parameters on historical forecast–observation pairs *before* scoring 2025 (avoid train/test leakage). |
| **offline prep** | Build a static product first (e.g. climatological normals from many years of obs/reanalysis), then apply for 2025 inits. |
| **straightforward run** | Fetch/archive output, bilinearly interpolate to station, score — no project-specific fitting step (AI methods use pretrained weights; setup may still be heavy). |

**Baselines**
- #2 — [Climatology baseline](https://github.com/weatherloo/weatherloo-internal/issues/2) — **offline prep**
- #3 — [Persistence baseline](https://github.com/weatherloo/weatherloo-internal/issues/3) — **straightforward run**

**Statistical Methods**
- #4 — [Mean bias correction](https://github.com/weatherloo/weatherloo-internal/issues/4) — **training/fitting**
- #5 — [Linear regression](https://github.com/weatherloo/weatherloo-internal/issues/5) — **training/fitting**
- #6 — [Model Output Statistics (MOS)](https://github.com/weatherloo/weatherloo-internal/issues/6) — **training/fitting**

**NWP (Numerical Weather Prediction)**
- #7 — [GEFS ensemble mean](https://github.com/weatherloo/weatherloo-internal/issues/7) (used as a deterministic forecast) — **straightforward run**
- #8 — [GFS interpolated at station](https://github.com/weatherloo/weatherloo-internal/issues/8) — **straightforward run**
- #9 — [GFS analysis](https://github.com/weatherloo/weatherloo-internal/issues/9) — **straightforward run**
- #19 — [HRDPS interpolated at station](https://github.com/weatherloo/weatherloo-internal/issues/19) — **straightforward run**
- #20 — [HRDPS analysis](https://github.com/weatherloo/weatherloo-internal/issues/20) — **straightforward run**
- #21 — [HRRR interpolated at station](https://github.com/weatherloo/weatherloo-internal/issues/21) — **straightforward run**
- #22 — [HRRR analysis](https://github.com/weatherloo/weatherloo-internal/issues/22) — **straightforward run**

**AI Models**
- #10 — [ECMWF AIFS](https://github.com/weatherloo/weatherloo-internal/issues/10) — **straightforward run** (pretrained inference; data access / licensing may dominate effort)
- #11 — [GraphCast (Google DeepMind)](https://github.com/weatherloo/weatherloo-internal/issues/11) — **straightforward run** (pretrained inference)
- #12 — [Pangu-Weather (Huawei)](https://github.com/weatherloo/weatherloo-internal/issues/12) — **straightforward run** (pretrained inference)

**Infrastructure & Website**
- #18 — [Deploy website](https://github.com/weatherloo/weatherloo-internal/issues/18)
- #13 — [Show sample counts on selection](https://github.com/weatherloo/weatherloo-internal/issues/13)
- #14 — [Time range filter for charts](https://github.com/weatherloo/weatherloo-internal/issues/14)
- #17 — [Init-cycle filter (00/06/12/18Z)](https://github.com/weatherloo/weatherloo-internal/issues/17)
- #16 — [Summary / leaderboard table](https://github.com/weatherloo/weatherloo-internal/issues/16)
- #15 — [Multi-method comparison on charts](https://github.com/weatherloo/weatherloo-internal/issues/15)
- `#N` — Website styling & UX polish *(ongoing; sub-issue not yet created)*
---
### Coding agents notes

Read **[`benchmarking-site/AGENTS.md`](https://github.com/weatherloo/weatherloo-internal/blob/main/benchmarking-site/AGENTS.md)** for repo paths, JSON schema, bilinear grid-to-station interpolation, metrics, definition of done, and NPZ export.
