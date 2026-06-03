# Benchmark implementation guide (coding agents)

Read this before implementing any method sub-issue of epic [#1](https://github.com/weatherloo/weatherloo-internal/issues/1).

All paths below are **relative to this directory** (`benchmarking-site/`).

## Repo paths

| What | Path |
|------|------|
| Ground truth (use as-is unless extending) | `data/observations/<station_id>/observations_6h_2025.json` |
| Method benchmark output | `data/<method_id>/` |
| Sample output shape | `data/gfs_interpolated/gfs_interpolated_sample.json` |
| Local dashboard | Run here: `python3 -m http.server 8080` |
| Regenerate observations | `python3 scripts/fetch_station_observations.py --year 2025` (from repo root) |
| Observations spec | `data/observations/README.md` |

## Station IDs and coordinates (use exactly)

| `station_id` | Lat | Lon | Ground-truth source |
|--------------|-----|-----|---------------------|
| `cyyz` | 43.6777 | -79.6248 | ECCC hourly **`STN_ID` 51459** (TORONTO INTL A). API filter: **`UTC_YEAR`**, not `LOCAL_YEAR`. |
| `eric_d_soulis` | 43.4668 | -80.5164 | UW Soulis archive. 2015+: `Hobo_15minutedata_2025.csv` only (see observations README). |

## Time rules (UTC only)

- **365 initializations:** `2025-01-01T00:00:00Z` through `2025-12-31T00:00:00Z` (daily at **00:00 UTC**).
- **Lead times (hours):** `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]` — length **12** for every metric array.
- **Forecast valid time:** `initialization + lead_time_hours` (e.g. init `2025-01-15T00:00:00Z`, 6h lead → `2025-01-15T06:00:00Z`).
- **Truth lookup:** `observations_6h_2025.json` entry where `valid_time` equals that UTC instant.
- Do **not** use ECCC `LOCAL_DATE` for timing.

## Variables

| JSON key | Meaning | Units | Notes |
|----------|---------|-------|--------|
| `t2m` | 2 m air temperature | °C | |
| `wind_speed` | 10 m wind speed | km/h | If only u/v components: `wind_speed = sqrt(u² + v²)` |

## Output contract

- **One file per initialization:** `data/<method_id>/2025-MM-DD.json` (365 files for 2025).
- Optional **`index.json`:** `{ "files": ["2025-01-01.json", ...] }` listing all files.
- Top-level `"method"` must match folder name (`<method_id>`).
- JSON shape per init: see epic #1 and `data/gfs_interpolated/gfs_interpolated_sample.json`.
- **Done on site:** method appears in dropdown; both stations show 8 charts (4 metrics × 2 variables) without load errors.

## Metrics (deterministic)

Per initialization, per station, per variable, per lead time — compare forecast vs truth, then store arrays of length 12 in the init JSON:

| Key | Description |
|-----|-------------|
| `rmse` | Root mean square error |
| `mae` | Mean absolute error |
| `bias` | Mean forecast − observation |
| `acc` | Anomaly correlation coefficient (document anomaly baseline/climatology used) |

**Sanity checks:** RMSE and MAE should increase with lead time; ACC should decrease. Bias sign should be stable if the model has a systematic error.

## Definition of done

- [ ] 365 JSON files (or `index.json` + 365 files) under `data/<method_id>/`
- [ ] Both `cyyz` and `eric_d_soulis`, both `t2m` and `wind_speed`, all 12 lead times
- [ ] Spot-check one init (e.g. `2025-01-15`) at `cyyz` for one lead time vs hand calculation
- [ ] Dashboard loads method and renders charts

## Method IDs (folder names)

| Method | `method_id` |
|--------|-------------|
| Climatology | `climatology` |
| Persistence | `persistence` |
| Mean bias correction | `mean_bias_correction` |
| Linear regression | `linear_regression` |
| MOS | `mos` |
| GEFS ensemble mean | `gefs_mean` |
| GFS interpolated | `gfs_interpolated` |
| GFS past-hour analysis | `gfs_analysis` |
| ECMWF AIFS | `ecmwf_aifs` |
| GraphCast | `graphcast` |
| Pangu-Weather | `pangu` |
