# Benchmarking site guide

Internal dashboard for [epic #1](https://github.com/weatherloo/weatherloo-internal/issues/1): compare deterministic forecast skill at **Eric D. Soulis** and **Toronto Pearson (CYYZ)** for **t2m** and **10 m wind speed** across lead times.

**Implementing a benchmark method?** Everything agents need is in this file.

All paths below are **relative to this directory** (`benchmarking-site/`).

## Run the dashboard locally

From `benchmarking-site/`:

```bash
npm install
npm run dev
```

Open http://localhost:5173 — click a station on the map, pick a method, view RMSE / MAE / bias / ACC vs lead time. The UI **averages** metrics across all loaded init files for the selected location.

Production build (serves `dist/` with benchmark JSON copied in):

```bash
npm run build
npm run preview
```

Without `data/<method_id>/index.json`, the app falls back to `<method_id>_sample.json`.

The UI is a **React** app (`src/`) built with Vite. Benchmark JSON stays in `data/` at the repo root of this folder.

## Repo paths

| What | Path |
|------|------|
| Ground truth (use as-is unless extending) | `data/observations/<station_id>/observations_6h_2025.json` |
| Method benchmark output | `data/<method_id>/` |
| Method compute script (when present) | `data/<method_id>/compute_benchmark.py` |
| Method Python deps (when present) | `data/<method_id>/requirements.txt` |
| Consolidated analysis (optional) | `data/<method_id>/<method_id>_<year>.npz` |
| Sample output shape | `data/climatology/climatology_sample.json` |
| Observations spec | `data/observations/README.md` |
| Regenerate observations | `python3 scripts/fetch_station_observations.py --year 2025` (from repo root) |
| GRIB download cache (GFS) | `.cache/gfs_grib/` (repo root; not committed) |

## Compute scripts

Each method's pipeline lives **alongside its output** under `data/<method_id>/`:

- `compute_benchmark.py` — fetches/processes forecasts, writes JSON (+ optional NPZ) into the same folder
- `requirements.txt` — method-specific Python dependencies

Run from repo root or from the method folder:

```bash
python3 -m venv .venv
.venv/bin/pip install -r benchmarking-site/data/gfs_interpolated/requirements.txt
.venv/bin/python benchmarking-site/data/gfs_interpolated/compute_benchmark.py --workers 6
```

Rebuild NPZ from existing JSON without re-fetching:

```bash
.venv/bin/python benchmarking-site/data/gfs_interpolated/compute_benchmark.py --export-npz-only
```

| `method_id` | Script | Notes |
|-------------|--------|-------|
| `gfs_interpolated` | `data/gfs_interpolated/compute_benchmark.py` | GFS 0.25° at **00/06/12/18Z**; bilinear interp; wind from 10 m u/v. `--resume` skips existing init JSONs. Use `--workers 2` if AWS connection resets; downloads retry automatically. |

Add a row here when implementing other methods.

## Station IDs and coordinates (use exactly)

| `station_id` | Lat | Lon | Ground-truth source |
|--------------|-----|-----|---------------------|
| `cyyz` | 43.6777 | -79.6248 | ECCC hourly **`STN_ID` 51459** (TORONTO INTL A). API filter: **`UTC_YEAR`**, not `LOCAL_YEAR`. |
| `eric_d_soulis` | 43.4668 | -80.5164 | UW Soulis archive. 2015+: `Hobo_15minutedata_2025.csv` only (see observations README). |

## Time rules (UTC only)

- **1460 initializations (2025):** every **00, 06, 12, 18 UTC** from `2025-01-01T00:00:00Z` through `2025-12-31T18:00:00Z` (365 days × 4 cycles).
- **Lead times (hours):** `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]` — length **12** for every metric array.
- **Forecast valid time:** `initialization + lead_time_hours` (e.g. init `2025-01-15T06:00:00Z`, 6h lead → `2025-01-15T12:00:00Z`).
- **Truth lookup:** `observations_6h_2025.json` entry where `valid_time` equals that UTC instant.
- Do **not** use ECCC `LOCAL_DATE` for timing.

## Partial / missing data (expected)

Full-year, every-cycle coverage is the **target**, but gaps are normal and **acceptable**:

- **Init cycles** — some 00/06/12/18Z runs may be missing (archive not published, download failure, `--resume` partial run).
- **Lead times / forecast hours** — a given init may lack model output for one or more steps.
- **Ground truth** — `observations_6h_2025.json` may have no row for some valid times (station outage, archive span, QC gaps).

When forecast or observation is missing for a (init, station, variable, lead) pair, keep arrays length **12** and use **`null`** in JSON (see existing `gfs_interpolated` files). Use **`NaN`** in NPZ. Omit an init file only if the whole run failed; partial inits are fine.

The dashboard **averages over whatever loads**; sparse or in-progress datasets are OK for development and for methods with limited archives.

## Variables

| JSON key | Meaning | Units | Notes |
|----------|---------|-------|--------|
| `t2m` | 2 m air temperature | °C | |
| `wind_speed` | 10 m wind speed | km/h | If only u/v components: `wind_speed = sqrt(u² + v²)` |

## Output contract

- **One file per initialization:** `data/<method_id>/2025-MM-DDTHHZ.json` where `HH` is `00`, `06`, `12`, or `18` (e.g. `2025-01-15T06Z.json`). **1460 files** for 2025.
- Top-level `"initialization"` must be full ISO8601 UTC (e.g. `"2025-01-15T06:00:00Z"`).
- Optional **`index.json`:** `{ "files": ["2025-01-01T00Z.json", "2025-01-01T06Z.json", ...] }` — list **new-format** filenames only (not legacy `2025-MM-DD.json`).
- Top-level `"method"` must match folder name (`<method_id>`).
- JSON shape per init: see epic #1 and `data/climatology/climatology_sample.json`.
- **Done on site:** method appears in dropdown; both stations show 8 charts (4 metrics × 2 variables) without load errors.

### Optional NPZ export (analysis)

Compute scripts may also write a **single consolidated NPZ** alongside the JSON files. The dashboard does **not** read NPZ; it is for Python/numpy aggregation across all inits.

- **Filename:** `data/<method_id>/<method_id>_<year>.npz` (e.g. `gfs_interpolated_2025.npz`).
- **Document in** `data/<method_id>/metadata.json` via `"npz_file"` and `"npz_schema": "see benchmarking-site/AGENTS.md"`.

**Arrays** (each metric array has shape `(n_init, n_station, n_variable, n_lead)`; use `NaN` for missing values):

| Key | Shape / dtype | Description |
|-----|---------------|-------------|
| `rmse`, `mae`, `bias`, `acc` | `(1460, 2, 2, 12)` float64 | Same values as the per-init JSON |
| `station_ids` | `(2,)` str | `["cyyz", "eric_d_soulis"]` — axis 1 index order |
| `variables` | `(2,)` str | `["t2m", "wind_speed"]` — axis 2 index order |
| `lead_times_hours` | `(12,)` int | `[6, 12, …, 72]` — axis 3 index order |
| `initializations` | `(1460,)` str | ISO8601 init times, sorted — axis 0 index order |
| `method_id` | scalar str | e.g. `"gfs_interpolated"` |
| `metrics` | `(4,)` str | `["rmse", "mae", "bias", "acc"]` |

**Example:**

```python
import numpy as np

d = np.load("data/gfs_interpolated/gfs_interpolated_2025.npz")
# Year-mean RMSE at CYYZ for t2m across lead times:
cyyz_t2m_rmse = d["rmse"][:, 0, 0, :].mean(axis=0)
```

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

Full coverage below is the goal for a **finished** method; incomplete cycles, lead times, or ground truth (see **Partial / missing data**) do not block committing useful partial output.

- [ ] 1460 JSON files (or `index.json` + 1460 files) under `data/<method_id>/`
- [ ] Both `cyyz` and `eric_d_soulis`, both `t2m` and `wind_speed`, all 12 lead times
- [ ] Spot-check one init (e.g. `2025-01-15T06:00:00Z`) at `cyyz` for one lead time vs hand calculation
- [ ] Dashboard loads method and renders charts
- [ ] (Optional) Consolidated `<method_id>_<year>.npz` for numpy analysis — see NPZ section above

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
