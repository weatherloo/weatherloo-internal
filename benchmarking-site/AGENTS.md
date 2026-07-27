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

When a method has a consolidated `<method_id>_2025.npz`, the dashboard loads its precomputed **static aggregate** (`data/<method_id>/aggregate.json`, built by `scripts/build_static_aggregates.py` — run automatically by `npm run build`, or manually via `bash scripts/build_static_aggregates.sh`). There is no API server; cycle/month filters recombine the aggregate's per-(month × cycle) partial sums client-side. Without NPZ (or for non-month-aligned custom ranges), it falls back to per-init JSON (via `index.json` or sample file).

Rebuild the consolidated NPZ from existing JSON without re-fetching:

```bash
python3 data/gfs_interpolated/compute_benchmark.py --export-npz-only
```

After (re)building an NPZ, refresh the static aggregate:

```bash
bash scripts/build_static_aggregates.sh --methods <method_id>
```

Production build (bundles `dist/` with `aggregate.json` files copied in; see `DEPLOY.md` for Vercel):

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
| Consolidated analysis | `data/<method_id>/<method_id>_<year>.npz` |
| Sample output shape | `data/climatology/climatology_sample.json` |
| Observations spec | `data/observations/README.md` |
| Regenerate observations | `python3 scripts/fetch_station_observations.py --year 2025` (from repo root) |
| GRIB download cache (GFS) | `.cache/gfs_grib/` (repo root; not committed) |
| GRIB download cache (GEFS) | `.cache/gefs_grib/` (repo root; not committed) |
| GRIB download cache (HRDPS) | `.cache/hrdps_grib/` (repo root; not committed) |

## Compute scripts

Each method's pipeline lives **alongside its output** under `data/<method_id>/`:

- `compute_benchmark.py` — fetches/processes forecasts, writes per-init JSON and consolidated NPZ into the same folder
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
| `gfs_analysis` | `data/gfs_analysis/compute_benchmark.py` | GFS **f000** analysis; one value per init reused at all leads. Same interp/wind rules as `gfs_interpolated`. |
| `hrdps_analysis` | `data/hrdps_analysis/compute_benchmark.py` | HRDPS **PT000H** from MSC Datamart; geographic bilinear interp on curvilinear grid; one analysis value per init at all leads. Datamart ~30-day retention — cache under `.cache/hrdps_grib/` for reruns. **Live archive:** run `data/hrdps_analysis/cache_daily.sh` on a cron (see `crontab.example`). |
| `climatology` | `data/climatology/compute_benchmark.py` | Multi-year (2010-2024) DOY+UTC-hour station climatology; no GRIB needed. `--fetch-historical` downloads historical obs. ACC is always null (forecast = climatology). |
| `ecmwf_aifs` | `data/ecmwf_aifs/compute_benchmark.py` | ECMWF AIFS Single 0.25° at **00/06/12/18Z** via [dynamical.org catalog](https://dynamical.org/catalog/ecmwf-aifs-single-forecast/) (`dynamical-catalog`); 6-hourly steps; bilinear interp of `temperature_2m` / `wind_u_10m` / `wind_v_10m`. Archive 2024-04-01–present includes full 2025. `--resume` skips existing init JSONs. |
| `gefs_mean` | `data/gefs_mean/compute_benchmark.py` | GEFS **ensemble mean** (`geavg`) at **0.5°** from AWS `noaa-gefs-pds`; 00/06/12/18Z; bilinear interp; wind from 10 m u/v. Pre-averaged 21-member mean on grid — no per-member downloads. `--resume` skips existing init JSONs. |
| `unet` | `data/unet/compute_benchmark.py` | **U-Net post-processing of GFS** (`models/unet/`). Thin adapter over the real model code; `corrected = GFS − predicted_residual` on the 21×41 southern Ontario grid, then interpolated to the station. Scored over the checkpoint’s recorded `sample_space` (now 00/06/12/18Z, f006–f048); other cells null unless `--all-cells`. Reuses `run_pipeline.py fetch` output via `--data-dir`. See **U-Net post-processing** below. |

Add a row here when implementing other methods.

### U-Net post-processing

All of the modelling lives in `models/unet/` — `data/unet/compute_benchmark.py`
is a thin **adapter** that imports it (`model/unet.py`, `data/dataset.py`,
`config.yaml`) rather than reimplementing it, so the dashboard can never
disagree with `models/unet/evaluate.py` about geometry, normalization, or sign.

**Sign — the easy way to get this wrong.** The network predicts the *error*
`GFS − ERA5`, so the correction is **subtracted**:

```
x_norm    = (gfs − gfs.mean) / gfs.std
r_norm    = unet(x_norm)
corrected = gfs − (r_norm * residual.std + residual.mean)
```

Adding it roughly doubles the error instead of removing it.

- **Grid:** the southern Ontario box in `models/unet/config.yaml`
  (41–46 °N, −84 to −74 °E at 0.25°) = **21 × 41**. This is baked into the
  checkpoint — `ResidualUNet` reflect-pads 21×41 → 24×48 internally and crops
  back, so a different crop is not interchangeable.
- **Channels:** `t2m` in **°C** (GFS `TMP` is Kelvin), `u10`/`v10` in m/s.
- **Wind:** correct `u10`/`v10` separately, then derive `sqrt(u² + v²) * 3.6`.
  Never correct speed directly — the model was not trained on it.
- **Interpolation order:** correct the *grid*, then bilinearly interpolate the
  corrected grid to the station — not the other way round.
- **Reusing fetched data:** pass `--data-dir` (or set `$UNET_DATA_DIR`) to the
  cache built by `models/unet/run_pipeline.py fetch`; cached samples need no
  download. Either the parent or the `unet_training/` directory itself works.
  The run prints its cache-hit count on startup — if that says `0`, the
  `--data-dir` is wrong and the job is about to re-download the year from AWS.

**Running a full year on WATcloud Slurm:**

```bash
sbatch benchmarking-site/data/unet/run_benchmark.slurm            # single node
sbatch --array=1-8 benchmarking-site/data/unet/run_benchmark.slurm  # sharded
sbatch --dependency=afterok:<jobid> \
    benchmarking-site/data/unet/finalize_benchmark.slurm
```

`DATA_DIR`, `YEAR`, and `WORKERS` are overridable (`--export=ALL,YEAR=2024`).
Array tasks pass `--shard i/N --no-export`, so they take disjoint slices and
leave `index.json` / the NPZ to the finalize job — concurrent shards would
otherwise race and each publish a partial index. Failed inits land in
`failures.json` rather than killing the job; re-run with `--resume` to retry
just those.

- **Lead time is an input channel.** The network takes **4** channels — the
  three normalized GFS fields plus a constant plane encoding the lead — so one
  model covers f006–f048 without over-correcting short leads. Build the tensor
  with `dataset.build_model_input`, never by hand.

**Coverage.** Currently **00/06/12/18Z at f006–f048**. The remaining cells of
the site's 00/06/12/18Z × 6–72 h matrix (f054–f072) are written as `null`
rather than silently extrapolated. `--all-cells` runs them anyway, at the cost
of feeding the model inputs far outside its training distribution.

The range comes from the **checkpoint's** recorded `sample_space`, not from
`config.yaml` — config describes the next training run, so after widening it a
checkpoint trained on the old set would otherwise be scored on cells it never
saw. Checkpoints predating that record fall back to config and say so.

> **Not yet a win.** `models/unet/README.md` records a held-out 2021 evaluation
> where this model *loses* to raw GFS on t2m (CYYZ RMSE 1.289 → 1.404) and to a
> zero-parameter seasonal-mean-bias baseline. Treat its dashboard numbers as a
> baseline to beat, not a finished result.

### HRDPS analysis live GRIB archive

MSC Datamart only keeps HRDPS on the server for about **30 days**. To build a local archive for backfill:

```bash
chmod +x benchmarking-site/data/hrdps_analysis/cache_daily.sh
benchmarking-site/data/hrdps_analysis/cache_daily.sh
```

Install cron (four times daily, shortly after each 00/06/12/18Z cycle):

```bash
# Edit paths, then paste from benchmarking-site/data/hrdps_analysis/crontab.example
crontab -e
```

Backfill benchmark JSON from cached GRIB (when obs year matches):

```bash
.venv/bin/python benchmarking-site/data/hrdps_analysis/compute_benchmark.py \
  --start-date 2026-06-01 --end-date 2026-06-07 --resume
```

Historical 2025 beyond Datamart retention requires ECCC archive retrieval (`ec.dps-client.ec@canada.ca`); see issue #20.

## Station IDs and coordinates (use exactly)

| `station_id` | Lat | Lon | Ground-truth source |
|--------------|-----|-----|---------------------|
| `cyyz` | 43.6777 | -79.6248 | ECCC hourly **`STN_ID` 51459** (TORONTO INTL A). API filter: **`UTC_YEAR`**, not `LOCAL_YEAR`. |
| `eric_d_soulis` | 43.4668 | -80.5164 | UW Soulis archive. 2015+: `Hobo_15minutedata_2025.csv` only (see observations README). |

## Grid-to-station interpolation

When a method reads **gridded** model output (NWP, AI global models, ensemble means, etc.), extract values at each station using **bilinear interpolation** in latitude/longitude:

- **t2m:** bilinear interp on the 2 m temperature grid.
- **Wind:** bilinear interp on **u** and **v** separately at 10 m, then `wind_speed = √(u² + v²)` — do not interpolate speed directly.

Reference implementation: `data/gfs_interpolated/compute_benchmark.py` (`scipy.interpolate.RegularGridInterpolator` with `method="linear"` on a 2D lat/lon grid).

Analysis methods (`*_analysis`) use the same rule on the init-time analysis field (`f000`).

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

### NPZ export

Compute scripts write a **single consolidated NPZ** alongside the per-init JSON files. The dashboard reads NPZ **indirectly via the static aggregate** `data/<method_id>/aggregate.json` built from it by `scripts/build_static_aggregates.py` (part of `npm run build`); without NPZ it falls back to per-init JSON. `server/npz_api.py` is a legacy local tool (run manually with `bash server/run_api.sh`) kept for ad-hoc NPZ inspection; nothing in the site calls it anymore.

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
- [ ] Consolidated `<method_id>_<year>.npz` with `npz_file` / `npz_schema` in `metadata.json` — see NPZ section above

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
| GFS analysis | `gfs_analysis` |
| HRDPS interpolated | `hrdps_interpolated` |
| HRDPS analysis | `hrdps_analysis` |
| HRRR interpolated | `hrrr_interpolated` |
| HRRR analysis | `hrrr_analysis` |
| ECMWF AIFS | `ecmwf_aifs` |
| GraphCast | `graphcast` |
| Pangu-Weather | `pangu` |
| U-Net post-processing (GFS) | `unet` |
