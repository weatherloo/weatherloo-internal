# LSTM Bias Correction — Coding Agent Runbook

You are a coding agent. A team member will tell you a **station** and **variable**. Your job is to run the Optuna sweep for every method × every lead time, collect results, update the task table below, and commit. No other input is needed.

---

## What the team member says to you

> "Train cyyz t2m" — or any combination of station and variable.

That's all. Do the rest yourself.

---

## Step-by-step

### 1. Confirm inputs

| | Valid values |
|---|---|
| `station` | `cyyz` or `eric_d_soulis` |
| `variable` | `t2m` or `wind_speed` |

### 2. Install dependencies (once per environment)

```bash
pip install -r lstm_training/requirements.txt
```

Run all commands below from the **repo root**.

### 3. Available methods and their NPZ paths

| Method | NPZ path |
|---|---|
| `climatology` | `benchmarking-site/data/climatology/climatology_2025.npz` |
| `ecmwf_aifs` | `benchmarking-site/data/ecmwf_aifs/ecmwf_aifs_2025.npz` |
| `gefs_mean` | `benchmarking-site/data/gefs_mean/gefs_mean_2025.npz` |
| `gfs_analysis` | `benchmarking-site/data/gfs_analysis/gfs_analysis_2025.npz` |
| `gfs_interpolated` | `benchmarking-site/data/gfs_interpolated/gfs_interpolated_2025.npz` |
| `graphcast` | `benchmarking-site/data/graphcast/graphcast_2025.npz` |
| `hrrr_interpolated` | `benchmarking-site/data/hrrr_interpolated/hrrr_interpolated_2025.npz` |
| `linear_regression` | `benchmarking-site/data/linear_regression/linear_regression_2025.npz` |
| `persistence` | `benchmarking-site/data/persistence/persistence_2025.npz` |

All methods have all 12 lead times: **6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72**

### 4. Run the sweep for every method × every lead time

Command pattern:
```bash
python lstm_training/sweep.py \
  --npz <npz_path> \
  --station <station> \
  --variable <variable> \
  --lead_time <lead> \
  --n_trials 100 \
  --out_dir lstm_training/output/<station>_<variable>/<method>_<lead>h
```

Run sequentially (108 sweeps total per station+variable pair). Name the output dir exactly as shown — `output/<station>_<variable>/<method>_<lead>h`.

### 5. Collect best RMSE values

After all sweeps finish:

```bash
python -c "
import json, glob, os
base = 'lstm_training/output'
for f in sorted(glob.glob(f'{base}/*/*/sweep_results.json')):
    d = json.load(open(f))
    parts = f.replace('\\\\', '/').split('/')
    combo, run = parts[-3], parts[-2]
    print(f'{combo}/{run}: {d[\"best_rmse\"]:.4f}')
"
```

### 6. Update the task table in this file

Fill in **Best RMSE** for every row you trained. Mark Status as `done`.

### 7. Commit

```bash
git add lstm_training/output/<station>_<variable>/
git add lstm_training/AGENTS.md
git commit -m "feat: lstm sweep <station> <variable> all methods all leads"
```

---

## Task table

One row per (station, variable, method, lead). Claim a block by adding your name to the Assigned column for that station+variable group.

### cyyz — t2m

| Method | Lead (h) | Assigned | Status | Best RMSE |
|---|---|---|---|---|
| climatology | 6 | — | — | — |
| climatology | 12 | — | — | — |
| climatology | 18 | — | — | — |
| climatology | 24 | — | — | — |
| climatology | 30 | — | — | — |
| climatology | 36 | — | — | — |
| climatology | 42 | — | — | — |
| climatology | 48 | — | — | — |
| climatology | 54 | — | — | — |
| climatology | 60 | — | — | — |
| climatology | 66 | — | — | — |
| climatology | 72 | — | — | — |
| ecmwf_aifs | 6 | — | — | — |
| ecmwf_aifs | 12 | — | — | — |
| ecmwf_aifs | 18 | — | — | — |
| ecmwf_aifs | 24 | — | — | — |
| ecmwf_aifs | 30 | — | — | — |
| ecmwf_aifs | 36 | — | — | — |
| ecmwf_aifs | 42 | — | — | — |
| ecmwf_aifs | 48 | — | — | — |
| ecmwf_aifs | 54 | — | — | — |
| ecmwf_aifs | 60 | — | — | — |
| ecmwf_aifs | 66 | — | — | — |
| ecmwf_aifs | 72 | — | — | — |
| gefs_mean | 6 | — | — | — |
| gefs_mean | 12 | — | — | — |
| gefs_mean | 18 | — | — | — |
| gefs_mean | 24 | — | — | — |
| gefs_mean | 30 | — | — | — |
| gefs_mean | 36 | — | — | — |
| gefs_mean | 42 | — | — | — |
| gefs_mean | 48 | — | — | — |
| gefs_mean | 54 | — | — | — |
| gefs_mean | 60 | — | — | — |
| gefs_mean | 66 | — | — | — |
| gefs_mean | 72 | — | — | — |
| gfs_analysis | 6 | — | — | — |
| gfs_analysis | 12 | — | — | — |
| gfs_analysis | 18 | — | — | — |
| gfs_analysis | 24 | — | — | — |
| gfs_analysis | 30 | — | — | — |
| gfs_analysis | 36 | — | — | — |
| gfs_analysis | 42 | — | — | — |
| gfs_analysis | 48 | — | — | — |
| gfs_analysis | 54 | — | — | — |
| gfs_analysis | 60 | — | — | — |
| gfs_analysis | 66 | — | — | — |
| gfs_analysis | 72 | — | — | — |
| gfs_interpolated | 6 | — | — | — |
| gfs_interpolated | 12 | — | — | — |
| gfs_interpolated | 18 | — | — | — |
| gfs_interpolated | 24 | — | — | — |
| gfs_interpolated | 30 | — | — | — |
| gfs_interpolated | 36 | — | — | — |
| gfs_interpolated | 42 | — | — | — |
| gfs_interpolated | 48 | — | — | — |
| gfs_interpolated | 54 | — | — | — |
| gfs_interpolated | 60 | — | — | — |
| gfs_interpolated | 66 | — | — | — |
| gfs_interpolated | 72 | — | — | — |
| graphcast | 6 | — | — | — |
| graphcast | 12 | — | — | — |
| graphcast | 18 | — | — | — |
| graphcast | 24 | — | — | — |
| graphcast | 30 | — | — | — |
| graphcast | 36 | — | — | — |
| graphcast | 42 | — | — | — |
| graphcast | 48 | — | — | — |
| graphcast | 54 | — | — | — |
| graphcast | 60 | — | — | — |
| graphcast | 66 | — | — | — |
| graphcast | 72 | — | — | — |
| hrrr_interpolated | 6 | — | — | — |
| hrrr_interpolated | 12 | — | — | — |
| hrrr_interpolated | 18 | — | — | — |
| hrrr_interpolated | 24 | — | — | — |
| hrrr_interpolated | 30 | — | — | — |
| hrrr_interpolated | 36 | — | — | — |
| hrrr_interpolated | 42 | — | — | — |
| hrrr_interpolated | 48 | — | — | — |
| hrrr_interpolated | 54 | — | — | — |
| hrrr_interpolated | 60 | — | — | — |
| hrrr_interpolated | 66 | — | — | — |
| hrrr_interpolated | 72 | — | — | — |
| linear_regression | 6 | — | — | — |
| linear_regression | 12 | — | — | — |
| linear_regression | 18 | — | — | — |
| linear_regression | 24 | — | — | — |
| linear_regression | 30 | — | — | — |
| linear_regression | 36 | — | — | — |
| linear_regression | 42 | — | — | — |
| linear_regression | 48 | — | — | — |
| linear_regression | 54 | — | — | — |
| linear_regression | 60 | — | — | — |
| linear_regression | 66 | — | — | — |
| linear_regression | 72 | — | — | — |
| persistence | 6 | — | — | — |
| persistence | 12 | — | — | — |
| persistence | 18 | — | — | — |
| persistence | 24 | — | — | — |
| persistence | 30 | — | — | — |
| persistence | 36 | — | — | — |
| persistence | 42 | — | — | — |
| persistence | 48 | — | — | — |
| persistence | 54 | — | — | — |
| persistence | 60 | — | — | — |
| persistence | 66 | — | — | — |
| persistence | 72 | — | — | — |

### cyyz — wind_speed

| Method | Lead (h) | Assigned | Status | Best RMSE |
|---|---|---|---|---|
| climatology | 6 | — | — | — |
| climatology | 12 | — | — | — |
| climatology | 18 | — | — | — |
| climatology | 24 | — | — | — |
| climatology | 30 | — | — | — |
| climatology | 36 | — | — | — |
| climatology | 42 | — | — | — |
| climatology | 48 | — | — | — |
| climatology | 54 | — | — | — |
| climatology | 60 | — | — | — |
| climatology | 66 | — | — | — |
| climatology | 72 | — | — | — |
| ecmwf_aifs | 6 | — | — | — |
| ecmwf_aifs | 12 | — | — | — |
| ecmwf_aifs | 18 | — | — | — |
| ecmwf_aifs | 24 | — | — | — |
| ecmwf_aifs | 30 | — | — | — |
| ecmwf_aifs | 36 | — | — | — |
| ecmwf_aifs | 42 | — | — | — |
| ecmwf_aifs | 48 | — | — | — |
| ecmwf_aifs | 54 | — | — | — |
| ecmwf_aifs | 60 | — | — | — |
| ecmwf_aifs | 66 | — | — | — |
| ecmwf_aifs | 72 | — | — | — |
| gefs_mean | 6 | — | — | — |
| gefs_mean | 12 | — | — | — |
| gefs_mean | 18 | — | — | — |
| gefs_mean | 24 | — | — | — |
| gefs_mean | 30 | — | — | — |
| gefs_mean | 36 | — | — | — |
| gefs_mean | 42 | — | — | — |
| gefs_mean | 48 | — | — | — |
| gefs_mean | 54 | — | — | — |
| gefs_mean | 60 | — | — | — |
| gefs_mean | 66 | — | — | — |
| gefs_mean | 72 | — | — | — |
| gfs_analysis | 6 | — | — | — |
| gfs_analysis | 12 | — | — | — |
| gfs_analysis | 18 | — | — | — |
| gfs_analysis | 24 | — | — | — |
| gfs_analysis | 30 | — | — | — |
| gfs_analysis | 36 | — | — | — |
| gfs_analysis | 42 | — | — | — |
| gfs_analysis | 48 | — | — | — |
| gfs_analysis | 54 | — | — | — |
| gfs_analysis | 60 | — | — | — |
| gfs_analysis | 66 | — | — | — |
| gfs_analysis | 72 | — | — | — |
| gfs_interpolated | 6 | — | — | — |
| gfs_interpolated | 12 | — | — | — |
| gfs_interpolated | 18 | — | — | — |
| gfs_interpolated | 24 | — | — | — |
| gfs_interpolated | 30 | — | — | — |
| gfs_interpolated | 36 | — | — | — |
| gfs_interpolated | 42 | — | — | — |
| gfs_interpolated | 48 | — | — | — |
| gfs_interpolated | 54 | — | — | — |
| gfs_interpolated | 60 | — | — | — |
| gfs_interpolated | 66 | — | — | — |
| gfs_interpolated | 72 | — | — | — |
| graphcast | 6 | — | — | — |
| graphcast | 12 | — | — | — |
| graphcast | 18 | — | — | — |
| graphcast | 24 | — | — | — |
| graphcast | 30 | — | — | — |
| graphcast | 36 | — | — | — |
| graphcast | 42 | — | — | — |
| graphcast | 48 | — | — | — |
| graphcast | 54 | — | — | — |
| graphcast | 60 | — | — | — |
| graphcast | 66 | — | — | — |
| graphcast | 72 | — | — | — |
| hrrr_interpolated | 6 | — | — | — |
| hrrr_interpolated | 12 | — | — | — |
| hrrr_interpolated | 18 | — | — | — |
| hrrr_interpolated | 24 | — | — | — |
| hrrr_interpolated | 30 | — | — | — |
| hrrr_interpolated | 36 | — | — | — |
| hrrr_interpolated | 42 | — | — | — |
| hrrr_interpolated | 48 | — | — | — |
| hrrr_interpolated | 54 | — | — | — |
| hrrr_interpolated | 60 | — | — | — |
| hrrr_interpolated | 66 | — | — | — |
| hrrr_interpolated | 72 | — | — | — |
| linear_regression | 6 | — | — | — |
| linear_regression | 12 | — | — | — |
| linear_regression | 18 | — | — | — |
| linear_regression | 24 | — | — | — |
| linear_regression | 30 | — | — | — |
| linear_regression | 36 | — | — | — |
| linear_regression | 42 | — | — | — |
| linear_regression | 48 | — | — | — |
| linear_regression | 54 | — | — | — |
| linear_regression | 60 | — | — | — |
| linear_regression | 66 | — | — | — |
| linear_regression | 72 | — | — | — |
| persistence | 6 | — | — | — |
| persistence | 12 | — | — | — |
| persistence | 18 | — | — | — |
| persistence | 24 | — | — | — |
| persistence | 30 | — | — | — |
| persistence | 36 | — | — | — |
| persistence | 42 | — | — | — |
| persistence | 48 | — | — | — |
| persistence | 54 | — | — | — |
| persistence | 60 | — | — | — |
| persistence | 66 | — | — | — |
| persistence | 72 | — | — | — |

### eric_d_soulis — t2m

| Method | Lead (h) | Assigned | Status | Best RMSE |
|---|---|---|---|---|
| climatology | 6 | — | — | — |
| climatology | 12 | — | — | — |
| climatology | 18 | — | — | — |
| climatology | 24 | — | — | — |
| climatology | 30 | — | — | — |
| climatology | 36 | — | — | — |
| climatology | 42 | — | — | — |
| climatology | 48 | — | — | — |
| climatology | 54 | — | — | — |
| climatology | 60 | — | — | — |
| climatology | 66 | — | — | — |
| climatology | 72 | — | — | — |
| ecmwf_aifs | 6 | — | — | — |
| ecmwf_aifs | 12 | — | — | — |
| ecmwf_aifs | 18 | — | — | — |
| ecmwf_aifs | 24 | — | — | — |
| ecmwf_aifs | 30 | — | — | — |
| ecmwf_aifs | 36 | — | — | — |
| ecmwf_aifs | 42 | — | — | — |
| ecmwf_aifs | 48 | — | — | — |
| ecmwf_aifs | 54 | — | — | — |
| ecmwf_aifs | 60 | — | — | — |
| ecmwf_aifs | 66 | — | — | — |
| ecmwf_aifs | 72 | — | — | — |
| gefs_mean | 6 | — | — | — |
| gefs_mean | 12 | — | — | — |
| gefs_mean | 18 | — | — | — |
| gefs_mean | 24 | — | — | — |
| gefs_mean | 30 | — | — | — |
| gefs_mean | 36 | — | — | — |
| gefs_mean | 42 | — | — | — |
| gefs_mean | 48 | — | — | — |
| gefs_mean | 54 | — | — | — |
| gefs_mean | 60 | — | — | — |
| gefs_mean | 66 | — | — | — |
| gefs_mean | 72 | — | — | — |
| gfs_analysis | 6 | — | — | — |
| gfs_analysis | 12 | — | — | — |
| gfs_analysis | 18 | — | — | — |
| gfs_analysis | 24 | — | — | — |
| gfs_analysis | 30 | — | — | — |
| gfs_analysis | 36 | — | — | — |
| gfs_analysis | 42 | — | — | — |
| gfs_analysis | 48 | — | — | — |
| gfs_analysis | 54 | — | — | — |
| gfs_analysis | 60 | — | — | — |
| gfs_analysis | 66 | — | — | — |
| gfs_analysis | 72 | — | — | — |
| gfs_interpolated | 6 | — | — | — |
| gfs_interpolated | 12 | — | — | — |
| gfs_interpolated | 18 | — | — | — |
| gfs_interpolated | 24 | — | — | — |
| gfs_interpolated | 30 | — | — | — |
| gfs_interpolated | 36 | — | — | — |
| gfs_interpolated | 42 | — | — | — |
| gfs_interpolated | 48 | — | — | — |
| gfs_interpolated | 54 | — | — | — |
| gfs_interpolated | 60 | — | — | — |
| gfs_interpolated | 66 | — | — | — |
| gfs_interpolated | 72 | — | — | — |
| graphcast | 6 | — | — | — |
| graphcast | 12 | — | — | — |
| graphcast | 18 | — | — | — |
| graphcast | 24 | — | — | — |
| graphcast | 30 | — | — | — |
| graphcast | 36 | — | — | — |
| graphcast | 42 | — | — | — |
| graphcast | 48 | — | — | — |
| graphcast | 54 | — | — | — |
| graphcast | 60 | — | — | — |
| graphcast | 66 | — | — | — |
| graphcast | 72 | — | — | — |
| hrrr_interpolated | 6 | — | — | — |
| hrrr_interpolated | 12 | — | — | — |
| hrrr_interpolated | 18 | — | — | — |
| hrrr_interpolated | 24 | — | — | — |
| hrrr_interpolated | 30 | — | — | — |
| hrrr_interpolated | 36 | — | — | — |
| hrrr_interpolated | 42 | — | — | — |
| hrrr_interpolated | 48 | — | — | — |
| hrrr_interpolated | 54 | — | — | — |
| hrrr_interpolated | 60 | — | — | — |
| hrrr_interpolated | 66 | — | — | — |
| hrrr_interpolated | 72 | — | — | — |
| linear_regression | 6 | — | — | — |
| linear_regression | 12 | — | — | — |
| linear_regression | 18 | — | — | — |
| linear_regression | 24 | — | — | — |
| linear_regression | 30 | — | — | — |
| linear_regression | 36 | — | — | — |
| linear_regression | 42 | — | — | — |
| linear_regression | 48 | — | — | — |
| linear_regression | 54 | — | — | — |
| linear_regression | 60 | — | — | — |
| linear_regression | 66 | — | — | — |
| linear_regression | 72 | — | — | — |
| persistence | 6 | — | — | — |
| persistence | 12 | — | — | — |
| persistence | 18 | — | — | — |
| persistence | 24 | — | — | — |
| persistence | 30 | — | — | — |
| persistence | 36 | — | — | — |
| persistence | 42 | — | — | — |
| persistence | 48 | — | — | — |
| persistence | 54 | — | — | — |
| persistence | 60 | — | — | — |
| persistence | 66 | — | — | — |
| persistence | 72 | — | — | — |

### eric_d_soulis — wind_speed

| Method | Lead (h) | Assigned | Status | Best RMSE |
|---|---|---|---|---|
| climatology | 6 | — | — | — |
| climatology | 12 | — | — | — |
| climatology | 18 | — | — | — |
| climatology | 24 | — | — | — |
| climatology | 30 | — | — | — |
| climatology | 36 | — | — | — |
| climatology | 42 | — | — | — |
| climatology | 48 | — | — | — |
| climatology | 54 | — | — | — |
| climatology | 60 | — | — | — |
| climatology | 66 | — | — | — |
| climatology | 72 | — | — | — |
| ecmwf_aifs | 6 | — | — | — |
| ecmwf_aifs | 12 | — | — | — |
| ecmwf_aifs | 18 | — | — | — |
| ecmwf_aifs | 24 | — | — | — |
| ecmwf_aifs | 30 | — | — | — |
| ecmwf_aifs | 36 | — | — | — |
| ecmwf_aifs | 42 | — | — | — |
| ecmwf_aifs | 48 | — | — | — |
| ecmwf_aifs | 54 | — | — | — |
| ecmwf_aifs | 60 | — | — | — |
| ecmwf_aifs | 66 | — | — | — |
| ecmwf_aifs | 72 | — | — | — |
| gefs_mean | 6 | — | — | — |
| gefs_mean | 12 | — | — | — |
| gefs_mean | 18 | — | — | — |
| gefs_mean | 24 | — | — | — |
| gefs_mean | 30 | — | — | — |
| gefs_mean | 36 | — | — | — |
| gefs_mean | 42 | — | — | — |
| gefs_mean | 48 | — | — | — |
| gefs_mean | 54 | — | — | — |
| gefs_mean | 60 | — | — | — |
| gefs_mean | 66 | — | — | — |
| gefs_mean | 72 | — | — | — |
| gfs_analysis | 6 | — | — | — |
| gfs_analysis | 12 | — | — | — |
| gfs_analysis | 18 | — | — | — |
| gfs_analysis | 24 | — | — | — |
| gfs_analysis | 30 | — | — | — |
| gfs_analysis | 36 | — | — | — |
| gfs_analysis | 42 | — | — | — |
| gfs_analysis | 48 | — | — | — |
| gfs_analysis | 54 | — | — | — |
| gfs_analysis | 60 | — | — | — |
| gfs_analysis | 66 | — | — | — |
| gfs_analysis | 72 | — | — | — |
| gfs_interpolated | 6 | — | — | — |
| gfs_interpolated | 12 | — | — | — |
| gfs_interpolated | 18 | — | — | — |
| gfs_interpolated | 24 | — | — | — |
| gfs_interpolated | 30 | — | — | — |
| gfs_interpolated | 36 | — | — | — |
| gfs_interpolated | 42 | — | — | — |
| gfs_interpolated | 48 | — | — | — |
| gfs_interpolated | 54 | — | — | — |
| gfs_interpolated | 60 | — | — | — |
| gfs_interpolated | 66 | — | — | — |
| gfs_interpolated | 72 | — | — | — |
| graphcast | 6 | — | — | — |
| graphcast | 12 | — | — | — |
| graphcast | 18 | — | — | — |
| graphcast | 24 | — | — | — |
| graphcast | 30 | — | — | — |
| graphcast | 36 | — | — | — |
| graphcast | 42 | — | — | — |
| graphcast | 48 | — | — | — |
| graphcast | 54 | — | — | — |
| graphcast | 60 | — | — | — |
| graphcast | 66 | — | — | — |
| graphcast | 72 | — | — | — |
| hrrr_interpolated | 6 | — | — | — |
| hrrr_interpolated | 12 | — | — | — |
| hrrr_interpolated | 18 | — | — | — |
| hrrr_interpolated | 24 | — | — | — |
| hrrr_interpolated | 30 | — | — | — |
| hrrr_interpolated | 36 | — | — | — |
| hrrr_interpolated | 42 | — | — | — |
| hrrr_interpolated | 48 | — | — | — |
| hrrr_interpolated | 54 | — | — | — |
| hrrr_interpolated | 60 | — | — | — |
| hrrr_interpolated | 66 | — | — | — |
| hrrr_interpolated | 72 | — | — | — |
| linear_regression | 6 | — | — | — |
| linear_regression | 12 | — | — | — |
| linear_regression | 18 | — | — | — |
| linear_regression | 24 | — | — | — |
| linear_regression | 30 | — | — | — |
| linear_regression | 36 | — | — | — |
| linear_regression | 42 | — | — | — |
| linear_regression | 48 | — | — | — |
| linear_regression | 54 | — | — | — |
| linear_regression | 60 | — | — | — |
| linear_regression | 66 | — | — | — |
| linear_regression | 72 | — | — | — |
| persistence | 6 | — | — | — |
| persistence | 12 | — | — | — |
| persistence | 18 | — | — | — |
| persistence | 24 | — | — | — |
| persistence | 30 | — | — | — |
| persistence | 36 | — | — | — |
| persistence | 42 | — | — | — |
| persistence | 48 | — | — | — |
| persistence | 54 | — | — | — |
| persistence | 60 | — | — | — |
| persistence | 66 | — | — | — |
| persistence | 72 | — | — | — |

---

## Reference: what each sweep produces

Each `output/<station>_<variable>/<method>_<lead>h/` contains:

| File | Contents |
|---|---|
| `best_model.pt` | PyTorch weights at lowest val loss |
| `config.json` | All hyperparams, normalization `mean`/`std`, split sizes, test metrics (normalized + original scale), baseline metrics, timestamp |
| `predictions.npz` | `predictions`, `targets` (denormalized), `train_losses`, `val_losses` per epoch |
| `sweep_results.json` | `best_rmse`, `best_params`, all 100 trial results, `swept_at` timestamp |

The `mean` and `std` in `config.json` are required at inference time to denormalize predictions.
