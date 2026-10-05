# RUNBOOK — U-Net pipeline on WATcloud (or any Linux box)

Self-contained instructions to fetch the training data and train the
residual-correction U-Net **without any local setup from the original repo**.
Everything is driven by two commands and two directories you choose.

Both data sources are public — **no credentials, API keys, or cloud accounts
needed**:

| Source | What | Access |
|---|---|---|
| GFS 0.25° forecasts (model input) | AWS Open Data `noaa-gfs-bdp-pds` | anonymous HTTPS byte-range |
| ERA5 reanalysis (ground truth) | GCP ARCO `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | anonymous (`token="anon"`) |

## 0. Setup (once)

Requirements: **Python 3.11+**, ~25 GB free disk for the full 2022–2025 fetch
(or ~1 GB with `--prune-grib`), outbound HTTPS. GPU optional (see §2).

```bash
cd weatherloo-internal
python3 -m venv .venv
.venv/bin/pip install -r models/unet/requirements.txt
```

`requirements.txt` is complete for both commands (pyyaml, xarray, gcsfs,
zarr<3, fsspec, cfgrib, eccodes, scipy, numpy, torch). The `eccodes` pip
package ships prebuilt binaries for Linux x86_64; if `import cfgrib` fails on
an unusual platform, install the system library instead
(`apt install libeccodes0` or `conda install -c conda-forge eccodes`).

Quick smoke test (network, ~1 min):

```bash
.venv/bin/python models/unet/data/fetch_era5.py
```

## 1. Fetch — download + cache all training samples

```bash
.venv/bin/python models/unet/run_pipeline.py fetch \
    --start 2022-01-01 --end 2025-12-31 \
    --data-dir /path/to/data --workers 6
```

What it does: enumerates every (init date × {00Z,12Z} × {f006,f012,f018,f024})
sample in the range, downloads the GFS fields via byte-range GRIB (retry with
exponential backoff on 429/5xx) and the matching ERA5 truth from the zarr
store, and writes one `.npz` region grid per sample under
`<data-dir>/unet_training/`. The end date is automatically clamped to real
ERA5 coverage (see "ERA5 coverage clamping" below), so a too-late `--end`
fails safe instead of fetching empty data.

**What to expect (2022-01-01 .. 2025-12-31):**

| | |
|---|---|
| Samples | ~11,700 (1461 days × 2 cycles × 4 lead times) |
| Disk | ~20 GB GRIB intermediates + ~250 MB `.npz`; **~250 MB total with `--prune-grib`** |
| Time | roughly **1–2 h** at `--workers 6` on a well-connected node (progress + ETA printed every 25 samples) |

**Resumable:** interrupt/re-run freely — cached samples are skipped, so a
second run only retries failures. A handful of failures is normal:

- transient network / HTTP 5xx → just re-run the same command;
- HTTP 404 → permanent gap in the GFS archive for that cycle; ignore
  (the train step automatically uses only what's cached).

Exit code is 0 only when every sample in range is cached, so it's easy to
loop in a shell script until clean.

## 2. Train

```bash
.venv/bin/python models/unet/run_pipeline.py train \
    --data-dir /path/to/data \
    --output-dir /path/to/checkpoints --epochs 50
```

Trains **only on the cached samples** (zero network IO), defaulting to the
full cached date range. Chronological 80/20 train/val split by default
(`--split-mode interleaved_month` for the per-month split). Normalization
stats are computed from the train split on first run and saved next to the
checkpoint.

**Hardware:** the model is tiny (~500k params on a 21×41 grid). A CUDA GPU is
auto-detected and used if present, but CPU-only works — expect on the order of
minutes/epoch on ~9,300 train samples with a modern multi-core CPU. RAM: a few
GB. Set `--workers 4` (DataLoader workers) on multi-core nodes.

**Outputs in `--output-dir`:**

| File | What |
|---|---|
| `best_model.pt` | best-val-loss checkpoint: model weights + normalization stats + epoch metadata |
| `training_log.csv` | per-epoch `train_loss,val_loss,lr` |
| `stats.json` | per-channel normalization stats (also embedded in the checkpoint) |

Training prints per-epoch losses, early-stops after 10 epochs without val
improvement, then reports **denormalized validation skill**: raw-GFS t2m RMSE
vs corrected t2m RMSE (the headline number — corrected < raw means the model
helps; the 2021 baselines sit ~1.4–1.5 °C at CYYZ).

## 3. Recommended first run (tiny end-to-end check, ~5 min)

```bash
.venv/bin/python models/unet/run_pipeline.py fetch \
    --start 2022-06-01 --end 2022-06-07 --data-dir /tmp/unet-test --workers 4
.venv/bin/python models/unet/run_pipeline.py train \
    --data-dir /tmp/unet-test --output-dir /tmp/unet-test-ckpt --epochs 3
```

~56 samples; confirms network access, GRIB decoding, and the training loop
before committing to the full multi-hour fetch.

## ERA5 coverage clamping (why a too-late `--end` is safe)

The ARCO ERA5 store's time axis is **padded far into the future** (to
2050-12-31) with empty NaN chunks — the raw axis end is *not* the data
boundary. The pipeline therefore reads the store's `valid_time_stop`
attribute at runtime (`dataset.era5_last_valid_time()`) and drops any sample
whose valid time falls past it. Two properties matter:

- **Fails safe, not silent:** an `--end` past the boundary just clamps —
  out-of-range samples are never enumerated, so you cannot silently train
  on NaN "ground truth" from the padded region.
- **Final ERA5 only:** the boundary used is `valid_time_stop` (final,
  quality-assured ERA5), deliberately **not** `valid_time_stop_era5t` (the
  preliminary ERA5T product, ~2.5 months fresher) — preliminary data is not
  used as training truth.

**Confirmed safe range: `--start 2022-01-01 --end 2025-12-31`.** As of
2026-07-21 the store's final-ERA5 boundary is 2026-04-30T23Z (verified by
probing actual values at the boundary), giving the full run **~4 months of
margin**. The boundary advances automatically as ECMWF finalizes more months,
so this margin only grows.

## Notes / gotchas

- `--data-dir` and `--output-dir` are the **only** locations written to
  (plus `stats.json`/logs inside `--output-dir`). Nothing touches the repo.
- The GFS 0.25° AWS archive starts ~2021-03-23 — don't request earlier dates.
- Re-running `fetch` over an existing `--data-dir` is always safe (idempotent).
- `run_pipeline.py train` never downloads: if it errors with "no cached
  samples", the fetch step hasn't been run against that `--data-dir`.
