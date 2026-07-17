# Raw HRRR + ERA5 downloaders

Bulk raw training data for the bias-correction model described in
`lit-review/weather-forecasting.md`: **HRRR** 0–48 h forecasts (model input) and
**ERA5** reanalysis (ground truth), both subset to the Kitchener-Waterloo
bounding box and kept on their **native grids** (no regridding — that's a
downstream modelling step).

| | HRRR (`download_hrrr.py`) | ERA5 (`download_era5.py`) |
|---|---|---|
| Source | AWS `noaa-hrrr-bdp-pds` (GRIB2, `.idx` byte-range) | ARCO-ERA5 Zarr on GCS (anonymous) |
| Resolution | 3 km CONUS Lambert-conformal | 0.25° global |
| Cadence | 00/06/12/18Z, hourly leads f00–f48 | hourly |
| Variables | `t2m`, `u10`, `v10`, `tp` | same (renamed to match) |
| Output | `{data_root}/hrrr/{YYYY}/{YYYYMMDD}/hrrr_{YYYYMMDD}_t{HH}z.nc` | `{data_root}/era5/{YYYY}/era5_{YYYYMM}.nc` |

## Setup

```bash
pip install -r scripts/requirements.txt
```

## Where data goes (`--data-root`)

Precedence: `--data-root` flag → `WEATHERLOO_DATA_ROOT` env → repo `data/`.
On WATcloud, point it at HDD-backed bulk storage (`/mnt/wato-drive*`), **not**
`/home` (SSD, small files only):

```bash
export WEATHERLOO_DATA_ROOT=/mnt/wato-drive/<you>/weatherloo-data
```

## Usage

```bash
# smoke tests first (single unit)
python scripts/download_hrrr.py --dry-run --data-root /tmp/wxtest
python scripts/download_era5.py --dry-run --data-root /tmp/wxtest

# full backfill 2018 -> present (resumable; safe to restart)
python scripts/download_hrrr.py --resume            # hours–days, ~1M+ requests
python scripts/download_era5.py --resume            # minutes–tens of minutes
```

Common flags: `--start-date`, `--end-date`, `--resume`, `--dry-run`.
HRRR also: `--cycles`, `--max-lead`, `--workers` (default 3), `--download-retries`.
ERA5 also: `--store` (override the ARCO zarr path).

## Notes

- **HRRR 48 h leads exist only from HRRRv4 (~Dec 2020).** Earlier dates (HRRRv3,
  from 2018-07-13) cap at f36; the missing leads are probed from the archive and
  skipped — no version dates are hard-coded.
- **`tp`** for HRRR is **1-hour precip** (kg/m^2): APCP is de-accumulated
  lead-to-lead on write. ERA5 `tp` is hourly accumulation in metres.
- **ERA5 / ARCO is bandwidth-heavy for a small region.** The ARCO store is
  chunked one-hour-x-whole-globe, so extracting the KW box still pulls a full
  global field per hour (~10 GB transferred per month for a few MB of output).
  Reads are egress-free (public GCS), but **run the ERA5 backfill on the cluster**,
  not a home connection. Months not yet reanalysed (~2–3 month lag) are detected
  by a cheap first-hour probe and skipped.
- HRRR GRIB messages are cached under a **private** subdir of
  `{data_root}/.cache/hrrr_grib/` (`job-$SLURM_JOB_ID/` or `pid-$PID/`, plus
  `proc-$PID/` per process worker) so parallel Slurm/process workers do not
  race on NFS (that segfaults eccodes). After each lead is read into memory the
  GRIB blob is deleted; the private dir is removed when the process exits.
  `--max-cache-gb` (default 8) prunes oldest files inside that private dir.
  Cycle parallelism uses processes (`--workers`, default 4) because eccodes is
  not thread-safe; HTTP field fetches within a lead use threads. Override the
  cache base with `--cache-dir` / `$WEATHERLOO_HRRR_CACHE`; pass `--shared-cache`
  only for debugging.
- Outputs under `data/hrrr/` and `data/era5/` (or your `--data-root`) are
  gitignored.

## Sanity-check GIFs

Animate one finished cycle over lead hours (`t2m`, wind speed, `tp`):

```bash
python scripts/sanity_gif_hrrr.py \
  /mnt/wato-drive/$USER/weatherloo-data/hrrr/2018/20180713/hrrr_20180713_t12z.nc
# writes {data_root}/viz/hrrr/hrrr_20180713_t12z_{t2m,wind,tp}.gif
```

## SLURM (WATcloud)

```bash
# one ~month worker (unbuffered logs under logs/)
sbatch --partition=compute --cpus-per-task=4 --mem=32G --time=12:00:00 \
  --job-name=weatherloo-raw-download-smoke \
  --output=logs/%x-%j.out --error=logs/%x-%j.err \
  --wrap='bash scripts/download_raw_training_data_slurm.sh /mnt/wato-drive/$USER/weatherloo-data --start-date 2018-07-13 --end-date 2018-07-13 --resume'

# full backfill: one job per ~month window (resumable, batched)
# submits 10 jobs at a time by default, waits for each batch to finish
BATCH_SIZE=10 scripts/submit_raw_training_data.sh /mnt/wato-drive/$USER/weatherloo-data
```

Worker sets `PYTHONUNBUFFERED=1` and runs `python -u` so Slurm `.out`/`.err` files get live progress.
`submit_raw_training_data.sh` batches submissions (`BATCH_SIZE`, default 10) to avoid hammering NFS with ~100 concurrent writers.
