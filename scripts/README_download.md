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
- **`tp`** is HRRR's accumulated APCP as-is (bucket resets over the run; f00 is
  the 0-0 h accumulation = zeros); ERA5 `tp` is accumulated precip in metres.
  De-accumulate downstream.
- **ERA5 / ARCO is bandwidth-heavy for a small region.** The ARCO store is
  chunked one-hour-x-whole-globe, so extracting the KW box still pulls a full
  global field per hour (~10 GB transferred per month for a few MB of output).
  Reads are egress-free (public GCS), but **run the ERA5 backfill on the cluster**,
  not a home connection. Months not yet reanalysed (~2–3 month lag) are detected
  by a cheap first-hour probe and skipped.
- HRRR GRIB messages are cached under `.cache/hrrr_grib/` (gitignored). Outputs
  under `data/hrrr/` and `data/era5/` are gitignored.
