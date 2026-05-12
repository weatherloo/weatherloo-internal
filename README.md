# Waterloo HRRR Post-Processing

This repository contains a first-pass implementation of a Waterloo-region
HRRR post-processing experiment.

The current v1 setup follows a sparse-supervision plan:

- target variable: `2 m temperature`
- training target: station observations only
- model output: a dense corrected field over a local Waterloo HRRR patch
- time split: January 2026 train/validation, February 2026 test

## Repository layout

- `data/stations/README.md`: station sources and hourly QC/alignment rules
- `src/data/stations.py`: station ingestion and hourly alignment
- `src/data/hrrr.py`: HRRR patch loading utilities
- `src/data/dataset.py`: Jan/Feb sample construction and PyTorch dataset
- `src/models/hrrr_cnn.py`: dense residual CNN/U-Net and station-loss helpers
- `src/evaluation.py`: station metrics and simple baselines
- `src/train.py`: end-to-end training and evaluation entry point
- `reports/jan_feb_2026_eval.md`: evaluation checklist and expected outputs

## Install

Create an environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## Data inputs

The code expects:

- the UW 2026 15-minute CSV
- hourly Weatherstats CSV exports for the nearby auxiliary stations
- an HRRR dataset with Waterloo-region coverage in NetCDF or Zarr format,
  either local or a public `s3://...` Zarr store

The station URLs and default v1 station metadata live in `src/data/stations.py`.
The HRRR loader is intentionally format-flexible, but it assumes the dataset
contains a target variable named `t2m` plus a few context variables.

## Run

```bash
python3 -m src.train \
  --uw-csv "path/to/Hobo_15minutedata_2026.csv" \
  --hrrr-path "path/to/hrrr_dataset.zarr" \
  --output-dir "artifacts/v1"
```

You can also point `--hrrr-path` at a public HRRR Zarr store on S3, for example
within NOAA's public HRRR archives on AWS:

```bash
python3 -m src.train \
  --hrrr-path "s3://hrrrzarr/path/to/store" \
  --output-dir "artifacts/v1"
```

Use `--help` on `python3 -m src.train` for the full argument list.

The training run writes metrics and plots into the chosen output directory,
including:

- `training_history.csv`
- `loss_curves.png`
- `*_metrics.json`
- `*_*_by_lead.csv`
- `*_model_by_lead_baseline_comparison.csv`
- `*_metrics_by_lead.png`
- `*_model_by_lead_baseline_comparison.png`
- `*_metrics_summary.png`
- `*_prediction_scatter.png`
- `*_station_timeseries_lead*.png`
- `validation_spatial_epoch*_example*_lead*.png`
- `test_spatial_example*_lead*.png`

## Notes

- This v1 keeps station evaluation as the source of truth.
- Dense-field quality should be treated as qualitative unless the number of
  usable stations increases materially.
- If too few nearby stations survive QC, the code can be adapted to a
  station-query output head with minimal changes to the data pipeline.

