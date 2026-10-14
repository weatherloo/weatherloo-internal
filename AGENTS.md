# WATcloud storage standard (weatherloo-internal)

This repository uses one portable root: `WEATHERLOO_DATA_ROOT`.

- On WATcloud: set it to your allocation (example: `/mnt/wato-drive/$USER/weatherloo-data`)
- Local development: if unset, scripts may fall back to repo-local `data/`, `.cache/`, and `artifacts/`

## Canonical layout under `WEATHERLOO_DATA_ROOT`

```text
${WEATHERLOO_DATA_ROOT}/
  raw/                  # immutable/raw datasets
    hrrr/
    era5/
    observations/
  processed/            # derived reusable datasets
    hrrr_bias_correction/
  cache/                # reusable fetch/intermediate caches
  experiments/          # run-specific working sets (e.g., unet)
  checkpoints/          # train checkpoints and resumable model state
  logs/                 # long-lived operational logs
  published/            # publishable artifacts (dashboard NPZ/JSON, exported models)
  tmp/                  # scratch, safe to delete
```

## Inventory of existing path usage (current repo)

### Python/config
- `src/hrrr_bias_correction/config.py`
- `src/hrrr_bias_correction/config.default.json`
- `src/hrrr_bias_correction/make_data.py`
- `benchmarking-site/data/cnn_lstm_bias_correction/compute_benchmark.py`
- `models/unet-hrrr/data/hrrr_dataset.py`
- `models/unet-hrrr/train.py`
- `scripts/weather_download_common.py`

### Shell/Slurm
- `scripts/submit_raw_training_data.sh`
- `scripts/babysit_raw_downloads.sh`
- `scripts/cron_update_weatherloo_data.sh`
- `scripts/download_raw_training_data_slurm.sh`
- `models/unet/submit_pipeline.sh`
- `models/unet/fetch.slurm`
- `models/unet/train.slurm`
- `models/unet-hrrr/train_hrrr.slurm`
- `benchmarking-site/data/unet/run_benchmark.slurm`
- `benchmarking-site/data/unet/finalize_benchmark.slurm`

### Repository-local locations still used for local workflows
- `.cache/`
- `data/`
- `artifacts/`
- `logs/`

## Ownership and permissions

- **User-owned paths** (`experiments/`, `tmp/`, most of `cache/`): owner-only write (`u+rwx`, no world write).
- **Shared datasets** (`raw/`, selected `processed/`, selected `published/`): readable by project group; write access limited to maintainers or designated data jobs.
- Avoid world-writable directories; prefer group-based sharing (`chmod 2775` + shared group) where collaboration is required.

## Retention and cleanup

- `raw/`: retain; do not mutate in-place.
- `processed/`: retain versions used by active experiments; remove superseded versions after validation.
- `cache/` and `tmp/`: periodic cleanup allowed.
- `logs/`: keep operationally relevant history; rotate/prune old logs.
- `published/`: keep files referenced by dashboard/docs/releases.

## Migration / linking tool

Use `scripts/migrate_weatherloo_storage.py`.

- Default is dry-run and prints explicit source→destination actions.
- `--apply` executes copy/link actions.
- Resumable/idempotent: existing identical files are skipped.
- Validation: size + SHA256 checks are enforced before any replace/remove operation.
- `--remove-source` removes sources only after validated destination contents exist.

Examples:

```bash
# plan only
python3 scripts/migrate_weatherloo_storage.py \
  --data-root "$WEATHERLOO_DATA_ROOT" \
  --source-root "/mnt/wato-drive/$USER/weatherloo-data"

# apply, including repo-local .cache/data/artifacts inputs
python3 scripts/migrate_weatherloo_storage.py \
  --data-root "$WEATHERLOO_DATA_ROOT" \
  --source-root "/mnt/wato-drive/$USER/weatherloo-data" \
  --include-repo-local \
  --apply
```
# Weatherloo Repository Guide

Weatherloo is a weather-forecast benchmarking and post-processing project. This
repository contains the benchmarking web application, forecast and observation
data pipelines, machine-learning models, training workflows, and supporting
research and operational scripts.

## Instruction scope

This file contains repository-wide guidance. More specific `AGENTS.md` files
take precedence for files within their directories.

- `benchmarking-site/AGENTS.md` — benchmarking methods, dashboard data contracts,
  station identifiers, metrics, interpolation, and benchmark-specific workflows.
- `lstm_training/AGENTS.md` — LSTM bias-correction sweeps and training workflow.

When working in one of those directories, read its `AGENTS.md` before making
changes.

## Repository map

- `benchmarking-site/` — React/Vite benchmarking dashboard and benchmark method
  data/pipelines.
- `src/hrrr_bias_correction/` — HRRR bias-correction source code.
- `models/` — model-specific implementations and training/evaluation workflows,
  including U-Net models.
- `lstm_training/` — LSTM bias-correction training and hyperparameter sweeps.
- `scripts/` — repository-level data download, preprocessing, benchmark-building,
  and Slurm helper scripts.
- `artifacts/` — generated model/pipeline artifacts; do not commit.
- `logs/` — runtime and Slurm logs.
- `lit-review/` — research/literature-review material.
- `meeting_notes/` — project meeting notes.
- `viz/` — generated or supporting visualizations.
- `tpm/` — project-management/supporting material.

## Development environments

The repository contains multiple Python and JavaScript workloads with separate
dependency files. There is not currently one repository-wide environment.

- The benchmarking dashboard uses Node.js/npm; see `benchmarking-site/AGENTS.md`.
- LSTM training uses `lstm_training/requirements.txt`; see
  `lstm_training/AGENTS.md`.
- Model-specific dependencies live with the model, for example
  `models/unet/requirements.txt` and `models/unet-hrrr/requirements.txt`.
- Raw HRRR/ERA5 downloader dependencies are in `scripts/requirements.txt`.
- `src/hrrr_bias_correction/` has its own `requirements.txt`.

For WATcloud jobs, keep bulk weather data on HDD-backed storage such as
`/mnt/wato-drive/...`, not in the repository or home directory.

## Common workflows

Run commands from the repository root unless a nested `AGENTS.md` says
otherwise.

For the benchmarking dashboard, follow `benchmarking-site/AGENTS.md`. To verify
that the frontend builds successfully:

```bash
cd benchmarking-site
npm run build
```

For LSTM training and sweeps, follow `lstm_training/AGENTS.md`. A lightweight
sequence-cutoff test can be run from the repository root:

```bash
python lstm_training/test_sequence_cutoff.py
```

Do not assume repository-wide `test`, `lint`, or setup commands exist. Use the
instructions and dependency files for the component being changed.

## Data and generated files

Do not commit local environments, caches, bulk datasets, or generated model
artifacts. In particular, `.venv/`, `.cache/`, `artifacts/`, `data/hrrr/`,
`data/era5/`, Python `__pycache__/`, `.DS_Store`, and notebook checkpoints are
ignored by Git.

Keep large weather datasets outside the repository when possible. WATcloud
workflows should use the configured bulk-storage data root.

Before adding generated benchmark data, model checkpoints, logs, or
visualizations, check the relevant nested documentation and existing Git
tracking conventions.

## Forecasting and reproducibility rules

- Use UTC for forecast initialization, valid times, and observation matching.
- Preserve the exact station identifiers and units documented in
  `benchmarking-site/AGENTS.md`.
- Avoid data leakage. Training inputs may only use information that would have
  been available at the forecast issuance time.
- Preserve configuration and split information needed to reproduce model results.

## Validation

Use the smallest relevant validation for the component being changed. Follow
nested `AGENTS.md` files for component-specific checks.

Do not claim a command or workflow is supported unless it exists and has been
validated in the current repository.

## Safety and expensive operations

Ask before starting large weather-data downloads, submitting Slurm jobs or large
training/hyperparameter sweeps, changing dataset layouts, performing
repository-wide migrations, or deleting potentially valuable data or artifacts.

Prefer dry runs, smoke tests, and existing cached data when validating changes.

Never commit credentials, tokens, private keys, machine-specific absolute paths,
or other secrets.
