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