# Weatherloo Repository Guide

Weatherloo contains the benchmarking dashboard, forecast-data pipelines,
bias-correction models, training workflows, and their published benchmark
products. This guide defines ownership of the repository's top-level homes;
`docs/repository-layout.md` records the migration from legacy paths.

More specific instructions take precedence within their directories:

- `benchmarking-site/AGENTS.md` — benchmark methods, dashboard data contracts,
  station IDs, metrics, interpolation, and published benchmark workflows.
- `lstm_training/AGENTS.md` — LSTM sweeps, run outputs, and training workflow.

## Repository layout

| Path | Owns |
|---|---|
| `benchmarking-site/` | React/Vite web application and development server. |
| `data/benchmarks/` | Versioned benchmark records and observations consumed by the site. Producer code does not live here. |
| `src/hrrr_bias_correction/` | Shared Python package for HRRR CNN-LSTM bias correction. |
| `pipelines/benchmarking/<method>/` | Benchmark computation, method dependencies, and benchmark-specific operational entry points. |
| `models/<model>/` | Model-specific source, configs, dependencies, and runbooks (not checkpoints or logs). |
| `lstm_training/` | LSTM training and inference source, dependencies, tests, and instructions. |
| `scripts/` | Shared download, preprocessing, data-building, and repository-level operations. |
| `outputs/` | Generated checkpoints, run outputs, logs, visualizations, and other reproducibility artifacts. New run output is ignored by Git. |
| `docs/` | Repository layout/migration map, research, meeting notes, project-management notes, and observation data documentation. |

Configuration belongs with the component it configures (`models/<model>/`,
`src/hrrr_bias_correction/`, or a benchmark method under
`pipelines/benchmarking/`). `data/benchmarks/` is the intentional exception to
the generated-output rule: it holds published, versioned inputs served by the
dashboard. Raw weather data and caches belong outside the repository, normally
under `$WEATHERLOO_DATA_ROOT` on shared storage or `~/weatherloo-data` locally;
local artifacts belong under `outputs/`.

## Path and workflow rules

- Resolve repository paths from `Path(__file__).resolve()` or the script's
  directory, never from the caller's current working directory. CLI paths
  supplied by users may remain relative to their invocation directory.
- Run documented repository-level commands from the repository root. The site
  has a separate `benchmarking-site/AGENTS.md` workflow.
- Keep benchmark producer code under `pipelines/benchmarking/` and its data in
  `data/benchmarks/`. The site serves this data at `/data/`; its build only
  copies the assets selected in `benchmarking-site/vite.config.js`.
- Old `benchmarking-site/data/<method>/compute_benchmark.py` entry points are
  temporary compatibility wrappers. New references must use
  `pipelines/benchmarking/<method>/compute_benchmark.py`; wrappers may be
  removed once external callers have migrated.
- Do not stage new contents from `outputs/`. Existing versioned historical
  artifacts were moved there intact; do not delete or replace them casually.
- Use UTC for forecast initialization, valid times, and observation matching.
  Preserve exact station identifiers, units, data splits, and reproducibility
  metadata. Training may only use information available at forecast issuance.

## Development and validation

The repository has multiple independent Python and JavaScript workloads; there
is no repository-wide environment or test command.

Benchmarking dashboard:

```bash
cd benchmarking-site
npm install
npm run dev
npm run build
```

LSTM sequence-cutoff smoke test (repository root):

```bash
python lstm_training/test_sequence_cutoff.py
```

Use the nested guides and component dependency files for other workflows. Run
the smallest relevant validation, and do not claim a workflow is supported
unless its entry point has been checked.

## Data, outputs, and safety

- Do not commit local environments, caches, raw weather data, generated run
  outputs, or new model artifacts. `.gitignore` covers common local data,
  caches, and `outputs/`.
- Keep bulk weather data on HDD-backed storage such as
  `/mnt/wato-drive/...`, not in the repository or home directory. Set
  `$WEATHERLOO_DATA_ROOT` for download/training scripts where applicable.
- Before producing or publishing benchmark data, consult
  `benchmarking-site/AGENTS.md` and preserve its output contract.
- Ask before large weather-data downloads, Slurm submissions, large training
  sweeps, dataset-layout changes, broad migrations, or deleting valuable
  artifacts.
- Never commit credentials, tokens, private keys, machine-specific absolute
  paths, or other secrets.
