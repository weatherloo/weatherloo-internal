# Repository layout and migration map

This repository separates producer source from applications, published data,
and local run products. The root `AGENTS.md` defines ownership and workflows;
component details stay with their component.

## Target layout

| Home | Ownership |
|---|---|
| `benchmarking-site/` | React/Vite UI, development server, and deploy configuration. |
| `data/benchmarks/` | Versioned benchmark JSON/NPZ, static aggregates, and observations used by the site. |
| `pipelines/benchmarking/<method>/` | Benchmark producer code, per-method Python requirements, and benchmark-specific Slurm/cron entry points. |
| `src/hrrr_bias_correction/` | Shared HRRR bias-correction package. |
| `models/<model>/` | Model implementation, configuration, dependencies, and runbooks. |
| `lstm_training/` | LSTM source, training/inference entry points, dependencies, tests, and runbooks. |
| `scripts/` | Shared data download, preprocessing, benchmark-building, and operational helpers. |
| `outputs/` | Local/generated checkpoints, run results, logs, visualizations, and locks. New output is ignored by Git. |
| `docs/` | Repository/migration guidance, research, meeting notes, project-management notes, and observation documentation. |

Method configuration and dependencies remain beside each producer/model.
Raw weather data and temporary caches stay outside the repository. Downloads
default to `~/weatherloo-data` locally or can target a shared storage location
through `WEATHERLOO_DATA_ROOT`.

The dashboard serves `data/benchmarks/` at `/data/` during development and
copies the small production subset selected in `benchmarking-site/vite.config.js`.
The full published dataset remains versioned in the repository because it is
the dashboard's checked-in input, not a local training cache.

## Migration map

| Legacy path | Canonical path | Notes |
|---|---|---|
| `benchmarking-site/data/<method>/**` benchmark records | `data/benchmarks/<method>/**` | Existing JSON, NPZ, metadata, and published method records were moved intact. |
| `benchmarking-site/data/aggregates/**` | `data/benchmarks/aggregates/**` | Vite still publishes these files under `/data/aggregates/`. |
| `benchmarking-site/data/observations/**` | `data/benchmarks/observations/**` | `vite.config.js` serves/copies required files at the same frontend URLs. |
| `benchmarking-site/data/<method>/compute_benchmark.py` | `pipelines/benchmarking/<method>/compute_benchmark.py` | Old Python paths remain as thin compatibility wrappers; retire them once external invocations have migrated. |
| `benchmarking-site/data/<method>/requirements.txt` | `pipelines/benchmarking/<method>/requirements.txt` | Producer dependencies are owned with producer source. |
| Benchmark Slurm/cron helpers under `benchmarking-site/data/` | `pipelines/benchmarking/<method>/` | Benchmark-specific job entry points now call the canonical producer. |
| `artifacts/` | `outputs/artifacts/` | Historical model artifacts were preserved. |
| `logs/` and model/benchmark `slurm_logs/` | `outputs/logs/` | Historical logs were preserved; future output is ignored. |
| `models/unet/checkpoints/`, `models/unet-hrrr/checkpoints/` | `outputs/models/<model>/checkpoints/` | Model code reads/writes the new canonical checkpoint directories. |
| `models/<model>/eval_results/` | `outputs/models/<model>/eval_results/` | Evaluation outputs remain available, outside model source. |
| `models/<model>/training_log*.csv` and `models/<model>/data/stats*.json` | `outputs/models/<model>/training_logs/` and `outputs/models/<model>/stats/` | Training products are separate from datasets and implementation. |
| `lstm_training/output/` | `outputs/lstm_training/sweeps/` | Sweep outputs retained; new runs are local and ignored. |
| `lstm_training/retrain_output/` | `outputs/lstm_training/retrained/` | Retrained checkpoints and metrics retained. |
| `viz/` and `scripts/era5_test/t2m.gif` | `outputs/viz/` | Generated visualizations retained. |
| `lit-review/` | `docs/research/lit-review/` | Research notes. |
| `meeting_notes/` | `docs/meeting-notes/` | Meeting notes/presentation. |
| `tpm/` | `docs/project-management/` | Project-management notes. |
| Root `package-lock.json` | Removed | Empty orphan npm lockfile; the supported frontend lockfile is `benchmarking-site/package-lock.json`. |
| `.locks/` | `outputs/.locks/` at runtime | Stale committed locks were removed; operational scripts create fresh locks as needed. |

## Compatibility and path resolution

New documentation and automation must use canonical paths. The wrappers at
`benchmarking-site/data/<method>/compute_benchmark.py` exist only to avoid
breaking existing invocations while callers transition. They can be removed
when external scheduled jobs and contributor instructions no longer use them.

Python entry points derive repository/data/output locations from their own
resolved file path. They must not rely on the caller's current working
directory. Relative paths explicitly supplied as CLI arguments retain their
normal meaning relative to the invoking shell.
