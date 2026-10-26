# Repository structure

Source code, applications, pipelines, configuration, documentation, tests, and
generated outputs each have one home. Bulk data follows the
`WEATHERLOO_DATA_ROOT` layout in the root `AGENTS.md`; the repo-local `data/`,
`.cache/`, `artifacts/`, and `logs/` directories are ignored fallbacks for local
runs only.

## Top-level ownership

| Path | Owns | Rules |
|---|---|---|
| `benchmarking-site/` | React/Vite dashboard, optional API server, and the **published** benchmark data it serves (`data/<method>/`, `data/aggregates/`, `data/observations/`) | No pipeline code. Vercel root directory and `npm run build` are unchanged. |
| `pipelines/download/` | Raw HRRR/ERA5/observation download and preprocessing CLIs, shared downloader helpers | Writes to `WEATHERLOO_DATA_ROOT`, never into the repo. |
| `pipelines/benchmarks/` | `build_*.py` benchmark builders; `methods/<method>/` compute entry points, requirements, and method-specific Slurm/cron helpers | Outputs go to `benchmarking-site/data/<method>/`. |
| `src/` | Shared Python packages (`hrrr_bias_correction`) | Importable code and its default config only. |
| `models/` | Model implementations plus their training/evaluation entry points, configs, and runbooks (`unet`, `unet-hrrr`, `lstm`) | Checkpoints, logs, and run outputs are ignored (see below). |
| `ops/slurm/` | Slurm submit/cron/babysit scripts | Scripts `cd` to the repo root themselves; run from anywhere. |
| `ops/storage/` | Storage migration tooling | |
| `tests/` | Repo-level tests | Model-local tests stay with the model. |
| `docs/` | Literature review, meeting notes, TPM material, visualizations, this document | |

Ignored local locations: `data/`, `.cache/`, `artifacts/`, `logs/`, `.locks/`,
`**/slurm_logs/`, `models/*/checkpoints/`, `models/*/eval_results/`,
`models/*/training_log*.csv`, `models/lstm/retrain_output/`, and
`models/lstm/output/` (except the small shared `sweep_results.json` files).

## Migration map

| Old path | New path |
|---|---|
| `scripts/download_*.py`, `fetch_*.py`, `weather_download_common.py`, `verify_hrrr_era5_units.py`, `sanity_gif_hrrr.py`, `era5_test/`, `requirements.txt`, `README_download.md` | `pipelines/download/` |
| `scripts/build_hindcast.py`, `build_lstm_benchmark.py`, `build_site_aggregates.py` | `pipelines/benchmarks/` |
| `scripts/submit_*.sh`, `babysit_raw_downloads.sh`, `cron_update_weatherloo_data.sh`, `download_raw_training_data_slurm.sh`, `train_hrrr_bias_correction_slurm.sh` | `ops/slurm/` |
| `scripts/migrate_weatherloo_storage.py` | `ops/storage/migrate_weatherloo_storage.py` |
| `scripts/tests/` | `tests/` |
| `benchmarking-site/data/<method>/{compute_benchmark.py,requirements.txt,*.slurm,*.sh,crontab.example}` | `pipelines/benchmarks/methods/<method>/` |
| `lstm_training/` | `models/lstm/` |
| `lstm_training/test_sequence_cutoff.py` | `tests/test_sequence_cutoff.py` |
| `lit-review/`, `meeting_notes/`, `tpm/`, `viz/` | `docs/lit-review/`, `docs/meeting_notes/`, `docs/tpm/`, `docs/viz/` |
| `benchmarking-site/data/unet/slurm_logs/` | `logs/slurm/unet-benchmark/` (ignored) |

Removed from version control (still in Git history; regenerate or restore from
`WEATHERLOO_DATA_ROOT`): `logs/`, `.locks/`, `artifacts/`, the empty root
`package-lock.json`, `models/*/checkpoints/`, `models/*/slurm_logs/`,
`models/*/eval_results/`, `models/*/training_log*.csv`,
`lstm_training/{retrain_output,*.zip,*.docx,ensemble_results.png}`, stray
`lstm_training/{climatology,ecmwifs}/`, `lstm_training/output/` files other than
`sweep_results.json`, and `scripts/era5_test/t2m.gif`.

## Compatibility

No wrappers were added: every moved entry point resolves the repo root from its
own location, so old paths fail fast instead of silently working from a stale
location. Update crontabs, Slurm wrappers, and shell history to the new paths
above. If a temporary shim is ever needed, place it at the old path and delete
it once crontabs and docs no longer reference it.

## Follow-ups (intentionally not done)

- Moving `benchmarking-site/` under `apps/` would require changing the Vercel
  root directory.
- `models/lstm/` output defaults remain repo-relative; moving them under
  `WEATHERLOO_DATA_ROOT/experiments/` would change workflow behavior.
