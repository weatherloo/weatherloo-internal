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
