# Training framework

One set of commands to **train, infer, evaluate and publish** any bias-correction model, locally or on WATcloud (Slurm).

- Your model's code lives in an **adapter**: one Python file with five methods.
- The framework provides everything else: config and overrides, dry-run, a standard run directory, protection against overwriting good checkpoints, reproducibility metadata, and export to the benchmarking site.

Two adapters ship today:

| Adapter | Model | Needs |
|---|---|---|
| `lstm_bias_correction` | `BiasLSTM` from `lstm_training/train.py` | torch |
| `ridge_bias_correction` | Ridge regression on lagged bias (baseline, CI and template) | numpy |

---

## 1. Setup (once per machine)

Run all commands from the **repo root**.

```bash
pip install -r lstm_training/requirements.txt         # torch + numpy (ridge only needs numpy)
cp configs/env.local.example.json configs/env.local.json
```

**Get the data.** Benchmark data isn't in git (see [Limitations](#7-limitations)). The framework reads the method NPZs, for example `benchmarking-site/data/gfs_interpolated/gfs_interpolated_2025.npz`. Copy them from WATcloud or a teammate, or regenerate them with that method's `benchmarking-site/data/<method>/compute_benchmark.py`. Check with:

```bash
ls benchmarking-site/data/*/*.npz
```

---

## 2. Train locally

```bash
CFG=configs/lstm_cyyz_t2m_gfs_interpolated_6h.json

# 1. Dry-run: checks config, data and run-dir safety; writes nothing
python -m training_framework.cli train --config $CFG --run-id first --dry-run

# 2. Train
python -m training_framework.cli train --config $CFG --run-id first

# 3. Re-score the checkpoint / re-run predictions
python -m training_framework.cli evaluate --config $CFG --run-id first
python -m training_framework.cli infer    --config $CFG --run-id first

# 4. Publish to the benchmarking site
python -m training_framework.cli export-benchmark --config $CFG --run-id first
```

Change anything without editing the config by passing `--set` (repeatable; values are parsed as JSON):

```bash
python -m training_framework.cli train --config $CFG --run-id lr-test \
    --set model.lr=0.0005 --set data.lead_time=12 --set data.station=eric_d_soulis
```

To train a different station, variable, method or lead time permanently, copy the config and edit its `data` block.

---

## 3. Train on WATcloud (Slurm)

The configs and commands are the same as locally. Only the env file changes.

```bash
# once: fill in your own paths in place of CHANGE_ME
cp configs/env.watcloud.example.json configs/env.watcloud.json

# submit (runs the same CLI with --env watcloud)
sbatch scripts/slurm_framework.sh train  configs/lstm_cyyz_t2m_gfs_interpolated_6h.json --run-id first
sbatch scripts/slurm_framework.sh evaluate configs/lstm_cyyz_t2m_gfs_interpolated_6h.json --run-id first

# more time or memory: pass sbatch flags, don't edit the script
sbatch --time=04:00:00 --mem=16G scripts/slurm_framework.sh train <cfg> --run-id big

# interactive node / no Slurm: just run the CLI directly
python -m training_framework.cli train --config <cfg> --run-id first --env watcloud
```

- Logs go to `logs/<job-name>-<jobid>.out` and `.err`.
- The script creates or reuses a venv at `lstm_training/.venv`. Set `WL_VENV=/path/to/venv` to use another one, and `WL_ENV=<name>` to use a different env file.

---

## 4. Commands

| Command | Does | Writes into the run dir |
|---|---|---|
| `train` | Fits the model and scores the held-out (most recent) split | `checkpoint.*`, `predictions.npz`, `metrics.json`, `manifest.json`, `config.resolved.json` |
| `infer` | Runs the recorded checkpoint over the held-out split | `predictions.npz` |
| `evaluate` | Re-scores the recorded checkpoint | `eval_metrics.json` |
| `export-benchmark` | Publishes `predictions.npz` to the site | `<benchmarking_data_root>/<method_id>/` (per-init JSON, `index.json`, `metadata.json`) |

Flags common to all commands:
- `--config` (required)
- `--env local|watcloud|path/to/env.json` (default `local`)
- `--run-id` (required except for `train`, where it defaults to a timestamp)
- `--set k=v`
- `--dry-run`

`train` also takes `--resume` and `--force`.

### Config files

| File | Contains | In git? |
|---|---|---|
| `configs/<name>.json` | `model` (must include `adapter`), `data`, `run.seed`, optional `export.method_id` / `export.model_label`. Paths use `${data_root}`. | yes |
| `configs/env.<name>.json` | `data_root`, `runs_root`, `benchmarking_data_root`: the only machine-specific values | no (copy from `.example.json`) |

---

## 5. Run directory, safety and reproducibility

```
<runs_root>/<adapter>/<station>_<variable>_<method>_<lead>h/<run_id>/
  manifest.json          status, config hash, timestamps, provenance, checkpoint sha256
  config.resolved.json   exact config used (after overrides and env substitution)
  checkpoint.pt|.npy     file name set by the adapter
  metrics.json  predictions.npz  [eval_metrics.json]
```

`manifest.json` → `provenance` records:
- git commit, branch and dirty flag
- data file path and sha256
- seed and env name
- Python version, platform and hostname
- Slurm job id
- numpy, torch, tensorflow and optuna versions

Together with `config.resolved.json`, that's enough to reproduce or audit a run.

**Safety rules (enforced, not advisory):**
- A `complete` run is never overwritten. Use a new `--run-id`, or pass `--force` deliberately.
- A `failed` or crashed run needs `--resume` (same config only) or `--force`.
- `infer` and `evaluate` refuse a run that isn't `complete`, and a checkpoint whose sha256 doesn't match the one recorded at train time. (`export-benchmark` only needs `predictions.npz` to exist.)

---

## 6. Adding your own model

The model doesn't have to be an LSTM or even PyTorch: TensorFlow, scikit-learn and XGBoost all work, because the adapter brings its own libraries. Copy `adapters/ridge_bias_correction.py` (about 100 lines) as a template.

1. Create `training_framework/adapters/my_model.py` with a class decorated `@register("my_model")`.
2. Add `"my_model": "training_framework.adapters.my_model"` to `_BUILTINS` in `registry.py`.
3. Write a config with `"model": {"adapter": "my_model", ...your hyperparameters}`.
4. Run it with the same commands as above.

Methods to implement:

| Method | Returns |
|---|---|
| `validate(cfg)` | Dict of cheap facts about the data; raise if it's unusable (used by `--dry-run`) |
| `load_dataset(cfg)` | `{"train": (X, y), "val": (X, y), "test": (X, y, timestamps), "normalization": {"mean", "std"} or None}` |
| `fit(cfg, dataset, run_dir)` | Writes the checkpoint and `predictions.npz` (`predictions`, `targets`, `timestamps`); returns a dict containing `metrics_original` or `metrics` |
| `load_checkpoint(cfg, path)` | `(model, device)` |
| `predict(cfg, model, device, dataset, normalization)` | `(predictions_in_original_units, timestamps)`, optionally plus a dict of per-row `station`, `variable` and `lead_time` arrays |

Optional class attributes: `checkpoint_name` (default `checkpoint.pt`), and `model_label`, which is shown in the benchmark `metadata.json`.

**Models covering many stations, variables or lead times:**
- Set `data.lead_time` (or `station`, or `variable`) to a list or `"all"`; the run dir becomes, for example, `cyyz_t2m_gfs_interpolated_all/`.
- Save per-row `station`, `variable` and `lead_time` arrays in `predictions.npz`, and return them from `predict`.
- Export fills each row's own slot.

**Publishing:**
- The site method name defaults to `<adapter prefix>_<data method>`, e.g. `lstm_gfs_interpolated`; override it with `export.method_id`.
- Export **merges** into existing files, so runs for different lead times can publish to one method without wiping each other.
- `metadata.json` lists what's filled in `covers`, and which runs contributed in `source_run_dirs`.
- A new station only needs adding to `benchmarking-site/data/observations/stations.json`.

---

## 7. Limitations

| Limitation | Effect | Workaround / status |
|---|---|---|
| **Benchmark data isn't in git** | A fresh clone has no NPZs, so `train` fails `--dry-run` until you get the data | Copy from WATcloud or regenerate with `compute_benchmark.py`. Shared WATcloud storage is a separate ticket |
| **`--resume` restarts training** | Re-runs `fit` from scratch in the same run dir; it doesn't continue from the last epoch | Safe, since only failed or unfinished runs can be resumed. True resume needs saved optimizer state per adapter |
| **No sweeps or batch runs** | `lstm_training/sweep.py`, `retrain_best.py` and `infer_recent.py` still run outside the framework, without run dirs or provenance; training 12 lead times means 12 commands | Loop over `--set data.lead_time=...` for now |
| **Point (station) metrics only** | Export writes per-station RMSE, MAE and bias; ACC is always `null`; there's no gridded output | Add when a gridded model needs it |
| **Single train/val/test split** | 70/15/15 temporal split, hardcoded in both adapters | Adapter-level choice; make it a config value if needed |
| **WATcloud path untested end to end** | Same code path as local (only the env file differs), but `slurm_framework.sh` hasn't been run on WATcloud yet | Run one job and report back |
| **Only one real model family** | The ridge adapter is a demo/baseline, not an existing team pipeline | Migrate the next real model when one exists |
| **Export doesn't write the consolidated NPZ** | The site falls back to per-init JSON for framework-exported methods | Add when a framework method needs to load fast on the site |

---

## 8. Checking it works

```bash
python tests/test_framework_e2e.py     # or: python -m pytest tests/
```

The test needs numpy only, takes a few seconds and doesn't need the real data. It builds a synthetic dataset in a temp directory and runs all four commands with the ridge adapter. It checks:
- dry-run writes nothing
- provenance is recorded
- a finished run can't be overwritten
- a tampered checkpoint is refused
- a multi-lead export merges into an existing method
