# LSTM Bias Correction

Trains an LSTM to predict the next-step forecast bias for a given station, variable, and lead time.

## Input data

NPZ files from `benchmarking-site/data/` with keys:
- `bias` shape `(n_inits, 2, 2, 12)` for `[inits, stations, variables, lead_times]`
- `station_ids`: `['cyyz', 'eric_d_soulis']`
- `variables`: `['t2m', 'wind_speed']`
- `lead_times_hours`: `[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]`
- `initializations`: ISO-8601 timestamp strings

## Local usage

```bash
# Install dependencies
pip install -r lstm_training/requirements.txt

# Train on GFS interpolated bias for CYYZ t2m at 6h lead
python lstm_training/train.py \
    --npz  benchmarking-site/data/gfs_interpolated/gfs_interpolated_2025.npz \
    --station    cyyz \
    --variable   t2m \
    --lead_time  6 \
    --out_dir    lstm_training/output
```

Logs a line every 5 epochs. Stops early once validation loss stops improving.

## Cluster usage (Slurm)

```bash
cd /path/to/weatherloo-internal
sbatch lstm_training/run_train.sh
```

The script creates a venv under `lstm_training/.venv/` on the first run and reuses it on subsequent runs. Logs go to `lstm_training/logs/<job_id>.out`.

## Outputs

All written to `--out_dir` (default `lstm_training/output/`):

| File | Contents |
|---|---|
| `best_model.pt` | Model weights at lowest validation loss |
| `config.json` | All hyperparams, normalization stats, split sizes, and test metrics |
| `predictions.npz` | `predictions`, `targets` (denormalized), `train_losses`, `val_losses` |

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--npz` | required | Path to benchmarking NPZ file |
| `--station` | `cyyz` | Station id |
| `--variable` | `t2m` | Variable name |
| `--lead_time` | `6` | Lead time in hours |
| `--seq_len` | `24` | Sliding window length |
| `--hidden_size` | `64` | LSTM hidden units |
| `--num_layers` | `2` | LSTM layers |
| `--dropout` | `0.1` | Dropout between LSTM layers |
| `--lr` | `0.001` | Adam learning rate |
| `--batch_size` | `32` | Mini-batch size |
| `--max_epochs` | `200` | Maximum training epochs |
| `--patience` | `15` | Early stopping patience |
| `--max_norm` | `1.0` | Gradient clipping max norm |
| `--out_dir` | `lstm_training/output` | Output directory |
