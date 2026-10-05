#!/bin/bash
# Train the residual-correction U-Net on whatever fetch.slurm cached.
#
#   sbatch models/unet/train.slurm
#   sbatch --dependency=afterok:<fetch_jobid> models/unet/train.slurm
#
# Reads only the cached .npz under DATA_DIR — zero network IO — so this can run
# on a node with no outbound access once the fetch is done.
#
# Tunables (override with --export=ALL,EPOCHS=80):
#   DATA_DIR    cache root produced by the fetch step
#   OUTPUT_DIR  where best_model.pt / training_log.csv / stats.json go
#   EPOCHS, BATCH_SIZE, LR, PATIENCE, SPLIT_MODE
#   START, END  restrict to a sub-range of what is cached (default: all)
#
#SBATCH --job-name=unet-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.out
#SBATCH --error=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.err

set -euo pipefail

REPO=/home/c52li/weatherloo-internal
cd "$REPO"
mkdir -p models/unet/slurm_logs

DATA_DIR="${DATA_DIR:-/mnt/wato-drive/c52li/weatherloo-data/unet}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/models/unet/checkpoints}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
SPLIT_MODE="${SPLIT_MODE:-chronological}"

source "$REPO/.venv/bin/activate"

# Training is IO-bound on small .npz reads, not compute (117k params), so spend
# the cores on DataLoader workers and keep BLAS single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
LOADER_WORKERS="${LOADER_WORKERS:-4}"

echo "job ${SLURM_JOB_ID} on $(hostname)"

mkdir -p "$OUTPUT_DIR"

# Two train jobs pointed at one --output-dir race on best_model.pt: the loser's
# early-epoch checkpoint lands on top of the winner's converged one, and
# training_log.csv is left describing a model that no longer exists on disk.
# Take an exclusive lock so the second job says so and exits instead.
exec {lockfd}>"$OUTPUT_DIR/.train.lock"
if ! flock -n "$lockfd"; then
  echo "ERROR: another training job holds $OUTPUT_DIR/.train.lock — refusing to"
  echo "       race it. Wait for it, or use OUTPUT_DIR=<other dir>."
  exit 1
fi

# Even serialized, a rerun would overwrite a good checkpoint with a worse one
# (best-so-far resets every run). Keep the old one before starting.
if [[ -f "$OUTPUT_DIR/best_model.pt" ]]; then
  backup="$OUTPUT_DIR/best_model.$(date -u +%Y%m%dT%H%M%SZ).pt"
  cp -p "$OUTPUT_DIR/best_model.pt" "$backup"
  cp -p "$OUTPUT_DIR/training_log.csv" "${backup%.pt}.training_log.csv" 2>/dev/null || true
  echo "existing checkpoint preserved -> $(basename "$backup")"
fi

ARGS=(
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --patience "$PATIENCE"
  --workers "$LOADER_WORKERS"
  --split-mode "$SPLIT_MODE"
)
[[ -n "${START:-}" ]] && ARGS+=(--start "$START")
[[ -n "${END:-}" ]] && ARGS+=(--end "$END")

srun python3 models/unet/run_pipeline.py train "${ARGS[@]}"

# stats.json auto-recomputes when config.yaml's cycle/lead set changes, but say
# what was actually used so the log records it alongside the checkpoint.
echo "--- stats.json sample space ---"
python3 -c "import json;print(json.load(open('$OUTPUT_DIR/stats.json')).get('sample_space'))" \
  2>/dev/null || echo "(stats.json not in $OUTPUT_DIR)"

# The checkpoint and the log are separate files; if a second run clobbered one
# of them they silently describe different models, and the benchmark would
# happily score whichever landed. Fail here instead of finding out later.
echo "--- checkpoint / log consistency ---"
python3 - "$OUTPUT_DIR" <<'PY'
import csv, sys, torch
from pathlib import Path
out = Path(sys.argv[1])
ck = torch.load(out / "best_model.pt", map_location="cpu", weights_only=True)
rows = list(csv.DictReader(open(out / "training_log.csv")))
best = min(rows, key=lambda r: float(r["val_loss"]))
ck_ep, ck_vl = int(ck["epoch"]), float(ck["val_loss"])
lg_ep, lg_vl = int(best["epoch"]), float(best["val_loss"])
print(f"  checkpoint: epoch {ck_ep} val_loss {ck_vl:.5f}")
print(f"  log best  : epoch {lg_ep} val_loss {lg_vl:.5f}")
if ck_ep != lg_ep or abs(ck_vl - lg_vl) > 1e-6:
    sys.exit("  MISMATCH: best_model.pt does not correspond to training_log.csv "
             "(concurrent or interrupted run clobbered one of them). Retrain.")
print(f"  OK — in_channels={ck.get('in_channels')} lead_channel={ck.get('lead_channel')} "
      f"sample_space={ck.get('sample_space')}")
PY
