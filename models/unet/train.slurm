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
