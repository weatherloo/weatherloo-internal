#!/bin/bash
#SBATCH --job-name=unet-train
# --- Partition + GPU (WATonomous cluster) ------------------------------------
# WATonomous has NO dedicated gpu partition: you stay on `compute` (max 1 day)
# and request the GPU via --gres. Use `compute_dense` (max 7 days) instead if a
# run could exceed 24h. Docs: https://cloud.watonomous.ca/docs/compute-cluster/slurm
#
# GPU options (pick ONE --gres line):
#   fractional (recommended, neighbour-friendly): --gres shard:8192,tmpdisk:0
#       8192 MiB VRAM is plenty for this ~117k-param model on a 21x41 grid.
#   whole GPU:            --gres gpu:1
#   whole GPU of a type:  --gres gpu:rtx_3090:1   (run `sinfo -o "%N %G"` to list)
#SBATCH --partition=compute
#SBATCH --gres=shard:8192
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.out
#SBATCH --error=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.err

set -euo pipefail

REPO=/home/c52li/weatherloo-internal
DATA_DIR=/mnt/wato-drive/c52li/weatherloo-data/unet
OUTPUT_DIR=/mnt/wato-drive/c52li/unet/checkpoints

cd "$REPO/models/unet"

# venv that actually has torch (+CUDA build). Same one used for fetch/train.
source "$REPO/.venv/bin/activate"

# DataLoader workers: match the CPUs we asked SLURM for.
WORKERS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== GPU visibility ==="
nvidia-smi || echo "WARNING: nvidia-smi failed — no GPU allocated? (check the --gres line)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

python3 run_pipeline.py train \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 50 \
  --batch-size 16 \
  --patience 10 \
  --workers "$WORKERS" \
  --split-mode chronological
