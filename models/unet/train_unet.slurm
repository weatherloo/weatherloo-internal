#!/bin/bash
#SBATCH --job-name=unet-train
# --- GPU resources -----------------------------------------------------------
# NOTE: confirm the partition name and gres syntax for your cluster.
#   run `sinfo -o "%P %G"` to list partitions and their GRES (gpu) resources,
#   then edit the two lines below to match (e.g. --partition=gpu, gpu:a100:1).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# -----------------------------------------------------------------------------
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
nvidia-smi || echo "WARNING: nvidia-smi failed — is this a GPU node? (check --partition/--gres)"
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
