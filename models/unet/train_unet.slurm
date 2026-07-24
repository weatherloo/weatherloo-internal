#!/bin/bash
#SBATCH --job-name=unet-train
# --- Partition + GPU (WATonomous cluster) ------------------------------------
# WATonomous has NO dedicated gpu partition: you stay on `compute` (max 1 day)
# and request the GPU via --gres. Use `compute_dense` (max 7 days) instead if a
# run could exceed 24h. Docs: https://cloud.watonomous.ca/docs/compute-cluster/slurm
#
# GPU: use a TYPED SHARD (fractional GPU pinned to a model). Two reasons:
#   1. Pin a MODERN type -> never land on the old GTX 1080 (Pascal sm_61), whose
#      kernels are absent from modern PyTorch wheels ("CUDA error: no kernel
#      image is available for execution on the device").
#   2. Fractional -> schedules right away by packing onto a partially-used GPU
#      (WATonomous docs recommend shard over whole GPUs).
#   This ~117k-param model on a 21x41 grid needs well under 4 GiB.
#
#   Pick a type that is (a) supported and (b) on an UP node. Check live state:
#       sinfo -p compute -N -o "%N %t %G %E"
#   As of last check: rtx_2080_ti (sm_75, delta-slurm1 up) and rtx_4090 (sm_89,
#   trpro-slurm2 idle) work; rtx_3090's node was drained (SlurmdSpoolDir full);
#   gtx_1080 is unsupported. 2080 Ti is the safest — sm_75 is in every modern
#   torch build. Swap the type below if the node map changes.
#   Alternatives:  --gres shard:rtx_4090:4096   (newest; needs torch w/ sm_89)
#                  --gres shard:rtx_3090:4096   (once trpro-slurm1 is undrained)
#SBATCH --partition=compute
#SBATCH --gres=shard:rtx_2080_ti:4096
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
