#!/bin/bash
#SBATCH --job-name=unet-hrrr-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/c52li/weatherloo-internal/models/unet-hrrr/slurm_logs/%x-%j.out
#SBATCH --error=/home/c52li/weatherloo-internal/models/unet-hrrr/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/c52li/weatherloo-internal/models/unet-hrrr

export UNET_DATA_ROOT=/mnt/wato-drive/c52li/weatherloo-data

# whichever venv actually has torch — your home one, or the mount one you created
source /home/c52li/weatherloo-internal/.venv/bin/activate

python3 train.py \
  --data-dir "$UNET_DATA_ROOT" \
  --output-dir /home/c52li/weatherloo-internal/models/unet-hrrr/checkpoints \
  --epochs 20 \
  --batch-size 4 \
  --patience 3 \
  --station-loss-weight 0.2 \
  --val-year 2025
