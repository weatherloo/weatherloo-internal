#!/bin/bash
#SBATCH --job-name=unet-hrrr-train
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=models/unet-hrrr/slurm_logs/%x-%j.out
#SBATCH --error=models/unet-hrrr/slurm_logs/%x-%j.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/models/unet-hrrr"
mkdir -p slurm_logs

export UNET_DATA_ROOT="${UNET_DATA_ROOT:-${WEATHERLOO_DATA_ROOT:+${WEATHERLOO_DATA_ROOT}/raw}}"
UNET_DATA_ROOT="${UNET_DATA_ROOT:-$REPO/data/raw}"

# whichever venv actually has torch — your home one, or the mount one you created
source "$REPO/.venv/bin/activate"

python3 train.py \
  --data-dir "$UNET_DATA_ROOT" \
  --output-dir "$REPO/models/unet-hrrr/checkpoints" \
  --epochs 20 \
  --batch-size 4 \
  --patience 3 \
  --station-loss-weight 0.2 \
  --val-year 2025
