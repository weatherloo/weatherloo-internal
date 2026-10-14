#!/bin/bash
# Publish step for a sharded U-Net benchmark run: collapse the per-init JSON
# written by the array into index.json + unet_<year>.npz, then rebuild the
# dashboard's static aggregate. Cheap and serial — no GRIB, no inference.
#
#   sbatch --dependency=afterok:<array_jobid> \
#       benchmarking-site/data/unet/finalize_benchmark.slurm
#
#SBATCH --job-name=unet-benchmark-finalize
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarking-site/data/unet/slurm_logs/%x-%j.out
#SBATCH --error=benchmarking-site/data/unet/slurm_logs/%x-%j.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
mkdir -p benchmarking-site/data/unet/slurm_logs

DATA_DIR="${DATA_DIR:-${WEATHERLOO_DATA_ROOT:-$REPO/data}/experiments/unet}"
YEAR="${YEAR:-2025}"

source "$REPO/.venv/bin/activate"

python3 benchmarking-site/data/unet/compute_benchmark.py \
  --year "$YEAR" --data-dir "$DATA_DIR" --export-npz-only

python3 scripts/build_site_aggregates.py

echo "Commit benchmarking-site/data/unet/ and benchmarking-site/data/aggregates/ to publish."
