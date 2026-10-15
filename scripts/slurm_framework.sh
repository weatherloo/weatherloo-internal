#!/bin/bash
# Generic Slurm wrapper for the training framework. Same CLI as local; only --env differs.
#
#   sbatch scripts/slurm_framework.sh train  configs/lstm_cyyz_t2m_gfs_interpolated_6h.json
#   sbatch scripts/slurm_framework.sh train  <config> --run-id r1 --resume
#   sbatch --time=02:00:00 scripts/slurm_framework.sh evaluate <config> --run-id r1
#
# Override resources on the sbatch command line rather than editing this file.
#SBATCH --job-name=wl-framework
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: sbatch $0 <train|infer|evaluate|export-benchmark> <config> [extra cli args...]" >&2
    exit 2
fi
VERB="$1"; CONFIG="$2"; shift 2

# sbatch copies the script to a spool dir, so BASH_SOURCE is useless there; use the submit dir.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

module load python/3.11 2>/dev/null || echo "python/3.11 module not available, using system python"
module load cuda/12    2>/dev/null || echo "cuda/12 module not available, proceeding without"

# Reuses the LSTM venv (has torch + numpy); WL_VENV overrides it.
VENV_DIR="${WL_VENV:-$REPO_DIR/lstm_training/.venv}"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/lstm_training/requirements.txt"
fi
source "$VENV_DIR/bin/activate"

echo "job=${SLURM_JOB_ID:-local} verb=$VERB config=$CONFIG started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -m training_framework.cli "$VERB" --config "$CONFIG" --env "${WL_ENV:-watcloud}" "$@"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
