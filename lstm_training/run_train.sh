#!/bin/bash
#SBATCH --job-name=lstm-bias
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$SCRIPT_DIR/logs"

# Load modules (failures are non-fatal)
module load python/3.11 2>/dev/null || echo "python/3.11 module not available, using system python"
module load cuda/12    2>/dev/null || echo "cuda/12 module not available, proceeding without"

# Create venv on first run
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

source "$VENV_DIR/bin/activate"

NPZ="$REPO_DIR/benchmarking-site/data/gfs_interpolated/gfs_interpolated_2025.npz"
OUT_DIR="$SCRIPT_DIR/output"

echo "NPZ:     $NPZ"
echo "Out dir: $OUT_DIR"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python "$SCRIPT_DIR/train.py" \
    --npz        "$NPZ"     \
    --station    cyyz        \
    --variable   t2m         \
    --lead_time  6           \
    --out_dir    "$OUT_DIR"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
