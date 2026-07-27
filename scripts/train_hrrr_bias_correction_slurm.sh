#!/usr/bin/env bash
# SLURM worker: train the HRRR bias-correction CNN-LSTM on one GPU shard.
# Usage: train_hrrr_bias_correction_slurm.sh [config.json] [extra train.py args...]
# Extra args are forwarded to train.py (e.g. --dry-run).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-src/hrrr_bias_correction/config.default.json}"
if [[ $# -gt 0 ]]; then shift; fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src/hrrr_bias_correction${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: missing venv python at $PYTHON" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r src/hrrr_bias_correction/requirements.txt" >&2
  exit 1
fi

mkdir -p logs

echo "=== weatherloo hrrr bias-correction training ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "extra_args=$*"
echo "python=$PYTHON"
"$PYTHON" -V
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU ---"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
fi

echo "--- train ---"
"$PYTHON" -u src/hrrr_bias_correction/train.py --config "$CONFIG" "$@"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
