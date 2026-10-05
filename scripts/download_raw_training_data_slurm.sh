#!/usr/bin/env bash
# SLURM worker: download HRRR (+ ERA5) for one date window into DATA_ROOT.
# Usage: download_raw_training_data_slurm.sh <data-root> [extra args...]
# Extra args are forwarded to both downloaders (e.g. --start-date --end-date --resume).
set -euo pipefail

DATA_ROOT="${1:?data-root required (e.g. /mnt/wato-drive/\$USER/weatherloo-data)}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export WEATHERLOO_DATA_ROOT="$DATA_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: missing venv python at $PYTHON" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" logs

echo "=== weatherloo raw download ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "data_root=$DATA_ROOT"
echo "args=$*"
echo "python=$PYTHON"
"$PYTHON" -V

echo "--- HRRR ---"
"$PYTHON" -u scripts/download_hrrr.py --data-root "$DATA_ROOT" "$@"
echo "--- ERA5 ---"
# ERA5 does not accept HRRR-only flags like --workers / --max-lead / --cycles.
ERA5_ARGS=()
skip_next=0
for arg in "$@"; do
  if [[ $skip_next -eq 1 ]]; then
    skip_next=0
    continue
  fi
  case "$arg" in
    --workers|--max-lead|--cycles|--download-retries|--max-cache-gb|--cache-dir)
      skip_next=1
      continue
      ;;
    --shared-cache)
      continue
      ;;
    --workers=*|--max-lead=*|--cycles=*|--download-retries=*|--max-cache-gb=*|--cache-dir=*)
      continue
      ;;
  esac
  ERA5_ARGS+=("$arg")
done
# ERA5 months() accepts YYYY-MM-DD by taking the first two dash fields.
"$PYTHON" -u scripts/download_era5.py --data-root "$DATA_ROOT" "${ERA5_ARGS[@]}"

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
