#!/usr/bin/env bash
# Submit one Slurm job per ~month window for the raw HRRR+ERA5 backfill.
# Submits BATCH_SIZE jobs at a time (default 10), waits for that batch to finish,
# then submits the next batch.
#
# Usage:
#   scripts/submit_raw_training_data.sh [data-root]
#   BATCH_SIZE=10 START_DATE=2018-07-13 scripts/submit_raw_training_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${1:-${WEATHERLOO_DATA_ROOT:-/mnt/wato-drive/${USER}/weatherloo-data}}"
START="${START_DATE:-2018-07-13}"   # first full HRRRv3 archive day
END="${END_DATE:-$(date -u +%Y-%m-%d)}"
BATCH_SIZE="${BATCH_SIZE:-6}"
POLL_SECS="${POLL_SECS:-60}"
WORKERS="${WORKERS:-4}"

mkdir -p logs "$DATA_ROOT"

SBATCH_OPTS=(
  --partition=compute
  --cpus-per-task=4
  --mem=32G
  --time=12:00:00
  --chdir="$REPO_ROOT"
  --export=ALL
  --output=logs/%x-%j.out
  --error=logs/%x-%j.err
)

wait_for_jobs() {
  local -a ids=("$@")
  if [[ ${#ids[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Waiting for batch: ${ids[*]}"
  while true; do
    local still_running=0
    for jid in "${ids[@]}"; do
      if squeue -j "$jid" -h 2>/dev/null | grep -q .; then
        still_running=$((still_running + 1))
      fi
    done
    if [[ $still_running -eq 0 ]]; then
      break
    fi
    echo "  $still_running/${#ids[@]} still running ($(date -u +%H:%M:%S)Z)"
    sleep "$POLL_SECS"
  done

  # Summarize batch outcomes
  sacct -j "$(IFS=,; echo "${ids[*]}")" \
    --format=JobID,JobName%35,State,ExitCode,Elapsed -n -P 2>/dev/null \
    | awk -F'|' 'NF && $1 !~ /\./ {printf "  %s %s exit=%s %s\n", $1, $3, $4, $5}'
}

# Build all monthly windows first.
declare -a WINDOWS=()
cur="$START"
while [[ "$(date -ud "$cur" +%Y%m%d)" -le "$(date -ud "$END" +%Y%m%d)" ]]; do
  y=$(date -ud "$cur" +%Y)
  m=$(date -ud "$cur" +%m)
  if [[ "$m" == "12" ]]; then
    next_y=$((y + 1))
    next_m=01
  else
    next_y=$y
    next_m=$(printf '%02d' $((10#$m + 1)))
  fi
  window_end="${next_y}-${next_m}-12"
  if [[ "$(date -ud "$window_end" +%Y%m%d)" -gt "$(date -ud "$END" +%Y%m%d)" ]]; then
    window_end="$END"
  fi
  WINDOWS+=("${y}|${m}|${cur}|${window_end}")
  cur="${next_y}-${next_m}-13"
done

total=${#WINDOWS[@]}
echo "Planning $total monthly windows from $START to $END -> $DATA_ROOT"
echo "Batch size: $BATCH_SIZE (poll every ${POLL_SECS}s); HRRR workers=$WORKERS (processes)"

submitted=0
batch_num=0
idx=0
while [[ $idx -lt $total ]]; do
  batch_num=$((batch_num + 1))
  declare -a batch_ids=()
  batch_start=$idx

  echo ""
  echo "=== Batch $batch_num: submitting up to $BATCH_SIZE jobs ==="
  while [[ ${#batch_ids[@]} -lt $BATCH_SIZE && $idx -lt $total ]]; do
    IFS='|' read -r y m window_start window_end <<< "${WINDOWS[$idx]}"
    job_name="weatherloo-raw-download-${y}-${m}"
    jid=$(sbatch --parsable \
      "${SBATCH_OPTS[@]}" \
      --job-name="$job_name" \
      --wrap="bash scripts/download_raw_training_data_slurm.sh $DATA_ROOT --start-date $window_start --end-date $window_end --resume --workers $WORKERS")
    echo "  $job_name  $window_start -> $window_end  job=$jid"
    batch_ids+=("$jid")
    submitted=$((submitted + 1))
    idx=$((idx + 1))
  done

  wait_for_jobs "${batch_ids[@]}"
  echo "Batch $batch_num done ($((idx - batch_start)) jobs)."
done

echo ""
echo "All done: submitted $submitted job(s) in $batch_num batch(es)."
echo "Logs: $REPO_ROOT/logs/weatherloo-raw-download-*-*.out"
