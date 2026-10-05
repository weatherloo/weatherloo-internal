#!/usr/bin/env bash
# Submit the whole U-Net pipeline to Slurm as one dependency chain:
#
#   fetch (array) -> train -> dashboard benchmark
#
# Each stage starts only if the previous one succeeded (afterok), so a failed
# fetch will not train on a half-populated cache.
#
# Usage:
#   models/unet/submit_pipeline.sh                    # full range, 8 fetch shards
#   SHARDS=16 models/unet/submit_pipeline.sh
#   START=2024-01-01 END=2025-12-31 models/unet/submit_pipeline.sh
#   STAGES=train,benchmark models/unet/submit_pipeline.sh   # skip the fetch
#
# Everything is resumable: fetch skips cached samples, and the benchmark takes
# --resume, so re-running after a failure costs only the missing work.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/mnt/wato-drive/c52li/weatherloo-data/unet}"
START="${START:-2021-03-23}"
END="${END:-2025-12-31}"
SHARDS="${SHARDS:-8}"
YEAR="${YEAR:-2025}"
STAGES="${STAGES:-fetch,train,benchmark}"

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# Passed through to every stage so all three agree on the cache root — the
# single most common way to get a silently empty run.
COMMON_EXPORT="ALL,DATA_DIR=$DATA_DIR"

echo "repo      : $REPO_ROOT"
echo "data dir  : $DATA_DIR"
echo "stages    : $STAGES"
has_stage fetch && echo "range     : $START .. $END ($SHARDS shards)"
echo

dep=""
add_dep() { dep="--dependency=afterok:$1"; }

if has_stage fetch; then
  fetch_id=$(sbatch --parsable --array="1-$SHARDS" \
    --export="$COMMON_EXPORT,START=$START,END=$END" \
    models/unet/fetch.slurm)
  echo "fetch     : job $fetch_id (array 1-$SHARDS)"
  add_dep "$fetch_id"
fi

if has_stage train; then
  # shellcheck disable=SC2086 — $dep is empty or one flag, both intended.
  train_id=$(sbatch --parsable $dep --export="$COMMON_EXPORT" models/unet/train.slurm)
  echo "train     : job $train_id${dep:+  (after $fetch_id)}"
  add_dep "$train_id"
fi

if has_stage benchmark; then
  # shellcheck disable=SC2086
  bench_id=$(sbatch --parsable $dep --export="$COMMON_EXPORT,YEAR=$YEAR" \
    benchmarking-site/data/unet/run_benchmark.slurm)
  echo "benchmark : job $bench_id${dep:+  (after ${train_id:-$fetch_id})}"
fi

echo
echo "watch:   squeue -u \$USER"
echo "logs:    models/unet/slurm_logs/  and  benchmarking-site/data/unet/slurm_logs/"
echo "after the benchmark finishes, commit benchmarking-site/data/unet/ to publish."
