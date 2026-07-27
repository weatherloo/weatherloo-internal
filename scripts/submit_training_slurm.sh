#!/usr/bin/env bash
# Submit one Slurm GPU job to train the HRRR bias-correction CNN-LSTM.
#
# GPUs on WATcloud are requested as shards (fractional GPU) via
#   --gres shard:<MiB_of_VRAM>[,tmpdisk:<MiB>]
# which is the preferred, GPU-sharing-friendly way to run on the
# GPU-constrained cluster. See:
#   https://cloud.watonomous.ca/docs/compute-cluster/slurm
#
# Usage:
#   scripts/submit_training_slurm.sh [config.json] [extra train.py args...]
#   VRAM_MIB=8192 TIME=08:00:00 scripts/submit_training_slurm.sh
#   scripts/submit_training_slurm.sh src/hrrr_bias_correction/config.default.json --dry-run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-src/hrrr_bias_correction/config.default.json}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS="$*"

PARTITION="${PARTITION:-compute}"   # override if your GPU partition differs
VRAM_MIB="${VRAM_MIB:-8192}"        # per-job GPU shard (MiB of VRAM)
TMPDISK_MIB="${TMPDISK_MIB:-}"      # optional scratch, appended to --gres
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
TIME="${TIME:-08:00:00}"
JOB_NAME="${JOB_NAME:-weatherloo-train-hrrr-bias}"

GRES="shard:${VRAM_MIB}"
if [[ -n "$TMPDISK_MIB" ]]; then
  GRES="${GRES},tmpdisk:${TMPDISK_MIB}"
fi

mkdir -p logs

SBATCH_OPTS=(
  --partition="$PARTITION"
  --gres="$GRES"
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
  --chdir="$REPO_ROOT"
  --export=ALL
  --job-name="$JOB_NAME"
  --output=logs/%x-%j.out
  --error=logs/%x-%j.err
)

echo "Submitting training job:"
echo "  partition=$PARTITION gres=$GRES cpus=$CPUS mem=$MEM time=$TIME"
echo "  config=$CONFIG extra_args=${EXTRA_ARGS:-<none>}"

jid=$(sbatch --parsable \
  "${SBATCH_OPTS[@]}" \
  --wrap="bash scripts/train_hrrr_bias_correction_slurm.sh $CONFIG $EXTRA_ARGS")

echo "Submitted job $jid"
echo "Monitor:  squeue -j $jid"
echo "Logs:     tail -f $REPO_ROOT/logs/${JOB_NAME}-${jid}.out"
echo "Summary:  sacct -j $jid --format=JobID,JobName,State,ExitCode,Elapsed"
