#!/bin/bash
# Fetch + cache GFS and ERA5 region grids for U-Net training, on WATcloud Slurm.
#
#   sbatch models/unet/fetch.slurm                    # whole range, one node
#   sbatch --array=1-8 models/unet/fetch.slurm        # split the range 8 ways
#
# Array tasks take contiguous, disjoint date chunks of [START, END]. The fetch
# is resumable (already-cached samples are skipped), so a task that dies can be
# resubmitted and will pick up where it left off.
#
# Tunables (override with --export=ALL,START=2023-01-01):
#   DATA_DIR     cache root (dataset.py appends unet_training/ to it)
#   START, END   init-date range; GFS 0.25° on this bucket starts ~2021-03-23
#   WORKERS      parallel download threads
#   PRUNE_GRIB   1 (default) deletes intermediate GRIB once the .npz is cached
#
#SBATCH --job-name=unet-fetch
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.out
#SBATCH --error=/home/c52li/weatherloo-internal/models/unet/slurm_logs/%x-%j.err

set -euo pipefail

REPO=/home/c52li/weatherloo-internal
cd "$REPO"
mkdir -p models/unet/slurm_logs

DATA_DIR="${DATA_DIR:-/mnt/wato-drive/c52li/weatherloo-data/unet}"
START="${START:-2021-03-23}"          # first day with 0.25° pgrb2 on noaa-gfs-bdp-pds
END="${END:-2025-12-31}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-6}}"
PRUNE_GRIB="${PRUNE_GRIB:-1}"

source "$REPO/.venv/bin/activate"

# Split [START, END] into SLURM_ARRAY_TASK_COUNT contiguous date chunks and take
# this task's. Chunking by date (not by sample) keeps each task's work
# contiguous, so a resubmitted task re-walks only its own range.
N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-1}"
total_days=$(( ( $(date -u -d "$END" +%s) - $(date -u -d "$START" +%s) ) / 86400 + 1 ))
per=$(( (total_days + N - 1) / N ))
off=$(( (I - 1) * per ))

if (( off >= total_days )); then
  echo "shard $I/$N: past the end of the range ($total_days days) — nothing to do"
  exit 0
fi

end_off=$(( off + per - 1 ))
(( end_off > total_days - 1 )) && end_off=$(( total_days - 1 ))
chunk_start=$(date -u -d "$START +$off days" +%F)
chunk_end=$(date -u -d "$START +$end_off days" +%F)

echo "job ${SLURM_JOB_ID} on $(hostname)"
echo "shard $I/$N: $chunk_start .. $chunk_end ($(( end_off - off + 1 )) days of $total_days)"

ARGS=(
  --data-dir "$DATA_DIR"
  --start "$chunk_start"
  --end "$chunk_end"
  --workers "$WORKERS"
)
# Without this the GRIB intermediates dwarf the .npz the training actually
# reads — roughly 1.8 MB/sample vs 21 KB.
[[ "$PRUNE_GRIB" == "1" ]] && ARGS+=(--prune-grib)

srun python3 models/unet/run_pipeline.py fetch "${ARGS[@]}"
