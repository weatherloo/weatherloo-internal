#!/bin/bash
# U-Net dashboard benchmark over a full year, on WATcloud Slurm.
#
#   sbatch pipelines/benchmarking/unet/run_benchmark.slurm
#
# Shard across nodes with a job array (each task takes a disjoint slice of the
# initializations; --no-export keeps them from racing on index.json/NPZ):
#
#   sbatch --array=1-8 pipelines/benchmarking/unet/run_benchmark.slurm
#
# then finalize once, after the array finishes:
#
#   sbatch --dependency=afterok:<array_jobid> \
#       pipelines/benchmarking/unet/finalize_benchmark.slurm
#
# Tunables (all overridable via --export, e.g. --export=ALL,YEAR=2024):
#   DATA_DIR  cache from `run_pipeline.py fetch` (parent of unet_training/)
#   YEAR      benchmark year (default 2025)
#   WORKERS   worker processes; keep <= --cpus-per-task
#
#SBATCH --job-name=unet-benchmark
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# %j, not %A_%a: for a non-array job %a expands to the NO_VAL sentinel and you
# get "...-12345_4294967294.out". %j is the per-task job id, so array tasks
# still land in separate files; the script logs its own shard index below.
#SBATCH --output=outputs/logs/slurm/benchmark-unet/%x-%j.out
#SBATCH --error=outputs/logs/slurm/benchmark-unet/%x-%j.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
mkdir -p outputs/logs/slurm/benchmark-unet

DATA_DIR="${DATA_DIR:-${WEATHERLOO_DATA_ROOT:-/mnt/wato-drive/${USER:-weatherloo}/weatherloo-data}/unet}"
YEAR="${YEAR:-2025}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-6}}"

source "$REPO/.venv/bin/activate"

# Each worker is a process; without this, BLAS spawns a thread pool per worker
# and they fight over the allocated cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ARGS=(
  --year "$YEAR"
  --data-dir "$DATA_DIR"
  --workers "$WORKERS"
  --resume
)

# In an array job take this task's slice and leave publishing to the finalize
# step; a lone job does the whole year and publishes itself.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" --no-export)
  echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} (array ${SLURM_ARRAY_JOB_ID})"
fi
echo "job ${SLURM_JOB_ID} on $(hostname) | log: ${SLURM_SUBMIT_DIR:-$REPO}/outputs/logs/slurm/benchmark-unet/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

srun python3 pipelines/benchmarking/unet/compute_benchmark.py "${ARGS[@]}"

# The dashboard reads aggregate.json in preference to the per-init JSON, so a
# run that stops after writing the NPZ still shows the previous run's numbers.
# Array tasks leave this to finalize_benchmark.slurm; a lone job finishes here.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python3 scripts/build_site_aggregates.py
fi
