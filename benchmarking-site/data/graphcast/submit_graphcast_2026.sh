#!/bin/bash
#SBATCH --job-name=graphcast-2026
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Backfills GraphCast forecast benchmark data for 2026 by reusing the
# existing compute_benchmark.py pipeline (same one used for 2025) — no new
# fetch logic, same station/variable/lead-time/output-format contract as
# every other method under benchmarking-site/data/.
#
# I/O-bound (network downloads + light GRIB decode), not compute-bound:
# CPU-only, no --gres=gpu, modest --mem. --time is generous because the
# bottleneck is NOAA S3 download latency, not CPU.
#
# Safe to resubmit: --resume skips any init whose output JSON already
# exists without re-fetching or re-decoding anything, so a time-limit kill
# or manual interruption just picks back up where it left off.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/logs"

# --- venv (created on first run, reused after) ---------------------------
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR"
    module load python/3.11 2>/dev/null || echo "python/3.11 module not available, using system python"
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

source "$VENV_DIR/bin/activate"

# --- IMPORTANT: verify eccodes is a native build before raising --workers -
# On the machine this wrapper was originally developed on (Windows, local),
# the pip-installed `eccodes` package turned out to be a WASM/Emscripten
# build (definitions loaded from an in-memory "/MEMFS/..." path), which
# crashed under ThreadPoolExecutor with >1 worker (concurrent access to the
# WASM runtime's shared state corrupted its GRIB definitions parser).
# --workers 1 avoided it there. That may or may not apply on this cluster —
# not yet verified here. Before bumping WORKERS below, manually check on
# this login/compute node:
#
#   source .venv/bin/activate
#   python -c "import eccodes; print(eccodes.__file__)"
#   python -c "import eccodes; print(eccodes.codes_get_api_version())"
#
# If `eccodes.__file__` points to a real installed native package (e.g.
# under site-packages with a compiled .so, not any "wasm"/"MEMFS"-adjacent
# path) AND a short manual test with --workers 2 on a handful of inits
# (e.g. `--dry-run`) completes cleanly with no "flex scanner" / "MEMFS"
# errors, it's likely safe to raise this. Until then, leave at 1.
WORKERS=1

echo "Workers:  $WORKERS  (see comment above before changing)"
echo "Out dir:  $SCRIPT_DIR"
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python "$SCRIPT_DIR/compute_benchmark.py" \
    --year 2026 \
    --resume \
    --workers "$WORKERS"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Output NPZ: $SCRIPT_DIR/graphcast_2026.npz"
