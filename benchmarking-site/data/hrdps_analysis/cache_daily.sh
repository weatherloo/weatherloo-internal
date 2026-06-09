#!/usr/bin/env bash
# Fetch HRDPS PT000H analysis GRIB (t2m, 10 m u/v) into .cache/hrdps_grib/.
#
# MSC Datamart keeps ~30 days on the server; this script preserves a local
# archive so compute_benchmark.py can backfill benchmark JSON later.
#
# Install (crontab -e), run ~2.5 h after each 00/06/12/18Z cycle:
#   30 2,8,14,20 * * * /path/to/weatherloo-internal/benchmarking-site/data/hrdps_analysis/cache_daily.sh >> /path/to/weatherloo-internal/.cache/hrdps_grib/cache.log 2>&1
#
# See crontab.example and benchmarking-site/AGENTS.md for details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

mkdir -p "${REPO_ROOT}/.cache/hrdps_grib"

exec "${PYTHON}" "${SCRIPT_DIR}/compute_benchmark.py" \
  --cache-only \
  --cache-days 3
