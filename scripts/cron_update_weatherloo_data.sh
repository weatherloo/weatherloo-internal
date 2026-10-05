#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${WEATHERLOO_DATA_ROOT:-/mnt/wato-drive/${USER}/weatherloo-data}"
LOG_DIR="$REPO_ROOT/logs"
LOCK_DIR="$REPO_ROOT/.locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR" "$DATA_ROOT"

timestamp() { date -u +%Y%m%dT%H%M%SZ; }

run_raw() {
  local lock="$LOCK_DIR/raw-babysit.lock"
  local log="$LOG_DIR/cron-raw-babysit-$(timestamp).log"
  flock -n "$lock" bash -lc "
    cd \"$REPO_ROOT\"
    export WEATHERLOO_DATA_ROOT=\"$DATA_ROOT\"
    export START_DATE=\${START_DATE:-2018-07-13}
    export END_DATE=\${END_DATE:-\$(date -u +%Y-%m-%d)}
    export BATCH_SIZE=\${BATCH_SIZE:-6}
    export WORKERS=\${WORKERS:-4}
    export POLL_SECS=\${POLL_SECS:-60}
    echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] cron raw babysit start DATA_ROOT=$DATA_ROOT\" >> \"$log\"
    bash \"$REPO_ROOT/scripts/babysit_raw_downloads.sh\" \"$DATA_ROOT\" >> \"$log\" 2>&1
  " || true
}

run_observations() {
  local lock="$LOCK_DIR/observations.lock"
  local log="$LOG_DIR/cron-observations-$(timestamp).log"
  flock -n "$lock" bash -lc "
    cd \"$REPO_ROOT\"
    export WEATHERLOO_DATA_ROOT=\"$DATA_ROOT\"
    echo \"[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] cron observations start DATA_ROOT=$DATA_ROOT\" >> \"$log\"
    python3 \"$REPO_ROOT/scripts/download_observations.py\" \
      --data-root \"$DATA_ROOT\" \
      --start 2018-07-13 \
      --end \"\$(date -u +%Y-%m-%d)\" \
      --pad-deg 0 \
      --workers 4 \
      --resume >> \"$log\" 2>&1
  " || true
}

mode="${1:-all}"
case "$mode" in
  raw) run_raw ;;
  observations) run_observations ;;
  all)
    run_raw
    run_observations
    ;;
  *)
    echo "Usage: $0 [raw|observations|all]" >&2
    exit 2
    ;;
esac

