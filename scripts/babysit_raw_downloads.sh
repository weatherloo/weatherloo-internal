#!/usr/bin/env bash
# Keep retrying incomplete HRRR+ERA5 windows until nothing fetchable remains.
# Uses --resume so finished/partial files are patched in place (not deleted).
#
# Usage:
#   nohup scripts/babysit_raw_downloads.sh >/path/to.log 2>&1 &
#   SKIP_PRUNE=1 nohup scripts/babysit_raw_downloads.sh ... &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${1:-${WEATHERLOO_DATA_ROOT:-/mnt/wato-drive/${USER}/weatherloo-data}}"
START="${START_DATE:-2018-07-13}"
END="${END_DATE:-$(date -u +%Y-%m-%d)}"
BATCH_SIZE="${BATCH_SIZE:-6}"
POLL_SECS="${POLL_SECS:-60}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
# Process workers inside each Slurm job (eccodes-safe ProcessPool).
WORKERS="${WORKERS:-4}"
# HRRRv3 typically 37 leads; HRRRv4 49. Drop suspiciously short files for refetch.
MIN_LEADS_PRE_V4="${MIN_LEADS_PRE_V4:-30}"
MIN_LEADS_V4="${MIN_LEADS_V4:-45}"
HRRR_V4_START="20201202"
SKIP_PRUNE="${SKIP_PRUNE:-0}"

# Single-flight with cron_update_weatherloo_data.sh raw
LOCK_FILE="${REPO_ROOT}/.locks/raw-babysit.lock"
mkdir -p "${REPO_ROOT}/.locks"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Another babysit holds $LOCK_FILE; exiting."
  exit 0
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
mkdir -p logs "$DATA_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

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
  [[ ${#ids[@]} -eq 0 ]] && return 0
  log "Waiting for batch: ${ids[*]}"
  while true; do
    local still_running=0
    for jid in "${ids[@]}"; do
      if squeue -j "$jid" -h 2>/dev/null | grep -q .; then
        still_running=$((still_running + 1))
      fi
    done
    [[ $still_running -eq 0 ]] && break
    log "  $still_running/${#ids[@]} still running"
    sleep "$POLL_SECS"
  done
  sacct -j "$(IFS=,; echo "${ids[*]}")" \
    --format=JobID,JobName%35,State,ExitCode,Elapsed -n -P 2>/dev/null \
    | awk -F'|' 'NF && $1 !~ /\./ {printf "  %s %s exit=%s %s\n", $1, $3, $4, $5}'
}

# Delete short/corrupt NetCDFs so --resume will refetch them.
# Do NOT delete files that only lack newly added variables — downloaders patch those.
prune_short_files() {
  if [[ "$SKIP_PRUNE" == "1" ]]; then
    log "Skipping prune (SKIP_PRUNE=1)"
    return 0
  fi
  log "Pruning short/corrupt HRRR NetCDFs..."
  "$PYTHON" - <<PY
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import netCDF4 as nc

root = Path("$DATA_ROOT") / "hrrr"
v4 = "$HRRR_V4_START"
min_pre, min_v4 = int("$MIN_LEADS_PRE_V4"), int("$MIN_LEADS_V4")
core = ("t2m", "u10", "v10", "tp")
paths = [str(p) for p in root.rglob("hrrr_*_t*z.nc")]

def check(p_s: str):
    p = Path(p_s)
    day = p.parent.name
    try:
        with nc.Dataset(p) as ds:
            n = int(ds.dimensions["lead"].size)
            for v in core:
                if v not in ds.variables:
                    return p_s, f"missing core {v}"
        need = min_v4 if day >= v4 else min_pre
        if n < need:
            return p_s, f"short leads={n} need>={need}"
    except Exception as exc:
        return p_s, str(exc)
    return None

removed = 0
with ProcessPoolExecutor(max_workers=16) as pool:
    futs = [pool.submit(check, p) for p in paths]
    for fut in as_completed(futs):
        bad = fut.result()
        if bad is None:
            continue
        p_s, reason = bad
        p = Path(p_s)
        print(f"  rm {p.name} ({reason})")
        p.unlink(missing_ok=True)
        removed += 1
print(f"pruned={removed}")
PY
}

list_incomplete_windows() {
  "$PYTHON" - <<PY
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
import netCDF4 as nc

root = Path("$DATA_ROOT")
hrrr, era5 = root / "hrrr", root / "era5"
start = date.fromisoformat("$START")
end = date.fromisoformat("$END")
cycles = ["00", "06", "12", "18"]
REQ = ("t2m", "u10", "v10", "tp", "q2", "psfc", "pblh", "hgt", "tsk", "ust")

def hrrr_file_ok(p: Path) -> bool:
    try:
        with nc.Dataset(p) as ds:
            for v in REQ:
                if v not in ds.variables:
                    return False
            if getattr(ds.variables["tp"], "accum", None) != "1-hour":
                return False
        return True
    except Exception:
        return False

def era5_file_ok(p: Path) -> bool:
    try:
        with nc.Dataset(p) as ds:
            return all(v in ds.variables for v in REQ)
    except Exception:
        return False

def window_row(item):
    ws_s, we_s, y, m = item
    ws, we = date.fromisoformat(ws_s), date.fromisoformat(we_s)
    d = ws
    while d <= we:
        day_dir = hrrr / f"{d.year}" / d.strftime("%Y%m%d")
        for c in cycles:
            p = day_dir / f"hrrr_{d.strftime('%Y%m%d')}_t{c}z.nc"
            if not p.is_file() or not hrrr_file_ok(p):
                return f"{y:04d}|{m:02d}|{ws.isoformat()}|{we.isoformat()}"
        d += timedelta(days=1)
    my, mm = ws.year, ws.month
    while (my, mm) <= (we.year, we.month):
        if date(my, mm, 1) > end.replace(day=1) - timedelta(days=100):
            my, mm = (my + 1, 1) if mm == 12 else (my, mm + 1)
            continue
        ep = era5 / f"{my}" / f"era5_{my}{mm:02d}.nc"
        if not ep.is_file() or not era5_file_ok(ep):
            return f"{y:04d}|{m:02d}|{ws.isoformat()}|{we.isoformat()}"
        my, mm = (my + 1, 1) if mm == 12 else (my, mm + 1)
    return None

windows = []
cur = start
while cur <= end:
    y, m = cur.year, cur.month
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    wend = date(ny, nm, 12)
    if wend > end:
        wend = end
    windows.append((cur.isoformat(), wend.isoformat(), y, m))
    cur = date(ny, nm, 13)

with ProcessPoolExecutor(max_workers=16) as pool:
    for row in pool.map(window_row, windows, chunksize=1):
        if row:
            print(row)
PY
}

coverage_summary() {
  "$PYTHON" - <<PY
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
import netCDF4 as nc

root = Path("$DATA_ROOT")
hrrr = root / "hrrr"
start = date.fromisoformat("$START")
end = date.fromisoformat("$END")
cycles = ["00", "06", "12", "18"]
REQ = ("t2m", "u10", "v10", "tp", "q2", "psfc", "pblh", "hgt", "tsk", "ust")
full = missing = days_any = 0
d = start
while d <= end:
    day_dir = hrrr / f"{d.year}" / d.strftime("%Y%m%d")
    have = set()
    if day_dir.is_dir():
        for p in day_dir.glob("hrrr_*_t*z.nc"):
            have.add(p.name.split("_t")[-1][:2])
    if have:
        days_any += 1
    if all(c in have for c in cycles):
        full += 1
    else:
        missing += sum(1 for c in cycles if c not in have)
    d += timedelta(days=1)
era = len(list((root / "era5").rglob("*.nc")))
nc = len(list(hrrr.rglob("*.nc")))

def hrrr_ok(p_s: str) -> bool:
    try:
        with nc.Dataset(p_s) as ds:
            if not all(v in ds.variables for v in REQ):
                return False
            return getattr(ds.variables["tp"], "accum", None) == "1-hour"
    except Exception:
        return False

paths = [str(p) for p in hrrr.rglob("hrrr_*_t*z.nc")]
ok = 0
# Sample up to 400 files for a cheap var-completeness estimate; full check is expensive.
sample = paths[:: max(1, len(paths) // 400)] if paths else []
with ProcessPoolExecutor(max_workers=16) as pool:
    ok = sum(1 for r in pool.map(hrrr_ok, sample) if r)
era_ok = 0
for p in (root / "era5").rglob("era5_*.nc"):
    try:
        with nc.Dataset(p) as ds:
            if all(v in ds.variables for v in REQ):
                era_ok += 1
    except Exception:
        pass
print(
    f"hrrr_nc={nc} full_days={full} days_any={days_any} missing_cycles={missing} "
    f"era5_months={era} era5_vars_ok={era_ok} "
    f"hrrr_vars_ok_sample={ok}/{len(sample)}"
)
PY
}

# Wait out any foreign download jobs from a prior submitter before we reclaim.
drain_foreign_jobs() {
  while true; do
    local n
    n=$(squeue -u "$USER" -h 2>/dev/null | grep -c weatherl || true)
    [[ "${n:-0}" -eq 0 ]] && break
    log "Waiting for $n existing weatherloo job(s) to finish before babysit takeover..."
    sleep "$POLL_SECS"
  done
}

log "Babysit start DATA_ROOT=$DATA_ROOT range=$START..$END batch=$BATCH_SIZE workers=$WORKERS SKIP_PRUNE=$SKIP_PRUNE"
drain_foreign_jobs

prev_incomplete=-1
stable_rounds=0

for round in $(seq 1 "$MAX_ROUNDS"); do
  log "===== ROUND $round ====="
  prune_short_files
  mapfile -t INCOMPLETE < <(list_incomplete_windows)
  total=${#INCOMPLETE[@]}
  coverage_summary
  log "Incomplete windows: $total"

  if [[ $total -eq 0 ]]; then
    log "All windows complete. Done."
    exit 0
  fi

  # Stall detection on incomplete *windows* (var-complete), not just file existence.
  if [[ "$total" -eq "$prev_incomplete" ]]; then
    stable_rounds=$((stable_rounds + 1))
  else
    stable_rounds=0
  fi
  prev_incomplete=$total
  if [[ $stable_rounds -ge 3 ]]; then
    log "No progress for 3 rounds (incomplete_windows=$total). Remaining gaps are likely unpublished. Stopping."
    list_incomplete_windows | head -40
    exit 0
  fi

  submitted=0
  idx=0
  batch_num=0
  while [[ $idx -lt $total ]]; do
    batch_num=$((batch_num + 1))
    declare -a batch_ids=()
    log "=== Round $round batch $batch_num ==="
    while [[ ${#batch_ids[@]} -lt $BATCH_SIZE && $idx -lt $total ]]; do
      IFS='|' read -r y m window_start window_end <<< "${INCOMPLETE[$idx]}"
      job_name="weatherloo-raw-download-${y}-${m}"
      jid=$(sbatch --parsable \
        "${SBATCH_OPTS[@]}" \
        --job-name="$job_name" \
        --wrap="bash scripts/download_raw_training_data_slurm.sh $DATA_ROOT --start-date $window_start --end-date $window_end --resume --workers $WORKERS")
      log "  $job_name $window_start->$window_end job=$jid"
      batch_ids+=("$jid")
      submitted=$((submitted + 1))
      idx=$((idx + 1))
    done
    wait_for_jobs "${batch_ids[@]}"
  done
  log "Round $round submitted $submitted jobs."
done

log "Hit MAX_ROUNDS=$MAX_ROUNDS without full completion."
coverage_summary
exit 1
