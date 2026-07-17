#!/usr/bin/env bash
# Build static aggregate.json files from consolidated NPZ benchmarks.
# Picks a Python with numpy (repo .venv first, then python3). On hosts with
# no numpy (e.g. Vercel's build image), falls back to committed aggregate.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"

PY=""
if [[ -x "$REPO/.venv/bin/python" ]] && "$REPO/.venv/bin/python" -c "import numpy" >/dev/null 2>&1; then
  PY="$REPO/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import numpy" >/dev/null 2>&1; then
  PY="python3"
fi

if [[ -z "$PY" ]]; then
  existing=$(find "$ROOT/data" -mindepth 2 -maxdepth 2 -name aggregate.json 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$existing" -gt 0 ]]; then
    echo "[build_static_aggregates] No Python with numpy found; keeping $existing committed aggregate.json file(s)." >&2
    exit 0
  fi
  echo "[build_static_aggregates] ERROR: no Python with numpy and no committed aggregate.json files." >&2
  echo "Run 'pip install numpy' (or use the repo .venv) and re-run npm run build." >&2
  exit 1
fi

exec "$PY" "$ROOT/scripts/build_static_aggregates.py" "$@"
