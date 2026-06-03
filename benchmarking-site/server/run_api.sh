#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  exec "$REPO/.venv/bin/python" "$ROOT/server/npz_api.py" "$@"
fi
exec python3 "$ROOT/server/npz_api.py" "$@"
