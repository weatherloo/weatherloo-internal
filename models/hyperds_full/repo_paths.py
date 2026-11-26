"""Put the shared repository modules on sys.path: station_labels (pipelines/download) and
site_scoring (pipelines/benchmarks). On Modal the images copy both files next to this one, so
the directories do not exist there and nothing is added."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
if len(_HERE.parents) > 2:  # in the repository (on Modal this file is /app/repo_paths.py)
    _ROOT = _HERE.parents[2]
    for _path in (_ROOT / "pipelines" / "download", _ROOT / "pipelines" / "benchmarks"):
        if _path.is_dir() and str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
