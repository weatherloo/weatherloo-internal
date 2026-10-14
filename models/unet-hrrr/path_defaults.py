from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def resolve_default_data_dir(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    if env.get("UNET_DATA_ROOT"):
        return str(Path(env["UNET_DATA_ROOT"]).expanduser())
    if env.get("WEATHERLOO_DATA_ROOT"):
        return str(Path(env["WEATHERLOO_DATA_ROOT"]).expanduser() / "raw")
    return str((HERE.parents[1] / "data" / "raw").resolve())
