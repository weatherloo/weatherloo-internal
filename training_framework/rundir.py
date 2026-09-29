"""Run-directory layout + crash-safe resume.

Every run lives at runs_root/<adapter>/<dataset_tag>/<run_id>/ and carries a
manifest.json tracking its status. The rule that matters: a run marked
"complete" is never silently overwritten.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class RunConflictError(Exception):
    """Raised when writing to a run dir would be unsafe given the flags passed."""


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(resolved_config):
    """Stable hash of a resolved config, excluding "_"-prefixed metadata keys
    (env name, source path) that don't affect what gets trained."""
    data = {k: v for k, v in resolved_config.items() if not k.startswith("_")}
    blob = json.dumps(data, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def dataset_tag(data_cfg):
    method = data_cfg.get("method", "data")
    return f"{data_cfg['station']}_{data_cfg['variable']}_{method}_{data_cfg['lead_time']}h"


def run_dir_path(runs_root, adapter_name, tag, run_id):
    return Path(runs_root) / adapter_name / tag / run_id


def manifest_path(run_dir):
    return Path(run_dir) / "manifest.json"


def read_manifest(run_dir):
    p = manifest_path(run_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_manifest(run_dir, manifest):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    manifest_path(run_dir).write_text(json.dumps(manifest, indent=2))


def check_can_write(run_dir, resolved_config, resume, force):
    """Decide whether it's safe to (re)train into run_dir.

    Returns "fresh" or "resume". Raises RunConflictError otherwise. Never
    creates or modifies anything -- safe to call from --dry-run.
    """
    existing = read_manifest(run_dir)
    if existing is None:
        return "fresh"

    same_config = existing.get("config_hash") == config_hash(resolved_config)

    if existing.get("status") == "complete":
        if force:
            return "fresh"
        raise RunConflictError(
            f"{run_dir} already holds a COMPLETE run "
            f"(config {'matches' if same_config else 'DIFFERS'} from this one). "
            "Pass --force to retrain over it, or use a different --run-id."
        )

    # status is "running" (crashed mid-flight) or "failed"
    if resume:
        if not same_config:
            raise RunConflictError(
                f"{run_dir} exists with a DIFFERENT config than requested -- "
                "can't --resume with changed hyperparameters. Use --force or a new --run-id."
            )
        return "resume"
    if force:
        return "fresh"
    raise RunConflictError(
        f"{run_dir} exists (status={existing.get('status')}) but neither --resume nor --force was given. "
        "Pass --resume to continue it or --force to restart it."
    )
