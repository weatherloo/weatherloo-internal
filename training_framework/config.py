"""Config loading: one JSON file per run, dot-path CLI overrides, and
path substitution so the same config works locally and on WatCloud.

ponytail: plain JSON, no YAML dependency -- stdlib covers this.
"""
import copy
import json
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def apply_overrides(cfg, overrides):
    """Apply ["model.lr=0.0005", "data.station=cyyz"] style overrides.

    Values are parsed as JSON when possible (so `--set model.lr=0.001` and
    `--set data.station=cyyz` both do the right thing), else kept as strings.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key.path=value")
        path, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def _substitute(value, env):
    if isinstance(value, str):
        for key, val in env.items():
            value = value.replace("${" + key + "}", str(val))
        return value
    if isinstance(value, dict):
        return {k: _substitute(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, env) for v in value]
    return value


def load_env(env_name, repo_root):
    env_path = Path(repo_root) / "configs" / f"env.{env_name}.json"
    if not env_path.exists():
        raise FileNotFoundError(
            f"No env file for '{env_name}': {env_path} "
            "(add configs/env.<name>.json with data_root / runs_root)"
        )
    env = load_json(env_path)
    env.setdefault("repo_root", str(repo_root))
    return env


def resolve_config(config_path, overrides, env_name, repo_root):
    """Load a run config, apply CLI overrides, then substitute ${data_root}
    style tokens using the chosen environment's path roots.

    Returns the resolved dict plus two "_"-prefixed keys carrying metadata
    that callers may need but that are excluded from the run's identity hash
    (see rundir.config_hash).
    """
    cfg = load_json(config_path)
    cfg = apply_overrides(cfg, overrides)
    env = load_env(env_name, repo_root)
    cfg = _substitute(cfg, env)
    cfg["_env_name"] = env_name
    cfg["_env_values"] = env
    cfg["_config_path"] = str(Path(config_path).resolve())
    return cfg
