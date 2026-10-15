"""One CLI, four verbs, same shape locally and on WatCloud:

    python -m training_framework.cli train    --config <path> [--env local|watcloud] [--dry-run] [--set k=v ...]
    python -m training_framework.cli evaluate  --config <path> --run-id <id> [--env ...]
    python -m training_framework.cli infer     --config <path> --run-id <id> [--env ...]
    python -m training_framework.cli export-benchmark --config <path> --run-id <id> [--env ...]

Model-specific behavior lives entirely in the adapter named by
config["model"]["adapter"]; this file only owns config resolution, the
run-directory/manifest lifecycle, and dispatch.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training_framework import benchmark_export, config as cfgmod, rundir
from training_framework.metrics import compute_metrics
from training_framework.registry import get as get_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def _add_common_args(p, require_run_id=False):
    p.add_argument("--config", required=True, help="Path to the run's JSON config")
    p.add_argument("--env", default="local", help="Which configs/env.<name>.json to resolve paths from")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override a config value, e.g. --set model.lr=0.0005 (repeatable)")
    p.add_argument("--run-id", default=None, required=require_run_id,
                   help="Run id under the run directory. Defaults to a fresh timestamp for `train`.")


def _resolve(args):
    return cfgmod.resolve_config(args.config, args.overrides, args.env, REPO_ROOT)


def _run_dir_for(resolved, run_id):
    adapter_name = resolved["model"]["adapter"]
    tag = rundir.dataset_tag(resolved["data"])
    runs_root = resolved["_env_values"]["runs_root"]
    return rundir.run_dir_path(runs_root, adapter_name, tag, run_id)


def _checkpoint_path(adapter, run_dir):
    # Adapters may override the filename (default keeps existing LSTM runs valid).
    return Path(run_dir) / getattr(adapter, "checkpoint_name", "checkpoint.pt")


def _verified_checkpoint(adapter, run_dir):
    """Checkpoint path, refusing one that isn't what `train` recorded (swapped,
    half-written, or from an unfinished run). Pre-provenance runs skip the hash check."""
    ckpt = _checkpoint_path(adapter, run_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}; train this run first")
    manifest = rundir.read_manifest(run_dir) or {}
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{run_dir} status is {manifest.get('status')!r}, not 'complete'")
    expected = manifest.get("checkpoint", {}).get("sha256")
    if expected and rundir.file_sha256(ckpt) != expected:
        raise RuntimeError(f"{ckpt} does not match the sha256 recorded at train time")
    return ckpt


def _test_targets(dataset):
    """Held-out targets in original units. Dataset contract: dataset["test"] is
    (X, y, timestamps); dataset["normalization"] ({mean, std}) is optional."""
    y = dataset["test"][1]
    norm = dataset.get("normalization")
    return y * norm["std"] + norm["mean"] if norm else y


def _predict(adapter, resolved, model, device, dataset):
    """-> (preds, timestamps, extra). `extra` holds optional per-row columns
    (station / variable / lead_time) for multi-slot models; predict may return 2 or 3 items."""
    out = adapter.predict(resolved, model, device, dataset, dataset.get("normalization"))
    return (*out, {}) if len(out) == 2 else out


def _default_run_id():
    return rundir.now_iso().replace(":", "").replace("-", "")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    resolved = _resolve(args)
    adapter = get_adapter(resolved["model"]["adapter"])
    run_id = args.run_id or resolved.get("run", {}).get("run_id") or _default_run_id()
    run_dir = _run_dir_for(resolved, run_id)

    if args.dry_run:
        print(f"[dry-run] config={args.config} env={args.env}")
        info = adapter.validate(resolved)
        print(f"[dry-run] data OK: {json.dumps(info)}")
        try:
            status = rundir.check_can_write(run_dir, resolved, args.resume, args.force)
        except rundir.RunConflictError as e:
            print(f"[dry-run] run dir CONFLICT: {e}")
            sys.exit(1)
        print(f"[dry-run] run dir: {run_dir}  -> would be a '{status}' run")
        print("[dry-run] OK -- nothing was trained or written.")
        return

    status = rundir.check_can_write(run_dir, resolved, args.resume, args.force)
    manifest = rundir.read_manifest(run_dir) if status == "resume" else {}
    manifest.update({
        "status": "running",
        "run_id": run_id,
        "adapter": resolved["model"]["adapter"],
        "config_hash": rundir.config_hash(resolved),
        "started_at": manifest.get("started_at") or rundir.now_iso(),
        "provenance": rundir.provenance(resolved, REPO_ROOT),
    })
    if status == "resume":
        manifest.setdefault("resumed_at", []).append(rundir.now_iso())
    rundir.write_manifest(run_dir, manifest)
    (Path(run_dir) / "config.resolved.json").write_text(json.dumps(resolved, indent=2))

    try:
        dataset = adapter.load_dataset(resolved)
        result = adapter.fit(resolved, dataset, str(run_dir))
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = rundir.now_iso()
        rundir.write_manifest(run_dir, manifest)
        raise

    ckpt = _checkpoint_path(adapter, run_dir)
    manifest.update({
        "status": "complete",
        "completed_at": rundir.now_iso(),
        # lets evaluate/infer/export prove they used the checkpoint this run produced
        "checkpoint": {"file": ckpt.name, "sha256": rundir.file_sha256(ckpt) if ckpt.exists() else None},
    })
    rundir.write_manifest(run_dir, manifest)
    (Path(run_dir) / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"Run complete: {run_dir}")
    print(json.dumps(result.get("metrics_original", result.get("metrics")), indent=2))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    resolved = _resolve(args)
    adapter = get_adapter(resolved["model"]["adapter"])
    run_dir = _run_dir_for(resolved, args.run_id)
    checkpoint = _verified_checkpoint(adapter, run_dir)

    if args.dry_run:
        info = adapter.validate(resolved)
        print(f"[dry-run] would evaluate checkpoint at {checkpoint} against: {json.dumps(info)}")
        return

    dataset = adapter.load_dataset(resolved)
    model, device = adapter.load_checkpoint(resolved, str(checkpoint))
    preds, ts, extra = _predict(adapter, resolved, model, device, dataset)
    metrics = compute_metrics(preds, _test_targets(dataset))
    print(json.dumps(metrics, indent=2))
    (Path(run_dir) / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def cmd_infer(args):
    resolved = _resolve(args)
    adapter = get_adapter(resolved["model"]["adapter"])
    run_dir = _run_dir_for(resolved, args.run_id)
    checkpoint = _verified_checkpoint(adapter, run_dir)

    if args.dry_run:
        info = adapter.validate(resolved)
        print(f"[dry-run] would run inference with checkpoint {checkpoint} on: {json.dumps(info)}")
        return

    dataset = adapter.load_dataset(resolved)
    model, device = adapter.load_checkpoint(resolved, str(checkpoint))
    preds, ts, extra = _predict(adapter, resolved, model, device, dataset)
    out_path = Path(run_dir) / "predictions.npz"
    np.savez(out_path, predictions=preds, targets=_test_targets(dataset), timestamps=ts, **extra)
    print(f"Wrote {len(preds)} predictions to {out_path}")


# ---------------------------------------------------------------------------
# export-benchmark
# ---------------------------------------------------------------------------

def cmd_export_benchmark(args):
    resolved = _resolve(args)
    run_dir = _run_dir_for(resolved, args.run_id)

    if args.dry_run:
        pred_path = Path(run_dir) / "predictions.npz"
        if not pred_path.exists():
            print(f"[dry-run] MISSING {pred_path} -- run `train` or `infer` first")
            sys.exit(1)
        print(f"[dry-run] would export {pred_path} to {benchmark_export.method_id_for(resolved)}/")
        return

    benchmarking_root = Path(resolved["_env_values"].get(
        "benchmarking_data_root", str(REPO_ROOT / "benchmarking-site" / "data")
    ))
    adapter = get_adapter(resolved["model"]["adapter"])
    result = benchmark_export.export_run(run_dir, resolved, benchmarking_root, adapter)
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="training_framework")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model from a config")
    _add_common_args(p_train)
    p_train.add_argument("--dry-run", action="store_true",
                          help="Validate config + data + run-dir safety, then exit without training")
    p_train.add_argument("--resume", action="store_true", help="Continue a running/failed run with the same config")
    p_train.add_argument("--force", action="store_true", help="Overwrite an existing (even complete) run")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Re-score an existing run's checkpoint")
    _add_common_args(p_eval, require_run_id=True)
    p_eval.add_argument("--dry-run", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Run a checkpoint over held-out data")
    _add_common_args(p_infer, require_run_id=True)
    p_infer.add_argument("--dry-run", action="store_true")
    p_infer.set_defaults(func=cmd_infer)

    p_export = sub.add_parser("export-benchmark", help="Publish a run's predictions to benchmarking-site")
    _add_common_args(p_export, require_run_id=True)
    p_export.add_argument("--dry-run", action="store_true")
    p_export.set_defaults(func=cmd_export_benchmark)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
