"""End-to-end check of the framework CLI on a synthetic dataset (numpy only, ~1s).

Run: python -m pytest tests/test_framework_e2e.py   (or: python tests/test_framework_e2e.py)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from training_framework.benchmark_export import export_run  # noqa: E402
from training_framework.fixture import make_tiny_npz  # noqa: E402
from training_framework.rundir import dataset_tag  # noqa: E402

CONFIG = REPO / "configs" / "ridge_cyyz_t2m_fixture_6h.json"


def _cli(verb, env, *extra, expect_ok=True):
    r = subprocess.run(
        [sys.executable, "-m", "training_framework.cli", verb, "--config", str(CONFIG),
         "--env", str(env), "--run-id", "ci", *extra],
        cwd=REPO, capture_output=True, text=True,
    )
    assert (r.returncode == 0) == expect_ok, r.stdout + r.stderr
    return r


def test_train_infer_evaluate_export_and_overwrite_guard():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data" / "fixture").mkdir(parents=True)
        make_tiny_npz(tmp / "data" / "fixture" / "tiny.npz")
        env = tmp / "env.json"
        env.write_text(json.dumps({"data_root": str(tmp / "data"), "runs_root": str(tmp / "runs"),
                                   "benchmarking_data_root": str(tmp / "site")}))

        _cli("train", env, "--dry-run")
        assert not (tmp / "runs").exists(), "dry-run must not write anything"

        _cli("train", env)
        run_dir = tmp / "runs" / "ridge_bias_correction" / "cyyz_t2m_fixture_6h" / "ci"
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        prov = manifest["provenance"]
        assert prov["data"]["sha256"] and prov["seed"] == 42 and prov["python"] and prov["packages"]["numpy"]
        assert manifest["checkpoint"]["sha256"]
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["metrics_original"]["rmse"] < metrics["baseline_metrics"]["rmse"]

        _cli("infer", env)
        _cli("evaluate", env)
        _cli("export-benchmark", env)
        site = tmp / "site" / "ridge_fixture"
        assert (site / "index.json").exists() and (site / "metadata.json").exists()
        assert len(json.loads((site / "index.json").read_text())["files"]) > 0

        # A multi-lead model (per-row lead_time/station arrays) exporting into the same
        # method must fill its slots without wiping the 6h run's slots.
        multi = tmp / "multi_run"
        multi.mkdir()
        first = sorted(json.loads((site / "index.json").read_text())["files"])[0]
        init = json.loads((site / first).read_text())["initialization"]
        np.savez(multi / "predictions.npz", predictions=np.array([1.0, 2.0]), targets=np.array([0.5, 0.5]),
                 timestamps=np.array([init, init]), lead_time=np.array([12, 18]),
                 station=np.array(["cyyz", "eric_d_soulis"]))
        cfg = {"model": {"adapter": "ridge_bias_correction"},
               "data": {"method": "fixture", "variable": "t2m", "lead_time": "all"}}
        export_run(multi, cfg, tmp / "site")
        merged = json.loads((site / first).read_text())["locations"]
        assert merged["cyyz"]["variables"]["t2m"]["rmse"][0] is not None   # 6h slot kept
        assert merged["cyyz"]["variables"]["t2m"]["rmse"][1] == 0.5        # 12h slot added
        assert merged["eric_d_soulis"]["variables"]["t2m"]["bias"][2] == 1.5
        covers = json.loads((site / "metadata.json").read_text())["covers"]
        assert len(covers) == 3

        ckpt_before = (run_dir / "checkpoint.npy").read_bytes()
        _cli("train", env, expect_ok=False)  # complete run must not be silently overwritten
        assert (run_dir / "checkpoint.npy").read_bytes() == ckpt_before

        (run_dir / "checkpoint.npy").write_bytes(ckpt_before + b"x")  # tampered checkpoint
        assert "does not match" in _cli("evaluate", env, expect_ok=False).stderr


def test_dataset_tag():
    base = {"method": "gfs_interpolated", "variable": "t2m"}
    assert dataset_tag({**base, "station": "cyyz", "lead_time": 6}) == "cyyz_t2m_gfs_interpolated_6h"  # unchanged
    assert dataset_tag({**base, "station": ["cyyz", "eric_d_soulis"], "lead_time": "all"}) == \
        "cyyz-eric_d_soulis_t2m_gfs_interpolated_all"
    assert dataset_tag({**base, "lead_time": [6, 12]}) == "all_t2m_gfs_interpolated_6-12h"


if __name__ == "__main__":
    test_dataset_tag()
    test_train_infer_evaluate_export_and_overwrite_guard()
    print("OK")
