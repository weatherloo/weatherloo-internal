"""CNN correction model: shapes and gradient flow."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import mos
from cnn_correction import build_model, run_cnn
from points import ALL_VARIABLES, WIDE_VARIABLES
from test_mos import synthetic


class CNNTest(unittest.TestCase):
    def test_forward_and_backward(self):
        model = build_model(15, 30)
        patch, scalars = torch.randn(4, 15, 17, 17), torch.randn(4, 60)
        out = model(patch, scalars)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_time_context_trajectory_and_wide_branches(self):
        model = build_model(3 * 15, 30, traj_channels=15, wide_channels=8)
        patch, scalars = torch.randn(4, 45, 17, 17), torch.randn(4, 60)
        traj, wide, lead = torch.randn(4, 15, 49), torch.randn(4, 8, 33, 37), torch.tensor([1, 6, 24, 48])
        out = model(patch, scalars, traj=traj, lead=lead, wide=wide)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        # The trajectory branch reads the predicted lead: changing it changes the output.
        with torch.no_grad():
            model.eval()
            other = model(patch, scalars, traj=traj, lead=torch.tensor([2, 7, 25, 47]), wide=wide)
        self.assertFalse(torch.allclose(out.detach(), other))


class RunCNNTest(unittest.TestCase):
    def test_end_to_end_with_every_option_on_synthetic_data(self):
        t = lambda *a: int(datetime(*a, tzinfo=timezone.utc).timestamp())
        bounds = {"TRAIN_END": t(2023, 12, 29), "VAL_END": t(2024, 1, 3), "TEST_END": t(2024, 1, 10)}
        with tempfile.TemporaryDirectory() as temp, mock.patch.multiple(mos, **bounds):
            root, values, inits, *_ = synthetic(temp, n_init=80)
            month = root / "context" / "2024-01"
            month.mkdir(parents=True)
            rng = np.random.default_rng(0)
            np.save(month / "patches.npy", rng.normal(size=(80, 49, len(ALL_VARIABLES), 9, 9)).astype(np.float16))
            np.save(month / "wide.npy", rng.normal(size=(80, 49, len(WIDE_VARIABLES), 6, 7)).astype(np.float16))
            np.save(month / "init_time.npy", inits)
            (month / "manifest.json").write_text(json.dumps({"samples": 80, "variables": list(ALL_VARIABLES),
                                                             "wide_variables": list(WIDE_VARIABLES)}))
            out = Path(temp) / "cnn.json"
            report = run_cnn(root / "points", root / "context", root / "stations", "eric_d_soulis", out=out,
                             seeds=1, max_epochs=2, batch_size=256, context=(-3, 0, 3), trajectory=True, wide=True,
                             variables=list(ALL_VARIABLES[:5]))
            self.assertEqual(report["options"]["context"], [-3, 0, 3])
            self.assertEqual(len(report["options"]["variables"]), 5)
            self.assertTrue(report["train_inits"] > 0 and report["val_inits"] > 0 and report["test_inits"] > 0)
            self.assertTrue(np.isfinite(report["methods"]["cnn"]["t2m"]["rmse"]))
            with np.load(out.with_suffix(".npz")) as saved:
                self.assertEqual(saved["cnn"].shape, (report["test_inits"], 48, 2))
                self.assertEqual(saved["val_cnn"].shape, (report["val_inits"], 48, 2))


if __name__ == "__main__":
    unittest.main()
