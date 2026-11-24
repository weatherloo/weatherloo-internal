"""The autoregressive rollout cannot see future HRRR frames or labels."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "temporal_station"))
from model import AutoregressiveModel  # noqa: E402


class AutoregressiveCausalityTest(unittest.TestCase):
    def test_future_frame_does_not_change_earlier_predictions(self):
        torch.manual_seed(0)
        model = AutoregressiveModel().eval()
        original = torch.randn(1, 6, 30, 30, 3)
        changed = original.clone()
        changed[:, 5] += 10
        with torch.no_grad():
            before = model(original)
            after = model(changed)
        torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[:, 5], after[:, 5]))


if __name__ == "__main__":
    unittest.main()
