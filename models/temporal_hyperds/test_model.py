"""Checks for the time-extended coordinate decoder's query and rollout behavior."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TemporalHyperDS  # noqa: E402


class TemporalHyperDSTest(unittest.TestCase):
    def test_queries_are_permutation_equivariant(self):
        torch.manual_seed(2)
        forecast = torch.randn(1, 4, 8, 8, 3)
        coordinates = torch.tensor([[[-0.5, 0.2], [0.8, -0.6], [0.0, 0.0]]])
        for autoregressive in (False, True):
            model = TemporalHyperDS(autoregressive=autoregressive)
            ordinary = model(forecast, coordinates)
            reversed_queries = model(forecast, coordinates.flip(1)).flip(2)
            self.assertEqual(ordinary.shape, (1, 4, 3, 2))
            torch.testing.assert_close(ordinary, reversed_queries)
            ordinary.sum().backward()
            self.assertIsNotNone(model.weight_generator.weight.grad)

    def test_autoregressive_rollout_uses_earlier_hour(self):
        torch.manual_seed(3)
        forecast = torch.randn(1, 5, 8, 8, 3)
        coordinates = torch.zeros(1, 1, 2)
        model = TemporalHyperDS(autoregressive=True)
        original = model(forecast, coordinates)
        changed = forecast.clone()
        changed[:, 0] += 4
        updated = model(changed, coordinates)
        self.assertFalse(torch.allclose(original[:, -1], updated[:, -1]))


if __name__ == "__main__":
    unittest.main()
