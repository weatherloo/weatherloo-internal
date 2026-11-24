"""CNN correction model: shapes and gradient flow."""

import unittest

import torch

from cnn_correction import build_model


class CNNTest(unittest.TestCase):
    def test_forward_and_backward(self):
        model = build_model(15, 30)
        patch, scalars = torch.randn(4, 15, 17, 17), torch.randn(4, 60)
        out = model(patch, scalars)
        self.assertEqual(tuple(out.shape), (4, 2))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
