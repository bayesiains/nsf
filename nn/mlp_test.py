"""Tests for multi-layer perceptrons."""

import torch
import torchtestcase
import unittest

from nn import mlp


class MLPTest(torchtestcase.TorchTestCase):

    def test_forward(self):
        batch_size = 10
        in_shape = [2, 3, 4]
        out_shape = [5, 6]
        inputs = torch.randn(batch_size, *in_shape)

        for hidden_sizes in [[20], [20, 30], [20, 30, 40]]:
            with self.subTest(hidden_sizes=hidden_sizes):
                model = mlp.MLP(
                    in_shape=in_shape,
                    out_shape=out_shape,
                    hidden_sizes=hidden_sizes,
                )
                outputs = model(inputs)
                self.assertIsInstance(outputs, torch.Tensor)
                self.assertEqual(outputs.shape, torch.Size([batch_size] + out_shape))
                self.assertFalse(torch.isnan(outputs).any())
                self.assertFalse(torch.isinf(outputs).any())

        with self.assertRaises(Exception):
            mlp.MLP(
                in_shape=in_shape,
                out_shape=out_shape,
                hidden_sizes=[],
            )


if __name__ == '__main__':
    unittest.main()
