import torch
import unittest

from nde import transforms
from nde.transforms.transform_test import TransformTest


class OneByOneConvolutionTest(TransformTest):
    def test_forward_and_inverse_are_consistent(self):
        batch_size = 10
        c, h, w = 3, 28, 28
        inputs = torch.randn(batch_size, c, h, w)
        transform = transforms.OneByOneConvolution(c)
        self.eps = 1e-6
        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == '__main__':
    unittest.main()
