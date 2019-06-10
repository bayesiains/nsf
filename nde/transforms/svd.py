import numpy as np
import torch

from torch import nn
from torch.nn import init

from nde import transforms
from nde.transforms.linear import Linear


class SVDLinear(Linear):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(self, features, num_householder, using_cache=False):
        super().__init__(features, using_cache)

        # First orthogonal matrix (U).
        self.orthogonal_1 = transforms.HouseholderSequence(
            features=features, num_transforms=num_householder)

        # Logs of diagonal entries of the diagonal matrix (S).
        self.log_diagonal = nn.Parameter(torch.zeros(features))

        # Second orthogonal matrix (V^T).
        self.orthogonal_2 = transforms.HouseholderSequence(
            features=features, num_transforms=num_householder)

        self._initialize()

    def _initialize(self):
        stdv = 1.0 / np.sqrt(self.features)
        init.uniform_(self.log_diagonal, -stdv, stdv)
        init.constant_(self.bias, 0.0)


    def forward_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs, _ = self.orthogonal_2(inputs)  # Ignore logabsdet as we know it's zero.
        outputs *= torch.exp(self.log_diagonal)
        outputs, _ = self.orthogonal_1(outputs)  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet = self.logabsdet() * torch.ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(outputs)  # Ignore logabsdet since we know it's zero.
        outputs *= torch.exp(-self.log_diagonal)
        outputs, _ = self.orthogonal_2.inverse(outputs)  # Ignore logabsdet since we know it's zero.
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * torch.ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal = torch.diag(torch.exp(self.log_diagonal))
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal_inv = torch.diag(torch.exp(-self.log_diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
        return weight_inv.t()

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_diagonal)
