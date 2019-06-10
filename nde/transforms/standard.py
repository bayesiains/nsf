"""Implementations of some standard transforms."""

import torch
from nde import transforms


class IdentityTransform(transforms.Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        logabsdet = torch.zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs, context=None):
        return self(inputs, context)


class AffineScalarTransform(transforms.Transform):
    """Computes X = X * scale + shift, where scale and shift are scalars, and scale is non-zero."""

    def __init__(self, shift=None, scale=None):
        super().__init__()

        if shift is None and scale is None:
            raise ValueError('At least one of scale and shift must be provided.')
        if scale == 0.:
            raise ValueError('Scale cannot be zero.')

        self.register_buffer('_shift', torch.tensor(shift if (shift is not None) else 0.))
        self.register_buffer('_scale', torch.tensor(scale if (scale is not None) else 1.))

    @property
    def _log_scale(self):
        return torch.log(torch.abs(self._scale))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = inputs * self._scale + self._shift
        logabsdet = torch.full([batch_size], self._log_scale * num_dims)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        num_dims = torch.prod(torch.tensor(inputs.shape[1:]), dtype=torch.float)
        outputs = (inputs - self._shift) / self._scale
        logabsdet = torch.full([batch_size], -self._log_scale * num_dims)
        return outputs, logabsdet
