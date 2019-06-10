import torch

from torch import distributions


class MixtureSameFamily(distributions.Distribution):
    def __init__(self, mixture_distribution, components_distribution):
        self.mixture_distribution = mixture_distribution
        self.components_distribution = components_distribution

        has_rsample=False

        super().__init__(
            batch_shape=self.components_distribution.batch_shape,
            event_shape=self.components_distribution.event_shape
        )

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    @property
    def arg_constraints(self):
        return dict(
            self.mixture_distribution.arg_constraints,
            **self.components_distribution.arg_constraints
        )

    @property
    def support(self):
        return self.components_distribution.support

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size()):
        mixture_mask = self.mixture_distribution.sample(sample_shape) # [S, B, D, M]
        if len(mixture_mask.shape) == 3:
            mixture_mask = mixture_mask[:, None, ...]
        components_samples = self.components_distribution.rsample(sample_shape) # [S, B, D, M]
        samples = torch.sum(mixture_mask * components_samples, dim=-1) # [S, B, D]
        return samples

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        # pad value for evaluation under component density
        value = value.permute(2, 0, 1) # [S, B, D]
        value = value[..., None].repeat(1, 1, 1, self.batch_shape[-1])  # [S, B, D, M]
        log_prob_components = self.components_distribution.log_prob(value).permute(1, 2, 3, 0)

        # calculate numerically stable log coefficients, and pad
        log_prob_mixture = self.mixture_distribution.logits
        log_prob_mixture = log_prob_mixture[..., None]
        return torch.logsumexp(log_prob_mixture + log_prob_components, dim=-2)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


def main():
    pass


if __name__ == '__main__':
    main()
