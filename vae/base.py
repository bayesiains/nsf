"""Basic definitions for VAEs."""

import torch

from torch import nn

import utils


class VariationalAutoencoder(nn.Module):
    """Implementation of a standard VAE."""

    def __init__(self, prior, approximate_posterior, likelihood, inputs_encoder=None):
        """
        Args:
            prior: a distribution object, the prior.
            approximate_posterior: a distribution object, the encoder.
            likelihood: a distribution object, the decoder.
        """
        super().__init__()
        self._prior = prior
        self._approximate_posterior = approximate_posterior
        self._likelihood = likelihood
        self._inputs_encoder = inputs_encoder

    def forward(self, *args):
        raise RuntimeError('Forward method cannot be called for a VAE object.')

    def stochastic_elbo(self, inputs, num_samples=1, kl_multiplier=1, keepdim=False):
        """Calculates an unbiased Monte-Carlo estimate of the evidence lower bound.

        Note: the KL term is also estimated via Monte Carlo.

        Args:
            inputs: Tensor of shape [batch_size, ...], the inputs.
            num_samples: int, number of samples to use for the Monte-Carlo estimate.

        Returns:
            A Tensor of shape [batch_size], an ELBO estimate for each input.
        """
        # Sample latents and calculate their log prob under the encoder.
        if self._inputs_encoder is None:
            posterior_context = inputs
        else:
            posterior_context = self._inputs_encoder(inputs)
        latents, log_q_z = self._approximate_posterior.sample_and_log_prob(
            num_samples,
            context=posterior_context
        )
        latents = utils.merge_leading_dims(latents, num_dims=2)
        log_q_z = utils.merge_leading_dims(log_q_z, num_dims=2)

        # Compute log prob of latents under the prior.
        log_p_z = self._prior.log_prob(latents)

        # Compute log prob of inputs under the decoder,
        inputs = utils.repeat_rows(inputs, num_reps=num_samples)
        log_p_x = self._likelihood.log_prob(inputs, context=latents)

        # Compute ELBO.
        # TODO: maybe compute KL analytically when possible?
        elbo = log_p_x + kl_multiplier * (log_p_z - log_q_z)
        elbo = utils.split_leading_dim(elbo, [-1, num_samples])
        if keepdim:
            return elbo
        else:
            return torch.sum(elbo, dim=1) / num_samples  # Average ELBO across samples.

    def log_prob_lower_bound(self, inputs, num_samples=100):
        elbo = self.stochastic_elbo(inputs, num_samples=num_samples, keepdim=True)
        log_prob_lower_bound = torch.logsumexp(elbo, dim=1) - torch.log(torch.Tensor([num_samples]))
        return log_prob_lower_bound

    def _decode(self, latents, mean):
        if mean:
            return self._likelihood.mean(context=latents)
        else:
            samples = self._likelihood.sample(num_samples=1, context=latents)
            return utils.merge_leading_dims(samples, num_dims=2)

    def sample(self, num_samples, mean=False):
        """Generates samples from the VAE.

        Args:
            num_samples: int, number of samples to generate.
            mean: bool, if True it uses the mean of the decoder instead of sampling from it.

        Returns:
            A tensor of shape [num_samples, ...], the samples.
        """
        latents = self._prior.sample(num_samples)
        return self._decode(latents, mean)

    def encode(self, inputs, num_samples=None):
        """Encodes inputs into the latent space.

        Args:
            inputs: Tensor of shape [batch_size, ...], the inputs to encode.
            num_samples: int or None, the number of latent samples to generate per input. If None,
                only one latent sample is generated per input.

        Returns:
            A Tensor of shape [batch_size, num_samples, ...] or [batch_size, ...] if num_samples
            is None, the latent samples for each input.
        """
        if num_samples is None:
            latents = self._approximate_posterior.sample(num_samples=1, context=inputs)
            latents = utils.merge_leading_dims(latents, num_dims=2)
        else:
            latents = self._approximate_posterior.sample(num_samples=num_samples, context=inputs)
        return latents

    def reconstruct(self, inputs, num_samples=None, mean=False):
        """Reconstruct given inputs.

        Args:
            inputs: Tensor of shape [batch_size, ...], the inputs to reconstruct.
            num_samples: int or None, the number of reconstructions to generate per input. If None,
                only one reconstruction is generated per input.
            mean: bool, if True it uses the mean of the decoder instead of sampling from it.

        Returns:
            A Tensor of shape [batch_size, num_samples, ...] or [batch_size, ...] if num_samples
            is None, the reconstructions for each input.
        """
        latents = self.encode(inputs, num_samples)
        if num_samples is not None:
            latents = utils.merge_leading_dims(latents, num_dims=2)
        recons = self._decode(latents, mean)
        if num_samples is not None:
            recons = utils.split_leading_dim(recons, [-1, num_samples])
        return recons
