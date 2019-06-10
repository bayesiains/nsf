"""Tests for VAEs."""

import torch
import torchtestcase
import unittest

from nde import distributions
from vae import base


class VariationalAutoencoderTest(torchtestcase.TorchTestCase):

    def test_stochastic_elbo(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        latent_shape = [5, 6]

        prior = distributions.StandardNormal(latent_shape)
        approximate_posterior = distributions.StandardNormal(latent_shape)
        likelihood = distributions.StandardNormal(input_shape)
        vae = base.VariationalAutoencoder(prior, approximate_posterior, likelihood)

        inputs = torch.randn(batch_size, *input_shape)
        for num_samples in [1, 10, 100]:
            with self.subTest(num_samples=num_samples):
                elbo = vae.stochastic_elbo(inputs, num_samples)
                self.assertIsInstance(elbo, torch.Tensor)
                self.assertFalse(torch.isnan(elbo).any())
                self.assertFalse(torch.isinf(elbo).any())
                self.assertEqual(elbo.shape, torch.Size([batch_size]))

    def test_sample(self):
        num_samples = 10
        input_shape = [2, 3, 4]
        latent_shape = [5, 6]

        prior = distributions.StandardNormal(latent_shape)
        approximate_posterior = distributions.StandardNormal(latent_shape)
        likelihood = distributions.StandardNormal(input_shape)
        vae = base.VariationalAutoencoder(prior, approximate_posterior, likelihood)

        for mean in [True, False]:
            with self.subTest(mean=mean):
                samples = vae.sample(num_samples, mean=mean)
                self.assertIsInstance(samples, torch.Tensor)
                self.assertFalse(torch.isnan(samples).any())
                self.assertFalse(torch.isinf(samples).any())
                self.assertEqual(samples.shape, torch.Size([num_samples] + input_shape))

    def test_encode(self):
        batch_size = 20
        input_shape = [2, 3, 4]
        latent_shape = [5, 6]
        inputs = torch.randn(batch_size, *input_shape)

        prior = distributions.StandardNormal(latent_shape)
        approximate_posterior = distributions.StandardNormal(latent_shape)
        likelihood = distributions.StandardNormal(input_shape)
        vae = base.VariationalAutoencoder(prior, approximate_posterior, likelihood)

        for num_samples in [None, 1, 10]:
            with self.subTest(num_samples=num_samples):
                encodings = vae.encode(inputs, num_samples)
                self.assertIsInstance(encodings, torch.Tensor)
                self.assertFalse(torch.isnan(encodings).any())
                self.assertFalse(torch.isinf(encodings).any())
                if num_samples is None:
                    self.assertEqual(encodings.shape, torch.Size([batch_size] + latent_shape))
                else:
                    self.assertEqual(
                        encodings.shape, torch.Size([batch_size, num_samples] + latent_shape))

    def test_reconstruct(self):
        batch_size = 20
        input_shape = [2, 3, 4]
        latent_shape = [5, 6]
        inputs = torch.randn(batch_size, *input_shape)

        prior = distributions.StandardNormal(latent_shape)
        approximate_posterior = distributions.StandardNormal(latent_shape)
        likelihood = distributions.StandardNormal(input_shape)
        vae = base.VariationalAutoencoder(prior, approximate_posterior, likelihood)

        for mean in [True, False]:
            for num_samples in [None, 1, 10]:
                with self.subTest(mean=mean, num_samples=num_samples):
                    recons = vae.reconstruct(inputs, num_samples=num_samples, mean=mean)
                    self.assertIsInstance(recons, torch.Tensor)
                    self.assertFalse(torch.isnan(recons).any())
                    self.assertFalse(torch.isinf(recons).any())
                    if num_samples is None:
                        self.assertEqual(recons.shape, torch.Size([batch_size] + input_shape))
                    else:
                        self.assertEqual(
                            recons.shape, torch.Size([batch_size, num_samples] + input_shape))


if __name__ == '__main__':
    unittest.main()
