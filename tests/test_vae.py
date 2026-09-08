"""Tests for VAE model, loss functions, and forward pass shapes."""

import torch
import torch.nn.functional as F

from vae import VAE, nb_loss, kl_divergence, vae_loss


class TestNBLoss:
    """Verify that nb_loss computes MSE and ignores dispersion."""

    def test_nb_loss_is_mse(self):
        x = torch.randn(8, 50)
        mean = torch.randn(8, 50)
        dispersion = torch.randn(8, 50)  # should be ignored

        loss = nb_loss(x, mean, dispersion, per_cell=False)
        expected = F.mse_loss(mean, x, reduction="none").sum(dim=1).mean()

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_nb_loss_ignores_dispersion(self):
        x = torch.randn(8, 50)
        mean = torch.randn(8, 50)

        loss_a = nb_loss(x, mean, torch.zeros(8, 50))
        loss_b = nb_loss(x, mean, torch.ones(8, 50) * 100)

        assert torch.allclose(loss_a, loss_b)

    def test_nb_loss_per_cell_mode(self):
        x = torch.randn(8, 50)
        mean = torch.randn(8, 50)
        dispersion = torch.zeros(8, 50)

        loss_per_cell = nb_loss(x, mean, dispersion, per_cell=True)
        expected = F.mse_loss(mean, x, reduction="mean")

        assert torch.allclose(loss_per_cell, expected, atol=1e-6)

    def test_nb_loss_zero_when_perfect(self):
        x = torch.randn(8, 50)
        loss = nb_loss(x, x, torch.zeros(8, 50))
        assert loss.item() == 0.0


class TestKLDivergence:
    """Verify KL divergence computation."""

    def test_kl_zero_for_standard_normal(self):
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)
        kl = kl_divergence(mu, logvar)
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_kl_positive_for_nonstandard(self):
        mu = torch.ones(8, 16)
        logvar = torch.zeros(8, 16)
        kl = kl_divergence(mu, logvar)
        assert kl.item() > 0

    def test_kl_known_value(self):
        # KL(N(1,1) || N(0,1)) = 0.5 * (1 + 0 - 1 - exp(0)) = 0.5 * (1-1) per dim
        # Actually: -0.5 * (1 + 0 - 1 - 1) = -0.5 * (-1) = 0.5 per dim
        mu = torch.ones(1, 1)
        logvar = torch.zeros(1, 1)
        kl = kl_divergence(mu, logvar)
        assert torch.allclose(kl, torch.tensor(0.5), atol=1e-6)

    def test_kl_free_bits_floor(self):
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)
        kl_no_free = kl_divergence(mu, logvar, free_bits=0.0)
        kl_with_free = kl_divergence(mu, logvar, free_bits=0.1)
        # With free_bits, KL should be >= free_bits * latent_dim
        assert kl_with_free.item() >= 0.1 * 16 - 1e-6
        assert kl_no_free.item() < kl_with_free.item()


class TestVAELoss:
    """Verify total loss = recon + beta * kl."""

    def test_loss_composition(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        output = small_vae(x, library_size=lib)
        total, recon, kl = vae_loss(x, output, likelihood="nb", beta=1.0)
        assert torch.allclose(total, recon + kl, atol=1e-5)

    def test_beta_scaling(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        output = small_vae(x, library_size=lib)
        total_b1, recon_b1, kl_b1 = vae_loss(x, output, likelihood="nb", beta=1.0)
        total_b2, recon_b2, kl_b2 = vae_loss(x, output, likelihood="nb", beta=2.0)
        assert torch.allclose(total_b2, recon_b2 + 2.0 * kl_b2, atol=1e-5)
        assert torch.allclose(recon_b1, recon_b2, atol=1e-6)


class TestDecoderOutput:
    """Verify decoder softmax and library_size behavior."""

    def test_softmax_sums_to_one(self, small_vae):
        z = torch.randn(4, 8)
        mean, _ = small_vae.decode(z)  # no library_size
        row_sums = mean.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_library_size_scaling(self, small_vae):
        z = torch.randn(4, 8)
        lib = torch.tensor([[100.0], [200.0], [300.0], [400.0]])
        mean, _ = small_vae.decode(z, library_size=lib)
        row_sums = mean.sum(dim=1)
        assert torch.allclose(row_sums, lib.squeeze(), atol=0.1)

    def test_dispersion_is_positive(self, small_vae):
        z = torch.randn(4, 8)
        _, dispersion = small_vae.decode(z)
        assert (dispersion > 0).all()


class TestVAEForwardShapes:
    """Verify output tensor shapes through the full forward pass."""

    def test_encode_shapes(self, small_vae, dummy_batch):
        x, _ = dummy_batch
        mu, logvar = small_vae.encode(x)
        assert mu.shape == (16, 8)
        assert logvar.shape == (16, 8)

    def test_forward_nb_shapes(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        mean, dispersion, mu, logvar, z = small_vae(x, library_size=lib)
        assert mean.shape == (16, 50)
        assert dispersion.shape == (16, 50)
        assert mu.shape == (16, 8)
        assert logvar.shape == (16, 8)
        assert z.shape == (16, 8)

    def test_forward_gaussian_shapes(self, small_vae_gaussian, dummy_batch):
        x, _ = dummy_batch
        recon_mu, recon_logvar, mu, logvar, z = small_vae_gaussian(x)
        assert recon_mu.shape == (16, 50)
        assert recon_logvar.shape == (16, 50)
        assert mu.shape == (16, 8)

    def test_reparameterize_deterministic(self, small_vae):
        mu = torch.randn(4, 8)
        logvar = torch.zeros(4, 8)
        torch.manual_seed(99)
        z1 = small_vae.reparameterize(mu, logvar)
        torch.manual_seed(99)
        z2 = small_vae.reparameterize(mu, logvar)
        assert torch.allclose(z1, z2)
