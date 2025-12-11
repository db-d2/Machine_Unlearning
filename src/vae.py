"""Variational Autoencoder with dual decoder heads (NB/ZINB and Gaussian).

References:
    - Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics.
      Nature Methods. https://doi.org/10.1038/s41592-018-0229-2 (scVI)
    - Kingma & Welling (2013). Auto-Encoding Variational Bayes.
      arXiv:1312.6114
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Normal
import numpy as np


class Encoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            else:
                layers.append(nn.BatchNorm1d(dims[i+1]))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.network(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class NBDecoder(nn.Module):
    """Negative Binomial decoder for scRNA-seq data."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()

        layers = []
        dims = [latent_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            else:
                layers.append(nn.BatchNorm1d(dims[i+1]))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # NB parameters: mean and dispersion
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softmax(dim=-1)
        )
        self.fc_dispersion = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, library_size=None):
        h = self.network(z)

        # Mean (normalized gene expression probabilities)
        mean = self.fc_mean(h)

        # Scale by library size (already shape [batch_size, 1])
        if library_size is not None:
            mean = mean * library_size

        # Dispersion (inverse overdispersion parameter)
        dispersion = torch.exp(self.fc_dispersion(h))

        return mean, dispersion


class GaussianDecoder(nn.Module):
    """Gaussian decoder for mixture model simulations."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()

        layers = []
        dims = [latent_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            else:
                layers.append(nn.BatchNorm1d(dims[i+1]))

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dims[-1], output_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.network(z)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class VAE(nn.Module):
    """
    Variational Autoencoder with switchable decoder heads.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        latent_dim: Latent space dimension
        likelihood: 'nb' for Negative Binomial or 'gaussian' for Gaussian
    """

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: list = [512, 128],
        latent_dim: int = 16,
        likelihood: str = 'nb',
        dropout: float = 0.1,
        use_layer_norm: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.likelihood = likelihood

        # Encoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout=dropout, use_layer_norm=use_layer_norm)

        # Decoder (symmetric architecture)
        decoder_hidden_dims = hidden_dims[::-1]

        if likelihood == 'nb':
            self.decoder = NBDecoder(latent_dim, decoder_hidden_dims, input_dim, dropout=dropout, use_layer_norm=use_layer_norm)
        elif likelihood == 'gaussian':
            self.decoder = GaussianDecoder(latent_dim, decoder_hidden_dims, input_dim, dropout=dropout, use_layer_norm=use_layer_norm)
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, library_size=None):
        """Decode latent code to reconstruction."""
        if self.likelihood == 'nb':
            return self.decoder(z, library_size)
        else:
            return self.decoder(z)

    def forward(self, x, library_size=None):
        """
        Forward pass through VAE.

        Args:
            x: Input data
            library_size: Library size for NB decoder (optional)

        Returns:
            Tuple of (reconstruction, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if self.likelihood == 'nb':
            mean, dispersion = self.decode(z, library_size)
            return mean, dispersion, mu, logvar, z
        else:
            recon_mu, recon_logvar = self.decode(z)
            return recon_mu, recon_logvar, mu, logvar, z


def nb_loss(x, mean, dispersion, eps=1e-8, per_cell=False):
    """
    Negative Binomial negative log-likelihood (simplified MSE for log-normalized data).

    For log-normalized single-cell data, MSE works well and is numerically stable.

    Args:
        x: True log-normalized counts
        mean: Predicted mean
        dispersion: Dispersion parameter (unused in MSE version)
        eps: Small constant for numerical stability
        per_cell: If True, average over genes (per-cell loss ~O(1)).
                  If False, sum over genes (per-gene loss ~O(n_genes)).

    Returns:
        MSE loss

    Note:
        Using MSE instead of full NB NLL for numerical stability with log-normalized data.
        This is a common practice in scRNA-seq VAE implementations.

        per_cell=True gives O(1) magnitude losses suitable for balancing with other O(1) terms.
    """
    mse = F.mse_loss(mean, x, reduction='none')
    if per_cell:
        return mse.mean()  # Average over both genes and batch → O(1)
    else:
        return mse.sum(dim=1).mean()  # Sum over genes, average over batch → O(n_genes)


def gaussian_loss(x, recon_mu, recon_logvar):
    """
    Gaussian negative log-likelihood.

    Args:
        x: True values
        recon_mu: Reconstructed mean
        recon_logvar: Reconstructed log variance

    Returns:
        NLL per sample
    """
    nll = 0.5 * (
        recon_logvar
        + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
        + np.log(2 * np.pi)
    )
    return nll.sum(dim=1).mean()


def kl_divergence(mu, logvar, free_bits=0.0, per_cell=False):
    """
    KL divergence between N(mu, var) and N(0, 1) with optional free-bits.

    Free-bits prevents posterior collapse by allowing each latent dimension
    to have a minimum KL of free_bits nats before contributing to the loss.

    Args:
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior
        free_bits: Minimum KL per dimension (nats) before penalty applies
        per_cell: If True, average over latent dims (per-cell loss ~O(1)).
                  If False, sum over latent dims (per-dim loss ~O(latent_dim)).

    Returns:
        KL divergence per sample
    """
    # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        # Apply free-bits: max(kl_per_dim, free_bits)
        kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))

    if per_cell:
        return kl_per_dim.mean()  # Average over both latent dims and batch → O(1)
    else:
        kl = torch.sum(kl_per_dim, dim=1)
        return kl.mean()  # Sum over latent dims, average over batch → O(latent_dim)


def vae_loss(x, model_output, likelihood='nb', library_size=None, beta=1.0, free_bits=0.0, per_cell=False):
    """
    Complete VAE loss (reconstruction + KL divergence).

    Args:
        x: Input data
        model_output: Tuple from VAE forward pass
        likelihood: 'nb' or 'gaussian'
        library_size: Library size for NB (optional)
        beta: Weight for KL term (beta-VAE)
        free_bits: Free-bits per dimension to prevent posterior collapse
        per_cell: If True, use per-cell averaging (O(1) loss magnitudes).
                  If False, use per-gene/per-dim summing (O(n_genes/latent_dim) magnitudes).

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)

    Note:
        per_cell=True is recommended for adversarial unlearning to balance loss magnitudes
        between privacy (O(1)) and utility (O(1)) terms.
    """
    if likelihood == 'nb':
        mean, dispersion, mu, logvar, z = model_output
        recon_loss = nb_loss(x, mean, dispersion, per_cell=per_cell)
    else:
        recon_mu, recon_logvar, mu, logvar, z = model_output
        recon_loss = gaussian_loss(x, recon_mu, recon_logvar)

    kl_loss = kl_divergence(mu, logvar, free_bits=free_bits, per_cell=per_cell)

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
