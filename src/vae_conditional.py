"""Conditional VAE with cluster-conditioned decoder output.

The decoder's fc_mean layer receives the hidden representation concatenated
with a one-hot cluster vector, giving the model a routing signal for
cluster-specific reconstruction. The fc_dispersion remains shared.
"""

import torch
import torch.nn as nn
from vae import VAE


class ConditionalVAE(VAE):
    """VAE with cluster-conditional decoder output layer."""

    def __init__(self, n_clusters=14, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters

        # Decoder hidden dims are encoder hidden dims reversed.
        # For hidden_dims=[1024, 512, 128], decoder uses [128, 512, 1024].
        # The last decoder hidden dimension = first encoder hidden dim.
        last_decoder_hidden = kwargs.get('hidden_dims', [1024, 512, 128])[0]
        output_dim = kwargs.get('input_dim', 2000)

        # Replace fc_mean to accept hidden + one-hot cluster
        self.decoder.fc_mean = nn.Sequential(
            nn.Linear(last_decoder_hidden + n_clusters, output_dim),
            nn.Softmax(dim=-1)
        )

    def decode(self, z, library_size=None, cluster_onehot=None):
        """Decode with optional cluster conditioning."""
        h = self.decoder.network(z)

        if cluster_onehot is not None:
            h_cond = torch.cat([h, cluster_onehot], dim=-1)
        else:
            h_cond = torch.cat([
                h, torch.zeros(h.shape[0], self.n_clusters, device=h.device)
            ], dim=-1)

        mean = self.decoder.fc_mean(h_cond)
        if library_size is not None:
            mean = mean * library_size

        dispersion = torch.exp(self.decoder.fc_dispersion(h))
        return mean, dispersion

    def forward(self, x, library_size=None, cluster_onehot=None):
        """Forward pass with optional cluster conditioning."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mean, dispersion = self.decode(z, library_size, cluster_onehot)
        return mean, dispersion, mu, logvar, z
