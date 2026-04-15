"""Synthetic augmentation utilities for privacy-aware training.

Generates synthetic cells for rare subpopulations to prevent memorization
during VAE training. Uses bootstrap + noise from real seed samples,
preserving sparsity and realistic expression distributions.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from vae import VAE, vae_loss


def generate_bootstrap_augmented(seed_samples, n_synthetic, noise_scale=0.1, seed=42):
    """Generate synthetic cells by bootstrap resampling + gaussian noise.

    Preserves sparsity structure of real cells. Only nonzero entries receive noise.

    Args:
        seed_samples: Real samples to bootstrap from [n_seeds, n_genes], numpy or tensor
        n_synthetic: Number of synthetic samples to generate
        noise_scale: Noise magnitude relative to per-gene std of seed samples
        seed: Random seed

    Returns:
        Synthetic samples as numpy array [n_synthetic, n_genes]
    """
    rng = np.random.RandomState(seed)

    if isinstance(seed_samples, torch.Tensor):
        seed_samples = seed_samples.numpy()

    n_seeds, n_genes = seed_samples.shape

    indices = rng.choice(n_seeds, size=n_synthetic, replace=True)
    synthetic = seed_samples[indices].copy()

    gene_std = seed_samples.std(axis=0)
    gene_std[gene_std == 0] = 1e-6

    for i in range(n_synthetic):
        nonzero_mask = synthetic[i] > 0
        noise = rng.normal(0, noise_scale * gene_std[nonzero_mask])
        synthetic[i, nonzero_mask] += noise
        synthetic[i] = np.maximum(synthetic[i], 0)

    return synthetic


def train_augmented_vae(X_retain, X_synthetic, config, device='cpu', seed=42):
    """Train a VAE on retain + synthetic data from scratch.

    Args:
        X_retain: Retain set expression [n_retain, n_genes], numpy array
        X_synthetic: Synthetic expression [n_synthetic, n_genes], numpy array
        config: VAE config dict
        device: torch device
        seed: Random seed

    Returns:
        (model, best_loss) tuple
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_all = np.concatenate([X_retain, X_synthetic], axis=0)
    X_tensor = torch.FloatTensor(X_all)
    lib_sizes = X_tensor.sum(dim=1, keepdim=True)

    dataset = TensorDataset(X_tensor, lib_sizes)
    batch_size = config.get('batch_size', 256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_config = {k: v for k, v in config.items()
                    if k in ['input_dim', 'latent_dim', 'hidden_dims',
                             'likelihood', 'dropout', 'use_layer_norm']}
    model = VAE(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))

    epochs = config.get('epochs', 100)
    kl_warmup = config.get('kl_warmup_epochs', 20)
    free_bits = config.get('free_bits', 0.03)
    patience = config.get('early_stopping_patience', 15)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        beta = min(1.0, (epoch + 1) / max(kl_warmup, 1))

        for x_batch, lib_batch in loader:
            x_batch = x_batch.to(device)
            lib_batch = lib_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch, library_size=lib_batch)
            loss, recon, kl = vae_loss(
                x_batch, output, likelihood=config.get('likelihood', 'nb'),
                beta=beta, free_bits=free_bits
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, best_loss
