"""Representation Alignment Unlearning (RAU).

Fine-tunes the baseline model so its posterior on forget samples matches
the retrain model's posterior. The retrain model provides the counterfactual:
what the representation SHOULD look like without memorization.

This operates in representation space, not parameter space, potentially
bypassing the Fisher overlap bottleneck identified in Proposition 1.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from vae import VAE, vae_loss
from train_scrub import kl_between_gaussians


def train_rau(
    baseline_model,
    retrain_model,
    X_forget,
    X_retain,
    lambda_align,
    config,
    device='cpu',
    epochs=50,
    lr=1e-4,
    patience=10,
    batch_size=256,
    seed=42,
):
    """Representation Alignment Unlearning.

    Fine-tunes a copy of the baseline model with two objectives:
    1. Utility: maintain ELBO on retain set
    2. Alignment: push posterior on forget samples toward retrain posterior

    Loss = ELBO(retain) + lambda_align * mean(KL(student || retrain)) on forget

    Args:
        baseline_model: Trained baseline VAE (will be deep-copied)
        retrain_model: Retrain VAE (frozen reference, never saw forget set)
        X_forget: Forget set expression [n_forget, n_genes], numpy
        X_retain: Retain set expression [n_retain, n_genes], numpy
        lambda_align: Weight for alignment loss
        config: VAE config dict
        device: torch device
        epochs: Max training epochs
        lr: Learning rate
        patience: Early stopping patience on retain ELBO
        batch_size: Batch size for retain loader
        seed: Random seed

    Returns:
        (model, history) tuple
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = deepcopy(baseline_model).to(device)
    student.train()

    retrain_model.eval()
    for p in retrain_model.parameters():
        p.requires_grad = False

    x_forget = torch.FloatTensor(X_forget).to(device)
    lib_forget = x_forget.sum(dim=1, keepdim=True)

    retain_tensor = torch.FloatTensor(X_retain)
    retain_lib = retain_tensor.sum(dim=1, keepdim=True)
    retain_dataset = TensorDataset(retain_tensor, retain_lib)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    likelihood = config.get('likelihood', 'nb')

    history = {'utility_loss': [], 'align_loss': [], 'total_loss': []}
    best_utility = float('inf')
    patience_counter = 0
    best_state = None

    with torch.no_grad():
        mu_retrain, logvar_retrain = retrain_model.encode(x_forget)

    for epoch in range(epochs):
        epoch_utility = 0
        epoch_align = 0
        n_batches = 0

        for x_batch, lib_batch in retain_loader:
            x_batch = x_batch.to(device)
            lib_batch = lib_batch.to(device)

            optimizer.zero_grad()

            output = student(x_batch, library_size=lib_batch)
            utility_loss, _, _ = vae_loss(x_batch, output, likelihood=likelihood, beta=1.0)

            mu_student, logvar_student = student.encode(x_forget)
            align_loss = kl_between_gaussians(
                mu_student, logvar_student, mu_retrain, logvar_retrain
            ).mean()

            total = utility_loss + lambda_align * align_loss
            total.backward()
            optimizer.step()

            epoch_utility += utility_loss.item()
            epoch_align += align_loss.item()
            n_batches += 1

        avg_utility = epoch_utility / n_batches
        avg_align = epoch_align / n_batches
        history['utility_loss'].append(avg_utility)
        history['align_loss'].append(avg_align)
        history['total_loss'].append(avg_utility + lambda_align * avg_align)

        if avg_utility < best_utility:
            best_utility = avg_utility
            patience_counter = 0
            best_state = deepcopy(student.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        student.load_state_dict(best_state)

    student.eval()
    return student, history
