"""Shared utilities for Fisher information overlap analysis.

Consolidates dataset classes, overlap metrics, Fisher computation functions,
and the deep MLP classifier used across notebooks 26-29 and scripts/.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class AnnDataDataset(Dataset):
    """Dataset returning (x, library_size) for VAE training/Fisher computation."""

    def __init__(self, adata, indices):
        self.adata = adata
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        row = self.adata.X[i]
        x = torch.FloatTensor(
            row.toarray().flatten() if hasattr(row, 'toarray')
            else np.asarray(row).flatten()
        )
        library_size = torch.FloatTensor([x.sum().item()])
        return x, library_size


class GeneExprDataset(Dataset):
    """Dataset returning (x, label) for classifier training/Fisher computation."""

    def __init__(self, adata, indices):
        X = adata.X[indices]
        self.X = torch.FloatTensor(
            X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        )
        self.y = torch.LongTensor(
            adata.obs['leiden'].values[indices].astype(int)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ConditionalAnnDataDataset(Dataset):
    """Dataset returning (x, library_size, cluster_onehot) for conditional VAE."""

    def __init__(self, adata, indices, n_clusters):
        self.adata = adata
        self.indices = indices
        self.n_clusters = n_clusters
        self.labels = adata.obs['leiden'].values[indices].astype(int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        row = self.adata.X[i]
        x = torch.FloatTensor(
            row.toarray().flatten() if hasattr(row, 'toarray')
            else np.asarray(row).flatten()
        )
        library_size = torch.FloatTensor([x.sum().item()])
        onehot = torch.zeros(self.n_clusters)
        onehot[self.labels[idx]] = 1.0
        return x, library_size, onehot


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def effective_rank(f):
    """Effective rank (exponential entropy of normalized distribution)."""
    p = f / f.sum()
    p = p.clamp(min=1e-30)
    return np.exp(-(p * p.log()).sum().item())


def top_k_overlap(a, b, k):
    """Fraction of top-k indices shared between two 1-D tensors."""
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / k


def vae_layer_category(name):
    """Classify VAE parameter name into layer category."""
    if 'fc_mu' in name or 'fc_logvar' in name:
        return 'Bottleneck'
    if name.startswith('encoder'):
        return 'Encoder'
    if 'fc_mean' in name or 'fc_dispersion' in name:
        return 'Decoder output'
    if name.startswith('decoder'):
        return 'Decoder hidden'
    return 'Other'


def classifier_layer_category(name):
    """Classify deep MLP classifier parameter into layer category."""
    if name.startswith('output'):
        return 'Class-specific output'
    return 'Shared hidden'


# ---------------------------------------------------------------------------
# Fisher computation (three model types)
# ---------------------------------------------------------------------------

def compute_vae_fisher(model, dataloader, device, damping=1e-8):
    """Diagonal Fisher information for a VAE using ELBO loss.

    Uses reduction='sum' then divides by n_samples (matching NB26 methodology).
    Returns (fisher_dict, n_samples).
    """
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    n_samples = 0

    for x_batch, lib_batch in dataloader:
        x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
        model.zero_grad()
        mu, logvar = model.encode(x_batch)
        z = model.reparameterize(mu, logvar)
        if model.likelihood == 'nb':
            mean, dispersion = model.decode(z, lib_batch)
            recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='sum')
        else:
            recon_mu, recon_logvar = model.decode(z)
            recon_loss = 0.5 * (
                recon_logvar
                + ((x_batch - recon_mu) ** 2) / torch.exp(recon_logvar)
                + np.log(2 * np.pi)
            ).sum()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
        n_samples += x_batch.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping
    model.train(False)
    return fisher, n_samples


def compute_classifier_fisher(model, dataloader, device, damping=1e-8):
    """Diagonal Fisher information for a classifier using cross-entropy loss.

    Returns (fisher_dict, n_samples).
    """
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    n_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        logits = model(x_batch)
        loss = nn.functional.cross_entropy(logits, y_batch, reduction='sum')
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
        n_samples += x_batch.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping
    model.train(False)
    return fisher, n_samples


def compute_conditional_vae_fisher(model, dataloader, device, damping=1e-8):
    """Diagonal Fisher information for a conditional VAE.

    Dataloader yields (x, library_size, cluster_onehot).
    Returns (fisher_dict, n_samples).
    """
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    n_samples = 0

    for x_batch, lib_batch, cluster_batch in dataloader:
        x_batch = x_batch.to(device)
        lib_batch = lib_batch.to(device)
        cluster_batch = cluster_batch.to(device)
        model.zero_grad()
        mu, logvar = model.encode(x_batch)
        z = model.reparameterize(mu, logvar)
        mean, dispersion = model.decode(z, lib_batch, cluster_onehot=cluster_batch)
        recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
        n_samples += x_batch.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping
    model.train(False)
    return fisher, n_samples


# ---------------------------------------------------------------------------
# Model: Deep MLP Classifier
# ---------------------------------------------------------------------------

class DeepMLPClassifier(nn.Module):
    """Deep MLP classifier with shared hidden layers.

    Architecture: input_dim -> hidden_dims -> n_classes
    Each hidden layer has LayerNorm + ReLU + Dropout.
    """

    def __init__(self, input_dim=2000, hidden_dims=None, n_classes=14, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        h = self.hidden(x)
        return self.output(h)
