"""Utility metrics for evaluating VAE model quality.

Extracted from notebook 18 (utility_suite). Provides held-out ELBO,
latent clustering metrics (silhouette, ARI), and marker gene correlation.
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from vae import VAE, vae_loss


def compute_held_out_elbo(model, X, device, batch_size=512):
    """Compute mean ELBO on held-out data.

    Args:
        model: Trained VAE model.
        X: Tensor of shape [n_cells, n_genes].
        device: torch device string.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with 'elbo', 'recon', 'kl' as mean per-cell values.
    """
    model.train(False)
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n = 0

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            lib_size = batch.sum(dim=1, keepdim=True)
            output = model(batch, library_size=lib_size)
            loss, recon, kl = vae_loss(batch, output, likelihood='nb', beta=1.0)

            batch_n = len(batch)
            total_loss += loss.item() * batch_n
            total_recon += recon.item() * batch_n
            total_kl += kl.item() * batch_n
            n += batch_n

    return {
        'elbo': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n
    }


def compute_latent_metrics(model, X, labels, device, batch_size=512):
    """Compute silhouette score and ARI from latent space.

    Args:
        model: Trained VAE model.
        X: Tensor of shape [n_cells, n_genes].
        labels: Array of cluster labels (string or int).
        device: torch device string.
        batch_size: Batch size for encoding.

    Returns:
        Dict with 'silhouette' and 'ari'.
    """
    model.train(False)
    latents = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_clusters = len(le.classes_)

    # Subsample for speed if needed
    np.random.seed(42)
    if len(latents) > 5000:
        sub_idx = np.random.choice(len(latents), 5000, replace=False)
        sil = silhouette_score(latents[sub_idx], y[sub_idx])
    else:
        sil = silhouette_score(latents, y)

    # ARI: k-means on latent space vs original labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latents)
    ari = adjusted_rand_score(y, pred_labels)

    return {'silhouette': float(sil), 'ari': float(ari)}


def compute_marker_correlation(model, X, marker_idx, gene_names, device,
                               batch_size=512):
    """Compute Pearson r between true and reconstructed marker gene expression.

    Args:
        model: Trained VAE model.
        X: Tensor of shape [n_cells, n_genes].
        marker_idx: List of column indices for marker genes.
        gene_names: List of gene name strings (used for output keys).
        device: torch device string.
        batch_size: Batch size for decoding.

    Returns:
        Dict with 'mean_r' and 'per_gene' (gene_name -> r).
    """
    model.train(False)
    recons = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            mu, logvar = model.encode(batch)
            z = mu  # deterministic for evaluation
            decode_out = model.decode(z)
            if isinstance(decode_out, tuple):
                recon = decode_out[0]
            else:
                recon = decode_out
            recons.append(recon.cpu().numpy())
    recons = np.concatenate(recons)

    X_np = X.numpy() if hasattr(X, 'numpy') else np.asarray(X)

    correlations = []
    for idx in marker_idx:
        r, _ = pearsonr(X_np[:, idx], recons[:, idx])
        correlations.append(r)

    return {
        'mean_r': float(np.mean(correlations)),
        'per_gene': {gene_names[i]: float(r) for i, r in zip(marker_idx, correlations)}
    }
