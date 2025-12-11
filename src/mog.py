"""
Mixture-of-Gaussians simulations for VAE unlearning.

This module implements:
- MoG data generator (K∈{3,5}, d∈{2,5,10}, scenarios)
- Gaussian VAE training and ARI alignment verification
- Forget scenarios (component removal, 50%, scattered, outliers)
- Evaluation (two-negative AUCs, ELBO gap, ARI/AMI, latent viz)
- Memorization study (rare vs common components)
- Dimensionality scaling (d∈{2,10,20})
- Retrain-floor logic application
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

from vae import VAE


# MoG data generator
def generate_mog_data(
    K: int,
    d: int,
    n: int,
    scenario: str = 'separated',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate Mixture-of-Gaussians data.

    Args:
        K: Number of components (3 or 5)
        d: Dimensionality (2, 5, 10, 20)
        n: Number of samples (5000-10000)
        scenario: 'separated' (default, well-separated) or 'overlapping' (Scenario C)
        seed: Random seed

    Returns:
        X: [n, d] data samples
        labels: [n] component assignments
        metadata: dict with component parameters
    """
    np.random.seed(seed)

    # Component proportions
    if scenario == 'separated':
        # Equal proportions for well-separated case
        pi = np.ones(K) / K
    elif scenario == 'overlapping':
        # Slightly unequal for overlap scenario
        pi = np.array([0.3, 0.3, 0.4]) if K == 3 else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Generate component means
    if scenario == 'separated':
        # Well-separated: place on grid with large spacing
        spacing = 5.0
        if K == 3:
            means = np.array([
                [0, 0] + [0] * (d - 2),
                [spacing, 0] + [0] * (d - 2),
                [spacing/2, spacing] + [0] * (d - 2)
            ])
        elif K == 5:
            means = np.array([
                [0, 0] + [0] * (d - 2),
                [spacing, 0] + [0] * (d - 2),
                [2*spacing, 0] + [0] * (d - 2),
                [spacing/2, spacing] + [0] * (d - 2),
                [3*spacing/2, spacing] + [0] * (d - 2)
            ])
    elif scenario == 'overlapping':
        # Overlapping: closer spacing
        spacing = 1.5
        if K == 3:
            means = np.array([
                [0, 0] + [0] * (d - 2),
                [spacing, 0] + [0] * (d - 2),
                [spacing/2, spacing] + [0] * (d - 2)
            ])
        elif K == 5:
            means = np.array([
                [0, 0] + [0] * (d - 2),
                [spacing, 0] + [0] * (d - 2),
                [2*spacing, 0] + [0] * (d - 2),
                [spacing/2, spacing] + [0] * (d - 2),
                [3*spacing/2, spacing] + [0] * (d - 2)
            ])

    # Generate covariances
    if scenario == 'separated':
        # Isotropic, moderate variance
        covs = [np.eye(d) * 0.5 for _ in range(K)]
    elif scenario == 'overlapping':
        # Larger variance for overlap
        covs = [np.eye(d) * 1.0 for _ in range(K)]

    # Sample from mixture
    labels = np.random.choice(K, size=n, p=pi)
    X = np.zeros((n, d))

    for i in range(n):
        k = labels[i]
        X[i] = multivariate_normal.rvs(mean=means[k], cov=covs[k])

    metadata = {
        'K': K,
        'd': d,
        'n': n,
        'scenario': scenario,
        'seed': seed,
        'means': means.tolist(),
        'proportions': pi.tolist()
    }

    return X, labels, metadata


# Forget scenario generators
def create_forget_set(
    labels: np.ndarray,
    scenario: str = 'component_removal',
    target_component: int = 0,
    fraction: float = 0.5,
    seed: int = 42
) -> np.ndarray:
    """
    Create forget set according to different scenarios.

    Args:
        labels: Component labels [n]
        scenario: One of:
            - 'component_removal': Remove entire component
            - 'partial_component': Remove fraction of one component
            - 'scattered': Remove scattered 20% across all components
            - 'outliers': Remove 5% outliers (farthest from means)
        target_component: Which component to target (for component scenarios)
        fraction: Fraction to remove (for partial/scattered)
        seed: Random seed

    Returns:
        forget_indices: Boolean mask [n] indicating forget set
    """
    np.random.seed(seed)
    n = len(labels)
    forget_mask = np.zeros(n, dtype=bool)

    if scenario == 'component_removal':
        # Remove all samples from target component
        forget_mask = (labels == target_component)

    elif scenario == 'partial_component':
        # Remove fraction of samples from target component
        component_indices = np.where(labels == target_component)[0]
        n_forget = int(len(component_indices) * fraction)
        forget_indices = np.random.choice(component_indices, size=n_forget, replace=False)
        forget_mask[forget_indices] = True

    elif scenario == 'scattered':
        # Remove scattered fraction across all components
        n_forget = int(n * fraction)
        forget_indices = np.random.choice(n, size=n_forget, replace=False)
        forget_mask[forget_indices] = True

    elif scenario == 'outliers':
        # Not implemented in this version - would require distance computation
        raise NotImplementedError("Outlier scenario requires X data, not just labels")

    else:
        raise ValueError(f"Unknown forget scenario: {scenario}")

    return forget_mask


# VAE training for MoG
def train_vae_mog(
    X_train: np.ndarray,
    hidden_dims: List[int],
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    beta: float = 1.0,
    kl_warmup_epochs: int = 0,
    free_bits: float = 0.0,
    print_every: int = 10
) -> VAE:
    """
    Train Gaussian VAE on MoG data.

    Args:
        X_train: Training data [n, d]
        hidden_dims: Hidden layer dimensions
        latent_dim: Latent dimension
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: torch device
        beta: KL weight
        kl_warmup_epochs: Linear warmup epochs for KL
        free_bits: Free-bits per dimension
        print_every: Print frequency

    Returns:
        model: Trained VAE
    """
    input_dim = X_train.shape[1]

    # Create model
    model = VAE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        likelihood='gaussian',
        dropout=0.1,
        use_layer_norm=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        # KL warmup
        if kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / kl_warmup_epochs) * beta
        else:
            kl_weight = beta

        for (x,) in loader:
            x = x.to(device)

            # Forward pass
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            recon_mu, recon_logvar = model.decode(z)

            # Reconstruction loss (Gaussian)
            recon_loss = 0.5 * (
                recon_logvar + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
            ).sum(dim=1).mean()

            # KL divergence with free-bits
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            if free_bits > 0:
                kl_div = torch.maximum(kl_div, torch.tensor(free_bits * latent_dim).to(device))
            kl_loss = kl_div.mean()

            # Total loss
            loss = recon_loss + kl_weight * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        if (epoch + 1) % print_every == 0:
            avg_loss = total_loss / n_batches
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, β={kl_weight:.3f}")

    return model


# Evaluation functions
def compute_elbo_mog(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 256) -> float:
    """Compute ELBO for MoG data."""
    model.eval()
    total_elbo = 0.0
    n_samples = 0

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.FloatTensor(X[i:i+batch_size]).to(device)

            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            recon_mu, recon_logvar = model.decode(z)

            # Reconstruction loss
            recon_loss = 0.5 * (
                recon_logvar + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
            ).sum(dim=1)

            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            elbo = -(recon_loss + kl_div)
            total_elbo += elbo.sum().item()
            n_samples += len(x)

    return total_elbo / n_samples


def get_latent_representations(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Get latent representations for data."""
    model.eval()
    latents = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.FloatTensor(X[i:i+batch_size]).to(device)
            mu, logvar = model.encode(x)
            latents.append(mu.cpu().numpy())

    return np.concatenate(latents, axis=0)


def compute_ari_ami(latents: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute ARI and AMI between latent clustering and true components.

    Uses k-means clustering on latents, compares to true component labels.
    """
    from sklearn.cluster import KMeans

    K = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(latents)

    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    return ari, ami


# Two-negative MIA evaluation for MoG
def train_mia_attacker_mog(
    model: VAE,
    X_forget: np.ndarray,
    X_unseen: np.ndarray,
    X_retain: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001
) -> Tuple[float, float]:
    """
    Train two-negative MIA attacker and return AUCs.

    Returns:
        auc_forget_vs_unseen: AUC for F vs Unseen
        auc_forget_vs_retain: AUC for F vs Retain
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Extract features (simple: use ELBO and latent norms)
    def extract_features(X):
        features = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x = torch.FloatTensor(X[i:i+batch_size]).to(device)
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                recon_mu, recon_logvar = model.decode(z)

                # Reconstruction loss
                recon_loss = 0.5 * (
                    recon_logvar + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
                ).sum(dim=1)

                # KL divergence
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

                # ELBO
                elbo = -(recon_loss + kl_div)

                # Feature vector: [elbo, recon, kl, ||mu||, ||z||]
                feat = torch.stack([
                    elbo,
                    -recon_loss,
                    -kl_div,
                    torch.norm(mu, dim=1),
                    torch.norm(z, dim=1)
                ], dim=1)

                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    # Extract features
    feat_f = extract_features(X_forget)
    feat_u = extract_features(X_unseen)
    feat_r = extract_features(X_retain)

    # Train F vs Unseen
    X_fu = np.concatenate([feat_f, feat_u], axis=0)
    y_fu = np.concatenate([np.ones(len(feat_f)), np.zeros(len(feat_u))])
    clf_fu = LogisticRegression(max_iter=1000, random_state=42)
    clf_fu.fit(X_fu, y_fu)
    auc_fu = roc_auc_score(y_fu, clf_fu.predict_proba(X_fu)[:, 1])

    # Train F vs Retain
    X_fr = np.concatenate([feat_f, feat_r], axis=0)
    y_fr = np.concatenate([np.ones(len(feat_f)), np.zeros(len(feat_r))])
    clf_fr = LogisticRegression(max_iter=1000, random_state=42)
    clf_fr.fit(X_fr, y_fr)
    auc_fr = roc_auc_score(y_fr, clf_fr.predict_proba(X_fr)[:, 1])

    return auc_fu, auc_fr


def run_mog_experiment(
    K: int,
    d: int,
    n: int,
    scenario: str,
    forget_scenario: str,
    output_dir: str,
    config: Dict,
    seed: int = 42
):
    """
    Run full MoG experiment: generate data, train VAE, create forget set,
    retrain, evaluate privacy/utility.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Running MoG experiment: K={K}, d={d}, n={n}, scenario={scenario}")
    print(f"Forget scenario: {forget_scenario}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Generate data
    print("Generating MoG data...")
    X, labels, metadata = generate_mog_data(K, d, n, scenario=scenario, seed=seed)

    # Split into train/test (80/20)
    n_train = int(0.8 * n)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    labels_train = labels[train_idx]
    X_test = X[test_idx]
    labels_test = labels[test_idx]

    print(f"Generated {n} samples, {n_train} train, {len(test_idx)} test")
    print(f"Component distribution: {np.bincount(labels_train)}")

    # Train baseline VAE
    print("\nTraining baseline Gaussian VAE...")
    start_time = time.time()
    baseline_model = train_vae_mog(
        X_train,
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        device=device,
        beta=config.get('beta', 1.0),
        kl_warmup_epochs=config.get('kl_warmup_epochs', 20),
        free_bits=config.get('free_bits', 0.03),
        print_every=config.get('print_every', 10)
    )
    baseline_time = time.time() - start_time

    # Verify ARI alignment
    print("\nVerifying ARI alignment with components...")
    baseline_latents = get_latent_representations(baseline_model, X_train, device)
    ari, ami = compute_ari_ami(baseline_latents, labels_train)
    print(f"Baseline ARI: {ari:.4f}, AMI: {ami:.4f}")

    # Create forget set
    print(f"\nCreating forget set (scenario={forget_scenario})...")
    if forget_scenario == 'component_removal':
        # Remove smallest component
        component_counts = np.bincount(labels_train)
        target_component = np.argmin(component_counts)
        print(f"Removing component {target_component} (n={component_counts[target_component]})")
        forget_mask_train = create_forget_set(
            labels_train,
            scenario='component_removal',
            target_component=target_component,
            seed=seed
        )
    elif forget_scenario == 'partial_component':
        # Remove 50% of first component
        forget_mask_train = create_forget_set(
            labels_train,
            scenario='partial_component',
            target_component=0,
            fraction=0.5,
            seed=seed
        )
    elif forget_scenario == 'scattered':
        # Remove scattered 20%
        forget_mask_train = create_forget_set(
            labels_train,
            scenario='scattered',
            fraction=0.2,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown forget scenario: {forget_scenario}")

    # Split train into F, R, U
    forget_idx = np.where(forget_mask_train)[0]
    retain_idx = np.where(~forget_mask_train)[0]

    # Use test set as Unseen
    X_forget = X_train[forget_idx]
    X_retain = X_train[retain_idx]
    X_unseen = X_test
    labels_forget = labels_train[forget_idx]
    labels_retain = labels_train[retain_idx]

    print(f"Split: F={len(X_forget)}, R={len(X_retain)}, U={len(X_unseen)}")

    # Baseline privacy audit
    print("\nBaseline privacy audit...")
    baseline_auc_fu, baseline_auc_fr = train_mia_attacker_mog(
        baseline_model, X_forget, X_unseen, X_retain, device
    )
    baseline_auc_avg = (baseline_auc_fu + baseline_auc_fr) / 2
    print(f"Baseline AUC (F vs U): {baseline_auc_fu:.4f}")
    print(f"Baseline AUC (F vs R): {baseline_auc_fr:.4f}")
    print(f"Baseline AUC (avg): {baseline_auc_avg:.4f}")

    # Retrain on D\F (gold standard / retrain floor)
    print("\nTraining retrain model (gold standard)...")
    start_time = time.time()
    retrain_model = train_vae_mog(
        X_retain,  # Train only on retain set
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        device=device,
        beta=config.get('beta', 1.0),
        kl_warmup_epochs=config.get('kl_warmup_epochs', 20),
        free_bits=config.get('free_bits', 0.03),
        print_every=config.get('print_every', 10)
    )
    retrain_time = time.time() - start_time

    # Retrain privacy audit (floor)
    print("\nRetrain privacy audit (floor)...")
    retrain_auc_fu, retrain_auc_fr = train_mia_attacker_mog(
        retrain_model, X_forget, X_unseen, X_retain, device
    )
    retrain_auc_avg = (retrain_auc_fu + retrain_auc_fr) / 2
    print(f"Retrain AUC (F vs U): {retrain_auc_fu:.4f}")
    print(f"Retrain AUC (F vs R): {retrain_auc_fr:.4f}")
    print(f"Retrain AUC (avg / floor): {retrain_auc_avg:.4f}")

    # Utility evaluation
    print("\nEvaluating utility (ELBO, ARI/AMI)...")

    # ELBO on retain set
    baseline_elbo_retain = compute_elbo_mog(baseline_model, X_retain, device)
    retrain_elbo_retain = compute_elbo_mog(retrain_model, X_retain, device)
    elbo_gap_pct = 100 * (retrain_elbo_retain - baseline_elbo_retain) / abs(baseline_elbo_retain)

    print(f"Baseline ELBO (R): {baseline_elbo_retain:.4f}")
    print(f"Retrain ELBO (R): {retrain_elbo_retain:.4f}")
    print(f"ELBO gap: {elbo_gap_pct:.2f}%")

    # ARI/AMI on retain set
    retrain_latents = get_latent_representations(retrain_model, X_retain, device)
    retrain_ari, retrain_ami = compute_ari_ami(retrain_latents, labels_retain)
    print(f"Retrain ARI: {retrain_ari:.4f}, AMI: {retrain_ami:.4f}")

    # Memorization study (component rarity)
    print("\nMemorization study (rare vs common components)...")
    component_counts = np.bincount(labels_train)
    print(f"Component sizes: {component_counts}")

    # Compute per-component ELBO on baseline
    per_component_elbo = {}
    for k in range(K):
        mask = (labels_train == k)
        if mask.sum() > 0:
            X_k = X_train[mask]
            elbo_k = compute_elbo_mog(baseline_model, X_k, device)
            per_component_elbo[k] = {
                'elbo': elbo_k,
                'count': int(mask.sum()),
                'is_rare': bool(component_counts[k] == component_counts.min())
            }
            print(f"Component {k} (n={component_counts[k]}): ELBO={elbo_k:.4f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'method_id': 'mog_simulation',
        'data': metadata,
        'forget_scenario': forget_scenario,
        'forget_set_size': int(len(X_forget)),
        'retain_set_size': int(len(X_retain)),
        'baseline': {
            'ari': float(ari),
            'ami': float(ami),
            'auc_forget_vs_unseen': float(baseline_auc_fu),
            'auc_forget_vs_retain': float(baseline_auc_fr),
            'auc_avg': float(baseline_auc_avg),
            'elbo_retain': float(baseline_elbo_retain),
            'training_time_seconds': float(baseline_time)
        },
        'retrain': {
            'ari': float(retrain_ari),
            'ami': float(retrain_ami),
            'auc_forget_vs_unseen': float(retrain_auc_fu),
            'auc_forget_vs_retain': float(retrain_auc_fr),
            'auc_avg': float(retrain_auc_avg),
            'auc_floor': float(retrain_auc_avg),
            'elbo_retain': float(retrain_elbo_retain),
            'elbo_gap_percent': float(elbo_gap_pct),
            'training_time_seconds': float(retrain_time)
        },
        'memorization': per_component_elbo
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    torch.save(baseline_model.state_dict(), os.path.join(output_dir, 'baseline_model.pt'))
    torch.save(retrain_model.state_dict(), os.path.join(output_dir, 'retrain_model.pt'))

    # Save latent visualizations if d=2 or project to 2D
    print("\nGenerating latent visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # True labels
    if d == 2:
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c=labels_train, cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title('True Data (colored by component)')
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_train)
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_train, cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title('True Data (PCA, colored by component)')

    # Baseline latents
    if baseline_model.latent_dim == 2:
        axes[1].scatter(baseline_latents[:, 0], baseline_latents[:, 1], c=labels_train, cmap='tab10', alpha=0.6, s=10)
        axes[1].set_title(f'Baseline Latents (ARI={ari:.3f})')
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        baseline_latents_2d = pca.fit_transform(baseline_latents)
        axes[1].scatter(baseline_latents_2d[:, 0], baseline_latents_2d[:, 1], c=labels_train, cmap='tab10', alpha=0.6, s=10)
        axes[1].set_title(f'Baseline Latents (PCA, ARI={ari:.3f})')

    # Retrain latents
    if retrain_model.latent_dim == 2:
        axes[2].scatter(retrain_latents[:, 0], retrain_latents[:, 1], c=labels_retain, cmap='tab10', alpha=0.6, s=10)
        axes[2].set_title(f'Retrain Latents (ARI={retrain_ari:.3f})')
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        retrain_latents_2d = pca.fit_transform(retrain_latents)
        axes[2].scatter(retrain_latents_2d[:, 0], retrain_latents_2d[:, 1], c=labels_retain, cmap='tab10', alpha=0.6, s=10)
        axes[2].set_title(f'Retrain Latents (PCA, ARI={retrain_ari:.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_viz.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nExperiment complete! Results saved to {output_dir}")
    print(f"Retrain floor (target for unlearning): AUC = {retrain_auc_avg:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='MoG simulations for VAE unlearning')
    parser.add_argument('--config', type=str, default='configs/mog.yaml', help='Config file')
    parser.add_argument('--output_dir', type=str, default='outputs/p3/mog', help='Output directory')
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)

    # Run experiments for different configurations
    experiments = [
        # Well-separated, K=3, d=2 (default, component removal)
        {'K': 3, 'd': 2, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'component_removal'},
        # Well-separated, K=5, d=5 (component removal)
        {'K': 5, 'd': 5, 'n': 8000, 'scenario': 'separated', 'forget_scenario': 'component_removal'},
        # Overlapping, K=3, d=2 (Scenario C)
        {'K': 3, 'd': 2, 'n': 5000, 'scenario': 'overlapping', 'forget_scenario': 'component_removal'},
        # Scattered forget
        {'K': 3, 'd': 5, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'scattered'},
        # Partial component
        {'K': 3, 'd': 5, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'partial_component'},
    ]

    # Dimensionality scaling (Scenario A only)
    scaling_experiments = [
        {'K': 3, 'd': 2, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'component_removal'},
        {'K': 3, 'd': 10, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'component_removal'},
        {'K': 3, 'd': 20, 'n': 5000, 'scenario': 'separated', 'forget_scenario': 'component_removal'},
    ]

    all_results = []

    print("Running main experiments...")
    for i, exp in enumerate(experiments):
        exp_name = f"K{exp['K']}_d{exp['d']}_n{exp['n']}_{exp['scenario']}_{exp['forget_scenario']}"
        exp_dir = os.path.join(output_base, exp_name)
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp_name}")
        print(f"{'='*80}")

        results = run_mog_experiment(
            K=exp['K'],
            d=exp['d'],
            n=exp['n'],
            scenario=exp['scenario'],
            forget_scenario=exp['forget_scenario'],
            output_dir=exp_dir,
            config=config,
            seed=config.get('seed', 42)
        )
        results['experiment_name'] = exp_name
        all_results.append(results)

    print("\nRunning dimensionality scaling experiments...")
    scaling_results = []
    for i, exp in enumerate(scaling_experiments):
        exp_name = f"scaling_d{exp['d']}"
        exp_dir = os.path.join(output_base, 'scaling', exp_name)
        print(f"\n{'='*80}")
        print(f"Scaling {i+1}/{len(scaling_experiments)}: d={exp['d']}")
        print(f"{'='*80}")

        results = run_mog_experiment(
            K=exp['K'],
            d=exp['d'],
            n=exp['n'],
            scenario=exp['scenario'],
            forget_scenario=exp['forget_scenario'],
            output_dir=exp_dir,
            config=config,
            seed=config.get('seed', 42)
        )
        results['experiment_name'] = exp_name
        scaling_results.append(results)

    # Save summary
    summary = {
        'method_id': 'mog_simulations',
        'description': 'Mixture-of-Gaussians simulations for VAE unlearning',
        'experiments': all_results,
        'scaling_experiments': scaling_results,
        'config': config
    }

    with open(os.path.join(output_base, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("All MoG experiments complete!")
    print(f"Summary saved to {output_base}/summary.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
