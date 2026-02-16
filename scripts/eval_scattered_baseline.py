#!/usr/bin/env python
"""
Evaluate scattered forget set baseline and retrain floor for Tabula Muris.

The structured forget set (Cluster 33, cardiac muscle cells) has a cell-type
confound: matched negatives are from different tissues/cell types, so the MIA
detects biology rather than memorization. The scattered forget set avoids this
because cells span multiple clusters and tissues.

Uses the already-trained baseline and retrain models from notebook 12.

Usage:
    PYTHONPATH=src python scripts/eval_scattered_baseline.py

Output:
    outputs/tabula_muris/scattered_baseline_results.json
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import json
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from utils import set_global_seed, GLOBAL_SEED, DEVICE
from vae import VAE
from attacker import MLPAttacker, build_attack_features
from train_attacker_conditioned import extract_features_for_split, train_attacker_epoch
from attacker_eval import matched_negative_evaluation


def get_latent_codes(model, adata, indices, device):
    """Extract latent codes for given indices."""
    X = adata.X[indices]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    x = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
    return z.cpu().numpy()


def load_vae(checkpoint_path, device):
    """Load a VAE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    model = VAE(
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        latent_dim=cfg['latent_dim'],
        likelihood=cfg['likelihood'],
        dropout=cfg.get('dropout', 0.1),
        use_layer_norm=cfg.get('use_layer_norm', True)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train(False)
    return model, ckpt


def train_attacker_on_features(forget_X, matched_X, feature_dim, device,
                                epochs=50, seed=42):
    """Train a fresh MIA attacker on given features."""
    set_global_seed(seed)

    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3,
        use_spectral_norm=True
    ).to(device)

    optimizer = optim.Adam(attacker.parameters(), lr=1e-3)

    member_labels = torch.ones(len(forget_X))
    nonmember_labels = torch.zeros(len(matched_X))
    all_X = torch.cat([forget_X, matched_X], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)
    perm = torch.randperm(len(all_X))

    train_dataset = TensorDataset(all_X[perm], all_labels[perm])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        train_loss = train_attacker_epoch(attacker, train_loader, optimizer, device)
        if (epoch + 1) % 10 == 0:
            attacker.train(False)
            with torch.no_grad():
                preds = torch.sigmoid(attacker(all_X.to(device))).cpu().numpy().flatten()
            auc = roc_auc_score(all_labels.numpy(), preds)
            attacker.train(True)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - AUC: {auc:.4f}")

    return attacker


def extract_and_build_features(model, adata, indices, retain_z, batch_size, device):
    """Extract VAE features and build attack feature vector."""
    feats, knn, _ = extract_features_for_split(
        model, adata, indices, batch_size, device,
        reference_z_retain=retain_z
    )
    return build_attack_features(feats, knn, None)


def main():
    DATA_DIR = Path('data')
    OUTPUT_DIR = Path('outputs/tabula_muris')
    BATCH_SIZE = 256
    K = 10

    # Load data and splits
    adata = sc.read_h5ad(DATA_DIR / 'tabula_muris_processed.h5ad')
    split_scattered = json.load(open(OUTPUT_DIR / 'split_scattered.json'))
    split_structured = json.load(open(OUTPUT_DIR / 'split_structured.json'))

    forget_idx = np.array(split_scattered['forget_indices'])
    unseen_idx = np.array(split_structured['unseen_indices'])  # Same unseen set
    retain_idx = np.array(split_scattered['retain_indices'])

    print("=" * 60)
    print("SCATTERED FORGET SET EVALUATION")
    print("=" * 60)
    print(f"Forget: {len(forget_idx)} random cells")
    print(f"Unseen: {len(unseen_idx)} cells")

    # Show tissue distribution of scattered forget set
    if 'tissue' in adata.obs.columns:
        print(f"\nScattered forget set tissues:")
        print(adata.obs.iloc[forget_idx]['tissue'].value_counts())

    # --- Baseline model ---
    print("\n--- Baseline Model ---")
    model, baseline_ckpt = load_vae(OUTPUT_DIR / 'baseline' / 'best_model.pt', DEVICE)
    print(f"Loaded baseline model from epoch {baseline_ckpt['epoch']}")

    # k-NN matched negatives in baseline latent space
    forget_z = get_latent_codes(model, adata, forget_idx, DEVICE)
    unseen_z = get_latent_codes(model, adata, unseen_idx, DEVICE)
    retain_z = get_latent_codes(model, adata, retain_idx[:5000], DEVICE)

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(unseen_z)
    distances, indices_knn = nbrs.kneighbors(forget_z)

    matched_local = np.unique(indices_knn.flatten())
    matched_indices = np.array([unseen_idx[i] for i in matched_local])

    mean_dist = distances.mean()
    print(f"Matched negatives: {len(matched_indices)} cells")
    print(f"Mean k-NN distance: {mean_dist:.4f}")

    if 'tissue' in adata.obs.columns:
        print(f"\nMatched negative tissues:")
        print(adata.obs.iloc[matched_indices]['tissue'].value_counts())

    # Extract features
    print("\nExtracting baseline features...")
    forget_X = extract_and_build_features(model, adata, forget_idx, retain_z, BATCH_SIZE, DEVICE)
    matched_X = extract_and_build_features(model, adata, matched_indices, retain_z, BATCH_SIZE, DEVICE)
    feature_dim = forget_X.shape[1]
    print(f"Feature dim: {feature_dim}")

    # Train attacker
    print("\nTraining attacker...")
    attacker = train_attacker_on_features(forget_X, matched_X, feature_dim, DEVICE)

    # Evaluate baseline
    baseline_metrics = matched_negative_evaluation(attacker, forget_X, matched_X, device=DEVICE)
    print(f"\n=== Scattered Baseline AUC: {baseline_metrics['auc']:.4f} ===")
    print(f"    95% CI: [{baseline_metrics['auc_ci_lower']:.4f}, {baseline_metrics['auc_ci_upper']:.4f}]")

    # --- Retrain model ---
    print("\n--- Retrain Model ---")
    retrain_model, retrain_ckpt = load_vae(OUTPUT_DIR / 'retrain' / 'best_model.pt', DEVICE)
    print(f"Loaded retrain model from epoch {retrain_ckpt['epoch']}")

    retrain_retain_z = get_latent_codes(retrain_model, adata, retain_idx[:5000], DEVICE)

    print("Extracting retrain features...")
    retrain_forget_X = extract_and_build_features(
        retrain_model, adata, forget_idx, retrain_retain_z, BATCH_SIZE, DEVICE
    )
    retrain_matched_X = extract_and_build_features(
        retrain_model, adata, matched_indices, retrain_retain_z, BATCH_SIZE, DEVICE
    )

    retrain_metrics = matched_negative_evaluation(
        attacker, retrain_forget_X, retrain_matched_X, device=DEVICE
    )
    print(f"\n=== Scattered Retrain Floor AUC: {retrain_metrics['auc']:.4f} ===")
    print(f"    95% CI: [{retrain_metrics['auc_ci_lower']:.4f}, {retrain_metrics['auc_ci_upper']:.4f}]")

    retrain_floor = retrain_metrics['auc']
    target_band = [retrain_floor - 0.03, retrain_floor + 0.03]
    print(f"    Target band: [{target_band[0]:.4f}, {target_band[1]:.4f}]")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'PBMC-33k':<15} {'Tabula Muris':<15}")
    print("-" * 55)
    print(f"{'Baseline AUC':<25} {'0.769':<15} {baseline_metrics['auc']:.3f}")
    print(f"{'Retrain floor':<25} {'0.481':<15} {retrain_floor:.3f}")
    print(f"{'Target band':<25} {'[0.451, 0.511]':<15} [{target_band[0]:.3f}, {target_band[1]:.3f}]")

    # Save results
    results = {
        'forget_type': 'scattered',
        'forget_size': int(len(forget_idx)),
        'baseline_auc': float(baseline_metrics['auc']),
        'baseline_ci': [float(baseline_metrics['auc_ci_lower']),
                        float(baseline_metrics['auc_ci_upper'])],
        'retrain_floor': float(retrain_floor),
        'retrain_ci': [float(retrain_metrics['auc_ci_lower']),
                       float(retrain_metrics['auc_ci_upper'])],
        'target_band': [float(x) for x in target_band],
        'matched_negatives': int(len(matched_indices)),
        'mean_knn_distance': float(mean_dist),
    }
    out_path = OUTPUT_DIR / 'scattered_baseline_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save matched negatives for scattered set
    matched_data = {
        'matched_indices': [int(i) for i in matched_indices],
        'method': 'latent_knn',
        'k': K,
        'mean_distance': float(mean_dist),
    }
    with open(OUTPUT_DIR / 'matched_negatives_scattered.json', 'w') as f:
        json.dump(matched_data, f, indent=2)


if __name__ == '__main__':
    main()
