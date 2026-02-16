#!/usr/bin/env python3
"""
Compute baseline MIA AUC for each scattered forget set size.

Evaluates the BASELINE model (trained on all data, no unlearning)
against each scattered forget set to determine the pre-unlearning
memorization level. This is needed to properly interpret Fisher
scattered size ablation results.

Attacker methodology (matches canonical NB03):
- Train fresh MLP attacker per size on BASELINE features: F vs matched neg
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims

Usage:
    PYTHONPATH=src python scripts/eval_scattered_baselines.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as functional
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

SRC_DIR = Path(__file__).parent.parent / 'src'
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

from vae import VAE
from attacker import (
    MLPAttacker, extract_vae_features, build_attack_features,
    compute_knn_distances
)

DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
SPLIT_DIR = BASE_DIR / 'outputs' / 'p4' / 'size_ablation' / 'splits'

SIZES = [10, 30, 50, 100, 500]
KNN_RETAIN_SAMPLE = 5000
DEVICE = 'cpu'


def load_vae_model(checkpoint_path):
    """Load a VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config.get('likelihood', 'nb'),
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True),
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train(False)
    return model, config


def get_retain_latent_codes(model, adata, retain_idx, n_sample=KNN_RETAIN_SAMPLE):
    """Get latent codes for retain set sample (for kNN features)."""
    np.random.seed(42)
    sample_idx = np.random.choice(
        retain_idx, size=min(n_sample, len(retain_idx)), replace=False
    )
    X = adata.X[sample_idx]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    x_tensor = torch.FloatTensor(np.asarray(X)).to(DEVICE)

    model.train(False)
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor)
        z = model.reparameterize(mu, logvar)
    return z.cpu().numpy()


def get_features(model, adata, indices, retain_z=None):
    """Extract 70-dim attack features (v1 + kNN distance to retain)."""
    X = adata.X[indices]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    x_tensor = torch.FloatTensor(np.asarray(X)).to(DEVICE)
    lib = x_tensor.sum(dim=1, keepdim=True)

    with torch.no_grad():
        features = extract_vae_features(model, x_tensor, lib, device=DEVICE)

    knn_dist = None
    if retain_z is not None:
        query_z = features['z'].numpy()
        knn_dist = compute_knn_distances(query_z, retain_z, k=5)

    attack_features = build_attack_features(
        features, knn_dist_retain=knn_dist, variant='v1'
    )
    return attack_features


def train_fresh_attacker(model, adata, forget_idx, matched_neg_idx,
                         retain_idx, label=""):
    """Train fresh attacker on F vs matched features from this model.

    Matches canonical NB03 methodology.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    retain_z = get_retain_latent_codes(model, adata, retain_idx)

    forget_X = get_features(model, adata, forget_idx, retain_z=retain_z)
    matched_X = get_features(model, adata, matched_neg_idx, retain_z=retain_z)

    feature_dim = forget_X.shape[1]

    member_labels = torch.ones(len(forget_X))
    nonmember_labels = torch.zeros(len(matched_X))
    all_X = torch.cat([forget_X, matched_X], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)

    perm = torch.randperm(len(all_X))
    n_train = int(0.8 * len(all_X))
    train_X = all_X[perm[:n_train]]
    train_y = all_labels[perm[:n_train]]

    print("  Training fresh attacker for n={}:".format(label))
    print("    Samples: {} ({} forget + {} matched)".format(
        len(all_X), len(forget_X), len(matched_X)))

    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3,
        use_spectral_norm=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(attacker.parameters(), lr=1e-3)
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(50):
        attacker.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = attacker(batch_x).squeeze()
            loss = functional.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Report AUC on full dataset
    attacker.train(False)
    with torch.no_grad():
        all_logits = attacker(all_X.to(DEVICE)).squeeze()
        all_preds = torch.sigmoid(all_logits).cpu().numpy()
    full_auc = roc_auc_score(all_labels.numpy(), all_preds)

    # Also compute TPR at low FPR
    fpr, tpr, _ = roc_curve(all_labels.numpy(), all_preds)
    tpr_at_1pct = float(tpr[np.searchsorted(fpr, 0.01)]) if np.any(
        fpr <= 0.01) else 0.0
    tpr_at_5pct = float(tpr[np.searchsorted(fpr, 0.05)]) if np.any(
        fpr <= 0.05) else 0.0

    print("    Baseline AUC: {:.4f}".format(full_auc))

    return {
        'baseline_auc': float(full_auc),
        'tpr_at_1pct_fpr': tpr_at_1pct,
        'tpr_at_5pct_fpr': tpr_at_5pct,
    }


def main():
    import scanpy as sc

    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)

    print("Loading baseline model...")
    model, config = load_vae_model(BASELINE_CHECKPOINT)

    print()
    print("=" * 60)
    print("SCATTERED BASELINE AUC (fresh attacker per size)")
    print("=" * 60)

    results = {}

    for size in SIZES:
        split_path = SPLIT_DIR / 'split_n{}.json'.format(size)
        matched_path = SPLIT_DIR / 'matched_neg_n{}.json'.format(size)

        if not split_path.exists():
            print("  n={}: split not found".format(size))
            continue

        with open(split_path) as f:
            split = json.load(f)
        with open(matched_path) as f:
            matched = json.load(f)

        forget_indices = split['forget_indices']
        retain_indices = split['retain_indices']
        matched_indices = matched['matched_indices']

        metrics = train_fresh_attacker(
            model, adata, forget_indices, matched_indices, retain_indices,
            label=str(size)
        )
        metrics['n_forget'] = len(forget_indices)
        metrics['n_matched'] = len(matched_indices)
        results[str(size)] = metrics

    # Also check original scattered split (n=35)
    orig_split_path = BASE_DIR / 'outputs' / 'p1' / 'split_scattered.json'
    orig_matched_path = BASE_DIR / 'outputs' / 'p1.5' / 'scattered_matched_negatives.json'

    if orig_split_path.exists() and orig_matched_path.exists():
        with open(orig_split_path) as f:
            split = json.load(f)
        with open(orig_matched_path) as f:
            matched = json.load(f)

        forget_indices = split['forget_indices']
        retain_indices = split['retain_indices']
        matched_indices = matched['matched_indices']

        metrics = train_fresh_attacker(
            model, adata, forget_indices, matched_indices, retain_indices,
            label='35_original'
        )
        metrics['n_forget'] = len(forget_indices)
        metrics['n_matched'] = len(matched_indices)
        results['35_original'] = metrics

    # Summary comparison
    print()
    print("=" * 60)
    print("COMPARISON: Baseline vs Fisher-unlearned")
    print("=" * 60)

    fisher_summary_path = BASE_DIR / 'outputs' / 'p4' / 'size_ablation' / 'summary.json'
    if fisher_summary_path.exists():
        with open(fisher_summary_path) as f:
            fisher = json.load(f)

        print()
        print("{:<8} {:<15} {:<15} {:<15} {:<10}".format(
            'Size', 'Baseline AUC', 'Fisher AUC', 'Delta', 'Reduction%'))
        print("-" * 65)

        for size in SIZES:
            s = str(size)
            if s in results and s in fisher['results_by_size']:
                baseline = results[s]['baseline_auc']
                fisher_auc = fisher['results_by_size'][s]['mean']
                delta = baseline - fisher_auc
                excess = baseline - 0.5
                pct = (delta / excess) * 100 if excess > 0 else 0.0
                print("{:<8} {:<15.4f} {:<15.4f} {:<15.4f} {:<10.1f}%".format(
                    size, baseline, fisher_auc, delta, pct))

    # Save
    out_path = BASE_DIR / 'outputs' / 'p4' / 'size_ablation' / 'scattered_baselines.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to {}".format(out_path))


if __name__ == '__main__':
    main()
