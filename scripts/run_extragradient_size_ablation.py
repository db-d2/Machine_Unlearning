#!/usr/bin/env python3
"""
Extra-gradient size ablation on PBMC structured forget sets.

Tests extra-gradient lambda=10 at different structured forget set sizes
(n=10, 50, 100) with 3 seeds each.

Attacker methodology (matches canonical NB03):
- Train fresh MLP attacker per size on BASELINE features: F vs matched neg
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims
- Apply same attacker to all seeds for that size

Requires:
- Structured splits created by create_structured_size_splits.py
- Pre-trained attackers at outputs/p2/attackers/ (for EG co-training only)

Usage:
    PYTHONPATH=src python scripts/run_extragradient_size_ablation.py
    PYTHONPATH=src python scripts/run_extragradient_size_ablation.py --sizes 10 50 100
    PYTHONPATH=src python scripts/run_extragradient_size_ablation.py --evaluate-only
"""

import sys
import os
import subprocess
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as functional
from pathlib import Path
from datetime import datetime
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

# === Paths ===
DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
# Pre-trained attackers used only for EG co-training, NOT for evaluation
COTRAIN_ATTACKER_DIR = BASE_DIR / 'outputs' / 'p2' / 'attackers'
SPLIT_DIR = BASE_DIR / 'outputs' / 'p4' / 'extragradient_size' / 'splits'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'extragradient_size'

VARIANTS = ['v1', 'v2', 'v3']
ATTACKER_SEEDS = [42, 43, 44]
EXPERIMENT_SEEDS = [42, 123, 456]
LAMBDA = 10
KNN_RETAIN_SAMPLE = 5000
DEVICE = 'cpu'


def load_vae_model(checkpoint_path, config=None):
    """Load a VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if config is None:
        config = ckpt['config']
    state_dict = ckpt.get('model_state_dict', ckpt)

    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config.get('likelihood', 'nb'),
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
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


def train_fresh_attacker(baseline_model, adata, forget_idx, matched_neg_idx,
                         retain_idx, size_label=""):
    """Train fresh attacker on baseline F vs matched features.

    Matches canonical NB03 methodology.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    retain_z = get_retain_latent_codes(baseline_model, adata, retain_idx)

    forget_X = get_features(baseline_model, adata, forget_idx, retain_z=retain_z)
    matched_X = get_features(baseline_model, adata, matched_neg_idx,
                             retain_z=retain_z)

    feature_dim = forget_X.shape[1]

    member_labels = torch.ones(len(forget_X))
    nonmember_labels = torch.zeros(len(matched_X))
    all_X = torch.cat([forget_X, matched_X], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)

    perm = torch.randperm(len(all_X))
    n_train = int(0.8 * len(all_X))
    train_X = all_X[perm[:n_train]]
    train_y = all_labels[perm[:n_train]]

    print("  Training fresh attacker for n={}:".format(size_label))
    print("    Samples: {} ({} forget + {} matched)".format(
        len(all_X), len(forget_X), len(matched_X)))
    print("    Features: {} dims".format(feature_dim))

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

    attacker.train(False)
    with torch.no_grad():
        all_logits = attacker(all_X.to(DEVICE)).squeeze()
        all_preds = torch.sigmoid(all_logits).cpu().numpy()
    full_auc = roc_auc_score(all_labels.numpy(), all_preds)
    print("    Baseline AUC (F vs matched): {:.4f}".format(full_auc))

    return attacker, retain_z


def evaluate_with_attacker(model, attacker, adata, forget_idx, matched_neg_idx,
                           retain_idx):
    """Evaluate a model using a pre-trained fresh attacker.

    Features extracted from THIS model (not baseline), including
    kNN distances in this model's latent space.
    """
    retain_z = get_retain_latent_codes(model, adata, retain_idx)

    forget_feats = get_features(model, adata, forget_idx, retain_z=retain_z)
    unseen_feats = get_features(model, adata, matched_neg_idx, retain_z=retain_z)

    with torch.no_grad():
        forget_probs = torch.sigmoid(attacker(forget_feats)).cpu().numpy()
        unseen_probs = torch.sigmoid(attacker(unseen_feats)).cpu().numpy()

    y_true = np.concatenate([np.ones(len(forget_probs)),
                             np.zeros(len(unseen_probs))])
    y_score = np.concatenate([forget_probs.flatten(), unseen_probs.flatten()])
    auc = roc_auc_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_at_1pct = float(tpr[np.searchsorted(fpr, 0.01)]) if np.any(
        fpr <= 0.01) else 0.0
    tpr_at_5pct = float(tpr[np.searchsorted(fpr, 0.05)]) if np.any(
        fpr <= 0.05) else 0.0

    return {
        'auc': float(auc),
        'tpr_at_1pct_fpr': tpr_at_1pct,
        'tpr_at_5pct_fpr': tpr_at_5pct,
    }


def run_single(size, seed):
    """Run a single extra-gradient experiment."""
    output_dir = OUTPUT_BASE / 'n{}'.format(size) / 'seed{}'.format(seed)

    if (output_dir / 'best_model.pt').exists():
        print("    seed={}: already exists, skipping".format(seed))
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-trained attackers for co-training (correct usage)
    attacker_paths = [
        str(COTRAIN_ATTACKER_DIR / 'attacker_{}_seed{}.pt'.format(v, s))
        for v, s in zip(VARIANTS, ATTACKER_SEEDS)
    ]

    for p in attacker_paths:
        if not Path(p).exists():
            print("    ERROR: Attacker not found: {}".format(p))
            return False

    split_path = SPLIT_DIR / 'split_structured_n{}.json'.format(size)
    if not split_path.exists():
        print("    ERROR: Split not found: {}".format(split_path))
        return False

    cmd = [
        sys.executable, str(SRC_DIR / 'train_unlearn_extragradient.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(split_path),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--attacker_paths', *attacker_paths,
        '--lambda_retain', str(LAMBDA),
        '--epochs', '50',
        '--lr_vae', '0.0001',
        '--lr_critic', '0.00001',
        '--critic_steps', '2',
        '--abort_threshold', '3',
        '--batch_size', '256',
        '--output_dir', str(output_dir),
        '--seed', str(seed),
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    import scanpy as sc

    parser = argparse.ArgumentParser(
        description='Extra-gradient size ablation on PBMC')
    parser.add_argument('--sizes', nargs='+', type=int, default=[10, 50, 100])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=EXPERIMENT_SEEDS)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("EXTRA-GRADIENT SIZE ABLATION (PBMC)")
    print("=" * 60)
    print("Sizes: {}".format(args.sizes))
    print("Seeds: {}".format(args.seeds))
    print("Lambda: {}".format(LAMBDA))
    print("Attacker: fresh per-size (canonical NB03 methodology)")
    print()

    # Step 1: Train
    if not args.evaluate_only:
        print("-" * 60)
        print("TRAINING")
        print("-" * 60)
        for size in args.sizes:
            print("\n  n={}:".format(size))
            for seed in args.seeds:
                print("    Running seed={}...".format(seed))
                success = run_single(size, seed)
                if not success:
                    print("    FAILED")

    # Step 2: Evaluate with fresh attackers
    print("\n" + "-" * 60)
    print("POST-HOC EVALUATION (fresh attacker per size)")
    print("-" * 60)

    # Load data and baseline once
    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    baseline_model, config = load_vae_model(BASELINE_CHECKPOINT)

    results_by_size = {}
    for size in args.sizes:
        split_path = SPLIT_DIR / 'split_structured_n{}.json'.format(size)
        matched_path = SPLIT_DIR / 'matched_neg_structured_n{}.json'.format(
            size)

        if not split_path.exists():
            print("\n  n={}: split not found".format(size))
            continue

        with open(split_path) as f:
            split = json.load(f)
        with open(matched_path) as f:
            matched = json.load(f)

        forget_idx = split['forget_indices']
        retain_idx = split['retain_indices']
        matched_idx = matched['matched_indices']

        # Train fresh attacker for this size
        print()
        attacker, _ = train_fresh_attacker(
            baseline_model, adata, forget_idx, matched_idx, retain_idx,
            size_label=str(size)
        )

        # Evaluate each seed with this attacker
        aucs = []
        for seed in args.seeds:
            model_path = (OUTPUT_BASE / 'n{}'.format(size) /
                          'seed{}'.format(seed) / 'best_model.pt')
            if not model_path.exists():
                print("  n={}, seed={}: model not found".format(size, seed))
                continue

            model, _ = load_vae_model(model_path, config=config)
            metrics = evaluate_with_attacker(
                model, attacker, adata, forget_idx, matched_idx, retain_idx
            )
            aucs.append(metrics['auc'])
            print("  n={}, seed={}: AUC={:.4f}".format(
                size, seed, metrics['auc']))

        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            ci_low = mean_auc - 1.96 * std_auc / np.sqrt(len(aucs))
            ci_high = mean_auc + 1.96 * std_auc / np.sqrt(len(aucs))

            results_by_size[str(size)] = {
                'mean': float(mean_auc),
                'std': float(std_auc),
                'ci_95_lower': float(ci_low),
                'ci_95_upper': float(ci_high),
                'n_runs': len(aucs),
                'all_aucs': [float(a) for a in aucs],
            }

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    target_low, target_high = 0.451, 0.511
    retrain_floor = 0.481

    print("\nRetrain floor: {:.3f}".format(retrain_floor))
    print("Target band: [{:.3f}, {:.3f}]".format(target_low, target_high))
    print("PBMC baseline AUC: 0.769")
    print("PBMC n=30 extra-gradient (reference): 0.482")
    print()

    print("{:<8} {:<10} {:<10} {:<20} {:<15}".format(
        'Size', 'Mean AUC', 'Std', '95% CI', 'Status'))
    print("-" * 65)

    print("{:<8} {:<10} {:<10} {:<20} {:<15}".format(
        30, '0.482', '-', '(reference)', 'IN TARGET BAND'))

    for size in sorted(results_by_size.keys(), key=int):
        r = results_by_size[size]
        status = ("IN TARGET BAND" if target_low <= r['mean'] <= target_high
                  else ("over-unlearned" if r['mean'] < target_low
                        else "insufficient"))
        print("{:<8} {:<10.4f} {:<10.4f} [{:.4f}, {:.4f}]   {:<15}".format(
            int(size), r['mean'], r['std'],
            r['ci_95_lower'], r['ci_95_upper'], status))

    # Save
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'pbmc',
        'method': 'extra-gradient',
        'lambda': LAMBDA,
        'forget_type': 'structured',
        'attacker_methodology': 'Fresh per-size attacker on baseline F vs '
                                'matched (canonical NB03)',
        'seeds': args.seeds,
        'retrain_floor': retrain_floor,
        'target_band': [target_low, target_high],
        'reference_n30': {'auc': 0.482, 'status': 'in_target_band'},
        'results_by_size': results_by_size,
    }

    out_path = OUTPUT_BASE / 'summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to {}".format(out_path))


if __name__ == '__main__':
    main()
