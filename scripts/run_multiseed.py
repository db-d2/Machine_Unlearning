#!/usr/bin/env python3
"""
Multi-seed ablation study for Fisher unlearning confidence intervals.

Runs Fisher unlearning with different random seeds and computes
mean, std, and 95% confidence intervals for statistical significance.

Attacker methodology (matches canonical NB03):
- Train fresh MLP attacker per forget type on BASELINE features: F vs matched neg
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims
- Apply same attacker to all seeds for that forget type
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os
import numpy as np
import torch
import torch.nn.functional as functional
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from vae import VAE
from attacker import (
    MLPAttacker, extract_vae_features, build_attack_features,
    compute_knn_distances
)

# Configuration
SEEDS = [42, 123, 456]
FORGET_TYPES = ['structured', 'scattered']

# Default hyperparameters
DEFAULTS = {
    'scrub_lr': 0.0001,
    'scrub_steps': 100,
    'finetune_epochs': 10,
    'finetune_lr': 0.0001,
    'fisher_damping': 0.1,
    'batch_size': 256,
}

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'multiseed'
SPLIT_DIR = BASE_DIR / 'outputs' / 'p1'

# Matched negatives paths
MATCHED_NEGATIVES = {
    'structured': BASE_DIR / 'outputs' / 'p1.5' / 's1_matched_negatives.json',
    'scattered': BASE_DIR / 'outputs' / 'p1.5' / 'scattered_matched_negatives.json',
}

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
                         retain_idx, label=""):
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

    print("Training fresh attacker for {}:".format(label))
    print("  Samples: {} ({} forget + {} matched)".format(
        len(all_X), len(forget_X), len(matched_X)))
    print("  Features: {} dims".format(feature_dim))

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
    print("  Baseline AUC (F vs matched): {:.4f}".format(full_auc))

    return attacker


def evaluate_with_attacker(model, attacker, adata, forget_idx,
                           matched_neg_idx, retain_idx):
    """Evaluate model using a pre-trained fresh attacker.

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

    return {'auc': float(auc)}


def run_fisher_unlearn(split_path, output_dir, seed):
    """Run Fisher unlearning with given seed."""
    cmd = [
        sys.executable, str(BASE_DIR / 'src' / 'train_fisher_unlearn.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(split_path),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--output_dir', str(output_dir),
        '--scrub_lr', str(DEFAULTS['scrub_lr']),
        '--scrub_steps', str(DEFAULTS['scrub_steps']),
        '--finetune_epochs', str(DEFAULTS['finetune_epochs']),
        '--finetune_lr', str(DEFAULTS['finetune_lr']),
        '--fisher_damping', str(DEFAULTS['fisher_damping']),
        '--batch_size', str(DEFAULTS['batch_size']),
        '--seed', str(seed),
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode == 0


def compute_statistics(results):
    """Compute mean, std, and 95% CI."""
    aucs = [r['auc'] for r in results]
    n = len(aucs)
    mean = np.mean(aucs)
    std = np.std(aucs, ddof=1)

    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return {
        'mean': float(mean),
        'std': float(std),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'n_runs': n,
    }


def main():
    import argparse
    import scanpy as sc

    parser = argparse.ArgumentParser()
    parser.add_argument('--forget-type', choices=['structured', 'scattered', 'both'],
                       default='both', help='Which forget type to run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training, only re-evaluate existing models')
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    forget_types = FORGET_TYPES if args.forget_type == 'both' else [args.forget_type]

    # Load data and baseline once for evaluation
    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    baseline_model, config = load_vae_model(BASELINE_CHECKPOINT)

    all_results = {}

    for forget_type in forget_types:
        print("\n{}".format('=' * 70))
        print("MULTI-SEED EXPERIMENT: {}".format(forget_type.upper()))
        print("{}".format('=' * 70))

        split_path = SPLIT_DIR / 'split_{}.json'.format(forget_type)
        matched_path = MATCHED_NEGATIVES[forget_type]

        if not matched_path.exists():
            print("  Matched negatives not found: {}".format(matched_path))
            continue

        with open(split_path) as f:
            split = json.load(f)
        with open(matched_path) as f:
            matched = json.load(f)

        forget_idx = split['forget_indices']
        retain_idx = split['retain_indices']
        matched_idx = matched['matched_indices']

        # Train fresh attacker for this forget type
        attacker = train_fresh_attacker(
            baseline_model, adata, forget_idx, matched_idx, retain_idx,
            label=forget_type
        )

        results = []

        for seed in SEEDS:
            run_name = '{}_seed{}'.format(forget_type, seed)
            output_dir = OUTPUT_BASE / forget_type / run_name

            if args.dry_run:
                print("Would run: {}".format(run_name))
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            # Train if needed
            model_path = output_dir / 'unlearned_model.pt'
            if not args.evaluate_only and not model_path.exists():
                print("\n--- Running seed={} ---".format(seed))
                start_time = time.time()
                success = run_fisher_unlearn(split_path, output_dir, seed)
                elapsed = time.time() - start_time

                if not success:
                    print("  Training failed")
                    continue
            elif not model_path.exists():
                print("  seed={}: model not found, skipping".format(seed))
                continue

            # Evaluate
            try:
                model, _ = load_vae_model(model_path, config=config)
                eval_result = evaluate_with_attacker(
                    model, attacker, adata, forget_idx, matched_idx, retain_idx
                )
                print("  seed={}: AUC={:.4f}".format(seed, eval_result['auc']))

                results.append({
                    'seed': seed,
                    **eval_result
                })

                with open(output_dir / 'eval_v1.json', 'w') as f:
                    json.dump(eval_result, f, indent=2)

            except Exception as e:
                print("  Evaluation failed: {}".format(e))

        if not args.dry_run and results:
            stats_result = compute_statistics(results)

            print("\n{}".format('=' * 60))
            print("STATISTICS for {}:".format(forget_type.upper()))
            print("  Mean AUC: {:.4f}".format(stats_result['mean']))
            print("  Std: {:.4f}".format(stats_result['std']))
            print("  95% CI: [{:.4f}, {:.4f}]".format(
                stats_result['ci_95_lower'], stats_result['ci_95_upper']))
            print("{}".format('=' * 60))

            all_results[forget_type] = {
                'runs': results,
                'statistics': stats_result,
            }

            with open(OUTPUT_BASE / forget_type / 'results.json', 'w') as f:
                json.dump(all_results[forget_type], f, indent=2)

    # Save combined results
    if not args.dry_run and all_results:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'attacker_methodology': 'Fresh per-type attacker on baseline F vs '
                                    'matched (canonical NB03)',
            'seeds': SEEDS,
            'defaults': DEFAULTS,
            'results': all_results,
        }

        with open(OUTPUT_BASE / 'multiseed_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n\nFINAL SUMMARY:")
        print("{}".format('=' * 70))
        for ft, data in all_results.items():
            s = data['statistics']
            print("{}: AUC = {:.4f} +/- {:.4f} (95% CI: [{:.4f}, {:.4f}])".format(
                ft, s['mean'], s['std'], s['ci_95_lower'], s['ci_95_upper']))

        print("\nResults saved to {}".format(
            OUTPUT_BASE / 'multiseed_summary.json'))


if __name__ == '__main__':
    main()
