#!/usr/bin/env python3
"""Multi-seed evaluation for publication statistical rigor.

Evaluates all multiseed checkpoints plus baseline and retrain.
Produces per-seed JSONs, method summaries, and a master manifest.

Per checkpoint computes:
1. MLP attacker AUC + advantage + bootstrap CI
2. Utility metrics (ELBO, silhouette, ARI, marker correlation)

Attacker methodology (matches canonical NB03):
- Train fresh MLP attacker on BASELINE features: F (30) vs matched neg (194)
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims
- Apply same attacker to features from each model being evaluated

Usage:
    PYTHONPATH=src python scripts/eval_multiseed.py
    PYTHONPATH=src python scripts/eval_multiseed.py --methods extragradient
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn.functional as functional
import scanpy as sc
from pathlib import Path
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).parent.parent / 'src'
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

from vae import VAE, vae_loss
from attacker import (
    MLPAttacker, extract_vae_features, build_attack_features,
    compute_knn_distances
)
from attacker_eval import (
    matched_negative_evaluation, compute_confidence_interval,
    compute_advantage, compute_attack_metrics
)
from utility_metrics import (
    compute_held_out_elbo, compute_latent_metrics, compute_marker_correlation
)

# === Paths ===
DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
SPLIT_PATH = BASE_DIR / 'outputs' / 'p1' / 'split_structured.json'
MATCHED_NEG_PATH = BASE_DIR / 'outputs' / 'p1.5' / 's1_matched_negatives.json'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
RETRAIN_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'retrain_structured' / 'best_model.pt'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'multiseed'
EVAL_DIR = OUTPUT_BASE / 'eval'

# Seeds
EG_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
SIMPLE_SEEDS = [42, 123, 456, 789, 1011]

# Marker genes for PBMC
MARKER_GENES = ['CD3D', 'CD3E', 'MS4A1', 'CD79A', 'CD14', 'LYZ', 'NKG7',
                'GNLY']

# kNN reference size (sample retain set for speed, matching NB03)
KNN_RETAIN_SAMPLE = 5000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_vae_model(checkpoint_path):
    """Load a VAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config'] if isinstance(ckpt, dict) and 'config' in ckpt else None

    if config is None:
        config = {
            'input_dim': 2000, 'latent_dim': 32,
            'hidden_dims': [1024, 512, 128],
            'likelihood': 'nb', 'dropout': 0.1, 'use_layer_norm': True,
        }
        state_dict = ckpt
    else:
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
    """Get latent codes for a sample of the retain set (for kNN features).

    Matches NB03: retain_indices[:5000] sampled for speed.
    """
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
    """Extract 70-dim attack features (v1 + kNN distance to retain).

    Matches NB03 feature extraction:
    - v1 features: recon_nll, kl, elbo, mu(32), logvar(32), mu_norm, logvar_norm = 69 dims
    - kNN distance to retain (k=5) = 1 dim
    - Total: 70 dims
    """
    X = adata.X[indices]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    x_tensor = torch.FloatTensor(np.asarray(X)).to(DEVICE)
    lib = x_tensor.sum(dim=1, keepdim=True)

    with torch.no_grad():
        features = extract_vae_features(model, x_tensor, lib, device=DEVICE)

    # Compute kNN distances to retain set
    knn_dist = None
    if retain_z is not None:
        query_z = features['z'].numpy()
        knn_dist = compute_knn_distances(query_z, retain_z, k=5)

    attack_features = build_attack_features(
        features, knn_dist_retain=knn_dist, variant='v1'
    )
    return attack_features


def train_fresh_attacker(baseline_model, adata, forget_idx, matched_neg_idx,
                         retain_idx, seed=42):
    """Train fresh attacker on baseline F vs matched features.

    Matches canonical NB03 methodology:
    - Extract v1 + kNN features from BASELINE model
    - Train on F (30, label=1) vs matched neg (194, label=0)
    - 80/20 train/test split
    - MLP with spectral norm, [256,256], dropout 0.3
    - 50 epochs, Adam lr=1e-3, batch_size=128, BCE loss
    - Report AUC on full dataset (not just test set)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get retain latent codes from baseline for kNN
    retain_z = get_retain_latent_codes(baseline_model, adata, retain_idx)

    # Extract features from baseline model
    forget_X = get_features(baseline_model, adata, forget_idx, retain_z=retain_z)
    matched_X = get_features(baseline_model, adata, matched_neg_idx,
                             retain_z=retain_z)

    feature_dim = forget_X.shape[1]

    # Labels: 1=member (forget), 0=non-member (matched)
    member_labels = torch.ones(len(forget_X))
    nonmember_labels = torch.zeros(len(matched_X))
    all_X = torch.cat([forget_X, matched_X], dim=0)
    all_labels = torch.cat([member_labels, nonmember_labels], dim=0)

    # 80/20 train/test split
    perm = torch.randperm(len(all_X))
    n_train = int(0.8 * len(all_X))
    train_X = all_X[perm[:n_train]]
    train_y = all_labels[perm[:n_train]]

    print(f"Training fresh attacker on baseline F vs matched:")
    print(f"  Samples: {len(all_X)} ({len(forget_X)} forget + "
          f"{len(matched_X)} matched)")
    print(f"  Features: {feature_dim} dims")
    print(f"  Train: {n_train}, Test: {len(all_X) - n_train}")

    # Initialize attacker (spectral norm, matching NB03)
    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3,
        use_spectral_norm=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(attacker.parameters(), lr=1e-3)

    # Train
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

    # Report AUC on full dataset (matching NB03 methodology)
    attacker.train(False)
    with torch.no_grad():
        all_logits = attacker(all_X.to(DEVICE)).squeeze()
        all_preds = torch.sigmoid(all_logits).cpu().numpy()
    full_auc = roc_auc_score(all_labels.numpy(), all_preds)
    print(f"  Baseline AUC (F vs matched, full set): {full_auc:.4f}")
    print(f"  (Canonical NB03 value: ~0.769)")

    return attacker


def evaluate_privacy(model, attacker, adata, forget_idx, matched_neg_idx,
                     retain_idx):
    """Evaluate MLP attacker AUC with bootstrap CI.

    Features are extracted from the model being evaluated (not baseline),
    including kNN distances computed in that model's latent space.
    The attacker was trained on baseline features but is applied to
    this model's features -- if unlearning worked, forget features
    should look like matched negative features.
    """
    # Get retain latent codes from THIS model (matching NB03 cell 20)
    retain_z = get_retain_latent_codes(model, adata, retain_idx)

    forget_feats = get_features(model, adata, forget_idx, retain_z=retain_z)
    neg_feats = get_features(model, adata, matched_neg_idx, retain_z=retain_z)

    result = matched_negative_evaluation(
        attacker, forget_feats, neg_feats, device=DEVICE
    )
    return {
        'mlp_auc': result['auc'],
        'mlp_advantage': compute_advantage(result['auc']),
        'ci_lower': result.get('auc_ci_lower', None),
        'ci_upper': result.get('auc_ci_upper', None),
        'advantage_ci_lower': result.get('advantage_ci_lower', None),
        'advantage_ci_upper': result.get('advantage_ci_upper', None),
    }


def evaluate_utility(model, X_holdout, labels_holdout, marker_idx,
                     gene_names):
    """Compute utility metrics on held-out data."""
    elbo = compute_held_out_elbo(model, X_holdout, DEVICE)
    latent = compute_latent_metrics(model, X_holdout, labels_holdout, DEVICE)
    marker = compute_marker_correlation(model, X_holdout, marker_idx,
                                        gene_names, DEVICE)
    return {
        'elbo': elbo['elbo'],
        'recon': elbo['recon'],
        'kl': elbo['kl'],
        'silhouette': latent['silhouette'],
        'ari': latent['ari'],
        'marker_r': marker['mean_r'],
        'marker_per_gene': marker['per_gene'],
    }


def evaluate_checkpoint(checkpoint_path, method, seed, adata, X_holdout,
                        labels_holdout, forget_idx, matched_neg_idx,
                        retain_idx, marker_idx, gene_names, attacker):
    """Full assessment of a single checkpoint."""
    print(f"  Checking {method} seed={seed}...")
    model, config = load_vae_model(checkpoint_path)

    privacy = evaluate_privacy(model, attacker, adata, forget_idx,
                               matched_neg_idx, retain_idx)
    utility = evaluate_utility(model, X_holdout, labels_holdout, marker_idx,
                               gene_names)

    return {
        'method': method,
        'seed': seed,
        'checkpoint': str(checkpoint_path),
        'privacy': privacy,
        'utility': utility,
    }


def compute_summary(method, results):
    """Aggregate per-seed results into method summary with CIs."""
    n_seeds = len(results)
    summary = {'method': method, 'n_seeds': n_seeds, 'privacy': {},
               'utility': {}}

    from scipy import stats

    # Privacy aggregation
    for key in ['mlp_auc', 'mlp_advantage']:
        values = [r['privacy'][key] for r in results]
        values_arr = np.array(values)
        mean = float(values_arr.mean())
        std = float(values_arr.std())
        se = std / np.sqrt(n_seeds)
        if n_seeds > 1:
            t_val = stats.t.ppf(0.975, n_seeds - 1)
            ci = [float(mean - t_val * se), float(mean + t_val * se)]
        else:
            ci = [float(mean), float(mean)]
        summary['privacy'][key] = {
            'mean': mean, 'std': std, 'ci_95': ci,
            'values': [float(v) for v in values],
        }

    # Utility aggregation
    for key in ['elbo', 'recon', 'kl', 'silhouette', 'ari', 'marker_r']:
        values = [r['utility'][key] for r in results]
        values_arr = np.array(values)
        mean = float(values_arr.mean())
        std = float(values_arr.std())
        se = std / np.sqrt(n_seeds)
        if n_seeds > 1:
            t_val = stats.t.ppf(0.975, n_seeds - 1)
            ci = [float(mean - t_val * se), float(mean + t_val * se)]
        else:
            ci = [float(mean), float(mean)]
        summary['utility'][key] = {
            'mean': mean, 'std': std, 'ci_95': ci,
            'values': [float(v) for v in values],
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed assessment for publication')
    parser.add_argument('--methods', nargs='+',
                        default=['extragradient', 'retain_finetune',
                                 'gradient_ascent', 'baseline', 'retrain'],
                        help='Methods to assess')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='Skip seeds with existing JSONs')
    args = parser.parse_args()

    # Load data once
    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    X = torch.tensor(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        dtype=torch.float32
    )
    gene_names = adata.var_names.tolist()
    leiden_labels = adata.obs['leiden'].values

    # Load split
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    unseen_idx = split['unseen_indices']

    # Load matched negatives
    with open(MATCHED_NEG_PATH, 'r') as f:
        matched_data = json.load(f)
    matched_neg_idx = matched_data['matched_indices']

    # Holdout set for utility
    X_holdout = X[unseen_idx]
    labels_holdout = leiden_labels[unseen_idx]

    # Marker gene indices
    marker_idx = [gene_names.index(g) for g in MARKER_GENES
                  if g in gene_names]
    print(f"Marker genes: {len(marker_idx)}/{len(MARKER_GENES)}")
    print(f"Device: {DEVICE}")

    # Train fresh attacker on baseline F vs matched (canonical NB03 approach)
    print("\nLoading baseline model for attacker training...")
    baseline_model, _ = load_vae_model(BASELINE_CHECKPOINT)
    attacker = train_fresh_attacker(
        baseline_model, adata, forget_idx, matched_neg_idx, retain_idx
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Track all results for manifest
    manifest_methods = {}

    # === Assess multi-seed methods ===
    method_configs = {
        'extragradient': EG_SEEDS,
        'retain_finetune': SIMPLE_SEEDS,
        'gradient_ascent': SIMPLE_SEEDS,
    }

    for method, seeds in method_configs.items():
        if method not in args.methods:
            continue

        print(f"\n{'=' * 60}")
        print(f"ASSESSING: {method}")
        print(f"{'=' * 60}")

        method_eval_dir = EVAL_DIR / method
        method_eval_dir.mkdir(parents=True, exist_ok=True)

        results = []
        eval_files = []
        checkpoints = []

        for seed in seeds:
            eval_path = method_eval_dir / f'seed{seed}.json'
            checkpoint_path = (OUTPUT_BASE / method / f'seed{seed}' /
                               'best_model.pt')

            if not checkpoint_path.exists():
                print(f"  seed={seed}: checkpoint not found, skipping")
                continue

            checkpoints.append(str(checkpoint_path))
            eval_files.append(str(eval_path))

            if eval_path.exists() and args.skip_existing:
                print(f"  seed={seed}: result exists, loading")
                with open(eval_path, 'r') as f:
                    result = json.load(f)
                results.append(result)
                continue

            result = evaluate_checkpoint(
                checkpoint_path, method, seed, adata, X_holdout,
                labels_holdout, forget_idx, matched_neg_idx, retain_idx,
                marker_idx, gene_names, attacker
            )
            results.append(result)

            with open(eval_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"    AUC={result['privacy']['mlp_auc']:.3f}, "
                  f"ELBO={result['utility']['elbo']:.1f}")

        if results:
            summary = compute_summary(method, results)
            summary_path = method_eval_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n  Summary: AUC={summary['privacy']['mlp_auc']['mean']:.3f} "
                  f"+/- {summary['privacy']['mlp_auc']['std']:.3f}")

            manifest_methods[method] = {
                'n_seeds': len(results),
                'seeds': [r['seed'] for r in results],
                'checkpoints': checkpoints,
                'eval_files': eval_files,
                'summary': str(summary_path),
            }

    # === Assess single checkpoints (baseline, retrain) ===
    single_checkpoints = {
        'baseline': BASELINE_CHECKPOINT,
        'retrain': RETRAIN_CHECKPOINT,
    }

    for method, checkpoint_path in single_checkpoints.items():
        if method not in args.methods:
            continue
        if not checkpoint_path.exists():
            print(f"\n{method}: checkpoint not found, skipping")
            continue

        eval_path = EVAL_DIR / f'{method}.json'

        if eval_path.exists() and args.skip_existing:
            print(f"\n{method}: result exists, loading")
            with open(eval_path, 'r') as f:
                result = json.load(f)
        else:
            print(f"\nAssessing {method}...")
            result = evaluate_checkpoint(
                checkpoint_path, method, 42, adata, X_holdout,
                labels_holdout, forget_idx, matched_neg_idx, retain_idx,
                marker_idx, gene_names, attacker
            )
            with open(eval_path, 'w') as f:
                json.dump(result, f, indent=2)

        print(f"  AUC={result['privacy']['mlp_auc']:.3f}, "
              f"ELBO={result['utility']['elbo']:.1f}")

        manifest_methods[method] = {
            'n_seeds': 1,
            'checkpoints': [str(checkpoint_path)],
            'eval_files': [str(eval_path)],
        }

    # === Write manifest ===
    manifest = {
        'generated': datetime.now().isoformat(),
        'attacker_methodology': 'Fresh attacker trained on baseline F vs '
                                'matched (canonical NB03 approach)',
        'methods': manifest_methods,
        'prior_results': {
            'attack_suite': str(BASE_DIR / 'outputs' /
                                'attack_suite_results.json'),
            'utility_single_seed': str(BASE_DIR / 'outputs' /
                                       'utility_suite_results.json'),
            'fisher_structured_3seed': str(BASE_DIR / 'outputs' / 'p4' /
                                           'ablation_results.json'),
            'fisher_size_ablation_10seed': str(BASE_DIR / 'outputs' / 'p4' /
                                               'size_ablation' / 'fisher'),
            'eg_size_ablation_3seed': str(BASE_DIR / 'outputs' / 'p4' /
                                          'extragradient_size'),
            'tabula_muris': str(BASE_DIR / 'outputs' / 'tabula_muris'),
            'matched_negative_validation': str(
                BASE_DIR / 'notebooks' /
                '19_matched_negative_validation.ipynb'),
        },
    }

    manifest_path = OUTPUT_BASE / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
