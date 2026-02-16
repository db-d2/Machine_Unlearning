#!/usr/bin/env python3
"""
Lambda sweep for extra-gradient on Tabula Muris structured forget set.

Extra-gradient with lambda=10 failed (AUC=0.874 vs target ~0.48-0.52).
Sweep lambda=[5, 7, 15, 20] to determine if any lambda achieves target band.

Attacker methodology (matches canonical NB03):
- Train fresh MLP attacker on TM BASELINE features: F vs matched neg
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims
- Apply same attacker to all lambda values

Usage:
    PYTHONPATH=src python scripts/run_lambda_sweep.py
    PYTHONPATH=src python scripts/run_lambda_sweep.py --lambdas 5 7 15 20
    PYTHONPATH=src python scripts/run_lambda_sweep.py --lambdas 5 --evaluate-only
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
DATA_PATH = BASE_DIR / 'data' / 'tabula_muris_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'tabula_muris' / 'baseline' / 'best_model.pt'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'tabula_muris'
# Pre-trained attackers for co-training only, NOT evaluation
COTRAIN_ATTACKER_DIR = OUTPUT_BASE / 'attackers'
SPLIT_STRUCTURED = OUTPUT_BASE / 'split_structured.json'
MATCHED_NEG_STRUCTURED = OUTPUT_BASE / 'matched_negatives.json'

VARIANTS = ['v1', 'v2', 'v3']
SEEDS = [42, 43, 44]
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
                         retain_idx):
    """Train fresh attacker on TM baseline F vs matched features.

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

    print("Training fresh TM attacker:")
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


def evaluate_with_attacker(model, attacker, adata, forget_idx, matched_neg_idx,
                           retain_idx):
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


def run_extragradient_lambda(lam):
    """Run extra-gradient with a specific lambda value."""
    output_dir = OUTPUT_BASE / 'extragradient_lambda{}'.format(lam)

    if (output_dir / 'best_model.pt').exists():
        print("  lambda={}: already exists, skipping training".format(lam))
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Co-training attackers (correct usage for adversarial training)
    attacker_paths = [
        str(COTRAIN_ATTACKER_DIR / 'attacker_{}_seed{}.pt'.format(v, s))
        for v, s in zip(VARIANTS, SEEDS)
    ]

    for p in attacker_paths:
        if not Path(p).exists():
            print("  ERROR: Attacker not found: {}".format(p))
            return False

    cmd = [
        sys.executable, str(SRC_DIR / 'train_unlearn_extragradient.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(SPLIT_STRUCTURED),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--attacker_paths', *attacker_paths,
        '--lambda_retain', str(lam),
        '--epochs', '50',
        '--lr_vae', '0.0001',
        '--lr_critic', '0.00001',
        '--critic_steps', '2',
        '--abort_threshold', '3',
        '--batch_size', '256',
        '--output_dir', str(output_dir),
        '--seed', '42',
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    print("  Lambda: {}, Epochs: 50, LR_VAE: 1e-4, LR_Critic: 1e-5".format(lam))

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    import scanpy as sc

    parser = argparse.ArgumentParser(
        description='Lambda sweep for extra-gradient on Tabula Muris')
    parser.add_argument('--lambdas', nargs='+', type=float,
                        default=[5, 7, 15, 20],
                        help='Lambda values to sweep (default: 5 7 15 20)')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Skip training, only evaluate existing models')
    args = parser.parse_args()

    print("=" * 60)
    print("LAMBDA SWEEP: Extra-gradient on Tabula Muris")
    print("=" * 60)
    print("Lambdas to test: {}".format(args.lambdas))
    print("Structured forget set: Cluster 33 (cardiac muscle cells)")
    print("Attacker: fresh (canonical NB03 methodology)")
    print()

    all_lambdas = sorted(set(args.lambdas + [10]))

    # Step 1: Train
    if not args.evaluate_only:
        print("-" * 60)
        print("TRAINING")
        print("-" * 60)
        for lam in args.lambdas:
            print("\nRunning extra-gradient with lambda={}...".format(lam))
            success = run_extragradient_lambda(lam)
            if not success:
                print("  FAILED for lambda={}".format(lam))

    # Step 2: Load data and train fresh attacker
    print("\nLoading data...")
    adata = sc.read_h5ad(DATA_PATH)
    baseline_model, config = load_vae_model(BASELINE_CHECKPOINT)

    with open(SPLIT_STRUCTURED) as f:
        split = json.load(f)
    with open(MATCHED_NEG_STRUCTURED) as f:
        matched = json.load(f)

    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    matched_idx = matched['matched_indices']

    attacker = train_fresh_attacker(
        baseline_model, adata, forget_idx, matched_idx, retain_idx
    )

    # Step 3: Evaluate all
    print("\n" + "-" * 60)
    print("POST-HOC EVALUATION (fresh attacker)")
    print("-" * 60)

    results = {}
    for lam in all_lambdas:
        if lam == 10:
            model_path = OUTPUT_BASE / 'extragradient_structured' / 'best_model.pt'
        else:
            model_path = OUTPUT_BASE / 'extragradient_lambda{}'.format(lam) / 'best_model.pt'

        if not model_path.exists():
            print("\n  lambda={}: no model found, skipping".format(lam))
            continue

        print("\n  Evaluating lambda={}...".format(lam))
        try:
            model, _ = load_vae_model(model_path, config=config)
            metrics = evaluate_with_attacker(
                model, attacker, adata, forget_idx, matched_idx, retain_idx
            )
            results[lam] = metrics

            # Load metadata for training info
            if lam == 10:
                meta_path = OUTPUT_BASE / 'extragradient_structured' / 'metadata.json'
            else:
                meta_path = OUTPUT_BASE / 'extragradient_lambda{}'.format(lam) / 'metadata.json'

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                results[lam]['epochs_trained'] = meta.get('total_epochs', '?')
                results[lam]['early_stopped'] = meta.get('early_stopped', '?')

            print("    AUC: {:.4f}".format(metrics['auc']))
            print("    TPR@1%FPR: {:.4f}".format(metrics['tpr_at_1pct_fpr']))
            print("    TPR@5%FPR: {:.4f}".format(metrics['tpr_at_5pct_fpr']))
        except Exception as e:
            print("    FAILED: {}".format(e))
            import traceback
            traceback.print_exc()

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("LAMBDA SWEEP RESULTS")
    print("=" * 60)

    # Load baseline for reference
    baseline_results_path = OUTPUT_BASE / 'baseline_results.json'
    if baseline_results_path.exists():
        with open(baseline_results_path) as f:
            baseline = json.load(f)
        baseline_auc = baseline['baseline_auc']
        retrain_auc = baseline.get('retrain_floor',
                                   baseline.get('retrain_auc', 'N/A'))
    else:
        baseline_auc = 'N/A'
        retrain_auc = 'N/A'

    target_low = 0.45
    target_high = 0.52

    print("\nBaseline AUC: {}".format(baseline_auc))
    print("Retrain floor: {}".format(retrain_auc))
    print("Target band: [{}, {}]".format(target_low, target_high))
    print("PBMC reference: lambda=10 -> AUC 0.48 (in target band)")
    print()

    print("{:<10} {:<10} {:<10} {:<12} {}".format(
        'Lambda', 'AUC', 'Epochs', 'Early Stop', 'Status'))
    print("-" * 55)

    for lam in sorted(results.keys()):
        r = results[lam]
        auc = r['auc']
        epochs = r.get('epochs_trained', '?')
        early = r.get('early_stopped', '?')

        if target_low <= auc <= target_high:
            status = "IN TARGET BAND"
        elif auc < target_low:
            status = "over-unlearned"
        elif isinstance(baseline_auc, float) and auc < baseline_auc:
            status = "partial unlearning"
        else:
            status = "no effect"

        print("{:<10} {:<10.4f} {:<10} {:<12} {}".format(
            lam, auc, str(epochs), str(early), status))

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'tabula_muris',
        'experiment': 'lambda_sweep',
        'attacker_methodology': 'Fresh attacker on TM baseline F vs '
                                'matched (canonical NB03)',
        'forget_set': 'structured (Cluster 33, cardiac muscle cells)',
        'baseline_auc': baseline_auc,
        'target_band': [target_low, target_high],
        'pbmc_reference': {
            'lambda_10_auc': 0.48,
            'status': 'in_target_band'
        },
        'results': {str(k): v for k, v in results.items()},
    }

    out_path = OUTPUT_BASE / 'lambda_sweep_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to {}".format(out_path))

    # Verdict
    any_success = any(target_low <= r['auc'] <= target_high
                      for r in results.values())
    print("\n" + "=" * 60)
    if any_success:
        best = min(results.items(), key=lambda x: abs(x[1]['auc'] - 0.50))
        print("SUCCESS: lambda={} achieves AUC={:.4f}".format(
            best[0], best[1]['auc']))
    else:
        print("FINDING: No lambda value achieves target band on Tabula Muris")
        print("This suggests extra-gradient dynamics are dataset-dependent")
        closest = min(results.items(), key=lambda x: abs(x[1]['auc'] - 0.50))
        print("Closest: lambda={} -> AUC={:.4f}".format(
            closest[0], closest[1]['auc']))
    print("=" * 60)


if __name__ == '__main__':
    main()
