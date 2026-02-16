#!/usr/bin/env python3
"""
Run unlearning experiments on Tabula Muris for cross-dataset validation.

Replicates the PBMC unlearning method on a new dataset:
1. Train 3 attacker variants (v1, v2, v3) on baseline model (for EG co-training)
2. Run extra-gradient (lambda=10) on structured forget set
3. Run Fisher scrubbing on structured forget set
4. Run Fisher scrubbing on scattered forget set
5. Evaluate all post-hoc with fresh attacker (canonical NB03 methodology)

Attacker methodology for evaluation:
- Train fresh MLP attacker per forget type on TM BASELINE features: F vs matched
- 80/20 train/test split, 50 epochs, Adam lr=1e-3
- Spectral norm, [256,256] hidden, dropout 0.3
- Features: v1 (69 dims) + kNN distance to retain (k=5) = 70 dims

Usage:
    PYTHONPATH=src python scripts/run_tabula_muris_unlearning.py
    PYTHONPATH=src python scripts/run_tabula_muris_unlearning.py --step attackers
    PYTHONPATH=src python scripts/run_tabula_muris_unlearning.py --step extragradient
    PYTHONPATH=src python scripts/run_tabula_muris_unlearning.py --step fisher
    PYTHONPATH=src python scripts/run_tabula_muris_unlearning.py --step evaluate
"""

import sys
import os
import subprocess
import json
import time
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
# Co-training attackers (for adversarial EG training, NOT evaluation)
COTRAIN_ATTACKER_DIR = OUTPUT_BASE / 'attackers'
SPLIT_STRUCTURED = OUTPUT_BASE / 'split_structured.json'
SPLIT_SCATTERED = OUTPUT_BASE / 'split_scattered.json'
MATCHED_NEG_STRUCTURED = OUTPUT_BASE / 'matched_negatives.json'
MATCHED_NEG_SCATTERED = OUTPUT_BASE / 'matched_negatives_scattered.json'

# === Attacker config ===
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


def train_attackers():
    """Step 1: Train 3 attacker variants on baseline model (for EG co-training)."""
    from train_multi_attackers import train_single_attacker

    COTRAIN_ATTACKER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Training Co-Training Attacker Variants")
    print("=" * 60)

    for variant, seed in zip(VARIANTS, SEEDS):
        path = COTRAIN_ATTACKER_DIR / 'attacker_{}_seed{}.pt'.format(variant, seed)
        if path.exists():
            print("  {} already exists, skipping".format(path.name))
            continue

        print("\nTraining attacker: variant={}, seed={}".format(variant, seed))
        auc = train_single_attacker(
            vae_path=str(BASELINE_CHECKPOINT),
            data_path=str(DATA_PATH),
            split_path=str(SPLIT_STRUCTURED),
            variant=variant,
            output_path=str(path),
            seed=seed,
            epochs=100,
            batch_size=64,
            lr=0.001
        )
        print("  AUC (F vs R): {:.4f}".format(auc))

    print("\nCo-training attacker training complete.")
    print("Saved to {}/".format(COTRAIN_ATTACKER_DIR))


def run_extragradient():
    """Step 2: Run extra-gradient unlearning on structured forget set."""
    output_dir = OUTPUT_BASE / 'extragradient_structured'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 2: Extra-Gradient Unlearning (Structured)")
    print("=" * 60)

    attacker_paths = [
        str(COTRAIN_ATTACKER_DIR / 'attacker_{}_seed{}.pt'.format(v, s))
        for v, s in zip(VARIANTS, SEEDS)
    ]

    for p in attacker_paths:
        if not Path(p).exists():
            print("ERROR: Attacker not found: {}".format(p))
            print("Run --step attackers first")
            return False

    cmd = [
        sys.executable, str(SRC_DIR / 'train_unlearn_extragradient.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(SPLIT_STRUCTURED),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--attacker_paths', *attacker_paths,
        '--lambda_retain', '10',
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

    print("Lambda: 10, Epochs: 50, LR_VAE: 1e-4, LR_Critic: 1e-5")
    print("Output: {}".format(output_dir))
    print()

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def run_fisher(forget_type):
    """Step 3/4: Run Fisher unlearning."""
    split_path = SPLIT_STRUCTURED if forget_type == 'structured' else SPLIT_SCATTERED
    output_dir = OUTPUT_BASE / 'fisher_{}'.format(forget_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n{}".format('=' * 60))
    print("STEP: Fisher Unlearning ({})".format(forget_type))
    print("{}".format('=' * 60))

    cmd = [
        sys.executable, str(SRC_DIR / 'train_fisher_unlearn.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(split_path),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--scrub_lr', '0.0001',
        '--scrub_steps', '100',
        '--finetune_epochs', '10',
        '--finetune_lr', '0.0001',
        '--fisher_damping', '0.1',
        '--batch_size', '256',
        '--output_dir', str(output_dir),
        '--seed', '42',
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    print("Scrub LR: 1e-4, Steps: 100, Finetune: 10 epochs")
    print("Output: {}".format(output_dir))
    print()

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def evaluate_all():
    """Step 5: Post-hoc evaluation of all unlearned models with fresh attackers."""
    import scanpy as sc

    print("\n" + "=" * 60)
    print("STEP: Post-Hoc Evaluation (fresh attacker, canonical NB03)")
    print("=" * 60)

    # Load data and baseline
    adata = sc.read_h5ad(DATA_PATH)
    baseline_model, config = load_vae_model(BASELINE_CHECKPOINT)

    experiments = [
        {
            'name': 'extra-gradient (structured)',
            'model_path': OUTPUT_BASE / 'extragradient_structured' / 'best_model.pt',
            'split_path': SPLIT_STRUCTURED,
            'matched_neg': MATCHED_NEG_STRUCTURED,
        },
        {
            'name': 'fisher (structured)',
            'model_path': OUTPUT_BASE / 'fisher_structured' / 'unlearned_model.pt',
            'split_path': SPLIT_STRUCTURED,
            'matched_neg': MATCHED_NEG_STRUCTURED,
        },
        {
            'name': 'fisher (scattered)',
            'model_path': OUTPUT_BASE / 'fisher_scattered' / 'unlearned_model.pt',
            'split_path': SPLIT_SCATTERED,
            'matched_neg': MATCHED_NEG_SCATTERED,
        },
    ]

    # Group by forget type for shared attackers
    results = {}
    attacker_cache = {}

    for exp in experiments:
        if not Path(exp['model_path']).exists():
            print("\n  {}: model not found, skipping".format(exp['name']))
            continue

        if not Path(exp['matched_neg']).exists():
            print("\n  {}: matched negatives not found, skipping".format(
                exp['name']))
            continue

        # Train or reuse attacker for this split/matched combo
        cache_key = str(exp['split_path']) + str(exp['matched_neg'])
        if cache_key not in attacker_cache:
            with open(exp['split_path']) as f:
                split = json.load(f)
            with open(exp['matched_neg']) as f:
                matched = json.load(f)

            forget_idx = split['forget_indices']
            retain_idx = split['retain_indices']
            matched_idx = matched['matched_indices']

            attacker = train_fresh_attacker(
                baseline_model, adata, forget_idx, matched_idx, retain_idx,
                label=exp['name']
            )
            attacker_cache[cache_key] = (attacker, forget_idx, retain_idx,
                                         matched_idx)
        else:
            attacker, forget_idx, retain_idx, matched_idx = attacker_cache[
                cache_key]

        print("\n  Evaluating {}...".format(exp['name']))
        try:
            model, _ = load_vae_model(exp['model_path'], config=config)
            metrics = evaluate_with_attacker(
                model, attacker, adata, forget_idx, matched_idx, retain_idx
            )
            results[exp['name']] = metrics
            print("    AUC: {:.4f}".format(metrics['auc']))
            print("    TPR@1%FPR: {:.4f}".format(metrics['tpr_at_1pct_fpr']))
        except Exception as e:
            print("    FAILED: {}".format(e))
            import traceback
            traceback.print_exc()

    # Load baseline results for comparison
    baseline_results = {}
    for name, path in [('structured', 'baseline_results.json'),
                       ('scattered', 'scattered_baseline_results.json')]:
        p = OUTPUT_BASE / path
        if p.exists():
            with open(p) as f:
                baseline_results[name] = json.load(f)

    # Summary
    print("\n" + "=" * 60)
    print("CROSS-DATASET COMPARISON")
    print("=" * 60)

    print("\n{:<30} {:<10} {:<20}".format('Method', 'AUC', 'PBMC Equivalent'))
    print("-" * 60)

    if 'structured' in baseline_results:
        b = baseline_results['structured']
        print("{:<30} {:.4f}     PBMC: 0.769".format(
            'Baseline (structured)', b['baseline_auc']))

    if 'scattered' in baseline_results:
        b = baseline_results['scattered']
        print("{:<30} {:.4f}     PBMC: 0.769".format(
            'Baseline (scattered)', b['baseline_auc']))

    for name, metrics in results.items():
        pbmc_ref = ""
        if "extra-gradient" in name and "structured" in name:
            pbmc_ref = "PBMC: 0.48 (target)"
        elif "fisher" in name and "structured" in name:
            pbmc_ref = "PBMC: 0.81 (fails)"
        elif "fisher" in name and "scattered" in name:
            pbmc_ref = "PBMC: 0.50 (works)"
        print("{:<30} {:.4f}     {}".format(name, metrics['auc'], pbmc_ref))

    # Save combined results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'tabula_muris',
        'attacker_methodology': 'Fresh per-type attacker on TM baseline '
                                'F vs matched (canonical NB03)',
        'baseline': baseline_results,
        'unlearning': results,
        'pbmc_reference': {
            'baseline_auc': 0.769,
            'retrain_floor': 0.481,
            'extragradient_structured': 0.48,
            'fisher_structured': 0.81,
            'fisher_scattered': 0.50,
        }
    }

    out_path = OUTPUT_BASE / 'unlearning_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(
        description='Tabula Muris unlearning experiments')
    parser.add_argument('--step', choices=[
        'attackers', 'extragradient', 'fisher-structured',
        'fisher-scattered', 'evaluate', 'all'
    ], default='all', help='Which step to run')
    args = parser.parse_args()

    print("=" * 60)
    print("TABULA MURIS UNLEARNING EXPERIMENTS")
    print("=" * 60)
    print("Data: {}".format(DATA_PATH))
    print("Baseline: {}".format(BASELINE_CHECKPOINT))
    print("Step: {}".format(args.step))
    print()

    if args.step in ('attackers', 'all'):
        train_attackers()

    if args.step in ('extragradient', 'all'):
        run_extragradient()

    if args.step in ('fisher-structured', 'all'):
        run_fisher('structured')

    if args.step in ('fisher-scattered', 'all'):
        run_fisher('scattered')

    if args.step in ('evaluate', 'all'):
        evaluate_all()


if __name__ == '__main__':
    main()
