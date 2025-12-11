#!/usr/bin/env python3
"""
Multi-seed ablation study for confidence intervals.

Runs Fisher unlearning with different random seeds and computes
mean, std, and 95% confidence intervals for statistical significance.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os
import numpy as np
from scipy import stats

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

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
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline_v2' / 'best_model.pt'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'multiseed'
SPLIT_DIR = BASE_DIR / 'outputs' / 'p1'
ATTACKER_DIR = BASE_DIR / 'outputs' / 'p2' / 'attackers'

# Matched negatives paths
MATCHED_NEGATIVES = {
    'structured': BASE_DIR / 'outputs' / 'p1.5' / 's1_matched_negatives.json',
    'scattered': BASE_DIR / 'outputs' / 'p1.5' / 'scattered_matched_negatives.json',
}


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


def evaluate_model(model_path, split_path, matched_neg_path):
    """Evaluate unlearned model."""
    import torch
    import scanpy as sc
    from vae import VAE
    from attacker import MLPAttacker, extract_vae_features, build_attack_features
    from sklearn.metrics import roc_auc_score, roc_curve
    from learning_curve import get_feature_dim

    device = 'cpu'

    # Load data
    adata = sc.read_h5ad(DATA_PATH)
    with open(split_path) as f:
        split = json.load(f)
    with open(matched_neg_path) as f:
        matched = json.load(f)

    forget_indices = split['forget_indices']
    matched_indices = np.array(matched['matched_indices'])

    # Load model
    baseline_ckpt = torch.load(BASELINE_CHECKPOINT, map_location=device)
    config = baseline_ckpt['config']

    unlearn_ckpt = torch.load(model_path, map_location=device)
    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config['likelihood'],
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    ).to(device)
    model.load_state_dict(unlearn_ckpt['model_state_dict'])
    model.eval()

    # Load attacker
    attacker_path = ATTACKER_DIR / 'attacker_v1_seed42.pt'
    attacker_ckpt = torch.load(attacker_path, map_location=device)
    feature_dim = get_feature_dim(config['latent_dim'], 'v1')
    attacker_config = attacker_ckpt.get('config', {})

    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=attacker_config.get('hidden_dims', [256, 256]),
        dropout=attacker_config.get('dropout', 0.3),
        use_spectral_norm=attacker_config.get('use_spectral_norm', True)
    ).to(device)
    attacker.load_state_dict(attacker_ckpt['model_state_dict'])
    attacker.eval()

    # Extract features
    def get_features(indices):
        X = adata.X[indices]
        if hasattr(X, 'toarray'):
            X = X.toarray()
        x_tensor = torch.FloatTensor(X).to(device)
        lib = x_tensor.sum(dim=1, keepdim=True)
        with torch.no_grad():
            features = extract_vae_features(model, x_tensor, lib, device=device)
            attack_features = build_attack_features(features, variant='v1')
        return attack_features

    forget_feats = get_features(forget_indices)
    unseen_feats = get_features(matched_indices)

    with torch.no_grad():
        forget_probs = torch.sigmoid(attacker(forget_feats)).cpu().numpy()
        unseen_probs = torch.sigmoid(attacker(unseen_feats)).cpu().numpy()

    y_true = np.concatenate([np.ones(len(forget_probs)), np.zeros(len(unseen_probs))])
    y_score = np.concatenate([forget_probs.flatten(), unseen_probs.flatten()])
    auc = roc_auc_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_at_1pct = tpr[np.searchsorted(fpr, 0.01)] if np.any(fpr <= 0.01) else 0.0

    return {
        'auc': float(auc),
        'tpr_at_1pct_fpr': float(tpr_at_1pct),
    }


def compute_statistics(results):
    """Compute mean, std, and 95% CI."""
    aucs = [r['auc'] for r in results]
    n = len(aucs)
    mean = np.mean(aucs)
    std = np.std(aucs, ddof=1)  # Sample std

    # 95% CI using t-distribution
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--forget-type', choices=['structured', 'scattered', 'both'],
                       default='both', help='Which forget type to run')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    forget_types = FORGET_TYPES if args.forget_type == 'both' else [args.forget_type]
    all_results = {}

    for forget_type in forget_types:
        print(f"\n{'='*70}")
        print(f"MULTI-SEED EXPERIMENT: {forget_type.upper()}")
        print(f"{'='*70}")

        split_path = SPLIT_DIR / f'split_{forget_type}.json'
        matched_path = MATCHED_NEGATIVES[forget_type]

        results = []

        for seed in SEEDS:
            run_name = f'{forget_type}_seed{seed}'
            output_dir = OUTPUT_BASE / forget_type / run_name

            if args.dry_run:
                print(f"Would run: {run_name}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Running seed={seed} ---")

            start_time = time.time()
            success = run_fisher_unlearn(split_path, output_dir, seed)
            elapsed = time.time() - start_time

            if success:
                model_path = output_dir / 'unlearned_model.pt'
                try:
                    eval_result = evaluate_model(model_path, split_path, matched_path)
                    print(f"  AUC={eval_result['auc']:.4f} (time={elapsed:.1f}s)")

                    results.append({
                        'seed': seed,
                        'time_seconds': elapsed,
                        **eval_result
                    })

                    with open(output_dir / 'eval_v1.json', 'w') as f:
                        json.dump(eval_result, f, indent=2)

                except Exception as e:
                    print(f"  Evaluation failed: {e}")
            else:
                print(f"  Training failed")

        if not args.dry_run and results:
            # Compute statistics
            stats_result = compute_statistics(results)

            print(f"\n{'='*60}")
            print(f"STATISTICS for {forget_type.upper()}:")
            print(f"  Mean AUC: {stats_result['mean']:.4f}")
            print(f"  Std: {stats_result['std']:.4f}")
            print(f"  95% CI: [{stats_result['ci_95_lower']:.4f}, {stats_result['ci_95_upper']:.4f}]")
            print(f"{'='*60}")

            all_results[forget_type] = {
                'runs': results,
                'statistics': stats_result,
            }

            # Save
            with open(OUTPUT_BASE / forget_type / 'results.json', 'w') as f:
                json.dump(all_results[forget_type], f, indent=2)

    # Save combined results
    if not args.dry_run and all_results:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'seeds': SEEDS,
            'defaults': DEFAULTS,
            'results': all_results,
            'retrain_floor_auc': 0.864,
            'target_band': [0.834, 0.894],
        }

        with open(OUTPUT_BASE / 'multiseed_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n\nFINAL SUMMARY:")
        print(f"{'='*70}")
        for ft, data in all_results.items():
            s = data['statistics']
            print(f"{ft}: AUC = {s['mean']:.4f} ± {s['std']:.4f} (95% CI: [{s['ci_95_lower']:.4f}, {s['ci_95_upper']:.4f}])")

        print(f"\nResults saved to {OUTPUT_BASE / 'multiseed_summary.json'}")


if __name__ == '__main__':
    main()
