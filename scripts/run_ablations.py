#!/usr/bin/env python3
"""
Fisher Unlearning Ablation Study

Ablation dimensions:
1. Forget type: structured (rare cluster, 30 cells) vs scattered (random, 35 cells)
2. Scrub steps: 50, 100, 200
3. Fisher damping: 0.01, 0.1, 1.0

Uses existing evaluation infrastructure from the codebase.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

# Ablation configurations
ABLATION_CONFIGS = {
    # Ablation 1: Forget type (using default hyperparams)
    'forget_type': [
        {'name': 'structured', 'split': 'split_structured.json'},
        {'name': 'scattered', 'split': 'split_scattered.json'},
    ],

    # Ablation 2: Scrub steps
    'scrub_steps': [50, 100, 200],

    # Ablation 3: Fisher damping
    'fisher_damping': [0.01, 0.1, 1.0],
}

# Default hyperparameters
DEFAULTS = {
    'scrub_lr': 0.0001,
    'scrub_steps': 100,
    'finetune_epochs': 10,
    'finetune_lr': 0.0001,
    'fisher_damping': 0.1,
    'batch_size': 256,
    'seed': 42,
}

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline_v2' / 'best_model.pt'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'ablations'
SPLIT_DIR = BASE_DIR / 'outputs' / 'p1'
ATTACKER_DIR = BASE_DIR / 'outputs' / 'p2' / 'attackers'
MATCHED_NEG_PATH = BASE_DIR / 'outputs' / 'p1.5' / 's1_matched_negatives.json'


def run_fisher_unlearn(config, output_dir):
    """Run Fisher unlearning with given config."""
    cmd = [
        sys.executable, str(BASE_DIR / 'src' / 'train_fisher_unlearn.py'),
        '--data_path', str(DATA_PATH),
        '--split_path', str(config['split_path']),
        '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
        '--output_dir', str(output_dir),
        '--scrub_lr', str(config['scrub_lr']),
        '--scrub_steps', str(config['scrub_steps']),
        '--finetune_epochs', str(config['finetune_epochs']),
        '--finetune_lr', str(config['finetune_lr']),
        '--fisher_damping', str(config['fisher_damping']),
        '--batch_size', str(config['batch_size']),
        '--seed', str(config['seed']),
    ]

    print(f"Running: {' '.join(cmd)}")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Error: {result.stderr[-1000:]}")
        return False

    # Print last part of output
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return True


def evaluate_model(model_path, split_path, attacker_variant='v1'):
    """Evaluate unlearned model using MIA attacker."""
    import torch
    import scanpy as sc
    from vae import VAE
    from attacker import MLPAttacker, extract_vae_features, build_attack_features
    from sklearn.metrics import roc_auc_score, roc_curve
    from learning_curve import get_feature_dim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load unlearned model
    unlearn_ckpt = torch.load(model_path, map_location=device)

    # Load baseline to get config
    baseline_ckpt = torch.load(BASELINE_CHECKPOINT, map_location=device)
    config = baseline_ckpt['config']

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

    # Load data
    adata = sc.read_h5ad(DATA_PATH)
    with open(split_path) as f:
        split = json.load(f)

    # Load matched negatives
    with open(MATCHED_NEG_PATH) as f:
        matched = json.load(f)
    matched_indices = np.array(matched['matched_indices'])

    forget_indices = split['forget_indices']

    # Load attacker
    attacker_path = ATTACKER_DIR / f'attacker_{attacker_variant}_seed42.pt'
    attacker_ckpt = torch.load(attacker_path, map_location=device)

    feature_dim = get_feature_dim(config['latent_dim'], attacker_variant)

    # Get config from checkpoint
    attacker_config = attacker_ckpt.get('config', {})
    hidden_dims = attacker_config.get('hidden_dims', [256, 256])
    dropout = attacker_config.get('dropout', 0.3)
    use_spectral_norm = attacker_config.get('use_spectral_norm', True)  # Default True for trained attackers

    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_spectral_norm=use_spectral_norm
    ).to(device)
    attacker.load_state_dict(attacker_ckpt['model_state_dict'])
    attacker.eval()

    def get_features(indices):
        """Extract features for given indices."""
        X = adata.X[indices]
        if hasattr(X, 'toarray'):
            X = X.toarray()
        x_tensor = torch.FloatTensor(X).to(device)
        lib = x_tensor.sum(dim=1, keepdim=True)

        with torch.no_grad():
            features = extract_vae_features(model, x_tensor, lib, device=device)
            attack_features = build_attack_features(features, variant=attacker_variant)
        return attack_features

    # Get features
    forget_feats = get_features(forget_indices)
    unseen_feats = get_features(matched_indices)

    # Get predictions
    with torch.no_grad():
        forget_probs = torch.sigmoid(attacker(forget_feats.to(device))).cpu().numpy()
        unseen_probs = torch.sigmoid(attacker(unseen_feats.to(device))).cpu().numpy()

    # Compute AUC
    y_true = np.concatenate([np.ones(len(forget_probs)), np.zeros(len(unseen_probs))])
    y_score = np.concatenate([forget_probs.flatten(), unseen_probs.flatten()])
    auc = roc_auc_score(y_true, y_score)

    # Compute TPR @ low FPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    tpr_at_1pct = tpr[np.searchsorted(fpr, 0.01)] if np.any(fpr <= 0.01) else 0.0
    tpr_at_01pct = tpr[np.searchsorted(fpr, 0.001)] if np.any(fpr <= 0.001) else 0.0

    result = {
        'auc': float(auc),
        'tpr_at_1pct_fpr': float(tpr_at_1pct),
        'tpr_at_01pct_fpr': float(tpr_at_01pct),
        'attacker_variant': attacker_variant,
        'n_forget': len(forget_indices),
        'n_unseen': len(matched_indices),
    }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', choices=['all', 'forget_type', 'scrub_steps', 'damping'],
                       default='all', help='Which ablation to run')
    parser.add_argument('--dry-run', action='store_true', help='Print configs without running')
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Ablation 1: Forget type
    if args.ablation in ['all', 'forget_type']:
        print("\n" + "="*70)
        print("ABLATION 1: Forget Type (structured vs scattered)")
        print("="*70)

        results = []
        for forget_cfg in ABLATION_CONFIGS['forget_type']:
            config = DEFAULTS.copy()
            config['split_path'] = SPLIT_DIR / forget_cfg['split']

            run_name = f"forget_{forget_cfg['name']}"
            output_dir = OUTPUT_BASE / 'forget_type' / run_name

            if args.dry_run:
                print(f"Would run: {run_name}")
                print(f"  Config: {config}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running: {run_name}")
            print(f"{'='*60}")

            start_time = time.time()
            success = run_fisher_unlearn(config, output_dir)
            elapsed = time.time() - start_time

            if success:
                # Evaluate with primary attacker
                model_path = output_dir / 'unlearned_model.pt'
                try:
                    eval_result = evaluate_model(
                        model_path,
                        config['split_path'],
                        attacker_variant='v1'
                    )
                    print(f"  AUC={eval_result['auc']:.4f}")

                    results.append({
                        'forget_type': forget_cfg['name'],
                        'time_seconds': elapsed,
                        'auc': eval_result['auc'],
                        'tpr_at_1pct_fpr': eval_result['tpr_at_1pct_fpr'],
                        'config': {k: str(v) for k, v in config.items()},
                    })

                    # Save individual result
                    with open(output_dir / 'eval_v1.json', 'w') as f:
                        json.dump(eval_result, f, indent=2)

                except Exception as e:
                    print(f"  Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

        if not args.dry_run and results:
            all_results['forget_type'] = results
            (OUTPUT_BASE / 'forget_type').mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_BASE / 'forget_type' / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

    # Ablation 2: Scrub steps
    if args.ablation in ['all', 'scrub_steps']:
        print("\n" + "="*70)
        print("ABLATION 2: Scrub Steps")
        print("="*70)

        results = []
        for scrub_steps in ABLATION_CONFIGS['scrub_steps']:
            config = DEFAULTS.copy()
            config['scrub_steps'] = scrub_steps
            config['split_path'] = SPLIT_DIR / 'split_structured.json'

            run_name = f"scrub_{scrub_steps}"
            output_dir = OUTPUT_BASE / 'scrub_steps' / run_name

            if args.dry_run:
                print(f"Would run: {run_name}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running: {run_name}")
            print(f"{'='*60}")

            start_time = time.time()
            success = run_fisher_unlearn(config, output_dir)
            elapsed = time.time() - start_time

            if success:
                model_path = output_dir / 'unlearned_model.pt'
                try:
                    eval_result = evaluate_model(model_path, config['split_path'])
                    print(f"  AUC={eval_result['auc']:.4f}")

                    results.append({
                        'scrub_steps': scrub_steps,
                        'time_seconds': elapsed,
                        'auc': eval_result['auc'],
                        'tpr_at_1pct_fpr': eval_result['tpr_at_1pct_fpr'],
                    })
                except Exception as e:
                    print(f"  Evaluation failed: {e}")

        if not args.dry_run and results:
            all_results['scrub_steps'] = results
            with open(OUTPUT_BASE / 'scrub_steps' / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

    # Ablation 3: Fisher damping
    if args.ablation in ['all', 'damping']:
        print("\n" + "="*70)
        print("ABLATION 3: Fisher Damping")
        print("="*70)

        results = []
        for damping in ABLATION_CONFIGS['fisher_damping']:
            config = DEFAULTS.copy()
            config['fisher_damping'] = damping
            config['split_path'] = SPLIT_DIR / 'split_structured.json'

            run_name = f"damping_{damping}"
            output_dir = OUTPUT_BASE / 'damping' / run_name

            if args.dry_run:
                print(f"Would run: {run_name}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running: {run_name}")
            print(f"{'='*60}")

            start_time = time.time()
            success = run_fisher_unlearn(config, output_dir)
            elapsed = time.time() - start_time

            if success:
                model_path = output_dir / 'unlearned_model.pt'
                try:
                    eval_result = evaluate_model(model_path, config['split_path'])
                    print(f"  AUC={eval_result['auc']:.4f}")

                    results.append({
                        'fisher_damping': damping,
                        'time_seconds': elapsed,
                        'auc': eval_result['auc'],
                        'tpr_at_1pct_fpr': eval_result['tpr_at_1pct_fpr'],
                    })
                except Exception as e:
                    print(f"  Evaluation failed: {e}")

        if not args.dry_run and results:
            all_results['damping'] = results
            with open(OUTPUT_BASE / 'damping' / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

    # Save combined results
    if not args.dry_run and all_results:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ablations': all_results,
            'defaults': DEFAULTS,
            'retrain_floor_auc': 0.864,
            'target_band': [0.834, 0.894],
        }
        with open(OUTPUT_BASE / 'ablation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {OUTPUT_BASE / 'ablation_summary.json'}")

        # Print summary table
        print("\n" + "="*70)
        print("ABLATION SUMMARY")
        print("="*70)
        print(f"Retrain floor AUC: 0.864")
        print(f"Target band: [0.834, 0.894]")
        print()

        for ablation_name, results in all_results.items():
            print(f"\n{ablation_name.upper()}:")
            for r in results:
                param_key = [k for k in r.keys() if k not in ['time_seconds', 'auc', 'tpr_at_1pct_fpr', 'config']][0]
                auc = r['auc']
                gap = auc - 0.864
                in_band = 0.834 <= auc <= 0.894
                status = "✓" if in_band else "✗"
                print(f"  {param_key}={r[param_key]}: AUC={auc:.4f} (gap={gap:+.4f}) [{status}]")


if __name__ == '__main__':
    main()
