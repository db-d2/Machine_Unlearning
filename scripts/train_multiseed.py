#!/usr/bin/env python3
"""Multi-seed training for publication statistical rigor.

Trains three unlearning methods across multiple seeds:
- Extra-gradient lambda=10: 10 seeds
- Retain-FT: 5 seeds
- Gradient ascent: 5 seeds

Calls src/ training scripts via subprocess. Skips existing checkpoints.

Usage:
    PYTHONPATH=src python scripts/train_multiseed.py
    PYTHONPATH=src python scripts/train_multiseed.py --methods extragradient
    PYTHONPATH=src python scripts/train_multiseed.py --methods retain_finetune gradient_ascent
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / 'src'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'p4' / 'multiseed'

# Paths
DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
SPLIT_PATH = BASE_DIR / 'outputs' / 'p1' / 'split_structured.json'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
ATTACKER_DIR = BASE_DIR / 'outputs' / 'p2' / 'attackers'

# Seeds
EG_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
SIMPLE_SEEDS = [42, 123, 456, 789, 1011]

ATTACKER_PATHS = [
    str(ATTACKER_DIR / 'attacker_v1_seed42.pt'),
    str(ATTACKER_DIR / 'attacker_v2_seed43.pt'),
    str(ATTACKER_DIR / 'attacker_v3_seed44.pt'),
]


def run_command(cmd, description):
    """Run a subprocess command."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC_DIR)

    print(f"  Running: {description}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"  FAILED (returncode={result.returncode})")
        return False
    return True


def train_extragradient(seeds):
    """Train extra-gradient lambda=10 across seeds."""
    print("\n" + "=" * 60)
    print("EXTRA-GRADIENT lambda=10")
    print("=" * 60)

    for seed in seeds:
        output_dir = OUTPUT_BASE / 'extragradient' / f'seed{seed}'
        if (output_dir / 'best_model.pt').exists():
            print(f"  seed={seed}: exists, skipping")
            continue

        # Verify attacker files exist
        for p in ATTACKER_PATHS:
            if not Path(p).exists():
                print(f"  ERROR: Attacker not found: {p}")
                return

        cmd = [
            sys.executable, str(SRC_DIR / 'train_unlearn_extragradient.py'),
            '--data_path', str(DATA_PATH),
            '--split_path', str(SPLIT_PATH),
            '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
            '--attacker_paths', *ATTACKER_PATHS,
            '--lambda_retain', '10',
            '--epochs', '50',
            '--lr_vae', '0.0001',
            '--lr_critic', '0.00001',
            '--critic_steps', '2',
            '--abort_threshold', '3',
            '--batch_size', '256',
            '--output_dir', str(output_dir),
            '--seed', str(seed),
        ]
        run_command(cmd, f"EG seed={seed}")


def train_retain_finetune(seeds):
    """Train retain-only fine-tuning across seeds."""
    print("\n" + "=" * 60)
    print("RETAIN-ONLY FINE-TUNING")
    print("=" * 60)

    for seed in seeds:
        output_dir = OUTPUT_BASE / 'retain_finetune' / f'seed{seed}'
        if (output_dir / 'best_model.pt').exists():
            print(f"  seed={seed}: exists, skipping")
            continue

        cmd = [
            sys.executable, str(SRC_DIR / 'train_retain_finetune.py'),
            '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
            '--data_path', str(DATA_PATH),
            '--split_path', str(SPLIT_PATH),
            '--output_dir', str(output_dir),
            '--lr', '0.0001',
            '--epochs', '50',
            '--patience', '10',
            '--batch_size', '256',
            '--seed', str(seed),
        ]
        run_command(cmd, f"Retain-FT seed={seed}")


def train_gradient_ascent(seeds):
    """Train gradient ascent across seeds."""
    print("\n" + "=" * 60)
    print("GRADIENT ASCENT")
    print("=" * 60)

    for seed in seeds:
        output_dir = OUTPUT_BASE / 'gradient_ascent' / f'seed{seed}'
        if (output_dir / 'best_model.pt').exists():
            print(f"  seed={seed}: exists, skipping")
            continue

        cmd = [
            sys.executable, str(SRC_DIR / 'train_gradient_ascent.py'),
            '--baseline_checkpoint', str(BASELINE_CHECKPOINT),
            '--data_path', str(DATA_PATH),
            '--split_path', str(SPLIT_PATH),
            '--output_dir', str(output_dir),
            '--ascent_steps', '10',
            '--ascent_lr', '0.00001',
            '--max_grad_norm', '1.0',
            '--finetune_epochs', '30',
            '--finetune_lr', '0.0001',
            '--patience', '10',
            '--batch_size', '256',
            '--seed', str(seed),
        ]
        run_command(cmd, f"Grad-ascent seed={seed}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed training for publication')
    parser.add_argument('--methods', nargs='+',
                        default=['extragradient', 'retain_finetune',
                                 'gradient_ascent'],
                        choices=['extragradient', 'retain_finetune',
                                 'gradient_ascent'],
                        help='Methods to train')
    args = parser.parse_args()

    # Verify prerequisites
    for path, name in [
        (DATA_PATH, 'Data'),
        (SPLIT_PATH, 'Split'),
        (BASELINE_CHECKPOINT, 'Baseline checkpoint'),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    print("=" * 60)
    print("MULTI-SEED TRAINING")
    print("=" * 60)
    print(f"Methods: {args.methods}")
    print(f"Output: {OUTPUT_BASE}")

    if 'extragradient' in args.methods:
        train_extragradient(EG_SEEDS)

    if 'retain_finetune' in args.methods:
        train_retain_finetune(SIMPLE_SEEDS)

    if 'gradient_ascent' in args.methods:
        train_gradient_ascent(SIMPLE_SEEDS)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for method, seeds in [('extragradient', EG_SEEDS),
                          ('retain_finetune', SIMPLE_SEEDS),
                          ('gradient_ascent', SIMPLE_SEEDS)]:
        if method not in args.methods:
            continue
        found = sum(1 for s in seeds
                    if (OUTPUT_BASE / method / f'seed{s}' /
                        'best_model.pt').exists())
        print(f"  {method}: {found}/{len(seeds)} checkpoints")


if __name__ == '__main__':
    main()
