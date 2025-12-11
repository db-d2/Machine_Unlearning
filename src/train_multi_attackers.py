"""Train multiple attackers for multi-critic unlearning.

Pre-trains an ensemble of attackers with different feature variants
for use in privacy evaluation and adversarial unlearning.
"""

import sys
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import numpy as np
from sklearn.metrics import roc_auc_score

from vae import VAE
from attacker import MLPAttacker, extract_vae_features, build_attack_features


def train_single_attacker(
    vae_path,
    data_path,
    split_path,
    variant,
    output_path,
    seed=42,
    hidden_dims=[256, 256],
    epochs=100,
    batch_size=64,
    lr=0.001
):
    """Train a single attacker with specified feature variant.

    Args:
        vae_path: Path to trained VAE checkpoint
        data_path: Path to processed data (adata_processed.h5ad)
        split_path: Path to split JSON (forget/retain/unseen indices)
        variant: Feature variant ('v1', 'v2', or 'v3')
        output_path: Where to save trained attacker
        seed: Random seed
        hidden_dims: Attacker architecture
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"Training Attacker with Variant: {variant}")
    print(f"{'='*70}")

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(data_path)
    with open(split_path, 'r') as f:
        splits = json.load(f)

    forget_idx = np.array(splits['forget_indices'])
    retain_idx = np.array(splits['retain_indices'])

    print(f"  Forget: {len(forget_idx)} cells")
    print(f"  Retain: {len(retain_idx)} cells")

    # Load VAE
    print(f"\nLoading VAE from {vae_path}...")
    vae_ckpt = torch.load(vae_path, map_location=device)
    vae_config = vae_ckpt['config']

    # Handle missing input_dim in older checkpoints
    input_dim = vae_config.get('input_dim', adata.n_vars)

    vae = VAE(
        input_dim=input_dim,
        hidden_dims=vae_config['hidden_dims'],
        latent_dim=vae_config['latent_dim'],
        likelihood=vae_config['likelihood'],
        dropout=vae_config.get('dropout', 0.0),
        use_layer_norm=vae_config.get('use_layer_norm', False)
    ).to(device)

    if 'vae_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['vae_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt['model_state_dict'])

    vae.eval()
    print(f"  VAE loaded: z={vae_config['latent_dim']}, {vae_config['likelihood']}")

    # Extract features
    print(f"\nExtracting features (variant={variant})...")

    def get_features(indices):
        X = torch.FloatTensor(
            adata.X[indices].toarray() if hasattr(adata.X[indices], 'toarray')
            else adata.X[indices]
        )
        lib = torch.FloatTensor(X.sum(dim=1, keepdim=True))

        with torch.no_grad():
            vae_feats = extract_vae_features(vae, X.to(device), lib.to(device), device)
            attack_feats = build_attack_features(vae_feats, variant=variant)

        return attack_feats.cpu().numpy()

    forget_feats = get_features(forget_idx)
    retain_feats = get_features(retain_idx)

    feature_dim = forget_feats.shape[1]
    print(f"  Feature dimension: {feature_dim}")

    # Initialize attacker
    print(f"\nInitializing attacker...")
    attacker = MLPAttacker(
        feature_dim,
        hidden_dims,
        dropout=0.3,
        use_spectral_norm=True
    ).to(device)

    print(f"  Architecture: {feature_dim} -> {hidden_dims} -> 1")
    print(f"  Dropout: 0.3, Spectral norm: True")

    optimizer = torch.optim.Adam(attacker.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    train_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        attacker.train()

        # Sample batches
        f_idx = np.random.choice(len(forget_feats), min(batch_size, len(forget_feats)), replace=False)
        r_idx = np.random.choice(len(retain_feats), min(batch_size, len(retain_feats)), replace=False)

        f_batch = torch.FloatTensor(forget_feats[f_idx]).to(device)
        r_batch = torch.FloatTensor(retain_feats[r_idx]).to(device)

        optimizer.zero_grad()
        logits_f = attacker(f_batch)
        logits_r = attacker(r_batch)

        loss_f = F.binary_cross_entropy_with_logits(logits_f, torch.ones_like(logits_f))
        loss_r = F.binary_cross_entropy_with_logits(logits_r, torch.zeros_like(logits_r))
        loss = (loss_f + loss_r) / 2

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.4f} | Best: {best_loss:.4f}")

    # Final evaluation
    print("\nEvaluating...")
    attacker.eval()
    with torch.no_grad():
        forget_preds = torch.sigmoid(attacker(torch.FloatTensor(forget_feats).to(device))).cpu().numpy()
        retain_preds = torch.sigmoid(attacker(torch.FloatTensor(retain_feats).to(device))).cpu().numpy()

    y_true = np.concatenate([np.ones(len(forget_preds)), np.zeros(len(retain_preds))])
    y_score = np.concatenate([forget_preds.flatten(), retain_preds.flatten()])
    auc = roc_auc_score(y_true, y_score)

    print(f"  Training AUC (F vs R): {auc:.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': attacker.state_dict(),
        'config': {
            'feature_dim': feature_dim,
            'hidden_dims': hidden_dims,
            'dropout': 0.3,
            'use_spectral_norm': True,
            'variant': variant
        },
        'training': {
            'epochs': epochs,
            'final_loss': train_losses[-1],
            'best_loss': best_loss,
            'auc': auc,
            'seed': seed
        }
    }, output_path)

    print(f"\nSaved to: {output_path}")
    print(f"{'='*70}\n")

    return auc


def main():
    parser = argparse.ArgumentParser(description='Train multiple attackers for multi-critic unlearning')
    parser.add_argument('--vae_path', type=str, default='outputs/p1/baseline_v2/best_model.pt',
                        help='Path to baseline VAE checkpoint')
    parser.add_argument('--data_path', type=str, default='data/adata_processed.h5ad',
                        help='Path to processed data')
    parser.add_argument('--split_path', type=str, default='outputs/p1/split_structured.json',
                        help='Path to split JSON')
    parser.add_argument('--output_dir', type=str, default='outputs/p2/attackers',
                        help='Output directory for trained attackers')
    parser.add_argument('--variants', type=str, nargs='+', default=['v1', 'v2', 'v3'],
                        help='Feature variants to train (v1, v2, v3)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds (one per variant)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per attacker')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("MULTI-ATTACKER TRAINING")
    print("="*70)
    print(f"\nVariants: {args.variants}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs} per attacker")

    # Train each attacker
    results = {}
    for i, (variant, seed) in enumerate(zip(args.variants, args.seeds)):
        attacker_name = f"attacker_{variant}_seed{seed}"
        output_path = os.path.join(args.output_dir, f"{attacker_name}.pt")

        auc = train_single_attacker(
            vae_path=args.vae_path,
            data_path=args.data_path,
            split_path=args.split_path,
            variant=variant,
            output_path=output_path,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        results[attacker_name] = {
            'variant': variant,
            'seed': seed,
            'auc': float(auc),
            'path': output_path
        }

    # Save summary
    summary_path = os.path.join(args.output_dir, 'attackers_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'method_id': 'multi_attacker_training',
            'attackers': results,
            'config': {
                'vae_path': args.vae_path,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr
            }
        }, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, info in results.items():
        print(f"{name:30s} | AUC: {info['auc']:.4f}")
    print(f"\nSummary saved to: {summary_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
