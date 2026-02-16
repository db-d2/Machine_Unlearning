#!/usr/bin/env python3
"""Selective Synaptic Dampening (SSD) for VAEs.

Multiplicative dampening of parameters proportional to their Fisher importance
for the forget set, relative to the retain set.

Reference:
    Foster et al. (2024). Fast Machine Unlearning Without Retraining Through
    Selective Synaptic Dampening. AAAI 2024.

Usage:
    PYTHONPATH=src python src/train_ssd.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/ssd/seed42 \
        --seed 42
"""

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from copy import deepcopy
import scanpy as sc

from vae import VAE, vae_loss


def load_vae(checkpoint_path, device='cpu'):
    """Load VAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config.get('likelihood', 'nb'),
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def create_dataloader(X, indices, batch_size=256, shuffle=True):
    """Create DataLoader from indices."""
    data = X[indices]
    library_size = data.sum(dim=1, keepdim=True)
    dataset = TensorDataset(data, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_fisher(model, dataloader, device, damping=1e-5):
    """Compute diagonal Fisher information on a dataset.

    Returns dict mapping parameter names to Fisher diagonal tensors.
    """
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    n_samples = 0
    for x, lib_size in dataloader:
        x = x.to(device)
        lib_size = lib_size.to(device)

        model.zero_grad()
        output = model(x, library_size=lib_size)
        loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2

        n_samples += x.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping

    return fisher


def ssd_dampen(model, fisher_forget, fisher_retain, alpha=1.0, threshold=0.0):
    """Apply SSD multiplicative dampening.

    For each parameter:
        importance = F_forget / max(F_retain, eps)
        dampening = alpha * importance / max(importance)
        theta *= (1 - dampening)    where dampening > threshold

    Parameters with high forget-importance relative to retain-importance
    are dampened toward zero. Parameters important to the retain set
    are protected.

    Args:
        model: VAE model (modified in-place)
        fisher_forget: Fisher diagonal for forget set
        fisher_retain: Fisher diagonal for retain set
        alpha: Dampening strength in [0, 1]. Higher = more aggressive.
        threshold: Minimum dampening ratio to apply (filters noise).

    Returns:
        Dict with dampening statistics.
    """
    stats = {'n_dampened': 0, 'n_total': 0, 'mean_dampening': 0.0}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in fisher_forget:
                continue

            f_forget = fisher_forget[name]
            f_retain = fisher_retain[name]

            # Relative importance: how much more important is this param
            # for forget set vs retain set?
            importance = f_forget / (f_retain + 1e-10)

            # Normalize to [0, 1]
            max_imp = importance.max()
            if max_imp > 0:
                importance_norm = importance / max_imp
            else:
                continue

            # Dampening factor
            dampening = alpha * importance_norm

            # Apply threshold
            mask = dampening > threshold
            dampening = dampening * mask.float()

            # Apply: theta *= (1 - dampening)
            param.data.mul_(1.0 - dampening)

            stats['n_dampened'] += mask.sum().item()
            stats['n_total'] += param.numel()
            stats['mean_dampening'] += dampening.sum().item()

    if stats['n_total'] > 0:
        stats['frac_dampened'] = stats['n_dampened'] / stats['n_total']
        stats['mean_dampening'] /= max(stats['n_dampened'], 1)

    return stats


def train_ssd(baseline_checkpoint, data_path, split_path, output_dir,
              alpha=1.0, threshold=0.0, damping=1e-5,
              finetune_epochs=10, finetune_lr=1e-4, patience=10,
              batch_size=256, seed=42):
    """Run SSD unlearning.

    1. Compute Fisher on forget set
    2. Compute Fisher on retain set
    3. Dampen parameters by relative importance
    4. Fine-tune on retain set

    Returns:
        Path to saved checkpoint.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    adata = sc.read_h5ad(data_path)
    X = torch.tensor(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
        dtype=torch.float32
    )

    with open(split_path, 'r') as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']

    print(f"Data: {X.shape}, Forget: {len(forget_idx)}, "
          f"Retain: {len(retain_idx)}, Device: {device}")

    forget_loader = create_dataloader(X, forget_idx, batch_size=min(batch_size, len(forget_idx)))
    retain_loader = create_dataloader(X, retain_idx, batch_size=batch_size)

    # Load baseline
    model, config = load_vae(baseline_checkpoint, device)
    model = model.to(device)

    # Step 1: Fisher on forget set
    print("Computing Fisher on forget set...")
    fisher_forget = compute_fisher(model, forget_loader, device, damping=damping)

    # Step 2: Fisher on retain set
    print("Computing Fisher on retain set...")
    fisher_retain = compute_fisher(model, retain_loader, device, damping=damping)

    # Step 3: SSD dampening
    print(f"Applying SSD dampening (alpha={alpha}, threshold={threshold})...")
    dampen_stats = ssd_dampen(model, fisher_forget, fisher_retain,
                              alpha=alpha, threshold=threshold)
    print(f"  Dampened {dampen_stats.get('frac_dampened', 0):.1%} of parameters")
    print(f"  Mean dampening magnitude: {dampen_stats.get('mean_dampening', 0):.4f}")

    # Step 4: Fine-tune on retain set
    if finetune_epochs > 0:
        val_size = min(1000, len(retain_idx) // 10)
        val_idx = np.random.choice(retain_idx, size=val_size, replace=False)
        val_loader = create_dataloader(X, val_idx, batch_size=batch_size, shuffle=False)

        print(f"Fine-tuning on retain set ({finetune_epochs} epochs, lr={finetune_lr})...")
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
        best_val_loss = float('inf')
        best_state = deepcopy(model.state_dict())
        patience_counter = 0
        ft_history = {'train': [], 'val': []}

        for epoch in range(finetune_epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            for x, lib_size in retain_loader:
                x = x.to(device)
                lib_size = lib_size.to(device)
                optimizer.zero_grad()
                output = model(x, library_size=lib_size)
                loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= n_batches

            model.eval()
            val_loss = 0
            n_batches = 0
            with torch.no_grad():
                for x, lib_size in val_loader:
                    x = x.to(device)
                    lib_size = lib_size.to(device)
                    output = model(x, library_size=lib_size)
                    loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)
                    val_loss += loss.item()
                    n_batches += 1
            val_loss /= n_batches

            ft_history['train'].append(train_loss)
            ft_history['val'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: train={train_loss:.2f}, val={val_loss:.2f}")

        model.load_state_dict(best_state)
    else:
        ft_history = None
        best_val_loss = None

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'method': 'ssd',
        'seed': seed,
        'alpha': alpha,
        'threshold': threshold,
        'dampen_stats': dampen_stats,
        'finetune_epochs': len(ft_history['train']) if ft_history else 0,
        'best_val_loss': best_val_loss,
    }, output_dir / 'best_model.pt')

    history = {
        'dampen_stats': dampen_stats,
        'finetune': ft_history,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Selective Synaptic Dampening for VAEs')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Dampening strength [0, 1]')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Min dampening ratio to apply')
    parser.add_argument('--damping', type=float, default=1e-5,
                        help='Fisher damping for numerical stability')
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_ssd(
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        alpha=args.alpha,
        threshold=args.threshold,
        damping=args.damping,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
