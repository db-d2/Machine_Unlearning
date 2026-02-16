#!/usr/bin/env python3
"""Gradient ascent unlearning baseline for VAEs.

Two-phase approach:
1. Gradient ascent on forget set (maximize loss to damage memorization)
2. Fine-tune on retain set (recover utility)

Logic extracted from notebook 17.

Usage:
    PYTHONPATH=src python src/train_gradient_ascent.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/gradient_ascent/seed42 \
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


def gradient_ascent_phase(model, forget_loader, n_steps, lr, device,
                          max_grad_norm=1.0):
    """Phase 1: Gradient ascent on forget set to damage memorization."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    step = 0
    losses = []

    while step < n_steps:
        for x, lib_size in forget_loader:
            if step >= n_steps:
                break

            x = x.to(device)
            lib_size = lib_size.to(device)

            optimizer.zero_grad()
            output = model(x, library_size=lib_size)
            loss, recon, kl = vae_loss(x, output, likelihood='nb', beta=1.0)

            if torch.isnan(loss):
                print(f"  NaN loss at step {step}, stopping ascent")
                return losses if losses else [float('nan')]

            # ASCENT: negate loss to maximize it
            (-loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            losses.append(loss.item())
            step += 1

    return losses


def finetune_phase(model, retain_loader, val_loader, n_epochs, lr, patience,
                   device):
    """Phase 2: Fine-tune on retain set to recover utility."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = deepcopy(model.state_dict())
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for x, lib_size in retain_loader:
            x = x.to(device)
            lib_size = lib_size.to(device)

            optimizer.zero_grad()
            output = model(x, library_size=lib_size)
            loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)

            if torch.isnan(loss):
                print(f"  NaN loss at epoch {epoch}, stopping finetune")
                model.load_state_dict(best_state)
                return history, best_val_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Validate
        model.train(False)
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

        if np.isnan(val_loss):
            print(f"  NaN val loss at epoch {epoch}, stopping finetune")
            model.load_state_dict(best_state)
            return history, best_val_loss

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return history, best_val_loss


def train_gradient_ascent(baseline_checkpoint, data_path, split_path,
                          output_dir, ascent_steps=10, ascent_lr=1e-5,
                          max_grad_norm=1.0, finetune_epochs=30,
                          finetune_lr=1e-4, patience=10, batch_size=256,
                          seed=42):
    """Run gradient ascent unlearning.

    1. Deep-copies baseline model
    2. Gradient ascent on forget set (maximize loss, N steps)
    3. Fine-tune on retain set (minimize ELBO, M epochs, early stopping)
    4. Save checkpoint

    Returns:
        Path to saved checkpoint.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Seed
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

    # Dataloaders
    forget_loader = create_dataloader(X, forget_idx, batch_size=32,
                                      shuffle=True)
    retain_loader = create_dataloader(X, retain_idx, batch_size=batch_size,
                                      shuffle=True)
    val_size = min(1000, len(retain_idx) // 10)
    val_idx = np.random.choice(retain_idx, size=val_size, replace=False)
    val_loader = create_dataloader(X, val_idx, batch_size=batch_size,
                                   shuffle=False)

    # Clone baseline
    baseline_model, config = load_vae(baseline_checkpoint, device)
    model = deepcopy(baseline_model).to(device)

    # Phase 1: Gradient ascent
    print(f"Phase 1: Gradient ascent ({ascent_steps} steps, lr={ascent_lr})")
    ascent_losses = gradient_ascent_phase(
        model, forget_loader, ascent_steps, ascent_lr, device,
        max_grad_norm=max_grad_norm
    )
    print(f"  Start loss: {ascent_losses[0]:.2f}, "
          f"End loss: {ascent_losses[-1]:.2f}")

    # Phase 2: Fine-tune
    print(f"Phase 2: Fine-tune ({finetune_epochs} epochs max, "
          f"lr={finetune_lr})")
    ft_history, best_val = finetune_phase(
        model, retain_loader, val_loader, finetune_epochs, finetune_lr,
        patience, device
    )
    print(f"  Best val loss: {best_val:.2f}, "
          f"Epochs: {len(ft_history['train'])}")

    model.train(False)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'method': 'gradient_ascent',
        'seed': seed,
        'ascent_steps': ascent_steps,
        'ascent_lr': ascent_lr,
        'finetune_epochs': len(ft_history['train']),
        'best_val_loss': best_val,
    }, output_dir / 'best_model.pt')

    history = {
        'ascent_losses': ascent_losses,
        'finetune': ft_history,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Gradient ascent unlearning for VAEs')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ascent_steps', type=int, default=10)
    parser.add_argument('--ascent_lr', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_gradient_ascent(
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        ascent_steps=args.ascent_steps,
        ascent_lr=args.ascent_lr,
        max_grad_norm=args.max_grad_norm,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
