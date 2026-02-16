#!/usr/bin/env python3
"""Retain-only fine-tuning baseline for VAE unlearning.

Fine-tunes a copy of the baseline model on the retain set only, without
any explicit forgetting mechanism. Logic extracted from notebook 16.

Usage:
    PYTHONPATH=src python src/train_retain_finetune.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/retain_finetune/seed42 \
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


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, lib_size in dataloader:
        x = x.to(device)
        lib_size = lib_size.to(device)

        optimizer.zero_grad()
        output = model(x, library_size=lib_size)
        loss, recon, kl = vae_loss(x, output, likelihood='nb', beta=1.0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_epoch(model, dataloader, device):
    """Evaluate model on dataloader, return mean loss."""
    model.train(False)
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, lib_size in dataloader:
            x = x.to(device)
            lib_size = lib_size.to(device)

            output = model(x, library_size=lib_size)
            loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_retain_finetune(baseline_checkpoint, data_path, split_path,
                          output_dir, lr=1e-4, epochs=50, patience=10,
                          batch_size=256, seed=42):
    """Run retain-only fine-tuning.

    1. Deep-copies baseline model
    2. Fine-tunes on retain set (ELBO loss, early stopping)
    3. Saves checkpoint to output_dir/best_model.pt

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
    retain_idx = split['retain_indices']

    print(f"Data: {X.shape}, Retain: {len(retain_idx)}, Device: {device}")

    # Dataloaders
    retain_loader = create_dataloader(X, retain_idx, batch_size=batch_size,
                                      shuffle=True)
    val_size = min(1000, len(retain_idx) // 10)
    val_idx = np.random.choice(retain_idx, size=val_size, replace=False)
    val_loader = create_dataloader(X, val_idx, batch_size=batch_size,
                                   shuffle=False)

    # Clone baseline
    baseline_model, config = load_vae(baseline_checkpoint, device)
    model = deepcopy(baseline_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"Fine-tuning for up to {epochs} epochs (patience={patience})...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, retain_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or improved:
            marker = '*' if improved else ''
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, "
                  f"val={val_loss:.4f} {marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    model.train(False)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'method': 'retain_finetune',
        'seed': seed,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': best_val_loss,
    }, output_dir / 'best_model.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Retain-only fine-tuning for VAE unlearning')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_retain_finetune(
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
