#!/usr/bin/env python3
"""Contrastive Latent Unlearning for VAEs.

Pushes forget-set latent representations toward the prior N(0, I) while
preserving retain-set representations, followed by retain-only fine-tuning.

This is a VAE-specific unlearning approach that exploits the latent space
structure: if forget samples map to the prior, they become indistinguishable
from random noise and carry no membership signal.

Usage:
    PYTHONPATH=src python src/train_contrastive_unlearn.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/contrastive/seed42 \
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


def contrastive_loss_forget(model, x, lib_size, gamma=1.0):
    """Loss to push forget-set latent representations toward the prior.

    L_forget = -gamma * KL(q(z|x) || p(z))

    Minimizing this makes q(z|x) = N(mu, sigma) approach p(z) = N(0, I),
    which means mu -> 0, logvar -> 0.

    Args:
        model: VAE model
        x: Forget set batch
        lib_size: Library sizes
        gamma: Weight for the prior-matching term

    Returns:
        Scalar loss (negative KL, to be minimized)
    """
    mu, logvar = model.encode(x)
    # KL(N(mu, sigma) || N(0, I))
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1)
    # We want to MINIMIZE KL, so the forget set maps to the prior
    return gamma * kl.mean()


def contrastive_loss_retain(model, x, lib_size, original_model, lam=1.0):
    """Loss to preserve retain-set representations.

    L_retain = lam * ||mu(x) - mu_orig(x)||^2

    Keeps the encoder output for retain samples close to the original.

    Args:
        model: Current VAE model (being updated)
        x: Retain set batch
        lib_size: Library sizes
        original_model: Frozen copy of baseline model
        lam: Weight for representation preservation

    Returns:
        Scalar loss
    """
    mu, logvar = model.encode(x)

    with torch.no_grad():
        mu_orig, logvar_orig = original_model.encode(x)

    # MSE on mean representations
    mu_diff = torch.mean((mu - mu_orig).pow(2))
    # Also regularize variance to stay close
    var_diff = torch.mean((logvar - logvar_orig).pow(2))

    return lam * (mu_diff + 0.1 * var_diff)


def train_contrastive(baseline_checkpoint, data_path, split_path, output_dir,
                      gamma=1.0, lam=1.0, n_epochs=20, lr=1e-4,
                      finetune_epochs=10, finetune_lr=1e-4, patience=10,
                      batch_size=256, seed=42):
    """Run contrastive latent unlearning.

    Phase 1: Contrastive training
        - Push forget latents toward prior N(0,I)
        - Preserve retain latents near original

    Phase 2: Retain fine-tuning
        - Restore reconstruction quality on retain set

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

    forget_loader = create_dataloader(X, forget_idx,
                                      batch_size=min(batch_size, len(forget_idx)),
                                      shuffle=True)
    retain_loader = create_dataloader(X, retain_idx, batch_size=batch_size,
                                      shuffle=True)

    # Load baseline and create frozen copy
    model, config = load_vae(baseline_checkpoint, device)
    model = model.to(device)
    original_model = deepcopy(model)
    original_model.eval()
    for p in original_model.parameters():
        p.requires_grad_(False)

    # Phase 1: Contrastive training
    print(f"Phase 1: Contrastive training ({n_epochs} epochs, "
          f"gamma={gamma}, lam={lam})...")

    # Only train encoder parameters (decoder stays fixed during contrastive phase)
    encoder_params = list(model.encoder.parameters())
    optimizer = optim.Adam(encoder_params, lr=lr)

    contrastive_history = {'forget_kl': [], 'retain_dist': [], 'total': []}
    forget_iter = iter(forget_loader)

    for epoch in range(n_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0
        n_batches = 0

        for x_retain, lib_retain in retain_loader:
            x_retain = x_retain.to(device)
            lib_retain = lib_retain.to(device)

            # Get a forget batch (cycle through)
            try:
                x_forget, lib_forget = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                x_forget, lib_forget = next(forget_iter)
            x_forget = x_forget.to(device)
            lib_forget = lib_forget.to(device)

            optimizer.zero_grad()

            # Forget loss: push toward prior
            l_forget = contrastive_loss_forget(model, x_forget, lib_forget,
                                               gamma=gamma)
            # Retain loss: stay close to original
            l_retain = contrastive_loss_retain(model, x_retain, lib_retain,
                                               original_model, lam=lam)

            loss = l_forget + l_retain
            loss.backward()
            optimizer.step()

            epoch_forget_loss += l_forget.item()
            epoch_retain_loss += l_retain.item()
            n_batches += 1

        avg_forget = epoch_forget_loss / n_batches
        avg_retain = epoch_retain_loss / n_batches
        contrastive_history['forget_kl'].append(avg_forget)
        contrastive_history['retain_dist'].append(avg_retain)
        contrastive_history['total'].append(avg_forget + avg_retain)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: forget_kl={avg_forget:.4f}, "
                  f"retain_dist={avg_retain:.4f}")

    # Phase 2: Fine-tune on retain set (full model)
    if finetune_epochs > 0:
        val_size = min(1000, len(retain_idx) // 10)
        val_idx = np.random.choice(retain_idx, size=val_size, replace=False)
        val_loader = create_dataloader(X, val_idx, batch_size=batch_size,
                                       shuffle=False)

        print(f"Phase 2: Retain fine-tuning ({finetune_epochs} epochs, "
              f"lr={finetune_lr})...")
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
                print(f"  Epoch {epoch+1}: train={train_loss:.2f}, "
                      f"val={val_loss:.2f}")

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
        'method': 'contrastive',
        'seed': seed,
        'gamma': gamma,
        'lam': lam,
        'contrastive_epochs': n_epochs,
        'finetune_epochs': len(ft_history['train']) if ft_history else 0,
        'best_val_loss': best_val_loss,
    }, output_dir / 'best_model.pt')

    history = {
        'contrastive': contrastive_history,
        'finetune': ft_history,
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Contrastive Latent Unlearning for VAEs')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for forget prior-matching loss')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='Weight for retain representation preservation')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Contrastive training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for contrastive phase')
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_contrastive(
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        gamma=args.gamma,
        lam=args.lam,
        n_epochs=args.n_epochs,
        lr=args.lr,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
