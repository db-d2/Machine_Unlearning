#!/usr/bin/env python3
"""DP-SGD training for VAEs using Opacus.

Trains a VAE from scratch on the retain set with differential privacy
guarantees via per-sample gradient clipping and Gaussian noise injection.

This provides a formal privacy baseline: the forget set was never in training,
so membership inference should be at chance. The utility cost of DP noise
establishes a lower bound on what privacy guarantees cost.

Reference:
    Abadi et al. (2016). Deep Learning with Differential Privacy.
    CCS 2016.

Usage:
    PYTHONPATH=src python src/train_dp.py \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/dp_sgd/eps10_seed42 \
        --target_epsilon 10.0 \
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

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False

from vae import VAE, nb_loss, kl_divergence, vae_loss


class VAEWithLoss(torch.nn.Module):
    """VAE wrapper that computes loss inside forward() for Opacus compatibility.

    Opacus needs per-sample gradients, which requires the loss to flow through
    the model's forward pass. This wrapper embeds the ELBO computation so that
    model(x, lib) returns a scalar loss directly.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x, library_size):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        mean, dispersion = self.vae.decode(z, library_size)
        recon = nb_loss(x, mean, dispersion, per_cell=False)
        kl = kl_divergence(mu, logvar, per_cell=False)
        return recon + kl


def create_dp_compatible_vae(input_dim, latent_dim, hidden_dims, dropout=0.1):
    """Create a VAE compatible with Opacus DP-SGD.

    Returns a VAEWithLoss wrapper that computes loss inside forward().
    Opacus requires: no BatchNorm, no in-place operations, and all
    trainable parameters must participate in the computation graph.
    """
    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        likelihood='nb',
        dropout=dropout,
        use_layer_norm=True
    )

    # Freeze fc_dispersion: nb_loss uses MSE and ignores dispersion,
    # so this layer has no gradient flow. Opacus requires all trainable
    # parameters to have per-sample gradients, which fails for unused layers.
    vae.decoder.fc_dispersion.weight.requires_grad_(False)
    vae.decoder.fc_dispersion.bias.requires_grad_(False)

    wrapped = VAEWithLoss(vae)

    if HAS_OPACUS:
        errors = ModuleValidator.validate(wrapped, strict=False)
        if errors:
            print(f"  Fixing {len(errors)} Opacus compatibility issues...")
            wrapped = ModuleValidator.fix(wrapped)

    return wrapped


def create_dataloader(X, indices, batch_size=256, shuffle=True):
    """Create DataLoader from indices."""
    data = X[indices]
    library_size = data.sum(dim=1, keepdim=True)
    dataset = TensorDataset(data, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_dp(data_path, split_path, output_dir,
             target_epsilon=10.0, target_delta=1e-5,
             max_grad_norm=1.0, noise_multiplier=None,
             n_epochs=50, lr=1e-3, batch_size=256,
             latent_dim=32, hidden_dims=None,
             patience=15, seed=42):
    """Train a VAE from scratch on retain set with DP-SGD.

    If noise_multiplier is not specified, Opacus will calibrate it
    automatically to achieve target_epsilon within n_epochs.

    Returns:
        Path to saved checkpoint.
    """
    if not HAS_OPACUS:
        raise ImportError("Opacus is required for DP-SGD training. "
                          "Install with: pip install opacus")

    if hidden_dims is None:
        hidden_dims = [1024, 512, 128]

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
    retain_idx = split['retain_indices']

    print(f"Data: {X.shape}, Retain: {len(retain_idx)}, Device: {device}",
          flush=True)
    print(f"Target epsilon: {target_epsilon}, delta: {target_delta}",
          flush=True)

    # Train/val split from retain set
    np.random.shuffle(retain_idx)
    val_size = min(1000, len(retain_idx) // 10)
    val_idx = retain_idx[:val_size]
    train_idx = retain_idx[val_size:]

    train_loader = create_dataloader(X, train_idx, batch_size=batch_size,
                                     shuffle=True)
    val_loader = create_dataloader(X, val_idx, batch_size=batch_size,
                                   shuffle=False)

    # Create DP-compatible model
    input_dim = X.shape[1]
    print(f"Creating DP-compatible VAE ({input_dim} -> {hidden_dims} -> z={latent_dim})...",
          flush=True)
    model = create_dp_compatible_vae(input_dim, latent_dim, hidden_dims)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Calibrate noise multiplier for target epsilon using RDP accountant
    # (PRV accountant in Opacus 1.4.0 has numerical issues)
    sample_rate = batch_size / len(train_idx)
    steps = n_epochs * (len(train_idx) // batch_size)

    if noise_multiplier is None:
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        lo, hi = 0.01, 100.0
        for _ in range(64):
            mid = (lo + hi) / 2
            rdp = compute_rdp(q=sample_rate, noise_multiplier=mid,
                              steps=steps, orders=alphas)
            eps, _ = get_privacy_spent(orders=alphas, rdp=rdp, delta=target_delta)
            if eps > target_epsilon:
                lo = mid
            else:
                hi = mid
        noise_multiplier = hi
    actual_noise = noise_multiplier
    print(f"Calibrated noise_multiplier: {actual_noise:.4f} "
          f"(target_eps={target_epsilon}, steps={steps})", flush=True)

    # Attach Opacus privacy engine
    privacy_engine = PrivacyEngine(accountant='rdp')

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=actual_noise,
        max_grad_norm=max_grad_norm,
    )

    # Training loop
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train': [], 'val': [], 'epsilon': []}

    print(f"Training for up to {n_epochs} epochs "
          f"(~{len(train_idx)//batch_size} batches/epoch)...", flush=True)

    import time as _time
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        _epoch_t0 = _time.time()
        for x, lib_size in train_loader:
            x = x.to(device)
            lib_size = lib_size.to(device)

            optimizer.zero_grad()
            # VAEWithLoss.forward() returns scalar loss directly
            loss = model(x, lib_size)

            if torch.isnan(loss):
                print(f"  NaN loss at epoch {epoch}, skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print(f"  All NaN at epoch {epoch}, stopping")
            break
        train_loss /= n_batches

        # Validate (no DP noise during validation, use inner model)
        model.eval()
        val_loss = 0
        n_batches = 0
        with torch.no_grad():
            for x, lib_size in val_loader:
                x = x.to(device)
                lib_size = lib_size.to(device)
                # Use the wrapped model directly (no per-sample gradient needed)
                loss = model(x, lib_size)
                val_loss += loss.item()
                n_batches += 1
        val_loss /= n_batches

        # Track privacy budget
        current_epsilon = privacy_engine.get_epsilon(target_delta)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['epsilon'].append(current_epsilon)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the underlying model state (unwrap GradSampleModule)
            best_state = deepcopy(model._module.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        _epoch_elapsed = _time.time() - _epoch_t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train={train_loss:.2f}, "
                  f"val={val_loss:.2f}, eps={current_epsilon:.2f}, "
                  f"time={_epoch_elapsed:.0f}s", flush=True)

    final_epsilon = privacy_engine.get_epsilon(target_delta)
    print(f"\nFinal privacy budget: epsilon={final_epsilon:.2f}, "
          f"delta={target_delta}")

    # Restore best model into a clean VAE (unwrap GradSampleModule + VAEWithLoss)
    clean_vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        likelihood='nb',
        dropout=0.1,
        use_layer_norm=True,
    )
    if best_state is not None:
        # best_state is from model._module (VAEWithLoss) -> extract vae keys
        vae_state = {k.replace('vae.', ''): v for k, v in best_state.items()
                     if k.startswith('vae.')}
        clean_vae.load_state_dict(vae_state)
    else:
        # Unwrap: GradSampleModule._module -> VAEWithLoss -> .vae
        inner = model._module
        if hasattr(inner, 'vae'):
            clean_vae.load_state_dict(inner.vae.state_dict())
        else:
            # Fallback: try extracting vae keys from full state
            full_state = inner.state_dict()
            vae_state = {k.replace('vae.', ''): v for k, v in full_state.items()
                         if k.startswith('vae.')}
            clean_vae.load_state_dict(vae_state)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'likelihood': 'nb',
        'dropout': 0.1,
        'use_layer_norm': True,
    }

    torch.save({
        'model_state_dict': clean_vae.state_dict(),
        'config': config,
        'method': 'dp_sgd',
        'seed': seed,
        'target_epsilon': target_epsilon,
        'achieved_epsilon': final_epsilon,
        'target_delta': target_delta,
        'noise_multiplier': actual_noise,
        'max_grad_norm': max_grad_norm,
        'epochs_trained': len(history['train']),
        'best_val_loss': best_val_loss,
    }, output_dir / 'best_model.pt')

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='DP-SGD VAE training (retain set only)')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_epsilon', type=float, default=10.0,
                        help='Target epsilon for DP guarantee')
    parser.add_argument('--target_delta', type=float, default=1e-5,
                        help='Target delta for DP guarantee')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Per-sample gradient clipping norm')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=[1024, 512, 128])
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_dp(
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
