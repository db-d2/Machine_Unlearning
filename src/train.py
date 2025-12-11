"""Training script for baseline VAE."""

import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import numpy as np

from vae import VAE, vae_loss
from utils import set_global_seed, create_run_metadata, save_metadata, load_config, Timer
from logging_utils import ExperimentLogger, save_metrics_json


def create_dataloader(adata, indices, batch_size=256, shuffle=True):
    """Create PyTorch DataLoader from AnnData subset."""
    X = torch.FloatTensor(adata.X[indices].toarray() if hasattr(adata.X[indices], 'toarray') else adata.X[indices])

    # Library size (total counts per cell)
    library_size = torch.FloatTensor(X.sum(dim=1, keepdim=True))

    dataset = TensorDataset(X, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, dataloader, optimizer, device, likelihood='nb', beta=1.0, free_bits=0.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for x, lib_size in dataloader:
        x = x.to(device)
        lib_size = lib_size.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(x, library_size=lib_size)

        # Compute loss with free-bits
        loss, recon, kl = vae_loss(x, output, likelihood=likelihood, library_size=lib_size, beta=beta, free_bits=free_bits)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def eval_epoch(model, dataloader, device, likelihood='nb', beta=1.0, free_bits=0.0):
    """Evaluate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, lib_size in dataloader:
            x = x.to(device)
            lib_size = lib_size.to(device)

            output = model(x, library_size=lib_size)
            loss, recon, kl = vae_loss(x, output, likelihood=likelihood, library_size=lib_size, beta=beta, free_bits=free_bits)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def get_kl_weight(epoch, warmup_epochs):
    """
    Linear KL warm-up schedule from 0 to 1.

    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs to ramp KL weight from 0 to 1

    Returns:
        KL weight (beta) in [0, 1]
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def main(args):
    # Set seed
    set_global_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading preprocessed data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded: {adata.shape}")

    # Create splits (reconstruct from saved indices if needed)
    # For now, use simple train/val split
    n_cells = adata.n_obs
    n_train = int(0.85 * n_cells)

    indices = np.arange(n_cells)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Create dataloaders
    train_loader = create_dataloader(adata, train_indices, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(adata, val_indices, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("\nInitializing VAE...")
    model = VAE(
        input_dim=adata.n_vars,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        likelihood=args.likelihood,
        dropout=getattr(args, 'dropout', 0.0),
        use_layer_norm=getattr(args, 'use_layer_norm', False)
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: {adata.n_vars} -> {args.hidden_dims} -> z={args.latent_dim}")
    print(f"Dropout: {getattr(args, 'dropout', 0.0)}, LayerNorm: {getattr(args, 'use_layer_norm', False)}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logger
    logger = ExperimentLogger(log_dir=output_dir / "logs")

    # Save metadata
    config = vars(args)
    metadata = create_run_metadata("baseline_vae_training", config, args.seed)
    save_metadata(metadata, output_dir)

    # Training loop
    print("\nStarting training...")
    kl_warmup_epochs = getattr(args, 'kl_warmup_epochs', 0)
    free_bits = getattr(args, 'free_bits', 0.0)

    if kl_warmup_epochs > 0:
        print(f"KL warm-up: ramping from 0 to 1 over {kl_warmup_epochs} epochs")
    if free_bits > 0:
        print(f"Free-bits: {free_bits} nats per dimension")

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    with Timer("Training"):
        for epoch in range(args.epochs):
            # Compute KL weight with warm-up schedule
            kl_weight = get_kl_weight(epoch, kl_warmup_epochs)

            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, args.likelihood, beta=kl_weight, free_bits=free_bits)

            # Validate (use full KL weight for validation)
            val_metrics = eval_epoch(model, val_loader, device, args.likelihood, beta=1.0, free_bits=free_bits)

            # Log
            logger.log_metrics(train_metrics, epoch, prefix="train")
            logger.log_metrics(val_metrics, epoch, prefix="val")
            logger.log_scalar("kl_weight", kl_weight, epoch)

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            # Print progress
            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f} - "
                      f"Val Loss: {val_metrics['loss']:.4f} - "
                      f"KL weight: {kl_weight:.3f}")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config
                }
                torch.save(checkpoint, output_dir / "best_model.pt")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Save training history
    save_metrics_json(history, output_dir, "training_history.json")

    logger.close()

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="data/adata_processed.h5ad")
    parser.add_argument("--output_dir", type=str, default="outputs/p1/baseline")

    # Model
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 128])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--likelihood", type=str, default="nb", choices=["nb", "gaussian"])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_every", type=int, default=10)

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    main(args)
