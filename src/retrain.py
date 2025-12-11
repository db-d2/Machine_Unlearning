r"""Gold-standard retraining on D\F (dataset with forget set removed).

This script trains a VAE from scratch on the remaining data after removing
the forget set. This serves as the gold standard for evaluating unlearning methods.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import numpy as np

from vae import VAE, vae_loss
from utils import set_global_seed, create_run_metadata, save_metadata, load_config, Timer, DEVICE
from logging_utils import ExperimentLogger, save_metrics_json
from learning_curve import (
    LearningCurveTracker,
    compute_mia_auc,
    load_attacker_for_eval,
    prepare_eval_data,
    get_feature_dim
)


def load_forget_set(forget_set_path: str) -> np.ndarray:
    """Load forget set indices from JSON file."""
    with open(forget_set_path, 'r') as f:
        forget_data = json.load(f)

    # Handle both 'indices' and 'forget_indices' keys
    if 'forget_indices' in forget_data:
        indices = np.array(forget_data['forget_indices'])
    else:
        indices = np.array(forget_data['indices'])

    print(f"Loaded forget set: {len(indices)} cells")
    if 'checksum' in forget_data:
        print(f"  Checksum: {forget_data['checksum']}")

    return indices


def create_dataloader_exclude(adata, indices, exclude_indices, batch_size=256, shuffle=True):
    """Create PyTorch DataLoader excluding specified cells."""
    # Filter out cells in forget set
    mask = np.ones(len(adata), dtype=bool)
    mask[exclude_indices] = False
    filtered_indices = indices[np.isin(indices, np.where(mask)[0])]

    X = torch.FloatTensor(
        adata.X[filtered_indices].toarray()
        if hasattr(adata.X[filtered_indices], 'toarray')
        else adata.X[filtered_indices]
    )

    library_size = torch.FloatTensor(X.sum(dim=1, keepdim=True))

    dataset = TensorDataset(X, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, dataloader, optimizer, device, likelihood='nb', beta=1.0):
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

        output = model(x, library_size=lib_size)
        loss, recon, kl = vae_loss(x, output, likelihood=likelihood, library_size=lib_size, beta=beta)

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


def eval_epoch(model, dataloader, device, likelihood='nb', beta=1.0):
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
            loss, recon, kl = vae_loss(x, output, likelihood=likelihood, library_size=lib_size, beta=beta)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'recon': total_recon / n_batches,
        'kl': total_kl / n_batches
    }


def main(args):
    set_global_seed(args.seed)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load forget set
    print("\nLoading forget set...")
    forget_indices = load_forget_set(args.forget_set_path)

    # Load data
    print("\nLoading preprocessed data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Total cells: {adata.shape}")
    print(f"Cells to exclude: {len(forget_indices)}")
    print(f"Remaining cells (D\\F): {adata.shape[0] - len(forget_indices)}")

    # Create train/val/test splits from remaining data
    n_cells = adata.n_obs
    all_indices = np.arange(n_cells)

    # Remove forget set cells
    retain_mask = np.ones(n_cells, dtype=bool)
    retain_mask[forget_indices] = False
    retain_indices = all_indices[retain_mask]

    # Split retained data: 85% train, 15% val from D\F
    np.random.shuffle(retain_indices)
    n_train = int(0.85 * len(retain_indices))

    train_indices = retain_indices[:n_train]
    val_indices = retain_indices[n_train:]

    print(f"\nSplits from D\\F:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")

    # Create dataloaders (no exclusion needed since we already filtered indices)
    train_loader = create_dataloader_exclude(adata, train_indices, [], batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader_exclude(adata, val_indices, [], batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("\nInitializing VAE...")
    model = VAE(
        input_dim=adata.n_vars,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        likelihood=args.likelihood
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger = ExperimentLogger(log_dir=output_dir / "logs")

    # Set up learning curve tracking if requested
    tracker = None
    eval_fn = None

    if args.track_auc:
        if args.attacker_path is None:
            raise ValueError("--attacker_path required when --track_auc is set")

        print("\nSetting up AUC tracking...")
        tracker = LearningCurveTracker()

        # Load attacker
        feature_dim = get_feature_dim(args.latent_dim, args.feature_variant)
        attacker = load_attacker_for_eval(args.attacker_path, feature_dim, str(device))
        print(f"  Loaded attacker from {args.attacker_path}")

        # Prepare evaluation data
        # For retrain, we need to construct split_data dict
        split_data = {
            'forget_indices': forget_indices.tolist(),
            'retain_indices': retain_indices.tolist()
        }

        matched_indices = None
        if args.matched_negatives_path:
            with open(args.matched_negatives_path, 'r') as f:
                matched_data = json.load(f)
            matched_indices = np.array(matched_data['matched_indices'])
            print(f"  Loaded {len(matched_indices)} matched negatives")

        forget_x, forget_lib, unseen_x, unseen_lib = prepare_eval_data(
            adata, split_data, matched_indices
        )
        print(f"  Evaluation data: {len(forget_x)} forget, {len(unseen_x)} unseen")

        # Create evaluation function
        def eval_fn(model):
            return compute_mia_auc(
                model, forget_x, forget_lib, unseen_x, unseen_lib,
                attacker, str(device), args.feature_variant
            )

    # Save metadata
    config = vars(args)
    metadata = create_run_metadata("retrain_without_forget", config, args.seed)
    metadata['forget_set_size'] = int(len(forget_indices))
    metadata['train_size'] = int(len(train_indices))
    metadata['val_size'] = int(len(val_indices))
    save_metadata(metadata, output_dir)

    # Training loop
    print("\nStarting training on D\\F...")
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    # Start tracker if enabled
    if tracker:
        tracker.start()
        # Initial AUC evaluation
        if eval_fn is not None:
            auc = eval_fn(model)
            tracker.log(step=0, phase="train", auc=auc)
            print(f"  Epoch 0: AUC={auc:.4f}")

    with Timer("Retrain"):
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, device, args.likelihood, args.beta)
            val_metrics = eval_epoch(model, val_loader, device, args.likelihood, args.beta)

            logger.log_metrics(train_metrics, epoch, prefix="train")
            logger.log_metrics(val_metrics, epoch, prefix="val")

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f} - "
                      f"Val Loss: {val_metrics['loss']:.4f}")

            # Periodic AUC evaluation
            if tracker is not None and eval_fn is not None and (epoch + 1) % args.eval_interval == 0:
                auc = eval_fn(model)
                tracker.log(step=epoch+1, phase="train", auc=auc, loss=train_metrics['loss'])
                print(f"  Epoch {epoch+1}: AUC={auc:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config,
                    'forget_set_size': len(forget_indices)
                }
                torch.save(checkpoint, output_dir / "best_model.pt")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    save_metrics_json(history, output_dir, "training_history.json")

    logger.close()

    # Save learning curve if tracking
    if tracker is not None:
        learning_curve_path = output_dir / 'learning_curve.json'
        tracker.save(str(learning_curve_path))
        print(f"Saved learning curve to {learning_curve_path}")

    print(f"\nRetrain complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="data/adata_processed.h5ad")
    parser.add_argument("--forget_set_path", type=str, required=True,
                        help="Path to forget set JSON file")
    parser.add_argument("--output_dir", type=str, required=True)

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

    # Learning curve tracking arguments
    parser.add_argument('--track_auc', action='store_true',
                        help='Track AUC vs wall-clock time for learning curves')
    parser.add_argument('--attacker_path', type=str, default=None,
                        help='Path to trained attacker for AUC evaluation')
    parser.add_argument('--matched_negatives_path', type=str, default=None,
                        help='Path to matched negatives JSON for evaluation')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Epochs between AUC evaluations')
    parser.add_argument('--feature_variant', type=str, default='v1',
                        choices=['v1', 'v2', 'v3'],
                        help='Attacker feature variant')

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    main(args)
