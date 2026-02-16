"""Fisher-based machine unlearning for VAE.

Implements Fisher-diagonal scrubbing followed by retain-only fine-tuning.

References:
    - Golatkar et al. (2020). Eternal Sunshine of the Spotless Net: Selective Forgetting in DNNs.
      CVPR 2020.
    - Nguyen et al. (2022). Variational Bayesian Unlearning.
      NeurIPS 2022.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import scanpy as sc

from vae import VAE
from learning_curve import (
    LearningCurveTracker,
    compute_mia_auc,
    load_attacker_for_eval,
    prepare_eval_data,
    get_feature_dim
)


class AnnDataDataset(Dataset):
    """PyTorch Dataset wrapper for AnnData."""

    def __init__(self, adata, indices):
        self.adata = adata
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        row = self.adata.X[i]
        x = torch.FloatTensor(row.toarray().flatten() if hasattr(row, 'toarray') else np.asarray(row).flatten())
        library_size = torch.FloatTensor([x.sum().item()])
        return x, library_size


def compute_fisher_diagonal(model, dataloader, device, damping=1e-5):
    """Compute Fisher Information Matrix diagonal.

    Uses the diagonal approximation of the Fisher Information Matrix
    to measure parameter importance for the retain set.

    Args:
        model: VAE model
        dataloader: DataLoader for retain set (parameters to protect)
        device: Device to compute on
        damping: Damping term for numerical stability

    Returns:
        Dict mapping parameter names to Fisher diagonal tensors
    """
    model.train()
    fisher = {}

    # Initialize Fisher diagonal accumulators
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    n_samples = 0
    print("Computing Fisher diagonal on retain set...")

    for x_batch, lib_batch in dataloader:
        x_batch = x_batch.to(device)
        lib_batch = lib_batch.to(device)

        model.zero_grad()

        # Forward pass
        mu, logvar = model.encode(x_batch)
        z = model.reparameterize(mu, logvar)

        if model.likelihood == 'nb':
            mean, dispersion = model.decode(z, lib_batch)
            # NB NLL (MSE approximation for log-normalized data)
            recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='sum')
        else:
            recon_mu, recon_logvar = model.decode(z)
            # Gaussian NLL
            recon_loss = 0.5 * (
                recon_logvar
                + ((x_batch - recon_mu) ** 2) / torch.exp(recon_logvar)
                + np.log(2 * np.pi)
            ).sum()

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total ELBO loss (negative ELBO)
        loss = recon_loss + kl_loss

        # Backward to get gradients
        loss.backward()

        # Accumulate squared gradients (Fisher = E[∇²])
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2

        n_samples += x_batch.size(0)

    # Average and add damping
    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping

    print(f"  Computed Fisher diagonal over {n_samples} retain samples")

    return fisher


def fisher_scrub(model, forget_loader, fisher, device, scrub_lr=0.001, scrub_steps=100,
                 tracker=None, eval_fn=None, eval_interval=10):
    """Scrub forget samples using Fisher-weighted gradient ascent.

    Moves parameters in the direction that increases loss on the forget set,
    weighted by the inverse Fisher information to preserve retain set performance.

    Args:
        model: VAE model
        forget_loader: DataLoader for forget set
        fisher: Fisher diagonal dict
        device: Device
        scrub_lr: Learning rate for scrubbing
        scrub_steps: Number of scrubbing steps
        tracker: Optional LearningCurveTracker for AUC-vs-time logging
        eval_fn: Optional function to compute AUC (called with model)
        eval_interval: Steps between AUC evaluations

    Returns:
        List of forget losses during scrubbing
    """
    model.train()

    # Store original parameters
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()

    forget_losses = []
    print(f"Fisher scrubbing for {scrub_steps} steps...")

    # Initial AUC evaluation
    if tracker is not None and eval_fn is not None:
        auc = eval_fn(model)
        tracker.log(step=0, phase="scrub", auc=auc)
        print(f"  Step 0: AUC={auc:.4f}")

    for step in range(scrub_steps):
        step_loss = 0.0
        n_batches = 0

        for x_batch, lib_batch in forget_loader:
            x_batch = x_batch.to(device)
            lib_batch = lib_batch.to(device)

            model.zero_grad()

            # Forward pass
            mu, logvar = model.encode(x_batch)
            z = model.reparameterize(mu, logvar)

            if model.likelihood == 'nb':
                mean, dispersion = model.decode(z, lib_batch)
                recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='sum')
            else:
                recon_mu, recon_logvar = model.decode(z)
                recon_loss = 0.5 * (
                    recon_logvar
                    + ((x_batch - recon_mu) ** 2) / torch.exp(recon_logvar)
                    + np.log(2 * np.pi)
                ).sum()

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            # Backward to get gradients
            loss.backward()

            # Fisher-weighted gradient ASCENT (increase forget loss)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Ascent: θ ← θ + lr * F⁻¹ * ∇L
                        # Using diagonal approximation: F⁻¹ ≈ 1/diag(F)
                        update = scrub_lr * (param.grad.data / fisher[name])
                        # Clip update to prevent numerical instability
                        update = torch.clamp(update, -0.001, 0.001)
                        param.data.add_(update)

            step_loss += loss.item()
            n_batches += 1

        avg_loss = step_loss / max(n_batches, 1)
        forget_losses.append(avg_loss / len(forget_loader.dataset))

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  Step {step + 1}/{scrub_steps}: forget_loss={forget_losses[-1]:.4f}")

        # Periodic AUC evaluation
        if tracker is not None and eval_fn is not None and (step + 1) % eval_interval == 0:
            auc = eval_fn(model)
            tracker.log(step=step + 1, phase="scrub", auc=auc, loss=forget_losses[-1])
            print(f"  Step {step + 1}: AUC={auc:.4f}")

    return forget_losses


def retain_finetune(model, retain_loader, device, epochs=10, lr=0.0001,
                    tracker=None, eval_fn=None, eval_interval=2, scrub_steps=0):
    """Fine-tune on retain set to restore utility.

    After scrubbing, fine-tune on the retain set to restore model performance
    while maintaining the forgetting effect.

    Args:
        model: VAE model (post-scrubbing)
        retain_loader: DataLoader for retain set
        device: Device
        epochs: Number of fine-tune epochs
        lr: Learning rate
        tracker: Optional LearningCurveTracker for AUC-vs-time logging
        eval_fn: Optional function to compute AUC (called with model)
        eval_interval: Epochs between AUC evaluations
        scrub_steps: Number of scrub steps (for step numbering in tracker)

    Returns:
        List of retain losses during fine-tuning
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    retain_losses = []
    print(f"Fine-tuning on retain set for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, lib_batch in retain_loader:
            x_batch = x_batch.to(device)
            lib_batch = lib_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            mu, logvar = model.encode(x_batch)
            z = model.reparameterize(mu, logvar)

            if model.likelihood == 'nb':
                mean, dispersion = model.decode(z, lib_batch)
                recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='mean')
            else:
                recon_mu, recon_logvar = model.decode(z)
                recon_loss = 0.5 * (
                    recon_logvar
                    + ((x_batch - recon_mu) ** 2) / torch.exp(recon_logvar)
                    + np.log(2 * np.pi)
                ).mean()

            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + kl_loss

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        retain_losses.append(avg_loss)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: retain_loss={avg_loss:.4f}")

        # Periodic AUC evaluation
        if tracker is not None and eval_fn is not None and (epoch + 1) % eval_interval == 0:
            auc = eval_fn(model)
            # Use cumulative step count (scrub steps + finetune epochs)
            total_step = scrub_steps + epoch + 1
            tracker.log(step=total_step, phase="finetune", auc=auc, loss=avg_loss)
            print(f"  Epoch {epoch+1}: AUC={auc:.4f}")

    return retain_losses


def main():
    parser = argparse.ArgumentParser(description='Fisher-based unlearning for VAE')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--scrub_lr', type=float, default=0.0001)
    parser.add_argument('--scrub_steps', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=0.0001)
    parser.add_argument('--fisher_damping', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    # Learning curve tracking arguments
    parser.add_argument('--track_auc', action='store_true',
                        help='Track AUC vs wall-clock time for learning curves')
    parser.add_argument('--attacker_path', type=str, default=None,
                        help='Path to trained attacker for AUC evaluation')
    parser.add_argument('--matched_negatives_path', type=str, default=None,
                        help='Path to matched negatives JSON for evaluation')
    parser.add_argument('--eval_interval_scrub', type=int, default=10,
                        help='Steps between AUC evaluations during scrubbing')
    parser.add_argument('--eval_interval_finetune', type=int, default=2,
                        help='Epochs between AUC evaluations during finetuning')
    parser.add_argument('--feature_variant', type=str, default='v1',
                        choices=['v1', 'v2', 'v3'],
                        help='Attacker feature variant')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)

    # Load splits
    print(f"Loading splits from {args.split_path}...")
    with open(args.split_path, 'r') as f:
        split_data = json.load(f)

    forget_indices = split_data['forget_indices']
    retain_indices = split_data['retain_indices']

    print(f"Forget set: {len(forget_indices)} samples")
    print(f"Retain set: {len(retain_indices)} samples")

    # Create datasets and loaders
    forget_dataset = AnnDataDataset(adata, forget_indices)
    retain_dataset = AnnDataDataset(adata, retain_indices)

    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)

    # Load baseline VAE
    print(f"Loading baseline VAE from {args.baseline_checkpoint}...")
    checkpoint = torch.load(args.baseline_checkpoint, map_location=device)

    # Get architecture from checkpoint config
    config = checkpoint['config']
    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config['likelihood'],
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded baseline model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Set up learning curve tracking if requested
    tracker = None
    eval_fn = None

    if args.track_auc:
        if args.attacker_path is None:
            raise ValueError("--attacker_path required when --track_auc is set")

        print("\nSetting up AUC tracking...")
        tracker = LearningCurveTracker()

        # Load attacker
        feature_dim = get_feature_dim(config['latent_dim'], args.feature_variant)
        attacker = load_attacker_for_eval(args.attacker_path, feature_dim, device)
        print(f"  Loaded attacker from {args.attacker_path}")

        # Prepare evaluation data
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
                attacker, device, args.feature_variant
            )

    # Start timer
    start_time = time.time()
    if tracker:
        tracker.start()

    # Step 1: Compute Fisher diagonal on retain set
    fisher = compute_fisher_diagonal(
        model=model,
        dataloader=retain_loader,
        device=device,
        damping=args.fisher_damping
    )

    # Step 2: Fisher scrubbing
    scrub_losses = fisher_scrub(
        model=model,
        forget_loader=forget_loader,
        fisher=fisher,
        device=device,
        scrub_lr=args.scrub_lr,
        scrub_steps=args.scrub_steps,
        tracker=tracker,
        eval_fn=eval_fn,
        eval_interval=args.eval_interval_scrub
    )

    # Step 3: Retain fine-tuning
    finetune_losses = retain_finetune(
        model=model,
        retain_loader=retain_loader,
        device=device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        tracker=tracker,
        eval_fn=eval_fn,
        eval_interval=args.eval_interval_finetune,
        scrub_steps=args.scrub_steps
    )

    # End timer
    total_time = time.time() - start_time

    # Save unlearned model
    model_path = output_dir / 'unlearned_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scrub_losses': scrub_losses,
        'finetune_losses': finetune_losses
    }, model_path)
    print(f"\nSaved unlearned model to {model_path}")

    # Save training history
    history = {
        'scrub_losses': scrub_losses,
        'finetune_losses': finetune_losses
    }

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save metadata
    metadata = {
        'method_id': 'fisher_unlearning',
        'method': 'fisher_scrub_finetune',
        'scrub_lr': args.scrub_lr,
        'scrub_steps': args.scrub_steps,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'fisher_damping': args.fisher_damping,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'total_time_seconds': total_time,
        'forget_set_size': len(forget_indices),
        'retain_set_size': len(retain_indices)
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save learning curve if tracking
    if tracker is not None:
        learning_curve_path = output_dir / 'learning_curve.json'
        tracker.save(str(learning_curve_path))
        print(f"Saved learning curve to {learning_curve_path}")

    print(f"\nFisher unlearning completed in {total_time:.1f}s")
    print(f"Final scrub loss: {scrub_losses[-1]:.4f}")
    print(f"Final finetune loss: {finetune_losses[-1]:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
