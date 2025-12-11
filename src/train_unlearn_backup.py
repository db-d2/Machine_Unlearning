"""Adversarial unlearning training script for VAE.

Implements: J(theta) = sum_F phi(A(x;theta)) + lambda * sum_R loss(x;theta)

where:
- F is the forget set
- R is the retain set
- A is the membership attacker (adversary)
- phi is log-loss (BCE)
- lambda balances privacy vs utility

Uses TTUR (Two Time-scale Update Rule) with attacker:VAE ratio of 3:1,
spectral normalization on attacker, EMA for VAE, and gradient clipping.
"""

import argparse
from pathlib import Path
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import numpy as np

from vae import VAE, vae_loss
from attacker import MLPAttacker, extract_vae_features, build_attack_features, EMA
from utils import set_global_seed, create_run_metadata, save_metadata, load_config, Timer
from logging_utils import ExperimentLogger, save_metrics_json


def create_dataloader(adata, indices, batch_size=256, shuffle=True):
    """Create PyTorch DataLoader from AnnData subset."""
    X = torch.FloatTensor(
        adata.X[indices].toarray() if hasattr(adata.X[indices], 'toarray') else adata.X[indices]
    )
    library_size = torch.FloatTensor(X.sum(dim=1, keepdim=True))
    dataset = TensorDataset(X, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_unlearning_objective(
    vae, attacker, forget_loader, retain_loader, device, lambda_retain
):
    """
    Compute unlearning objective J(theta).

    J(theta) = sum_F phi(A(x;theta)) + lambda * sum_R loss(x;theta)

    where phi is binary cross-entropy for attacker predictions.

    Returns:
        j_total: Total unlearning objective
        j_privacy: Privacy term (attacker loss on forget set)
        j_utility: Utility term (VAE loss on retain set)
    """
    vae.eval()
    attacker.eval()

    j_privacy = 0.0
    j_utility = 0.0
    n_forget = 0
    n_retain = 0

    # Privacy term: attacker should fail on forget set (predict 0)
    with torch.no_grad():
        for x_f, lib_f in forget_loader:
            x_f = x_f.to(device)
            lib_f = lib_f.to(device)

            # Extract features for attacker
            vae_feats = extract_vae_features(vae, x_f, lib_f, device)
            attack_feats = build_attack_features(vae_feats)
            attack_feats = attack_feats.to(device)

            # Attacker predictions
            logits = attacker(attack_feats)

            # We want attacker to fail (predict 0 for forget set)
            # Binary cross-entropy with target=0 (non-member)
            privacy_loss = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction='sum'
            )

            j_privacy += privacy_loss.item()
            n_forget += len(x_f)

    # Utility term: VAE should still work well on retain set (per-cell for O(1) magnitude)
    with torch.no_grad():
        for x_r, lib_r in retain_loader:
            x_r = x_r.to(device)
            lib_r = lib_r.to(device)

            output = vae(x_r, library_size=lib_r)
            loss, _, _ = vae_loss(x_r, output, likelihood=vae.likelihood,
                                  library_size=lib_r, per_cell=True)

            j_utility += loss.item() * len(x_r)
            n_retain += len(x_r)

    j_privacy = j_privacy / n_forget if n_forget > 0 else 0.0
    j_utility = j_utility / n_retain if n_retain > 0 else 0.0
    j_total = j_privacy + lambda_retain * j_utility

    return j_total, j_privacy, j_utility


def train_attacker_step(attacker, vae, forget_loader, retain_loader, optimizer, device):
    """
    Train attacker for one step.

    Attacker learns to distinguish forget set (label=1) from retain set (label=0).
    """
    attacker.train()
    vae.eval()

    total_loss = 0.0
    n_samples = 0

    # Sample one batch from each set
    try:
        x_f, lib_f = next(iter(forget_loader))
        x_r, lib_r = next(iter(retain_loader))
    except StopIteration:
        return 0.0

    x_f, lib_f = x_f.to(device), lib_f.to(device)
    x_r, lib_r = x_r.to(device), lib_r.to(device)

    optimizer.zero_grad()

    # Extract features (no grad through VAE)
    with torch.no_grad():
        vae_feats_f = extract_vae_features(vae, x_f, lib_f, device)
        vae_feats_r = extract_vae_features(vae, x_r, lib_r, device)

    attack_feats_f = build_attack_features(vae_feats_f).to(device)
    attack_feats_r = build_attack_features(vae_feats_r).to(device)

    # Attacker predictions
    logits_f = attacker(attack_feats_f)
    logits_r = attacker(attack_feats_r)

    # Binary cross-entropy: forget=1, retain=0
    loss_f = F.binary_cross_entropy_with_logits(logits_f, torch.ones_like(logits_f))
    loss_r = F.binary_cross_entropy_with_logits(logits_r, torch.zeros_like(logits_r))
    loss = (loss_f + loss_r) / 2

    loss.backward()
    optimizer.step()

    return loss.item()


def train_vae_step(vae, attacker, forget_loader, retain_loader, optimizer, device, lambda_retain, grad_clip):
    """
    Train VAE for one unlearning step.

    Objective: J(theta) = sum_F phi(A(x;theta)) + lambda * sum_R loss(x;theta)
    """
    vae.train()
    attacker.eval()

    # Sample batches
    try:
        x_f, lib_f = next(iter(forget_loader))
        x_r, lib_r = next(iter(retain_loader))
    except StopIteration:
        return 0.0, 0.0, 0.0

    x_f, lib_f = x_f.to(device), lib_f.to(device)
    x_r, lib_r = x_r.to(device), lib_r.to(device)

    optimizer.zero_grad()

    # Privacy term: attacker loss on forget set (we want attacker to fail)
    vae_feats_f = extract_vae_features(vae, x_f, lib_f, device)
    attack_feats_f = build_attack_features(vae_feats_f).to(device)

    # Forward through attacker (gradients flow back to VAE, but attacker weights don't update)
    logits_f = attacker(attack_feats_f)

    # BCE loss when attacker should predict forget set as NON-members (target=0)
    # We minimize this to make VAE transform forget samples to look like non-members
    privacy_loss = F.binary_cross_entropy_with_logits(
        logits_f, torch.zeros_like(logits_f)
    )

    # Utility term: VAE reconstruction on retain set (per-cell for O(1) magnitude)
    output_r = vae(x_r, library_size=lib_r)
    utility_loss, _, _ = vae_loss(x_r, output_r, likelihood=vae.likelihood,
                                   library_size=lib_r, per_cell=True)

    # Total unlearning objective
    # Now both terms are O(1): privacy ~2 nats, utility ~0.18 nats
    total_loss = privacy_loss + lambda_retain * utility_loss

    total_loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)

    optimizer.step()

    return total_loss.item(), privacy_loss.item(), utility_loss.item()


def main(args):
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Data shape: {adata.shape}")

    # Load splits
    with open(args.split_path, 'r') as f:
        splits = json.load(f)

    forget_indices = np.array(splits['forget_indices'])
    retain_indices = np.array(splits['retain_indices'])
    unseen_indices = np.array(splits['unseen_indices'])

    print(f"Forget: {len(forget_indices)}, Retain: {len(retain_indices)}, Unseen: {len(unseen_indices)}")

    # Create dataloaders
    forget_loader = create_dataloader(adata, forget_indices, args.batch_size, shuffle=True)
    retain_loader = create_dataloader(adata, retain_indices, args.batch_size, shuffle=True)
    unseen_loader = create_dataloader(adata, unseen_indices, args.batch_size, shuffle=False)

    # Load baseline VAE
    print("\nLoading baseline VAE...")
    checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
    config = checkpoint['config']

    vae = VAE(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        likelihood=config['likelihood'],
        dropout=config.get('dropout', 0.0),
        use_layer_norm=config.get('use_layer_norm', False)
    ).to(device)

    vae.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded VAE: z={config['latent_dim']}, layers={config['hidden_dims']}")

    # Initialize attacker
    print("\nInitializing attacker...")

    # Feature dimension: mu (latent_dim) + logvar (latent_dim) + recon (1) + kl (1) + ELBO (1) + mu_norm (1) + logvar_norm (1)
    # From build_attack_features: recon_nll, kl, elbo, mu, logvar, mu_norm, logvar_norm
    feature_dim = 3 + 2 * config['latent_dim'] + 2  # 3 scalars + 2*latent + 2 norms

    attacker = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3,
        use_spectral_norm=args.use_spectral_norm
    ).to(device)

    print(f"Attacker: input_dim={feature_dim}, spectral_norm={args.use_spectral_norm}")

    # Optimizers with TTUR
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    attacker_optimizer = optim.Adam(attacker.parameters(), lr=args.attacker_lr)

    print(f"TTUR: attacker_lr={args.attacker_lr}, vae_lr={args.vae_lr} (ratio={args.attacker_lr/args.vae_lr:.1f}:1)")

    # EMA for VAE
    ema = EMA(vae, decay=args.ema_decay)
    print(f"EMA decay: {args.ema_decay}")

    # Logger
    logger = ExperimentLogger(log_dir=output_dir / "logs")

    # Save metadata
    metadata = create_run_metadata("adversarial_unlearning_backup", vars(args), args.seed)
    save_metadata(metadata, output_dir)

    # =========================================================================
    # PRE-TRAINING PHASE: Train attacker on baseline VAE
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: PRE-TRAINING ATTACKER ON BASELINE VAE (20 epochs)")
    print("="*70)
    print("This ensures attacker starts strong before adversarial unlearning.")
    print("Objective: Attacker learns to distinguish F (label=1) from R (label=0)")

    pretrain_epochs = 20
    pretrain_history = []

    with Timer("Attacker pre-training"):
        for epoch in range(pretrain_epochs):
            epoch_losses = []

            # Train attacker for 10 steps per epoch
            for _ in range(10):
                att_loss = train_attacker_step(
                    attacker, vae, forget_loader, retain_loader, attacker_optimizer, device
                )
                epoch_losses.append(att_loss)

            avg_loss = np.mean(epoch_losses)
            pretrain_history.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Pre-train epoch {epoch+1}/{pretrain_epochs}: Attacker loss = {avg_loss:.4f}")

    # Save pre-training history
    save_metrics_json({'pretrain_attacker_loss': pretrain_history}, output_dir, "pretrain_history.json")
    print(f"\nPre-training complete! Final attacker loss: {pretrain_history[-1]:.4f}")
    print("Attacker is now ready to guide unlearning.\n")

    # =========================================================================
    # PHASE 2: TWO-STAGE ADVERSARIAL UNLEARNING
    # =========================================================================
    print("="*70)
    print("PHASE 2: TWO-STAGE ADVERSARIAL UNLEARNING")
    print("="*70)
    print(f"Lambda: {args.lambda_retain}, Total Epochs: {args.epochs}")
    print(f"Stage-A: Freeze attacker for {args.stage_a_epochs} epochs (stabilize VAE)")
    print(f"Stage-B: Unfreeze attacker with LR_A = LR_VAE/10, TTUR = 1:1")

    # Stage-B attacker learning rate: LR_VAE / 10
    stage_b_attacker_lr = args.vae_lr / 10
    print(f"Stage-B attacker LR: {stage_b_attacker_lr} (VAE LR: {args.vae_lr})")

    # EMA tracking for early stopping on privacy loss
    ema_privacy_loss = None
    ema_alpha = 0.1  # For EMA tracking of privacy loss

    best_privacy = float('inf')
    patience_counter = 0
    history = {
        'j_total': [], 'j_privacy': [], 'j_utility': [],
        'stage': []  # Track which stage each epoch belongs to
    }

    with Timer("Adversarial training"):
        for epoch in range(args.epochs):
            # Determine current stage
            is_stage_a = epoch < args.stage_a_epochs
            stage_name = "Stage-A" if is_stage_a else "Stage-B"
            history['stage'].append(stage_name)

            if epoch == 0:
                print(f"\n>>> {stage_name}: Attacker FROZEN, training VAE only")
            elif epoch == args.stage_a_epochs:
                print(f"\n>>> {stage_name}: Attacker UNFROZEN, co-training VAE + attacker")
                print(f"    Adjusting attacker LR to {stage_b_attacker_lr}")
                # Update attacker optimizer with new learning rate
                attacker_optimizer = optim.Adam(attacker.parameters(), lr=stage_b_attacker_lr)
                # Reset early stopping to give adversarial game time to develop
                print(f"    Resetting early stopping patience (was {patience_counter})")
                patience_counter = 0
                best_privacy = float('inf')

            # Stage-B: Train attacker if past Stage-A
            if not is_stage_a:
                # Train attacker for one step
                att_loss = train_attacker_step(
                    attacker, vae, forget_loader, retain_loader, attacker_optimizer, device
                )

            # Train VAE (both stages)
            vae_loss_total, vae_loss_privacy, vae_loss_utility = train_vae_step(
                vae, attacker, forget_loader, retain_loader, vae_optimizer, device,
                args.lambda_retain, args.grad_clip
            )

            # Update EMA
            ema.update()

            # Evaluate unlearning objective with EMA weights
            ema.apply_shadow()
            j_total, j_privacy, j_utility = compute_unlearning_objective(
                vae, attacker, forget_loader, retain_loader, device, args.lambda_retain
            )
            ema.restore()

            # Update EMA tracking for privacy loss
            if ema_privacy_loss is None:
                ema_privacy_loss = j_privacy
            else:
                ema_privacy_loss = ema_alpha * j_privacy + (1 - ema_alpha) * ema_privacy_loss

            # Log metrics
            logger.log_scalar("j_total", j_total, epoch)
            logger.log_scalar("j_privacy", j_privacy, epoch)
            logger.log_scalar("j_utility", j_utility, epoch)
            logger.log_scalar("ema_privacy", ema_privacy_loss, epoch)
            logger.log_scalar("stage", 0 if is_stage_a else 1, epoch)

            history['j_total'].append(j_total)
            history['j_privacy'].append(j_privacy)
            history['j_utility'].append(j_utility)

            # Print progress
            if (epoch + 1) % args.print_every == 0:
                stage_marker = "[A]" if is_stage_a else "[B]"
                print(f"Epoch {epoch+1}/{args.epochs} {stage_marker} - "
                      f"J: {j_total:.4f} (priv: {j_privacy:.4f}, util: {j_utility:.4f}) "
                      f"EMA_priv: {ema_privacy_loss:.4f}")

            # Early stopping on EMA(privacy_loss)
            if ema_privacy_loss < best_privacy:
                best_privacy = ema_privacy_loss
                patience_counter = 0

                # Save best checkpoint with EMA weights
                ema.apply_shadow()
                checkpoint = {
                    'epoch': epoch,
                    'stage': stage_name,
                    'vae_state_dict': vae.state_dict(),
                    'attacker_state_dict': attacker.state_dict(),
                    'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                    'attacker_optimizer_state_dict': attacker_optimizer.state_dict(),
                    'j_total': j_total,
                    'j_privacy': j_privacy,
                    'ema_privacy': ema_privacy_loss,
                    'config': vars(args)
                }
                torch.save(checkpoint, output_dir / "best_model.pt")
                ema.restore()
            else:
                patience_counter += 1

                if patience_counter >= args.early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (patience={args.early_stop_patience})")
                    print(f"Best EMA(privacy): {best_privacy:.4f}")
                    break

    # Save final model
    ema.apply_shadow()
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'attacker_state_dict': attacker.state_dict(),
        'config': vars(args)
    }, output_dir / "final_model.pt")
    ema.restore()

    # Save history
    save_metrics_json(history, output_dir, "training_history.json")

    logger.close()

    print(f"\nUnlearning complete!")
    print(f"Best EMA(privacy): {best_privacy:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file (parse first to allow overrides)
    parser.add_argument("--config", type=str, default=None)

    # Data
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)

    # Unlearning objective
    parser.add_argument("--lambda_retain", type=float, default=0.5)

    # Adversarial setup
    parser.add_argument("--attacker_lr", type=float, default=0.0003)
    parser.add_argument("--vae_lr", type=float, default=0.0001)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_spectral_norm", type=bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_cooldown", type=int, default=3)
    parser.add_argument("--stage_a_epochs", type=int, default=5,
                        help="Stage-A: number of epochs to freeze attacker")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=1)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    main(args)
