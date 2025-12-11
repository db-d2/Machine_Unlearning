"""Adversarial unlearning training script for VAE.

Implements frozen-critic default with:
- Multi-step training per epoch (ceil(|R|/B) steps)
- Privacy gradient accumulation with K repeats
- Per-cell utility loss (O(1) magnitude)
- Optional EMA ratio scaling for gradient balance
- Optional latent MMD and F-only KL up-weight
- Max privacy objective (vs U_matched and vs R)
- Gradient norm logging
"""

import argparse
from pathlib import Path
import json
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import scanpy as sc
import numpy as np

from vae import VAE, vae_loss, kl_divergence
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


def compute_mmd_rbf(z1, z2, gamma=0.1):
    """Compute RBF kernel MMD between two sets of latent codes.

    Args:
        z1: Tensor of shape [N1, latent_dim]
        z2: Tensor of shape [N2, latent_dim]
        gamma: RBF kernel bandwidth

    Returns:
        MMD scalar
    """
    # Compute pairwise distances
    z1z1 = torch.cdist(z1, z1, p=2) ** 2
    z2z2 = torch.cdist(z2, z2, p=2) ** 2
    z1z2 = torch.cdist(z1, z2, p=2) ** 2

    # RBF kernel
    k11 = torch.exp(-gamma * z1z1).mean()
    k22 = torch.exp(-gamma * z2z2).mean()
    k12 = torch.exp(-gamma * z1z2).mean()

    mmd = k11 + k22 - 2 * k12
    return mmd


def compute_gradient_norms(model):
    """Compute L2 norm of model gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def load_attackers(attacker_paths, device):
    """Load ensemble of pre-trained attackers.

    Supports multi-critic evaluation with different feature variants.

    Args:
        attacker_paths: List of paths to attacker checkpoints
        device: Device to load on

    Returns:
        List of (attacker_model, variant, config) tuples
    """
    attackers = []
    for path in attacker_paths:
        ckpt = torch.load(path, map_location=device)
        config = ckpt['config']

        attacker = MLPAttacker(
            input_dim=config['feature_dim'],
            hidden_dims=config['hidden_dims'],
            dropout=config.get('dropout', 0.3),
            use_spectral_norm=config.get('use_spectral_norm', True)
        ).to(device)

        attacker.load_state_dict(ckpt['model_state_dict'])
        attacker.eval()  # Frozen

        variant = config.get('variant', 'v1')
        attackers.append((attacker, variant, config))

    return attackers


def train_attacker_step(attacker, vae, forget_loader, retain_loader, optimizer, device):
    """Train attacker for one step."""
    attacker.train()
    vae.eval()

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


def train_vae_multi_step(
    vae, attacker, retain_loader, forget_loader, unseen_loader,
    optimizer, device, args, ema_priv, ema_util, logger, epoch
):
    """Train VAE for multiple steps per epoch with privacy gradient accumulation.

    Args:
        vae: VAE model
        attacker: MLP attacker (frozen during training)
        retain_loader: DataLoader for retain set
        forget_loader: DataLoader for forget set
        unseen_loader: DataLoader for unseen set (optional, for matched negatives)
        optimizer: VAE optimizer
        device: torch device
        args: training arguments
        ema_priv: EMA tracker for privacy loss magnitude
        ema_util: EMA tracker for utility loss magnitude
        logger: experiment logger
        epoch: current epoch number

    Returns:
        Tuple of (avg_privacy_loss, avg_utility_loss, ema_priv, ema_util, grad_norm_priv, grad_norm_util)
    """
    vae.train()

    # Set attacker(s) to eval mode
    if isinstance(attacker, list):
        # Multi-critic mode
        for att_model, _, _ in attacker:
            att_model.eval()
    else:
        # Single attacker mode
        attacker.eval()

    # Create cyclic iterators for forget and unseen
    forget_iter = cycle(forget_loader)
    unseen_iter = cycle(unseen_loader) if unseen_loader is not None else None

    # Compute steps per epoch
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = math.ceil(len(retain_loader.dataset) / args.batch_size)

    total_privacy_loss = 0.0
    total_utility_loss = 0.0
    grad_norms_priv = []
    grad_norms_util = []

    retain_iter = iter(retain_loader)

    for step in range(steps_per_epoch):
        # Get retain batch
        try:
            x_r, lib_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            x_r, lib_r = next(retain_iter)

        x_r, lib_r = x_r.to(device), lib_r.to(device)

        optimizer.zero_grad()

        # === PRIVACY TERM with K repeats and gradient accumulation ===
        # Use list accumulation to preserve gradient graph
        privacy_losses = []

        # Multi-critic ensemble support
        if isinstance(attacker, list):
            # Multi-critic mode: max over critics
            attackers_ensemble = attacker  # Rename for clarity
        else:
            # Single-critic mode: wrap in list for compatibility
            attackers_ensemble = [(attacker, 'v1', None)]

        for k in range(args.privacy_repeats_k):
            # Sample forget batch
            x_f, lib_f = next(forget_iter)
            x_f, lib_f = x_f.to(device), lib_f.to(device)

            # Extract VAE features WITH gradients for backprop
            vae_feats_f = extract_vae_features(vae, x_f, lib_f, device, requires_grad=True)
            mu_f, logvar_f = vae_feats_f['mu'], vae_feats_f['logvar']

            # Compute loss for each critic and take max
            critic_losses = []
            for attacker_model, variant, _ in attackers_ensemble:
                attack_feats_f = build_attack_features(vae_feats_f, variant=variant).to(device)
                logits_f = attacker_model(attack_feats_f)
                loss_i = F.binary_cross_entropy_with_logits(logits_f, torch.zeros_like(logits_f))
                critic_losses.append(loss_i)

            # Max over critics
            if len(critic_losses) > 1:
                privacy_loss = torch.max(torch.stack(critic_losses))
            else:
                privacy_loss = critic_losses[0]

            # Optional: privacy vs unseen (if matched negatives available)
            mu_u = None
            if unseen_iter is not None and args.use_max_privacy:
                x_u, lib_u = next(unseen_iter)
                x_u, lib_u = x_u.to(device), lib_u.to(device)

                # Extract with gradients for MMD computation
                vae_feats_u = extract_vae_features(vae, x_u, lib_u, device, requires_grad=True)
                mu_u = vae_feats_u['mu']

                critic_losses_u = []
                for attacker_model, variant, _ in attackers_ensemble:
                    attack_feats_u = build_attack_features(vae_feats_u, variant=variant).to(device)
                    logits_u = attacker_model(attack_feats_u)
                    loss_u_i = F.binary_cross_entropy_with_logits(logits_u, torch.zeros_like(logits_u))
                    critic_losses_u.append(loss_u_i)

                privacy_vs_u = torch.max(torch.stack(critic_losses_u)) if len(critic_losses_u) > 1 else critic_losses_u[0]
                privacy_loss = torch.max(privacy_loss, privacy_vs_u)

            # Feature-space MMD (not latent-space)
            if args.mmd_gamma > 0 and mu_u is not None:
                # Use v1 (full) attacker features for MMD, not just latents
                # Reuse already-computed VAE features
                attack_feats_f_mmd = build_attack_features(vae_feats_f, variant='v1').to(device)
                attack_feats_u_mmd = build_attack_features(vae_feats_u, variant='v1').to(device)
                mmd_loss = compute_mmd_rbf(attack_feats_f_mmd, attack_feats_u_mmd, gamma=args.mmd_gamma)
                privacy_loss = privacy_loss + args.mmd_gamma * mmd_loss

            # Optional: F-only KL up-weight
            if args.alpha_f_kl > 0:
                kl_f = kl_divergence(mu_f, logvar_f, per_cell=True)
                privacy_loss = privacy_loss + args.alpha_f_kl * kl_f

            privacy_losses.append(privacy_loss)

        # Average privacy loss over K repeats
        privacy_loss_avg = torch.stack(privacy_losses).mean()

        # === UTILITY TERM ===
        output_r = vae(x_r, library_size=lib_r)
        utility_loss, _, _ = vae_loss(
            x_r, output_r, likelihood=vae.likelihood,
            library_size=lib_r, per_cell=True  # per-cell for O(1) magnitude
        )

        # Update EMA trackers
        if ema_priv is None:
            ema_priv = privacy_loss_avg.item()
            ema_util = utility_loss.item()
        else:
            ema_priv = 0.1 * privacy_loss_avg.item() + 0.9 * ema_priv
            ema_util = 0.1 * utility_loss.item() + 0.9 * ema_util

        # Optional: EMA ratio scaling
        scale_t = 1.0
        if args.balance_mode == "ema_ratio" and ema_util > 0:
            scale_t = max(1e-3, min(1e3, ema_priv / ema_util))

        # Separate gradient computation for proper logging
        # Method: Measure privacy grads, zero, measure utility grads, zero, then do actual backward

        # Measure privacy gradient norm
        privacy_loss_avg.backward(retain_graph=True)
        grad_norm_priv = compute_gradient_norms(vae)
        optimizer.zero_grad()

        # Measure utility gradient norm
        utility_loss_weighted = args.lambda_retain * scale_t * utility_loss
        utility_loss_weighted.backward(retain_graph=True)
        grad_norm_util = compute_gradient_norms(vae)
        optimizer.zero_grad()

        # Actual backward for optimization (total loss)
        total_loss = privacy_loss_avg + utility_loss_weighted
        total_loss.backward()

        grad_norms_priv.append(grad_norm_priv)
        grad_norms_util.append(grad_norm_util)

        # Gradient clipping and optimizer step
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)

        optimizer.step()

        # Accumulate losses
        total_privacy_loss += privacy_loss_avg.item()
        total_utility_loss += utility_loss.item()

    # Average over steps
    avg_privacy_loss = total_privacy_loss / steps_per_epoch
    avg_utility_loss = total_utility_loss / steps_per_epoch
    avg_grad_norm_priv = np.mean(grad_norms_priv)
    avg_grad_norm_util = np.mean(grad_norms_util)

    return avg_privacy_loss, avg_utility_loss, ema_priv, ema_util, avg_grad_norm_priv, avg_grad_norm_util


def compute_unlearning_objective(
    vae, attacker, forget_loader, retain_loader, device, lambda_retain
):
    """Compute unlearning objective J(theta) for evaluation."""
    vae.eval()

    # Set attacker(s) to eval mode
    if isinstance(attacker, list):
        # Multi-critic mode
        for att_model, _, _ in attacker:
            att_model.eval()
        attackers_ensemble = attacker
    else:
        # Single attacker mode
        attacker.eval()
        attackers_ensemble = [(attacker, 'v1', None)]

    j_privacy = 0.0
    j_utility = 0.0
    n_forget = 0
    n_retain = 0

    # Privacy term (max over critics for multi-critic mode)
    with torch.no_grad():
        for x_f, lib_f in forget_loader:
            x_f = x_f.to(device)
            lib_f = lib_f.to(device)

            vae_feats = extract_vae_features(vae, x_f, lib_f, device)

            # Compute max privacy loss over critics
            critic_losses = []
            for att_model, variant, _ in attackers_ensemble:
                attack_feats = build_attack_features(vae_feats, variant=variant).to(device)
                logits = att_model(attack_feats)
                loss = F.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(logits), reduction='sum'
                )
                critic_losses.append(loss.item())

            # Take max over critics (for multi-critic) or single value (for single-critic)
            privacy_loss = max(critic_losses)

            j_privacy += privacy_loss
            n_forget += len(x_f)

    # Utility term
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
    unseen_loader = create_dataloader(adata, unseen_indices, args.batch_size, shuffle=True) if args.use_matched_unseen_loader else None

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

    # Load attacker(s)
    use_multi_critic = False
    if args.attacker_paths is not None and len(args.attacker_paths) > 1:
        # Multi-critic mode
        print(f"\nLoading {len(args.attacker_paths)} pre-trained attackers (multi-critic)...")
        attacker = load_attackers(args.attacker_paths, device)
        for i, (att_model, variant, att_config) in enumerate(attacker):
            auc = att_config.get('training', {}).get('auc', 'N/A')
            print(f"  Critic {i + 1}: variant={variant}, AUC={auc:.4f}" if isinstance(auc, float) else f"  Critic {i + 1}: variant={variant}")
        use_multi_critic = True
        attacker_optimizer = None  # No optimizer for frozen multi-critic

    elif args.attacker_paths is not None and len(args.attacker_paths) == 1:
        # Single attacker from checkpoint
        print(f"\nLoading single pre-trained attacker from {args.attacker_paths[0]}...")
        attacker_list = load_attackers(args.attacker_paths, device)
        attacker, variant, att_config = attacker_list[0]
        auc = att_config.get('training', {}).get('auc', 'N/A')
        print(f"  Loaded: variant={variant}, AUC={auc:.4f}" if isinstance(auc, float) else f"  Loaded: variant={variant}")
        use_multi_critic = False
        attacker_optimizer = None  # Frozen attacker

    else:
        # Train attacker from scratch (original behavior)
        print("\nInitializing attacker...")
        feature_dim = 3 + 2 * config['latent_dim'] + 2
        attacker = MLPAttacker(
            input_dim=feature_dim,
            hidden_dims=[256, 256],
            dropout=0.3,
            use_spectral_norm=args.use_spectral_norm
        ).to(device)

        print(f"Attacker: input_dim={feature_dim}, spectral_norm={args.use_spectral_norm}")
        attacker_optimizer = optim.Adam(attacker.parameters(), lr=args.attacker_lr)
        print(f"Attacker LR: {args.attacker_lr}")
        use_multi_critic = False

    # VAE Optimizer
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
    print(f"VAE LR: {args.vae_lr}")

    # EMA for VAE
    ema = EMA(vae, decay=args.ema_decay)
    print(f"EMA decay: {args.ema_decay}")

    # Logger
    logger = ExperimentLogger(log_dir=output_dir / "logs")

    # Save metadata
    metadata = create_run_metadata("adversarial_unlearning", vars(args), args.seed)
    save_metadata(metadata, output_dir)

    # =========================================================================
    # PHASE 1: PRE-TRAINING ATTACKER
    # =========================================================================
    if attacker_optimizer is not None:
        # Only train if attacker is not pre-loaded
        print("\n" + "="*70)
        print("PHASE 1: PRE-TRAINING ATTACKER ON BASELINE VAE")
        print("="*70)
        print(f"Training attacker for {args.pretrain_epochs} epochs")

        pretrain_history = []

        with Timer("Attacker pre-training"):
            for epoch in range(args.pretrain_epochs):
                epoch_losses = []

                for _ in range(10):
                    att_loss = train_attacker_step(
                        attacker, vae, forget_loader, retain_loader, attacker_optimizer, device
                    )
                    epoch_losses.append(att_loss)

                avg_loss = np.mean(epoch_losses)
                pretrain_history.append(avg_loss)

                if (epoch + 1) % 5 == 0:
                    print(f"  Pre-train epoch {epoch+1}/{args.pretrain_epochs}: Attacker loss = {avg_loss:.4f}")

        save_metrics_json({'pretrain_attacker_loss': pretrain_history}, output_dir, "pretrain_history.json")
        print(f"\nPre-training complete! Final attacker loss: {pretrain_history[-1]:.4f}")
    else:
        print("\n" + "="*70)
        print("PHASE 1: SKIPPED (Using pre-trained attacker(s))")
        print("="*70)

    # =========================================================================
    # PHASE 2: ADVERSARIAL UNLEARNING (FROZEN-CRITIC DEFAULT)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: ADVERSARIAL UNLEARNING (FROZEN-CRITIC)")
    print("="*70)
    print(f"Lambda: {args.lambda_retain}, Epochs: {args.epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch or 'auto'}")
    print(f"Privacy repeats K: {args.privacy_repeats_k}")
    print(f"Balance mode: {args.balance_mode}")
    print(f"MMD gamma: {args.mmd_gamma}")
    print(f"F-only KL α: {args.alpha_f_kl}")

    best_privacy = float('inf')
    patience_counter = 0
    history = {
        'j_total': [], 'j_privacy': [], 'j_utility': [],
        'grad_norm_priv': [], 'grad_norm_util': []
    }

    ema_priv = None
    ema_util = None

    with Timer("Adversarial training"):
        for epoch in range(args.epochs):
            # Multi-step VAE training
            avg_priv, avg_util, ema_priv, ema_util, grad_norm_priv, grad_norm_util = train_vae_multi_step(
                vae, attacker, retain_loader, forget_loader, unseen_loader,
                vae_optimizer, device, args, ema_priv, ema_util, logger, epoch
            )

            # Update EMA
            ema.update()

            # Evaluate with EMA weights
            ema.apply_shadow()
            j_total, j_privacy, j_utility = compute_unlearning_objective(
                vae, attacker, forget_loader, retain_loader, device, args.lambda_retain
            )
            ema.restore()

            # Log metrics
            logger.log_scalar("j_total", j_total, epoch)
            logger.log_scalar("j_privacy", j_privacy, epoch)
            logger.log_scalar("j_utility", j_utility, epoch)
            logger.log_scalar("ema_privacy", ema_priv, epoch)
            logger.log_scalar("ema_utility", ema_util, epoch)
            logger.log_scalar("grad_norm_priv", grad_norm_priv, epoch)
            logger.log_scalar("grad_norm_util", grad_norm_util, epoch)

            history['j_total'].append(j_total)
            history['j_privacy'].append(j_privacy)
            history['j_utility'].append(j_utility)
            history['grad_norm_priv'].append(grad_norm_priv)
            history['grad_norm_util'].append(grad_norm_util)

            # Print progress
            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - "
                      f"J: {j_total:.4f} (priv: {j_privacy:.4f}, util: {j_utility:.4f}) "
                      f"EMA_priv: {ema_priv:.4f} "
                      f"Grad(priv/util): {grad_norm_priv:.2e}/{grad_norm_util:.2e}")

            # Early stopping on EMA(privacy_loss)
            if ema_priv < best_privacy:
                best_privacy = ema_priv
                patience_counter = 0

                # Save best checkpoint with EMA weights
                ema.apply_shadow()
                checkpoint = {
                    'epoch': epoch,
                    'vae_state_dict': vae.state_dict(),
                    'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                    'j_total': j_total,
                    'j_privacy': j_privacy,
                    'ema_privacy': ema_priv,
                    'config': vars(args)
                }

                # Only save attacker state if single-attacker mode (not multi-critic)
                if not isinstance(attacker, list):
                    checkpoint['attacker_state_dict'] = attacker.state_dict()

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
    final_checkpoint = {
        'vae_state_dict': vae.state_dict(),
        'config': vars(args)
    }

    # Only save attacker state if single-attacker mode (not multi-critic)
    if not isinstance(attacker, list):
        final_checkpoint['attacker_state_dict'] = attacker.state_dict()

    torch.save(final_checkpoint, output_dir / "final_model.pt")
    ema.restore()

    # Save history
    save_metrics_json(history, output_dir, "training_history.json")

    logger.close()

    print(f"\nUnlearning complete!")
    print(f"Best EMA(privacy): {best_privacy:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument("--config", type=str, default=None)

    # Data
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)

    # Unlearning objective
    parser.add_argument("--lambda_retain", type=float, default=10.0)

    # Adversarial setup
    parser.add_argument("--attacker_lr", type=float, default=0.0003)
    parser.add_argument("--vae_lr", type=float, default=0.0001)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_spectral_norm", type=bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Training schedule
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Auto-computed as ceil(|R|/B) if not specified")
    parser.add_argument("--privacy_repeats_k", type=int, default=8,
                        help="Number of privacy gradient accumulation steps")
    parser.add_argument("--pretrain_epochs", type=int, default=20)

    # Loss balancing
    parser.add_argument("--balance_mode", type=str, default="none",
                        choices=["none", "ema_ratio"],
                        help="Loss scale balancing method")

    # Privacy objective
    parser.add_argument("--use_max_privacy", type=bool, default=False,
                        help="Take max of F vs U and F vs R")
    parser.add_argument("--use_matched_unseen_loader", type=bool, default=False,
                        help="Use matched unseen negatives")
    parser.add_argument("--mmd_gamma", type=float, default=0.0,
                        help="Latent MMD weight")
    parser.add_argument("--alpha_f_kl", type=float, default=0.0,
                        help="F-only KL up-weight")

    # Multi-critic support
    parser.add_argument("--attacker_paths", type=str, nargs='+', default=None,
                        help="Paths to pre-trained attackers (multi-critic mode)")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=1)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    main(args)
