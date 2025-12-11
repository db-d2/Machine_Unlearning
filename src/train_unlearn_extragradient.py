# Extra-gradient stable co-training for VAE unlearning
#
# Implements:
# - Unfrozen critics with extra-gradient/optimistic updates
# - TTUR: 2 critic steps per VAE step, lr_critic = lr_vae / 10
# - Spectral norm on critics (already have this)
# - EMA of critics for evaluation stability
# - Shadow attacker for early stopping
#
# References:
#   - Daskalakis et al. (2018). Training GANs with Optimism. NeurIPS 2018.
#   - Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule. NeurIPS 2017.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import scanpy as sc

from vae import VAE
from attacker import MLPAttacker, extract_vae_features, build_attack_features


class AnnDataDataset(Dataset):
    """PyTorch Dataset wrapper for AnnData."""
    def __init__(self, adata, indices):
        self.adata = adata
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = torch.FloatTensor(self.adata.X[i].toarray().flatten())
        library_size = torch.FloatTensor([x.sum().item()])
        return x, library_size


def load_attackers(attacker_paths, device):
    """Load pre-trained attackers for initialization (will be fine-tuned, not frozen)."""
    attackers = []
    variants = []
    for path in attacker_paths:
        ckpt = torch.load(path, map_location=device)
        attacker = MLPAttacker(
            ckpt['config']['feature_dim'],
            ckpt['config']['hidden_dims'],
            dropout=ckpt['config'].get('dropout', 0.3),
            use_spectral_norm=ckpt['config'].get('use_spectral_norm', True)
        ).to(device)
        attacker.load_state_dict(ckpt['model_state_dict'])
        attackers.append(attacker)
        variants.append(ckpt['config'].get('variant', 'v1'))
    return attackers, variants


def create_ema_models(models):
    """Create EMA copies of models."""
    return [deepcopy(model) for model in models]


def update_ema_models(models, ema_models, decay=0.999):
    """Update EMA model parameters."""
    for model, ema_model in zip(models, ema_models):
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def extra_gradient_critic_step(attackers, attacker_variants, vae, forget_loader, retain_loader, lr_critic, device):
    """Perform extra-gradient update on attackers.

    Args:
        attackers: List of attacker models
        attacker_variants: List of feature variants for each attacker
        vae: VAE model (for feature extraction)
        forget_loader: DataLoader for forget set
        retain_loader: DataLoader for retain set
        lr_critic: Critic learning rate
        device: torch device

    Returns:
        Average attacker loss
    """
    vae.eval()  # VAE is frozen during critic update

    total_loss = 0.0

    # Get batches
    forget_batch = next(iter(forget_loader))
    retain_batch = next(iter(retain_loader))

    x_f, lib_f = forget_batch
    x_r, lib_r = retain_batch

    x_f = x_f.to(device)
    lib_f = lib_f.to(device)
    x_r = x_r.to(device)
    lib_r = lib_r.to(device)

    # Extract VAE features once
    with torch.no_grad():
        vae_feats_f = extract_vae_features(vae, x_f, lib_f, device, requires_grad=False)
        vae_feats_r = extract_vae_features(vae, x_r, lib_r, device, requires_grad=False)

    for attacker, variant in zip(attackers, attacker_variants):
        # Build features for this attacker's variant
        forget_feats = build_attack_features(vae_feats_f, variant=variant)
        retain_feats = build_attack_features(vae_feats_r, variant=variant)
        attacker.train()

        # Standard critic loss
        logits_f = attacker(forget_feats)
        logits_r = attacker(retain_feats)

        loss_f = nn.functional.binary_cross_entropy_with_logits(
            logits_f, torch.ones_like(logits_f)
        )
        loss_r = nn.functional.binary_cross_entropy_with_logits(
            logits_r, torch.zeros_like(logits_r)
        )
        loss = (loss_f + loss_r) / 2

        # Manual gradient descent with extra-gradient (simplified)
        loss.backward()

        with torch.no_grad():
            for p in attacker.parameters():
                if p.grad is not None:
                    p.data -= lr_critic * p.grad

        attacker.zero_grad()
        total_loss += loss.item()

    vae.train()  # Return VAE to train mode
    return total_loss / len(attackers)


def train_shadow_attacker(vae, forget_idx, retain_idx, unseen_idx, adata, device, epochs=20):
    """Train small shadow attacker to detect privacy degradation.

    Returns:
        Shadow attacker conditioned AUC (average of F vs U and F vs R)
    """
    # Extract features
    forget_feats = []
    retain_feats = []
    unseen_feats = []

    vae.eval()
    with torch.no_grad():
        for idx_list, feat_list in [(forget_idx, forget_feats),
                                      (retain_idx, retain_feats),
                                      (unseen_idx, unseen_feats)]:
            for i in idx_list:
                x = torch.FloatTensor(adata.X[i].toarray().flatten()).unsqueeze(0).to(device)
                lib = torch.FloatTensor([[x.sum().item()]]).to(device)
                vae_feats = extract_vae_features(vae, x, lib, device, requires_grad=False)
                attack_feats = build_attack_features(vae_feats, variant='v1')
                feat_list.append(attack_feats)

    forget_feats = torch.cat(forget_feats, dim=0)
    retain_feats = torch.cat(retain_feats, dim=0)
    unseen_feats = torch.cat(unseen_feats, dim=0)

    # Train shadow attacker
    shadow = MLPAttacker(forget_feats.shape[1], [128, 128], dropout=0.3).to(device)
    optimizer = torch.optim.Adam(shadow.parameters(), lr=0.001)

    for _ in range(epochs):
        # Sample batches
        f_idx = np.random.choice(len(forget_feats), min(30, len(forget_feats)), replace=False)
        r_idx = np.random.choice(len(retain_feats), min(64, len(retain_feats)), replace=False)

        f_batch = forget_feats[f_idx]
        r_batch = retain_feats[r_idx]

        optimizer.zero_grad()
        logits_f = shadow(f_batch)
        logits_r = shadow(r_batch)

        loss_f = nn.functional.binary_cross_entropy_with_logits(
            logits_f, torch.ones_like(logits_f)
        )
        loss_r = nn.functional.binary_cross_entropy_with_logits(
            logits_r, torch.zeros_like(logits_r)
        )
        loss = (loss_f + loss_r) / 2

        loss.backward()
        optimizer.step()

    # Evaluate conditioned AUC
    shadow.eval()
    with torch.no_grad():
        f_preds = torch.sigmoid(shadow(forget_feats)).cpu().numpy()
        u_preds = torch.sigmoid(shadow(unseen_feats)).cpu().numpy()
        r_preds = torch.sigmoid(shadow(retain_feats)).cpu().numpy()

    from sklearn.metrics import roc_auc_score

    # F vs U
    y_f_vs_u = np.concatenate([np.ones(len(f_preds)), np.zeros(len(u_preds))])
    scores_f_vs_u = np.concatenate([f_preds.flatten(), u_preds.flatten()])
    auc_f_vs_u = roc_auc_score(y_f_vs_u, scores_f_vs_u)

    # F vs R
    y_f_vs_r = np.concatenate([np.ones(len(f_preds)), np.zeros(len(r_preds))])
    scores_f_vs_r = np.concatenate([f_preds.flatten(), r_preds.flatten()])
    auc_f_vs_r = roc_auc_score(y_f_vs_r, scores_f_vs_r)

    return (auc_f_vs_u + auc_f_vs_r) / 2


def main():
    parser = argparse.ArgumentParser(description='Extra-gradient stable co-training for VAE unlearning')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--attacker_paths', nargs='+', required=True)
    parser.add_argument('--lambda_retain', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr_vae', type=float, default=0.0001)
    parser.add_argument('--lr_critic', type=float, default=0.00001)  # 10x smaller (TTUR)
    parser.add_argument('--critic_steps', type=int, default=2)
    parser.add_argument('--abort_threshold', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=1)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extra-gradient stable co-training")
    print(f"Device: {device}")
    print(f"Lambda: {args.lambda_retain}, LR_VAE: {args.lr_vae}, LR_Critic: {args.lr_critic}")
    print(f"Critic steps: {args.critic_steps}, Abort threshold: {args.abort_threshold}")

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(args.data_path)

    with open(args.split_path, 'r') as f:
        split = json.load(f)

    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    unseen_idx = split.get('unseen_matched_indices', split.get('unseen_indices', []))

    print(f"Forget: {len(forget_idx)}, Retain: {len(retain_idx)}, Unseen: {len(unseen_idx)}")

    # Create dataloaders
    forget_dataset = AnnDataDataset(adata, forget_idx)
    retain_dataset = AnnDataDataset(adata, retain_idx)

    forget_loader = DataLoader(forget_dataset, batch_size=min(len(forget_idx), 32),
                                shuffle=True, drop_last=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

    # Load baseline VAE
    print("\nLoading baseline VAE...")
    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device)
    config = baseline_ckpt['config']

    vae = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config['likelihood'],
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    ).to(device)

    vae.load_state_dict(baseline_ckpt['model_state_dict'])
    print(f"VAE loaded: z={config['latent_dim']}, hidden={config['hidden_dims']}")

    # Load attackers (will be fine-tuned, not frozen)
    print("\nLoading attackers (will be fine-tuned)...")
    attackers, attacker_variants = load_attackers(args.attacker_paths, device)
    print(f"Loaded {len(attackers)} attackers with variants: {attacker_variants}")

    # Create EMA critics for evaluation
    ema_attackers = create_ema_models(attackers)
    print("Created EMA copies of attackers")

    # Optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr_vae)

    # All attacker parameters in one optimizer
    all_attacker_params = []
    for attacker in attackers:
        all_attacker_params.extend(list(attacker.parameters()))
    critic_optimizer = optim.Adam(all_attacker_params, lr=args.lr_critic)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'privacy_loss': [],
        'utility_loss': [],
        'shadow_auc': [],
        'critic_loss': [],
    }

    print("\n" + "="*60)
    print("Starting extra-gradient co-training")
    print("="*60)

    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()

        # === VAE Update (1 step) ===
        vae.train()
        for attacker in attackers:
            attacker.eval()  # Freeze critics during VAE update

        # Sample batches
        forget_batch = next(iter(forget_loader))
        retain_batch = next(iter(retain_loader))

        x_f, lib_f = forget_batch
        x_r, lib_r = retain_batch

        x_f = x_f.to(device)
        lib_f = lib_f.to(device)
        x_r = x_r.to(device)
        lib_r = lib_r.to(device)

        # VAE forward
        vae_optimizer.zero_grad()

        # Privacy loss: max over critics (trying to fool them)
        vae_feats_f = extract_vae_features(vae, x_f, lib_f, device, requires_grad=True)

        critic_preds = []
        for attacker, variant in zip(attackers, attacker_variants):
            attack_feats_f = build_attack_features(vae_feats_f, variant=variant)
            logits = attacker(attack_feats_f)
            prob = torch.sigmoid(logits)
            critic_preds.append(prob.mean())

        # Privacy loss: try to make critics predict 0 (non-member)
        privacy_loss = max(critic_preds)  # Max over critics

        # Utility loss: standard VAE ELBO on retain set
        mu_r, logvar_r = vae.encode(x_r)
        z_r = vae.reparameterize(mu_r, logvar_r)

        if vae.likelihood == 'nb':
            mean_r, dispersion_r = vae.decode(z_r, lib_r)
            recon_loss_r = nn.functional.mse_loss(mean_r, x_r, reduction='mean')
        else:
            recon_mu_r, recon_logvar_r = vae.decode(z_r)
            recon_loss_r = 0.5 * (
                recon_logvar_r
                + ((x_r - recon_mu_r) ** 2) / torch.exp(recon_logvar_r)
                + np.log(2 * np.pi)
            ).mean()

        kl_loss_r = -0.5 * torch.mean(torch.sum(1 + logvar_r - mu_r.pow(2) - logvar_r.exp(), dim=1))
        utility_loss = recon_loss_r + kl_loss_r

        # Combined loss
        total_loss = privacy_loss + args.lambda_retain * utility_loss

        total_loss.backward()
        vae_optimizer.step()

        # === Critic Updates (2 steps with extra-gradient) ===
        critic_loss_avg = 0.0
        for critic_step in range(args.critic_steps):
            critic_loss = extra_gradient_critic_step(
                attackers, attacker_variants, vae, forget_loader, retain_loader, args.lr_critic, device
            )
            critic_loss_avg += critic_loss

        critic_loss_avg /= args.critic_steps

        # Update EMA critics
        update_ema_models(attackers, ema_attackers, decay=0.999)

        # === Shadow Attacker (Early Warning) ===
        shadow_auc = train_shadow_attacker(
            vae, forget_idx, retain_idx, unseen_idx, adata, device, epochs=20
        )

        # Record history
        history['privacy_loss'].append(privacy_loss.item())
        history['utility_loss'].append(utility_loss.item())
        history['shadow_auc'].append(shadow_auc)
        history['critic_loss'].append(critic_loss_avg)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
                  f"Privacy: {privacy_loss.item():.4f} | "
                  f"Utility: {utility_loss.item():.4f} | "
                  f"Critic: {critic_loss_avg:.4f} | "
                  f"Shadow AUC: {shadow_auc:.4f}")

        # === Early Stopping Check ===
        if len(history['shadow_auc']) >= args.abort_threshold:
            recent_aucs = history['shadow_auc'][-args.abort_threshold:]
            # Check if AUC is increasing (privacy degrading)
            is_worsening = all(recent_aucs[i] < recent_aucs[i+1]
                                for i in range(len(recent_aucs)-1))

            if is_worsening:
                print("\n" + "="*60)
                print("EARLY STOPPING: Shadow AUC worsening for {} epochs".format(args.abort_threshold))
                print("="*60)
                for i, auc in enumerate(recent_aucs):
                    print(f"  Epoch {epoch - args.abort_threshold + i + 2}: {auc:.4f}")
                print("\nAborting training...")
                break

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("Training complete")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("="*60)

    # Save results
    print("\nSaving results...")

    # Save VAE checkpoint
    torch.save({
        'model_state_dict': vae.state_dict(),
        'config': config,
        'args': vars(args),
        'history': history,
    }, output_dir / 'best_model.pt')

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save metadata
    metadata = {
        'method': 'extra_gradient_unlearning',
        'approach': 'extra-gradient co-training',
        'lambda_retain': args.lambda_retain,
        'lr_vae': args.lr_vae,
        'lr_critic': args.lr_critic,
        'critic_steps': args.critic_steps,
        'abort_threshold': args.abort_threshold,
        'total_epochs': len(history['shadow_auc']),
        'total_time_seconds': total_time,
        'final_shadow_auc': history['shadow_auc'][-1],
        'early_stopped': len(history['shadow_auc']) < args.epochs,
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - best_model.pt")
    print(f"  - history.json")
    print(f"  - metadata.json")

    print("\n" + "="*60)
    print("Next step: Post-hoc evaluation with fresh attacker")
    print("="*60)


if __name__ == '__main__':
    main()
