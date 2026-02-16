#!/usr/bin/env python3
"""SCRUB (Selective and Continuous Removal of Unlearned Biases) for VAEs.

Teacher-student distillation where the student matches the teacher on
retain data and diverges from the teacher on forget data.

Reference:
    Kurmanji et al. (2023). Towards Unbounded Machine Unlearning.
    NeurIPS 2023.

Adaptation for VAEs:
    - Teacher = frozen copy of baseline VAE
    - Student = copy of baseline VAE (being updated)
    - Retain loss: KL(student_posterior || teacher_posterior) + recon matching
    - Forget loss: -KL(student_posterior || teacher_posterior) (maximize divergence)
    - Alternating optimization: forget steps then retain steps per epoch

Usage:
    PYTHONPATH=src python src/train_scrub.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/scrub/seed42 \
        --seed 42
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
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


def kl_between_gaussians(mu1, logvar1, mu2, logvar2):
    """KL(N(mu1, sigma1) || N(mu2, sigma2)) per sample.

    Returns tensor of shape (batch_size,).
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    kl = 0.5 * (
        (logvar2 - logvar1)
        + var1 / var2
        + (mu1 - mu2).pow(2) / var2
        - 1
    ).sum(dim=1)
    return kl


def scrub_forget_step(student, teacher, x, lib_size, alpha_forget=1.0):
    """One step of forget-phase: maximize divergence from teacher on forget data.

    Loss = -alpha * KL(student || teacher)

    Minimizing this pushes the student's posterior away from the teacher's.
    """
    # Student forward
    mu_s, logvar_s = student.encode(x)

    # Teacher forward (no grad)
    with torch.no_grad():
        mu_t, logvar_t = teacher.encode(x)

    # KL between student and teacher posteriors
    kl_div = kl_between_gaussians(mu_s, logvar_s, mu_t, logvar_t)

    # Negative KL: maximize divergence
    loss = -alpha_forget * kl_div.mean()

    return loss


def scrub_retain_step(student, teacher, x, lib_size, alpha_retain=1.0):
    """One step of retain-phase: match teacher on retain data.

    Loss = alpha * KL(student || teacher) + ELBO(student, x)

    Minimizing this keeps the student close to the teacher on retain data
    while maintaining good reconstruction.
    """
    # Student forward
    mu_s, logvar_s = student.encode(x)

    # Teacher forward (no grad)
    with torch.no_grad():
        mu_t, logvar_t = teacher.encode(x)

    # KL between student and teacher posteriors
    kl_match = kl_between_gaussians(mu_s, logvar_s, mu_t, logvar_t)

    # Student ELBO (standard VAE loss for reconstruction quality)
    output = student(x, library_size=lib_size)
    elbo_loss, _, _ = vae_loss(x, output, likelihood='nb', beta=1.0)

    # Combined: match teacher + maintain ELBO
    loss = alpha_retain * kl_match.mean() + elbo_loss

    return loss


def train_scrub(baseline_checkpoint, data_path, split_path, output_dir,
                alpha_forget=1.0, alpha_retain=1.0,
                n_epochs=20, forget_steps_per_epoch=5,
                retain_steps_per_epoch=10,
                lr_forget=1e-4, lr_retain=1e-4,
                max_grad_norm=1.0,
                finetune_epochs=10, finetune_lr=1e-4, patience=10,
                batch_size=256, seed=42):
    """Run SCRUB unlearning.

    Alternating optimization:
    1. For each epoch:
       a. Forget phase: N steps of gradient descent on -KL(student || teacher)
       b. Retain phase: M steps of gradient descent on KL(student || teacher) + ELBO
    2. Optional: fine-tune on retain set for reconstruction quality

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

    # Load baseline as teacher (frozen)
    teacher, config = load_vae(baseline_checkpoint, device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student starts as copy of baseline
    student, _ = load_vae(baseline_checkpoint, device)
    student = student.to(device)

    # Separate optimizers for forget and retain phases
    optimizer_forget = optim.Adam(student.parameters(), lr=lr_forget)
    optimizer_retain = optim.Adam(student.parameters(), lr=lr_retain)

    # SCRUB alternating training
    print(f"SCRUB training: {n_epochs} epochs, "
          f"{forget_steps_per_epoch} forget + {retain_steps_per_epoch} retain steps/epoch")
    print(f"  alpha_forget={alpha_forget}, alpha_retain={alpha_retain}")

    history = {'forget_loss': [], 'retain_loss': []}
    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    for epoch in range(n_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0

        # Phase A: Forget steps
        student.train()
        for step in range(forget_steps_per_epoch):
            try:
                x_f, lib_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                x_f, lib_f = next(forget_iter)
            x_f = x_f.to(device)
            lib_f = lib_f.to(device)

            optimizer_forget.zero_grad()
            loss = scrub_forget_step(student, teacher, x_f, lib_f,
                                     alpha_forget=alpha_forget)

            if torch.isnan(loss):
                print(f"  NaN in forget step at epoch {epoch+1}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer_forget.step()
            epoch_forget_loss += loss.item()

        # Phase B: Retain steps
        for step in range(retain_steps_per_epoch):
            try:
                x_r, lib_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x_r, lib_r = next(retain_iter)
            x_r = x_r.to(device)
            lib_r = lib_r.to(device)

            optimizer_retain.zero_grad()
            loss = scrub_retain_step(student, teacher, x_r, lib_r,
                                     alpha_retain=alpha_retain)

            if torch.isnan(loss):
                print(f"  NaN in retain step at epoch {epoch+1}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer_retain.step()
            epoch_retain_loss += loss.item()

        avg_f = epoch_forget_loss / max(forget_steps_per_epoch, 1)
        avg_r = epoch_retain_loss / max(retain_steps_per_epoch, 1)
        history['forget_loss'].append(avg_f)
        history['retain_loss'].append(avg_r)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: forget={avg_f:.4f}, retain={avg_r:.4f}")

    model = student

    # Optional fine-tuning phase
    if finetune_epochs > 0:
        val_size = min(1000, len(retain_idx) // 10)
        val_idx = np.random.choice(retain_idx, size=val_size, replace=False)
        val_loader = create_dataloader(X, val_idx, batch_size=batch_size,
                                       shuffle=False)

        print(f"Fine-tuning on retain set ({finetune_epochs} epochs, "
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
        'method': 'scrub',
        'seed': seed,
        'alpha_forget': alpha_forget,
        'alpha_retain': alpha_retain,
        'scrub_epochs': n_epochs,
        'forget_steps_per_epoch': forget_steps_per_epoch,
        'retain_steps_per_epoch': retain_steps_per_epoch,
        'finetune_epochs': len(ft_history['train']) if ft_history else 0,
        'best_val_loss': best_val_loss,
    }, output_dir / 'best_model.pt')

    history['finetune'] = ft_history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir / 'best_model.pt'}")
    return output_dir / 'best_model.pt'


def main():
    parser = argparse.ArgumentParser(
        description='SCRUB unlearning for VAEs')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha_forget', type=float, default=1.0,
                        help='Weight for forget divergence loss')
    parser.add_argument('--alpha_retain', type=float, default=1.0,
                        help='Weight for retain matching loss')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of SCRUB epochs')
    parser.add_argument('--forget_steps_per_epoch', type=int, default=5)
    parser.add_argument('--retain_steps_per_epoch', type=int, default=10)
    parser.add_argument('--lr_forget', type=float, default=1e-4)
    parser.add_argument('--lr_retain', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_scrub(
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        split_path=args.split_path,
        output_dir=args.output_dir,
        alpha_forget=args.alpha_forget,
        alpha_retain=args.alpha_retain,
        n_epochs=args.n_epochs,
        forget_steps_per_epoch=args.forget_steps_per_epoch,
        retain_steps_per_epoch=args.retain_steps_per_epoch,
        lr_forget=args.lr_forget,
        lr_retain=args.lr_retain,
        max_grad_norm=args.max_grad_norm,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
