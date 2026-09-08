#!/usr/bin/env python3
"""Moon et al. (2024) feature unlearning, adapted to a gene-expression VAE.

Reference:
    Moon, Cho, Kim. Feature Unlearning for Pre-trained GANs and VAEs.
    AAAI 2024, 21420-21428. arXiv:2303.05699.

Original method (images): collect generated samples that contain a target
feature (positive) and samples that do not (negative), build a target-feature
direction z_e = mean(positive latents) - mean(negative latents), and fine-tune
the generator so that feature-present latents reconstruct as their feature-
removed projection z_mod = z - relu(proj(z) - t) * z_e_hat, while feature-absent
latents keep their original reconstruction.

Adaptation to this study: the "feature" is membership in the structured forget
subpopulation (positive = forget set, negative = a retain sample). The image-only
perceptual (MS-SSIM) term is dropped; the reconstruction terms use L1 on the
decoder mean output, exactly as in the paper. The method fine-tunes the decoder
(the generator), leaving the encoder fixed, which is faithful to a generator
feature-unlearning method.

Usage:
    PYTHONPATH=src python src/train_moon_unlearn.py \
        --baseline_checkpoint outputs/p1/baseline/best_model.pt \
        --data_path data/adata_processed.h5ad \
        --split_path outputs/p1/split_structured.json \
        --output_dir outputs/p2/moon/seed42 --seed 42
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

from vae import VAE


def load_vae(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    keep = ['input_dim', 'latent_dim', 'hidden_dims', 'likelihood', 'dropout', 'use_layer_norm']
    model = VAE(**{k: v for k, v in config.items() if k in keep})
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def create_dataloader(X, indices, batch_size=256, shuffle=True):
    data = X[indices]
    library_size = data.sum(dim=1, keepdim=True)
    dataset = TensorDataset(data, library_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def mean_latent(model, X, indices, device, chunk=2048):
    """Deterministic mean latent (z = mu) over a set of cells, eval mode."""
    model.eval()
    mus = []
    for start in range(0, len(indices), chunk):
        idx = indices[start:start + chunk]
        x = X[idx].to(device)
        mu, _ = model.encode(x)
        mus.append(mu)
    return torch.cat(mus, 0)


def train_moon(baseline_checkpoint, data_path, split_path, output_dir,
               alpha=1.0, n_epochs=20, lr=1e-4, batch_size=256, seed=42,
               retain_cap=8000):
    """Moon feature unlearning on the decoder.

    z_e is fixed from the frozen baseline encoder. Each step encodes a batch,
    forms z_mod by projecting feature-present latents back to the threshold,
    and trains the decoder so g(z) matches f(z_mod) on feature-present cells and
    f(z) on feature-absent cells (soft-weighted by cosine similarity to z_e).
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)

    adata = sc.read_h5ad(data_path)
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    split = json.load(open(split_path))
    forget_idx = list(split['forget_indices'])
    retain_idx = list(split['retain_indices'])
    print(f"Data: {X.shape}, Forget: {len(forget_idx)}, Retain: {len(retain_idx)}, Device: {device}", flush=True)

    # Frozen baseline (the generator f, and the encoder that fixes z_e)
    model, config = load_vae(baseline_checkpoint, device)
    model = model.to(device)
    frozen = deepcopy(model)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    # Target-feature direction z_e from the frozen encoder (deterministic z = mu)
    mu_forget = mean_latent(frozen, X, forget_idx, device).mean(0)
    retain_sample = retain_idx if len(retain_idx) <= retain_cap else list(
        np.random.choice(retain_idx, size=retain_cap, replace=False))
    mu_retain = mean_latent(frozen, X, retain_sample, device).mean(0)
    z_e = (mu_forget - mu_retain)
    z_e_hat = z_e / (z_e.norm() + 1e-8)

    # Threshold t: midpoint of mean forget/retain projections onto z_e_hat
    proj_forget = (mean_latent(frozen, X, forget_idx, device) @ z_e_hat).mean()
    proj_retain = (mean_latent(frozen, X, retain_sample, device) @ z_e_hat).mean()
    t = 0.5 * (proj_forget + proj_retain)
    print(f"z_e norm={z_e.norm():.3f}, proj_forget={proj_forget:.3f}, "
          f"proj_retain={proj_retain:.3f}, t={t:.3f}", flush=True)

    # Train the decoder (generator) only; encoder is fixed
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr)

    # Feature-present cells (forget) are rare, so oversample them alongside retain
    train_loader = create_dataloader(X, forget_idx + retain_idx, batch_size=batch_size, shuffle=True)

    history = {'unlearn': [], 'recon': [], 'total': []}
    for epoch in range(n_epochs):
        model.train()
        e_un = e_re = 0.0
        n = 0
        for x, lib in train_loader:
            x = x.to(device)
            lib = lib.to(device)
            with torch.no_grad():
                mu, _ = frozen.encode(x)          # z fixed by frozen encoder
                z = mu
                proj = z @ z_e_hat                 # scalar projection
                cos = (z @ z_e) / (z.norm(dim=1) * z_e.norm() + 1e-8)
                sim = cos.clamp(0.0, 1.0).unsqueeze(1)          # feature-present weight
                z_mod = z - torch.relu(proj - t).unsqueeze(1) * z_e_hat.unsqueeze(0)
                mean_f, _ = frozen.decode(z, lib)              # original reconstruction
                mean_f_mod, _ = frozen.decode(z_mod, lib)      # feature-removed target

            optimizer.zero_grad()
            mean_g, _ = model.decode(z, lib)
            l1_mod = (mean_g - mean_f_mod).abs().sum(1, keepdim=True)
            l1_keep = (mean_g - mean_f).abs().sum(1, keepdim=True)
            l_unlearn = (sim * l1_mod).mean()
            l_recon = ((1.0 - sim) * l1_keep).mean()
            loss = alpha * l_unlearn + l_recon
            loss.backward()
            optimizer.step()
            e_un += l_unlearn.item()
            e_re += l_recon.item()
            n += 1

        history['unlearn'].append(e_un / n)
        history['recon'].append(e_re / n)
        history['total'].append((alpha * e_un + e_re) / n)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: unlearn={e_un/n:.3f}, recon={e_re/n:.3f}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'method': 'moon',
        'seed': seed,
        'alpha': alpha,
        'n_epochs': n_epochs,
        'threshold': float(t),
    }, output_dir / 'best_model.pt')
    json.dump(history, open(output_dir / 'history.json', 'w'), indent=2)
    print(f"Saved to {output_dir / 'best_model.pt'}", flush=True)
    return output_dir / 'best_model.pt'


def main():
    p = argparse.ArgumentParser(description='Moon et al. (2024) feature unlearning for VAEs')
    p.add_argument('--baseline_checkpoint', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--split_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--alpha', type=float, default=1.0, help='Weight on the unlearning term')
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train_moon(args.baseline_checkpoint, args.data_path, args.split_path, args.output_dir,
               alpha=args.alpha, n_epochs=args.n_epochs, lr=args.lr,
               batch_size=args.batch_size, seed=args.seed)


if __name__ == '__main__':
    main()
