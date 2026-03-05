"""Fisher information overlap analysis between forget and retain sets.

Computes diagonal Fisher approximation for each set separately, then measures
overlap to quantify how much forget-set influence is entangled with retain-set
influence in parameter space. High overlap means selective parameter perturbation
will inevitably damage retain performance.

Usage:
    PYTHONPATH=src python scripts/fisher_overlap_analysis.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import scanpy as sc

from vae import VAE


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
        x = torch.FloatTensor(
            row.toarray().flatten() if hasattr(row, 'toarray') else np.asarray(row).flatten()
        )
        library_size = torch.FloatTensor([x.sum().item()])
        return x, library_size


def compute_fisher_diagonal(model, dataloader, device, damping=1e-5):
    """Compute diagonal Fisher approximation.

    F_ii = E[(dL/d theta_i)^2] averaged over the dataset.
    Identical to the implementation in src/train_fisher_unlearn.py.
    """
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    n_samples = 0
    for x_batch, lib_batch in dataloader:
        x_batch = x_batch.to(device)
        lib_batch = lib_batch.to(device)
        model.zero_grad()

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
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2

        n_samples += x_batch.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping

    return fisher, n_samples


def flatten_fisher(fisher):
    """Flatten a Fisher dict into a single 1-D vector."""
    return torch.cat([f.flatten() for f in fisher.values()])


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D tensors."""
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    return (dot / (norm_a * norm_b)).item()


def effective_rank(f):
    """Effective rank = exp(entropy of normalized Fisher).

    Measures how many parameters carry significant Fisher weight.
    High effective rank = influence is spread across many parameters.
    Low effective rank = influence is concentrated in few parameters.
    """
    f_flat = f.flatten()
    p = f_flat / f_flat.sum()
    # Clip to avoid log(0)
    p = p.clamp(min=1e-30)
    entropy = -(p * p.log()).sum().item()
    return np.exp(entropy)


def top_k_overlap(f_a, f_b, k):
    """Fraction of top-k parameters shared between two Fisher vectors."""
    top_a = set(torch.topk(f_a.flatten(), k).indices.tolist())
    top_b = set(torch.topk(f_b.flatten(), k).indices.tolist())
    return len(top_a & top_b) / k


def layer_category(name):
    """Categorize a parameter name into encoder/decoder/bottleneck."""
    if name.startswith('encoder'):
        if 'fc_mu' in name or 'fc_logvar' in name:
            return 'bottleneck'
        return 'encoder'
    elif name.startswith('decoder'):
        if 'fc_mean' in name or 'fc_dispersion' in name:
            return 'decoder_output'
        return 'decoder_hidden'
    return 'other'


def compute_overlap_metrics(f_forget, f_retain, name="global"):
    """Compute all overlap metrics between two Fisher vectors."""
    ff = f_forget.flatten()
    fr = f_retain.flatten()
    n_params = len(ff)

    metrics = {
        'name': name,
        'n_params': n_params,
        'cosine_similarity': cosine_similarity(ff, fr),
        'effective_rank_forget': effective_rank(ff),
        'effective_rank_retain': effective_rank(fr),
        'effective_rank_ratio_forget': effective_rank(ff) / n_params,
        'effective_rank_ratio_retain': effective_rank(fr) / n_params,
        'fisher_norm_forget': torch.norm(ff).item(),
        'fisher_norm_retain': torch.norm(fr).item(),
        'fisher_mean_forget': ff.mean().item(),
        'fisher_mean_retain': fr.mean().item(),
    }

    # Top-k overlap at various thresholds
    for k_frac in [0.001, 0.01, 0.1]:
        k = max(1, int(n_params * k_frac))
        metrics[f'top_{k_frac}_overlap'] = top_k_overlap(ff, fr, k)

    # Pearson correlation of log Fisher values
    log_ff = torch.log(ff.clamp(min=1e-30))
    log_fr = torch.log(fr.clamp(min=1e-30))
    corr = torch.corrcoef(torch.stack([log_ff, log_fr]))[0, 1].item()
    metrics['log_fisher_correlation'] = corr

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Fisher overlap analysis')
    parser.add_argument('--data_path', type=str, default='data/adata_processed.h5ad')
    parser.add_argument('--split_path', type=str, default='outputs/p1/split_structured.json')
    parser.add_argument('--matched_neg_path', type=str,
                        default='outputs/p1.5/s1_matched_negatives.json')
    parser.add_argument('--checkpoint', type=str, default='outputs/p1/baseline/best_model.pt')
    parser.add_argument('--output_dir', type=str, default='outputs/p6')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--damping', type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"  Data shape: {adata.shape}")

    # Load splits
    with open(args.split_path) as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    print(f"  Forget set: {len(forget_idx)} cells")
    print(f"  Retain set: {len(retain_idx)} cells")

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    vae_args = {k: v for k, v in config.items()
                if k in ['input_dim', 'latent_dim', 'hidden_dims', 'likelihood',
                          'dropout', 'use_layer_norm']}
    model = VAE(**vae_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataloaders
    forget_ds = AnnDataDataset(adata, forget_idx)
    retain_ds = AnnDataDataset(adata, retain_idx)
    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=False)

    # Compute Fisher diagonals
    print("\nComputing forget-set Fisher...")
    t0 = time.time()
    fisher_forget, n_forget = compute_fisher_diagonal(model, forget_loader, device, args.damping)
    t_forget = time.time() - t0
    print(f"  Done in {t_forget:.1f}s ({n_forget} samples)")

    print("\nComputing retain-set Fisher...")
    t0 = time.time()
    fisher_retain, n_retain = compute_fisher_diagonal(model, retain_loader, device, args.damping)
    t_retain = time.time() - t0
    print(f"  Done in {t_retain:.1f}s ({n_retain} samples)")

    # Global overlap metrics
    print("\n=== Global Overlap Metrics ===")
    ff_global = flatten_fisher(fisher_forget)
    fr_global = flatten_fisher(fisher_retain)
    global_metrics = compute_overlap_metrics(ff_global, fr_global, "global")
    print(f"  Cosine similarity:     {global_metrics['cosine_similarity']:.4f}")
    print(f"  Effective rank forget: {global_metrics['effective_rank_forget']:.0f} "
          f"({global_metrics['effective_rank_ratio_forget']:.4f})")
    print(f"  Effective rank retain: {global_metrics['effective_rank_retain']:.0f} "
          f"({global_metrics['effective_rank_ratio_retain']:.4f})")
    print(f"  Log-Fisher correlation: {global_metrics['log_fisher_correlation']:.4f}")
    print(f"  Top-0.1% overlap:      {global_metrics['top_0.001_overlap']:.4f}")
    print(f"  Top-1% overlap:        {global_metrics['top_0.01_overlap']:.4f}")
    print(f"  Top-10% overlap:       {global_metrics['top_0.1_overlap']:.4f}")

    # Per-layer overlap metrics
    print("\n=== Per-Layer Overlap Metrics ===")
    layer_metrics = {}
    for name in fisher_forget:
        if fisher_forget[name].numel() < 10:
            # Skip tiny bias terms
            continue
        metrics = compute_overlap_metrics(
            fisher_forget[name], fisher_retain[name], name
        )
        layer_metrics[name] = metrics
        category = layer_category(name)
        print(f"  {name:45s}  cos={metrics['cosine_similarity']:.4f}  "
              f"eff_rank_f={metrics['effective_rank_forget']:8.0f}  "
              f"category={category}")

    # Category-level aggregation
    print("\n=== Category-Level Overlap ===")
    categories = {}
    for name in fisher_forget:
        cat = layer_category(name)
        if cat not in categories:
            categories[cat] = {'forget': [], 'retain': []}
        categories[cat]['forget'].append(fisher_forget[name].flatten())
        categories[cat]['retain'].append(fisher_retain[name].flatten())

    category_metrics = {}
    for cat, tensors in categories.items():
        ff_cat = torch.cat(tensors['forget'])
        fr_cat = torch.cat(tensors['retain'])
        metrics = compute_overlap_metrics(ff_cat, fr_cat, cat)
        category_metrics[cat] = metrics
        print(f"  {cat:20s}  cos={metrics['cosine_similarity']:.4f}  "
              f"n_params={metrics['n_params']:>10,}  "
              f"eff_rank_ratio_f={metrics['effective_rank_ratio_forget']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'fisher_overlap.json'

    results = {
        'metadata': {
            'data_path': args.data_path,
            'split_path': args.split_path,
            'checkpoint': args.checkpoint,
            'n_forget': n_forget,
            'n_retain': n_retain,
            'damping': args.damping,
            'total_params': ff_global.numel(),
            'compute_time_forget_s': t_forget,
            'compute_time_retain_s': t_retain,
        },
        'global': global_metrics,
        'per_layer': layer_metrics,
        'per_category': category_metrics,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save raw Fisher tensors for figure generation
    torch.save({
        'fisher_forget': fisher_forget,
        'fisher_retain': fisher_retain,
    }, output_dir / 'fisher_tensors.pt')
    print(f"Fisher tensors saved to {output_dir / 'fisher_tensors.pt'}")


if __name__ == '__main__':
    main()
