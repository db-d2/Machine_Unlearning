"""Task 4: Train VAE with z=8 and compute Fisher overlap.

Same architecture as baseline except latent_dim=8 instead of 32.
Same training hyperparameters and data split.
Single seed sufficient (generalization check, not multi-seed experiment).
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from vae import VAE, vae_loss

OUTPUTS = Path(__file__).parent.parent / 'outputs'
DAMPING = 1e-8


def cosine_sim(a, b):
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def effective_rank(f):
    p = f / f.sum()
    p = p.clamp(min=1e-30)
    return np.exp(-(p * p.log()).sum().item())


def top_k_overlap(a, b, k):
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / k


def layer_cat(name):
    if 'fc_mu' in name or 'fc_logvar' in name:
        return 'Bottleneck'
    if name.startswith('encoder'):
        return 'Encoder'
    if 'fc_mean' in name or 'fc_dispersion' in name:
        return 'Decoder output'
    if name.startswith('decoder'):
        return 'Decoder hidden'
    return 'Other'


class AnnDataDataset(Dataset):
    def __init__(self, adata, indices):
        self.adata = adata
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        row = self.adata.X[i]
        x = torch.FloatTensor(
            row.toarray().flatten() if hasattr(row, 'toarray')
            else np.asarray(row).flatten()
        )
        library_size = torch.FloatTensor([x.sum().item()])
        return x, library_size


def set_inference_mode(model):
    """Put model in inference mode (equivalent to .eval())."""
    model.train(False)


def compute_fisher_diagonal(model, dataloader, device, damping=DAMPING):
    """Batch Fisher diagonal (same methodology as NB26)."""
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    n_samples = 0

    for x_batch, lib_batch in dataloader:
        x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
        model.zero_grad()
        mu, logvar = model.encode(x_batch)
        z = model.reparameterize(mu, logvar)
        if model.likelihood == 'nb':
            mean, dispersion = model.decode(z, lib_batch)
            recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='sum')
        else:
            recon_mu, recon_logvar = model.decode(z)
            recon_loss = 0.5 * (
                recon_logvar + ((x_batch - recon_mu)**2) / torch.exp(recon_logvar)
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
    set_inference_mode(model)
    return fisher, n_samples


def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    print('Loading data...')
    adata = sc.read_h5ad(str(OUTPUTS.parent / 'data' / 'adata_processed.h5ad'))
    with open(OUTPUTS / 'p1' / 'split_structured.json') as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    unseen_idx = split['unseen_indices']
    all_train_idx = forget_idx + retain_idx
    print(f'Train: {len(all_train_idx)}, Val: {len(unseen_idx)}')

    # Create dataloaders
    train_ds = AnnDataDataset(adata, all_train_idx)
    val_ds = AnnDataDataset(adata, unseen_idx)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Initialize VAE with z=8
    model = VAE(
        input_dim=2000,
        hidden_dims=[1024, 512, 128],
        latent_dim=8,
        likelihood='nb',
        dropout=0.1,
        use_layer_norm=True
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'VAE (z=8): {n_params:,} params')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print('\nTraining...')
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        n_train = 0
        for x_batch, lib_batch in train_loader:
            x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, library_size=lib_batch)
            loss, recon, kl = vae_loss(x_batch, output, likelihood='nb', library_size=lib_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
            n_train += len(x_batch)

        # Validate
        set_inference_mode(model)
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x_batch, lib_batch in val_loader:
                x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
                output = model(x_batch, library_size=lib_batch)
                loss, _, _ = vae_loss(x_batch, output, likelihood='nb', library_size=lib_batch)
                val_loss += loss.item() * len(x_batch)
                n_val += len(x_batch)

        avg_train = train_loss / n_train
        avg_val = val_loss / n_val

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}: train={avg_train:.2f}, val={avg_val:.2f}, best={best_val_loss:.2f}')

    # Load best model
    model.load_state_dict(best_state)
    print(f'\nBest val loss: {best_val_loss:.2f}')

    # Save checkpoint
    out_dir = OUTPUTS / 'p6'
    out_dir.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'config': {
            'input_dim': 2000, 'latent_dim': 8,
            'hidden_dims': [1024, 512, 128],
            'likelihood': 'nb', 'dropout': 0.1,
            'use_layer_norm': True
        },
        'val_loss': best_val_loss,
    }, out_dir / 'vae_z8_best_model.pt')

    # Compute Fisher overlap
    forget_ds = AnnDataDataset(adata, forget_idx)
    retain_ds = AnnDataDataset(adata, retain_idx)
    forget_loader = DataLoader(forget_ds, batch_size=128, shuffle=False)
    retain_loader = DataLoader(retain_ds, batch_size=256, shuffle=False)

    print('\nComputing forget-set Fisher...')
    fisher_forget, n_f = compute_fisher_diagonal(model, forget_loader, device)
    print(f'  {n_f} samples')

    print('Computing retain-set Fisher...')
    fisher_retain, n_r = compute_fisher_diagonal(model, retain_loader, device)
    print(f'  {n_r} samples')

    # Per-layer analysis
    print('\n=== Per-layer Fisher Overlap ===')
    layer_data = []
    categories = {}

    for name in fisher_forget:
        ff = fisher_forget[name].flatten()
        fr = fisher_retain[name].flatten()
        n = len(ff)
        if n < 10:
            continue
        cos = cosine_sim(ff, fr)
        er_f = effective_rank(ff)
        er_r = effective_rank(fr)
        cat = layer_cat(name)
        layer_data.append({
            'name': name, 'n_params': n, 'cosine': cos,
            'eff_rank_f': er_f, 'eff_rank_r': er_r, 'category': cat
        })
        if cat not in categories:
            categories[cat] = {'f': [], 'r': []}
        categories[cat]['f'].append(ff)
        categories[cat]['r'].append(fr)
        print(f'  {name:50s} n={n:10,d} cos={cos:.4f} [{cat}]')

    # Global metrics
    ff_global = torch.cat([f.flatten() for f in fisher_forget.values()])
    fr_global = torch.cat([f.flatten() for f in fisher_retain.values()])
    global_cos = cosine_sim(ff_global, fr_global)
    log_corr = torch.corrcoef(
        torch.stack([ff_global.log(), fr_global.log()])
    )[0, 1].item()

    n_total = len(ff_global)
    top1 = top_k_overlap(ff_global, fr_global, int(0.01 * n_total))
    top10 = top_k_overlap(ff_global, fr_global, int(0.1 * n_total))

    # Per-category metrics
    cat_results = {}
    for cat in ['Encoder', 'Bottleneck', 'Decoder hidden', 'Decoder output']:
        if cat in categories:
            ff_c = torch.cat(categories[cat]['f'])
            fr_c = torch.cat(categories[cat]['r'])
            cat_results[cat] = {
                'cosine': cosine_sim(ff_c, fr_c),
                'n_params': len(ff_c),
                'eff_rank_ratio_forget': effective_rank(ff_c) / len(ff_c),
                'eff_rank_ratio_retain': effective_rank(fr_c) / len(fr_c),
            }

    elapsed = time.time() - start_time
    print(f'\n=== Summary (elapsed: {elapsed:.0f}s) ===')
    print(f'Global cosine: {global_cos:.4f}')
    print(f'Log-Fisher correlation: {log_corr:.4f}')
    print(f'Top-1% overlap: {top1:.4f}')
    print(f'Top-10% overlap: {top10:.4f}')
    for cat, r in cat_results.items():
        print(f'  {cat}: cosine={r["cosine"]:.4f}, params={r["n_params"]:,}')

    # Compare with baseline (z=32)
    baseline_path = OUTPUTS / 'p6' / 'fisher_overlap_results.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_cos = baseline['vae']['global_cosine']
        print(f'\nBaseline VAE (z=32) global cosine: {baseline_cos:.4f}')
        print(f'VAE (z=8) global cosine: {global_cos:.4f}')
        print(f'Change: {global_cos - baseline_cos:+.4f}')

    # Save results
    results = {
        'global_cosine': global_cos,
        'log_fisher_correlation': log_corr,
        'top_1pct_overlap': top1,
        'top_10pct_overlap': top10,
        'total_params': n_total,
        'global_eff_rank_forget': effective_rank(ff_global),
        'global_eff_rank_retain': effective_rank(fr_global),
        'per_category': cat_results,
        'per_layer': layer_data,
        'val_loss': best_val_loss,
        'metadata': {
            'latent_dim': 8,
            'hidden_dims': [1024, 512, 128],
            'n_forget': len(forget_idx),
            'n_retain': len(retain_idx),
            'damping': DAMPING,
            'epochs': 100,
            'lr': 1e-3,
            'seed': 42,
            'elapsed_seconds': elapsed,
        }
    }

    out_path = out_dir / 'fisher_overlap_z8.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
