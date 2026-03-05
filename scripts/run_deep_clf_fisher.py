"""Task 3: Train deep MLP classifier on raw gene expression and compute Fisher overlap.

Architecture: 2000 -> [512, 128] -> 14 with LayerNorm + Dropout(0.1).
This has ~1.09M params (comparable to VAE scale) and shared hidden layers.

Expected result: shared hidden layers have high Fisher overlap (~0.2-0.4, like VAE),
while the class-specific output layer has low overlap (~0.02, like linear probe).
This proves the point is shared vs class-specific structure, not model size.
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

OUTPUTS = Path(__file__).parent.parent / 'outputs'
DAMPING = 1e-8


def cosine_sim(a, b):
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def effective_rank(f):
    p = f / f.sum()
    p = p.clamp(min=1e-30)
    return np.exp(-(p * p.log()).sum().item())


class GeneExprDataset(Dataset):
    """Dataset returning raw gene expression and cluster labels."""

    def __init__(self, adata, indices):
        X = adata.X[indices]
        self.X = torch.FloatTensor(
            X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        )
        self.y = torch.LongTensor(
            adata.obs['leiden'].values[indices].astype(int)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepMLPClassifier(nn.Module):
    """Deep MLP classifier with shared hidden layers matching VAE decoder dims."""

    def __init__(self, input_dim=2000, hidden_dims=None, n_classes=14, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        h = self.hidden(x)
        return self.output(h)


def layer_category(name):
    """Classify parameter as shared hidden or class-specific output."""
    if name.startswith('output'):
        return 'Class-specific output'
    return 'Shared hidden'


def compute_fisher_batch(model, dataloader, device, damping=DAMPING):
    """Batch Fisher diagonal (same methodology as NB26 VAE Fisher)."""
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)
    n_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        model.zero_grad()
        logits = model(x_batch)
        loss = nn.functional.cross_entropy(logits, y_batch, reduction='sum')
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data ** 2
        n_samples += x_batch.size(0)

    for name in fisher:
        fisher[name] /= n_samples
        fisher[name] += damping
    model.eval()
    return fisher, n_samples


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    print('Loading data...')
    adata = sc.read_h5ad(str(OUTPUTS.parent / 'data' / 'adata_processed.h5ad'))
    with open(OUTPUTS / 'p1' / 'split_structured.json') as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    all_train_idx = forget_idx + retain_idx
    print(f'Forget: {len(forget_idx)}, Retain: {len(retain_idx)}, Total: {len(all_train_idx)}')

    n_classes = int(adata.obs['leiden'].values[all_train_idx].astype(int).max()) + 1
    print(f'Classes: {n_classes}')

    # Create datasets
    all_ds = GeneExprDataset(adata, all_train_idx)
    all_loader = DataLoader(all_ds, batch_size=256, shuffle=True)

    # Train deep MLP classifier
    model = DeepMLPClassifier(
        input_dim=2000, hidden_dims=[512, 128],
        n_classes=n_classes, dropout=0.1
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Deep MLP classifier: {n_params:,} params')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print('\nTraining...')
    for epoch in range(200):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x_batch, y_batch in all_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(x_batch)
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.4f}')

    # Final accuracy
    model.train()  # keep in train mode for consistency
    test_loader = DataLoader(all_ds, batch_size=512, shuffle=False)
    model_eval_mode = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(x_batch)
    accuracy = correct / total
    print(f'\nFinal training accuracy: {accuracy:.4f}')

    # Compute Fisher on forget and retain subsets
    forget_ds = GeneExprDataset(adata, forget_idx)
    retain_ds = GeneExprDataset(adata, retain_idx)
    forget_loader = DataLoader(forget_ds, batch_size=128, shuffle=False)
    retain_loader = DataLoader(retain_ds, batch_size=256, shuffle=False)

    print('\nComputing forget-set Fisher...')
    fisher_forget, n_f = compute_fisher_batch(model, forget_loader, device)
    print(f'  {n_f} samples')

    print('Computing retain-set Fisher...')
    fisher_retain, n_r = compute_fisher_batch(model, retain_loader, device)
    print(f'  {n_r} samples')

    # Compute overlap metrics
    print('\n=== Per-layer Fisher Overlap ===')
    layer_data = []
    categories = {}

    for name in fisher_forget:
        ff = fisher_forget[name].flatten()
        fr = fisher_retain[name].flatten()
        n = len(ff)
        cos = cosine_sim(ff, fr)
        er_f = effective_rank(ff)
        er_r = effective_rank(fr)
        cat = layer_category(name)
        layer_data.append({
            'name': name, 'n_params': n, 'cosine': cos,
            'eff_rank_f': er_f, 'eff_rank_r': er_r, 'category': cat
        })
        if cat not in categories:
            categories[cat] = {'f': [], 'r': []}
        categories[cat]['f'].append(ff)
        categories[cat]['r'].append(fr)
        print(f'  {name:40s} n={n:10,d} cos={cos:.4f} [{cat}]')

    # Global metrics
    ff_global = torch.cat([f.flatten() for f in fisher_forget.values()])
    fr_global = torch.cat([f.flatten() for f in fisher_retain.values()])
    global_cos = cosine_sim(ff_global, fr_global)
    log_corr = torch.corrcoef(
        torch.stack([ff_global.log(), fr_global.log()])
    )[0, 1].item()

    # Per-category metrics
    cat_results = {}
    for cat, data in categories.items():
        ff_c = torch.cat(data['f'])
        fr_c = torch.cat(data['r'])
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
    for cat, r in cat_results.items():
        print(f'  {cat}: cosine={r["cosine"]:.4f}, params={r["n_params"]:,}')

    # Save results
    results = {
        'model': 'deep_mlp_classifier',
        'architecture': '2000 -> [512, 128] -> 14',
        'total_params': int(n_params),
        'accuracy': accuracy,
        'global_cosine': global_cos,
        'log_fisher_correlation': log_corr,
        'per_category': cat_results,
        'per_layer': layer_data,
        'metadata': {
            'n_forget': len(forget_idx),
            'n_retain': len(retain_idx),
            'damping': DAMPING,
            'hidden_dims': [512, 128],
            'dropout': 0.1,
            'epochs': 200,
            'lr': 1e-3,
            'elapsed_seconds': elapsed,
        }
    }

    out_path = OUTPUTS / 'p6' / 'fisher_overlap_deep_clf.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
