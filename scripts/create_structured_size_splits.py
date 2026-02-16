#!/usr/bin/env python3
"""
Create structured forget set splits of different sizes on PBMC.

Combines rare clusters to form larger structured forget sets:
  n=10:  10 cells from cluster 13
  n=30:  All 30 train cells from cluster 13 (baseline, already exists)
  n=50:  Cluster 13 (30) + 20 from cluster 12
  n=100: Cluster 13 (30) + cluster 12 (all train) + remainder from cluster 11

Uses only training-set cells. Generates k-NN matched negatives in latent space.

Usage:
    PYTHONPATH=src python scripts/create_structured_size_splits.py
"""

import sys
import json
import numpy as np
import torch
import scanpy as sc
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

SRC_DIR = Path(__file__).parent.parent / 'src'
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

DATA_PATH = BASE_DIR / 'data' / 'adata_processed.h5ad'
BASELINE_CHECKPOINT = BASE_DIR / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
EXISTING_SPLIT = BASE_DIR / 'outputs' / 'p1' / 'split_structured.json'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'p4' / 'extragradient_size' / 'splits'


def get_latent_embeddings(adata, model, device='cpu'):
    """Get VAE latent embeddings for all cells."""
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    x = torch.FloatTensor(np.asarray(X)).to(device)
    lib = x.sum(dim=1, keepdim=True)

    model.train(False)
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.cpu().numpy()


def compute_matched_negatives(forget_indices, unseen_indices, latent_embeddings, k=7):
    """Find k-NN matched negatives from unseen set for each forget cell."""
    forget_emb = latent_embeddings[forget_indices]
    unseen_emb = latent_embeddings[unseen_indices]

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(unseen_emb)
    distances, neighbor_idx = nn.kneighbors(forget_emb)

    matched = set()
    for row in neighbor_idx:
        for idx in row:
            matched.add(unseen_indices[idx])

    matched_list = sorted(matched)
    mean_dist = float(distances.mean())
    return matched_list, mean_dist


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Total cells: {adata.n_obs}")

    # Load existing split for train/unseen indices
    with open(EXISTING_SPLIT) as f:
        existing = json.load(f)

    train_indices = set(existing['retain_indices'] + existing['forget_indices'])
    all_indices = set(range(adata.n_obs))
    unseen_set = all_indices - train_indices
    unseen_indices_full = sorted(unseen_set)

    print(f"Train: {len(train_indices)}, Unseen: {len(unseen_indices_full)}")

    # Load VAE for latent embeddings
    print("Loading VAE for latent embeddings...")
    device = 'cpu'
    baseline_ckpt = torch.load(BASELINE_CHECKPOINT, map_location=device, weights_only=False)
    config = baseline_ckpt['config']

    from vae import VAE
    model = VAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        likelihood=config['likelihood'],
        dropout=config.get('dropout', 0.1),
        use_layer_norm=config.get('use_layer_norm', True)
    ).to(device)
    model.load_state_dict(baseline_ckpt['model_state_dict'])

    print("Computing latent embeddings...")
    latent = get_latent_embeddings(adata, model, device)

    # Identify train cells per cluster
    clusters_in_train = {}
    for idx in train_indices:
        cl = adata.obs['leiden'].iloc[idx]
        if cl not in clusters_in_train:
            clusters_in_train[cl] = []
        clusters_in_train[cl].append(idx)

    for cl in clusters_in_train:
        clusters_in_train[cl].sort()

    print("\nTrain cells per cluster (rare clusters):")
    for cl in sorted(clusters_in_train, key=lambda c: len(clusters_in_train[c])):
        n = len(clusters_in_train[cl])
        if n < 100:
            print(f"  Cluster {cl}: {n} train cells")

    cl13 = clusters_in_train['13']
    cl12 = clusters_in_train['12']
    cl11 = clusters_in_train['11']

    print(f"\nCluster 13: {len(cl13)} train cells")
    print(f"Cluster 12: {len(cl12)} train cells")
    print(f"Cluster 11: {len(cl11)} train cells")

    # Define forget sets
    np.random.seed(42)

    n50_from_cl12 = min(50 - len(cl13), len(cl12))
    n100_from_cl11 = min(100 - len(cl13) - len(cl12), len(cl11))

    sizes = {
        10: {
            'indices': sorted(np.random.choice(cl13, size=10, replace=False).tolist()),
            'description': 'Structured forget set: 10 cells from cluster 13 (megakaryocytes)',
            'clusters': ['13'],
        },
        50: {
            'indices': sorted(cl13 + sorted(np.random.choice(cl12, size=n50_from_cl12, replace=False).tolist())),
            'description': 'Structured forget set: cluster 13 ({}) + {} from cluster 12'.format(len(cl13), n50_from_cl12),
            'clusters': ['13', '12'],
        },
        100: {
            'indices': sorted(cl13 + cl12 + sorted(np.random.choice(cl11, size=n100_from_cl11, replace=False).tolist())),
            'description': 'Structured forget set: cluster 13 ({}) + cluster 12 ({}) + {} from cluster 11'.format(len(cl13), len(cl12), n100_from_cl11),
            'clusters': ['13', '12', '11'],
        },
    }

    for n, info in sizes.items():
        forget = info['indices']
        retain = sorted(train_indices - set(forget))

        print("\n" + "=" * 60)
        print("Size n={}: {} forget, {} retain".format(n, len(forget), len(retain)))
        print("  {}".format(info['description']))
        print("  Clusters: {}".format(info['clusters']))

        matched, mean_dist = compute_matched_negatives(
            forget, unseen_indices_full, latent, k=7
        )
        print("  Matched negatives: {} (mean k-NN dist: {:.4f})".format(len(matched), mean_dist))

        split = {
            'forget_indices': forget,
            'retain_indices': retain,
            'unseen_indices': unseen_indices_full,
            'n_forget': len(forget),
            'n_retain': len(retain),
            'n_unseen': len(unseen_indices_full),
            'description': info['description'],
            'forget_type': 'structured',
            'clusters': info['clusters'],
        }

        split_path = OUTPUT_DIR / 'split_structured_n{}.json'.format(n)
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        print("  Saved split: {}".format(split_path))

        matched_data = {
            'matched_indices': matched,
            'n_matched': len(matched),
            'k_neighbors': 7,
            'mean_knn_distance': mean_dist,
            'forget_size': len(forget),
        }
        matched_path = OUTPUT_DIR / 'matched_neg_structured_n{}.json'.format(n)
        with open(matched_path, 'w') as f:
            json.dump(matched_data, f, indent=2)
        print("  Saved matched negatives: {}".format(matched_path))

    print("\nAll splits saved to {}/".format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
