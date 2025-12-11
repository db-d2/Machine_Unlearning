"""Create and save train/forget/unseen splits for unlearning experiments."""

import json
import numpy as np
import scanpy as sc
from pathlib import Path

# Load data
adata = sc.read_h5ad("data/adata_processed.h5ad")
print(f"Data shape: {adata.shape}")

# Get Leiden clusters
if 'leiden' not in adata.obs.columns:
    print("Computing Leiden clustering...")
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=0.5)

# Split into train/test first (85/15)
n_cells = adata.n_obs
n_train = int(0.85 * n_cells)

np.random.seed(42)
indices = np.arange(n_cells)
np.random.shuffle(indices)

train_indices = indices[:n_train]
unseen_indices = indices[n_train:]

print(f"Train: {len(train_indices)}, Unseen: {len(unseen_indices)}")

# Structured forget set: entire Cluster 13
cluster_13_mask = adata.obs['leiden'] == '13'
cluster_13_indices = np.where(cluster_13_mask)[0]

# Intersect with train indices
forget_structured = np.intersect1d(cluster_13_indices, train_indices)
retain_structured = np.setdiff1d(train_indices, forget_structured)

print(f"\nStructured split:")
print(f"  Forget (Cluster 13): {len(forget_structured)}")
print(f"  Retain: {len(retain_structured)}")
print(f"  Unseen: {len(unseen_indices)}")

# Save structured split
split_structured = {
    'forget_indices': forget_structured.tolist(),
    'retain_indices': retain_structured.tolist(),
    'unseen_indices': unseen_indices.tolist(),
    'n_forget': int(len(forget_structured)),
    'n_retain': int(len(retain_structured)),
    'n_unseen': int(len(unseen_indices)),
    'description': 'Structured forget set: entire Cluster 13'
}

Path("outputs/p1").mkdir(parents=True, exist_ok=True)
with open("outputs/p1/split_structured.json", 'w') as f:
    json.dump(split_structured, f, indent=2)

print("Saved outputs/p1/split_structured.json")

# Scattered forget set: random 35 cells from train
np.random.seed(42)
forget_scattered = np.random.choice(train_indices, size=35, replace=False)
retain_scattered = np.setdiff1d(train_indices, forget_scattered)

print(f"\nScattered split:")
print(f"  Forget (random): {len(forget_scattered)}")
print(f"  Retain: {len(retain_scattered)}")
print(f"  Unseen: {len(unseen_indices)}")

# Save scattered split
split_scattered = {
    'forget_indices': forget_scattered.tolist(),
    'retain_indices': retain_scattered.tolist(),
    'unseen_indices': unseen_indices.tolist(),
    'n_forget': int(len(forget_scattered)),
    'n_retain': int(len(retain_scattered)),
    'n_unseen': int(len(unseen_indices)),
    'description': 'Scattered forget set: random 35 cells'
}

with open("outputs/p1/split_scattered.json", 'w') as f:
    json.dump(split_scattered, f, indent=2)

print("Saved outputs/p1/split_scattered.json")

