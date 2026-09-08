#!/usr/bin/env python3
"""Held-out utility for the Moon (2024) feature-unlearning checkpoints.

score_moon.py only records the privacy scores. The paper and the figure script quote
Moon's ELBO and marker correlation, so those must also trace to a file. This reuses
the canonical utility evaluator from eval_multiseed on the 4,964-cell held-out set
and writes per-seed values plus the mean.

Run: PYTHONPATH=src python scripts/score_moon_utility.py
Output: outputs/p2/moon/utility.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'  # utility is deterministic on CPU; matches the canonical scoring device
from eval_multiseed import (load_vae_model, evaluate_utility, MARKER_GENES,
                            DATA_PATH, SPLIT_PATH)

OUT = BASE / 'outputs' / 'p2' / 'moon' / 'utility.json'
FIELDS = ['elbo', 'kl', 'marker_r', 'ari', 'silhouette']


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH))
    unseen_idx = split['unseen_indices']
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                     dtype=torch.float32)
    X_holdout = X[unseen_idx]
    labels_holdout = adata.obs['leiden'].values[unseen_idx]
    gene_names = list(adata.var_names)
    marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]

    per_seed = {}
    for sd_dir in sorted((BASE / 'outputs' / 'p2' / 'moon').glob('seed*')):
        model, _ = load_vae_model(str(sd_dir / 'best_model.pt'))
        util = evaluate_utility(model, X_holdout, labels_holdout, marker_idx, gene_names)
        per_seed[sd_dir.name] = {f: float(util[f]) for f in FIELDS}
        per_seed[sd_dir.name]['marker_per_gene'] = {
            k: float(v) for k, v in util.get('marker_per_gene', {}).items()}
        print(f"  moon {sd_dir.name}: ELBO={util['elbo']:.1f} marker_r={util['marker_r']:.3f} "
              f"ARI={util['ari']:.3f}", flush=True)

    mean = {f: float(np.mean([v[f] for v in per_seed.values()])) for f in FIELDS}
    json.dump({'method': 'moon', 'holdout_size': len(unseen_idx), 'device': 'cpu',
               'per_seed': per_seed, 'mean': mean}, open(OUT, 'w'), indent=2)
    print(f"mean: {mean}\nsaved {OUT}")


if __name__ == '__main__':
    main()
