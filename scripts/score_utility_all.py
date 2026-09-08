#!/usr/bin/env python3
"""Held-out utility for every current PBMC checkpoint in one CPU pass.

Uses the same checkpoint map as save_persample_scores.py so privacy and utility always
describe the same files. Writes outputs/p4/multiseed/utility_all.json with per-seed
values and per-method means for ELBO, reconstruction, KL, silhouette, ARI, marker r.

Run: PYTHONPATH=src python scripts/score_utility_all.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
import nested_bootstrap_retrain as NBR
NBR.DEVICE = 'cpu'
from nested_bootstrap_retrain import load_retrain_model
from eval_multiseed import load_vae_model, evaluate_utility, MARKER_GENES, DATA_PATH, SPLIT_PATH
from save_persample_scores import checkpoint_map

OUT = BASE / 'outputs' / 'p4' / 'multiseed' / 'utility_all.json'
FIELDS = ['elbo', 'recon', 'kl', 'silhouette', 'ari', 'marker_r']


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH))
    unseen_idx = split['unseen_indices']
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    X_holdout, labels_holdout = X[unseen_idx], adata.obs['leiden'].values[unseen_idx]
    gene_names = list(adata.var_names)
    marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]

    results = {}
    for method, seeds in checkpoint_map().items():
        per_seed = {}
        for sd, path in seeds:
            model = load_retrain_model(str(path)) if method == 'retrain_leaky' else load_vae_model(str(path))[0]
            u = evaluate_utility(model, X_holdout, labels_holdout, marker_idx, gene_names)
            per_seed[str(sd)] = {f: float(u[f]) for f in FIELDS}
            per_seed[str(sd)]['marker_per_gene'] = {k: float(v) for k, v in u['marker_per_gene'].items()}
            per_seed[str(sd)]['path'] = str(path.relative_to(BASE))
            print(f"  {method} {sd}: ELBO={u['elbo']:.1f} KL={u['kl']:.2f} marker_r={u['marker_r']:.3f} ARI={u['ari']:.3f}", flush=True)
        results[method] = {'per_seed': per_seed,
                           'mean': {f: float(np.mean([v[f] for v in per_seed.values()])) for f in FIELDS},
                           'n_seeds': len(per_seed)}
    json.dump({'holdout_size': len(unseen_idx), 'device': 'cpu', 'methods': results}, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
