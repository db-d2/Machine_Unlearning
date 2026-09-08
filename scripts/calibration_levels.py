#!/usr/bin/env python3
"""How the baseline and the clean retrains fit the forget cells and each calibration set of notebook 41.

Deterministic estimator (src/likelihood_audit.py), CPU. Writes outputs/p7/calibration_levels.json.
Run: PYTHONPATH=src python scripts/calibration_levels.py
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
from eval_multiseed import load_vae_model, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT
from likelihood_audit import neg_elbo_per_cell, N_SAMPLES, SEED
from train_targeted_unlearn import matched_neighbors

OUT = BASE / 'outputs/p7/calibration_levels.json'


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH)); matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    xf, xm, xu = X[split['forget_indices']], X[matched], X[split['unseen_indices']]
    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    calib = torch.load(BASE / 'outputs/p7/ltu_disjoint/seed42/best_model.pt', map_location='cpu', weights_only=False)['calibration_indices']
    xc = X[calib]
    nb = matched_neighbors(base, xf, xm, 'cpu', k=10).numpy()
    models = {'baseline': base}
    for s in range(42, 47):
        models[f'retrain_clean_seed_{s}'], _ = load_vae_model(str(BASE / f'outputs/p1/retrain_clean/seed_{s}/best_model.pt'))
    out = {'estimator': f'likelihood_audit.neg_elbo_per_cell, eval mode, {N_SAMPLES} seeded z samples (seed {SEED}), CPU',
           'sets': {'forget': len(xf), 'matched_negatives': len(xm), 'ten_nearest_matched_per_forget_cell': int(nb.shape[1]),
                    'disjoint_lookalikes_ranks_11_to_30': len(xc), 'unseen_first_2000': 2000},
           'mean_neg_elbo': {}}
    for name, m in models.items():
        nf, nm, nc, nu = neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm), neg_elbo_per_cell(m, xc), neg_elbo_per_cell(m, xu[:2000])
        out['mean_neg_elbo'][name] = {'forget': float(nf.mean()), 'matched_negatives': float(nm.mean()),
                                      'ten_nearest_matched_per_forget_cell': float(nm[nb].mean()),
                                      'disjoint_lookalikes': float(nc.mean()), 'unseen_first_2000': float(nu.mean())}
        print(f"{name:24s} forget={nf.mean():6.1f} matched={nm.mean():6.1f} 10NN={nm[nb].mean():6.1f} disjoint={nc.mean():6.1f} unseen={nu.mean():6.1f}")
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
