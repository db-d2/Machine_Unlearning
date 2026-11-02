#!/usr/bin/env python3
"""Phase audit for two-phase methods: does the retain fine-tune undo the forget-set edit?

Proposition 1 says forget-set and retain-set Fisher information share directions, so a
scrub that raises the forget-set loss should also damage the retain set, and the retain
fine-tune that repairs the damage should also restore the forget set's fit. This script
measures forget-set negative ELBO, retain-set negative ELBO (fixed sample of 2,000 retain
cells), held-out negative ELBO, the reference-model LR AUC, and the canonical MLP AUC at
three points: the baseline, after the scrub (or ascent) phase, and after the fine-tune.

Fisher at max_step 1e-3, 3e-3, 1e-2 and gradient ascent at the paper's setting. Writes
outputs/p6/phase_audit.json. Run: PYTHONPATH=src python scripts/fisher_phase_audit.py
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features,
                            DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT)
from fisher_utils import AnnDataDataset
from train_fisher_unlearn import compute_fisher_diagonal, fisher_scrub, retain_finetune
from train_gradient_ascent import gradient_ascent_phase, finetune_phase

import os
# CPU by default (2026-09-08): the scrub at large steps leaves the model where MPS nondeterminism
# changes the fine-tune endpoint (forget negELBO 518 in one MPS run, 1,076 in another at step 1e-2).
DEV = os.environ.get('PHASE_AUDIT_DEVICE', 'cpu')
REF = BASE / 'outputs/p1/retrain_clean/seed_42/best_model.pt'
OUT = BASE / 'outputs/p6/phase_audit.json'


def neg_elbo(model, x, device='cpu'):
    """Deterministic estimator shared by every likelihood instrument (src/likelihood_audit.py)."""
    return neg_elbo_per_cell(model.to(device), x)


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH)); f_idx, r_idx, u_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
    m_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    rs = np.random.RandomState(0).choice(r_idx, 2000, replace=False)
    xf, xm, xr, xu = X[f_idx], X[m_idx], X[rs], X[u_idx[:2000]]
    labels = np.r_[np.ones(len(f_idx)), np.zeros(len(m_idx))]
    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    attacker = train_fresh_attacker(base, adata, f_idx, m_idx, r_idx); attacker.eval()
    ref, _ = load_vae_model(str(REF)); ref_f, ref_m = neg_elbo(ref, xf), neg_elbo(ref, xm)

    def measure(model, tag):
        model = model.to('cpu')
        rz = get_retain_latent_codes(model, adata, r_idx)
        Xa = torch.cat([get_features(model, adata, f_idx, retain_z=rz), get_features(model, adata, m_idx, retain_z=rz)], 0)
        with torch.no_grad():
            s = torch.sigmoid(attacker(Xa).squeeze()).numpy()
        nf, nm = neg_elbo(model, xf), neg_elbo(model, xm)
        rec = {'forget_neg_elbo': float(nf.mean()), 'matched_neg_elbo': float(nm.mean()),
               'retain_neg_elbo': float(neg_elbo(model, xr).mean()), 'unseen_neg_elbo': float(neg_elbo(model, xu).mean()),
               'mlp_auc': float(roc_auc_score(labels, s)), 'lr_auc': float(roc_auc_score(labels, np.r_[ref_f - nf, ref_m - nm])),
               'lr_auc_ci': lr_auc_ci(ref_f, ref_m, nf, nm)['ci95']}
        print(f"  {tag:34s} forget={rec['forget_neg_elbo']:.1f} matched={rec['matched_neg_elbo']:.1f} retain={rec['retain_neg_elbo']:.1f} "
              f"unseen={rec['unseen_neg_elbo']:.1f} | MLP {rec['mlp_auc']:.3f} LR {rec['lr_auc']:.3f}", flush=True)
        return rec

    out = {'reference': str(REF.relative_to(BASE)), 'runs': {}}
    out['runs']['baseline'] = measure(base, 'baseline')
    forget_loader = DataLoader(AnnDataDataset(adata, f_idx), batch_size=256, shuffle=False)
    retain_loader = DataLoader(AnnDataDataset(adata, r_idx), batch_size=256, shuffle=True)
    fisher_loader = DataLoader(AnnDataDataset(adata, r_idx), batch_size=128, shuffle=False)
    model = deepcopy(base).to(DEV)
    fisher = compute_fisher_diagonal(model, fisher_loader, DEV, damping=0.1,
                                     cache=BASE / 'outputs/p2/fisher_unlearn/fisher_retain_persample.pt')
    for step in [1e-3, 3e-3, 1e-2]:
        torch.manual_seed(42); np.random.seed(42)
        model = deepcopy(base).to(DEV)
        fisher_scrub(model, forget_loader, {k: v.to(DEV) for k, v in fisher.items()}, DEV, scrub_lr=1e-4, scrub_steps=100, max_step=step)
        after_scrub = measure(model, f'fisher step {step:g}: after scrub')
        retain_finetune(model.to(DEV), retain_loader, DEV, epochs=10, lr=1e-4)
        after_ft = measure(model, f'fisher step {step:g}: after fine-tune')
        out['runs'][f'fisher_step{step:g}'] = {'after_scrub': after_scrub, 'after_finetune': after_ft}
    torch.manual_seed(42); np.random.seed(42)
    model = deepcopy(base).to(DEV)
    gradient_ascent_phase(model, forget_loader, 10, 1e-5, DEV)
    after_asc = measure(model, 'gradient ascent: after ascent')
    from torch.utils.data import TensorDataset
    val_loader = DataLoader(TensorDataset(xu, xu.sum(1, keepdim=True)), batch_size=256)
    finetune_phase(model.to(DEV), retain_loader, val_loader, 30, 1e-4, 10, DEV)
    after_ft = measure(model, 'gradient ascent: after fine-tune')
    out['runs']['gradient_ascent'] = {'after_ascent': after_asc, 'after_finetune': after_ft}
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
