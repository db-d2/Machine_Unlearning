#!/usr/bin/env python3
"""Predicted survival of the forget Fisher mass under the SGD arm of the restoration intervention.

Proposition 1 (Restoration) gives, for gradient descent with step eta on the retain loss, a residual delta_j (1 - eta
F_R,j)^T after T steps, so the share of an edit's forget-Fisher-weighted damage that survives is
S(t) = sum_j F_F,j exp(-t F_R,j / mean F_R) / sum_j F_F,j with t = 2 eta T mean(F_R), the fine-tuning time in units in
which a typical retain coordinate decays by e^-1. The SGD arm of scripts/restricted_scrub_test.py runs eta = 1e-3 for
10, 50 and 200 epochs of batches of 256 over the retain set; this script evaluates S(t) at those t, for the whole model
and split by the retain-null mask of the intervention (retain Fisher below tau = 1e-2 times its mean), with the retain
Fisher from the scrub's cache and the forget Fisher from the same estimator (per-sample diagonal, damping 0). The
step-size probe (one epoch from the baseline) found SGD stable to about 2.5e-3 against the proposition's bound
1/max F_R of 0.013, so the empirical Fisher underestimates the top curvature by about five; S(5t) is given beside S(t).

Writes outputs/p6/sgd_survival_prediction.json. Run: PYTHONPATH=src python scripts/sgd_survival_prediction.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model
from fisher_utils import AnnDataDataset, compute_vae_fisher
from retain_null_mass import SETTINGS

OUT = BASE / 'outputs/p6/sgd_survival_prediction.json'
ETA, BATCH, EPOCHS, TAU, CAL = 1e-3, 256, (10, 50, 200), 1e-2, 5.0


def main():
    out = {'eta': ETA, 'batch_size': BATCH, 'epochs': list(EPOCHS), 'tau': TAU, 'curvature_calibration': CAL, 'settings': {}}
    for name in ('pbmc33k', 'pbmc68k_mk'):
        cfg = SETTINGS[name]; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
        model, _ = load_vae_model(str(cfg['baseline']))
        raw = torch.load(cfg['cache'], map_location='cpu', weights_only=False)
        F_R = {k: v.cpu().double().numpy() for k, v in raw['fisher'].items()}
        loader = DataLoader(AnnDataDataset(adata, split['forget_indices']), batch_size=128, shuffle=False)
        F_F, n = compute_vae_fisher(model, loader, 'cpu', damping=0.0)
        ff = np.concatenate([F_F[k].cpu().double().numpy().ravel() for k in F_F]); fr = np.concatenate([F_R[k].ravel() for k in F_F])
        m_r = fr.mean(); nb = int(np.ceil(len(split['retain_indices']) / BATCH)); null = fr < TAU * m_r
        def S(mask, t):
            return float((ff[mask] * np.exp(-t * fr[mask] / m_r)).sum() / ff[mask].sum())
        rec = {'n_forget': int(n), 'mean_retain_fisher': float(m_r), 'max_retain_fisher': float(fr.max()), 'eta_bound': float(1 / fr.max()), 'batches_per_epoch': nb,
               'null_mass': float(ff[null].sum() / ff.sum()), 'epochs': {}}
        for ep in EPOCHS:
            t = 2 * ETA * ep * nb * m_r; every = np.ones_like(null)
            rec['epochs'][str(ep)] = {'t': float(t), 'survival': S(every, t), 'survival_active': S(~null, t), 'survival_null': S(null, t),
                                      'survival_calibrated': S(every, CAL * t), 'survival_active_calibrated': S(~null, CAL * t), 'survival_null_calibrated': S(null, CAL * t)}
        out['settings'][name] = rec
        print(name, json.dumps(rec, indent=1), flush=True)
    json.dump(out, open(OUT, 'w'), indent=1); print(f'saved {OUT}')


if __name__ == '__main__':
    main()
