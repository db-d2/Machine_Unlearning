#!/usr/bin/env python3
"""Reference-model likelihood-ratio audit for Tabula Muris and TCGA-BRCA.

Same instrument as scripts/attack_lr_reference.py: score = negELBO under the cohort's
clean retrain minus negELBO under the target model, forget cells vs matched negatives,
CPU, deterministic estimator of src/likelihood_audit.py (eval mode, 16 seeded z samples). Both cohorts' retrains train on the retain split only (notebooks 12 and 32).
There is no second retrain seed on these cohorts, so the retrain row is the reference
scored against itself (AUC 0.5 by construction) and is reported only as a placeholder.

Writes outputs/attack_lr_reference_multi.json.
Run: PYTHONPATH=src python scripts/attack_lr_reference_multi.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
from vae import VAE
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci, N_SAMPLES, SEED

OUT = BASE / 'outputs' / 'attack_lr_reference_multi.json'
COHORTS = {
    'tabula_muris': {
        'data': BASE / 'data/tabula_muris_processed.h5ad',
        'split': BASE / 'outputs/tabula_muris/split_structured.json',
        'matched': BASE / 'outputs/tabula_muris/matched_negatives.json',
        'reference': BASE / 'outputs/tabula_muris/retrain/best_model.pt',
        'targets': {'baseline': BASE / 'outputs/tabula_muris/baseline/best_model.pt',
                    'extragradient': BASE / 'outputs/tabula_muris/extragradient_structured/best_model.pt',
                    'fisher': BASE / 'outputs/tabula_muris/fisher_structured/unlearned_model.pt'},
    },
    'tcga_brca': {
        'data': BASE / 'data/tcga_brca/tcga_brca_processed.h5ad',
        'split': BASE / 'outputs/tcga_brca/split_structured.json',
        'matched': BASE / 'outputs/tcga_brca/matched_negatives.json',
        'reference': BASE / 'outputs/tcga_brca/retrain_structured/best_model.pt',
        'targets': {'baseline': BASE / 'outputs/tcga_brca/baseline/best_model.pt',
                    **{m: BASE / f'outputs/tcga_brca/unlearning/{d}/best_model.pt' for m, d in
                       [('fisher', 'fisher'), ('retain_finetune', 'retain_ft'), ('gradient_ascent', 'gradient_ascent'),
                        ('ssd', 'ssd'), ('scrub', 'scrub'), ('contrastive', 'contrastive')]}},
    },
}


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = {k: v for k, v in ck['config'].items() if k in ['input_dim', 'latent_dim', 'hidden_dims', 'likelihood', 'dropout', 'use_layer_norm']}
    m = VAE(**cfg); m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


def neg_elbo(model, x):
    """Deterministic estimator shared by every likelihood instrument (src/likelihood_audit.py)."""
    return neg_elbo_per_cell(model, x)


def main():
    out = {}
    for cohort, c in COHORTS.items():
        adata = sc.read_h5ad(c['data'])
        split = json.load(open(c['split'])); matched = json.load(open(c['matched']))['matched_indices']
        Xm = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        xf, xm = torch.FloatTensor(np.asarray(Xm[split['forget_indices']])), torch.FloatTensor(np.asarray(Xm[matched]))
        labels = np.r_[np.ones(len(xf)), np.zeros(len(xm))]
        ref = load(c['reference']); rf, rm = neg_elbo(ref, xf), neg_elbo(ref, xm)
        res = {'n_forget': len(xf), 'n_matched': len(xm), 'reference': str(c['reference'].relative_to(BASE)), 'models': {}}
        for name, path in c['targets'].items():
            if not path.exists():
                print(f"  [{cohort}] missing {name}"); continue
            m = load(path); tf, tm = neg_elbo(m, xf), neg_elbo(m, xm); s = np.r_[rf - tf, rm - tm]
            auc = float(roc_auc_score(labels, s)); c = lr_auc_ci(rf, rm, tf, tm)
            res['models'][name] = {'lr_auc': auc, 'advantage': 2 * abs(auc - 0.5), 'ci95': c['ci95'], 'null95': c['null95']}
            print(f"  [{cohort}] {name:18s} LR AUC={auc:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]", flush=True)
        out[cohort] = res
    out['estimator'] = f'likelihood_audit.neg_elbo_per_cell: eval mode, {N_SAMPLES} seeded z samples (seed {SEED}), CPU'
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
