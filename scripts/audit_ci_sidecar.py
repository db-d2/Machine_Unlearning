#!/usr/bin/env python3
"""Audit intervals for checkpoints whose result files carry point audits only: the likelihood-targeted
models of notebook 41 (outputs/p7/{ltu,ltu_avg,ltu_disjoint}) and the 200-epoch retain fine-tunes of
notebook 42 (outputs/p8/{pbmc,tcga}). Same estimator and reference as the paper (clean retrain seed 42
on PBMC-33k against the 194 matched negatives; the structured retrain on TCGA-BRCA against the 89
cross-subtype negatives), with the cell bootstrap and nested interval of src/likelihood_audit.py.

Writes outputs/audit_ci_sidecar.json. Run: PYTHONPATH=src python scripts/audit_ci_sidecar.py
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
from eval_multiseed import load_vae_model, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci, nested_auc_ci

OUT = BASE / 'outputs/audit_ci_sidecar.json'


def cohort_pbmc():
    adata = sc.read_h5ad(DATA_PATH); split = json.load(open(SPLIT_PATH)); matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    X = lambda idx: torch.FloatTensor(np.asarray(adata.X[idx].toarray() if hasattr(adata.X[idx], 'toarray') else adata.X[idx]))
    ref, _ = load_vae_model(str(BASE / 'outputs/p1/retrain_clean/seed_42/best_model.pt'))
    return X(split['forget_indices']), X(matched), ref


def cohort_tcga():
    adata = sc.read_h5ad(BASE / 'data/tcga_brca/tcga_brca_processed.h5ad'); split = json.load(open(BASE / 'outputs/tcga_brca/split_structured.json'))
    matched = json.load(open(BASE / 'outputs/tcga_brca/matched_negatives.json'))['matched_indices']
    Xm = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = lambda idx: torch.FloatTensor(np.asarray(Xm[idx]))
    ck = torch.load(BASE / 'outputs/tcga_brca/retrain_structured/best_model.pt', map_location='cpu', weights_only=False)
    from vae import VAE
    cfg = {k: v for k, v in ck['config'].items() if k in ['input_dim', 'latent_dim', 'hidden_dims', 'likelihood', 'dropout', 'use_layer_norm']}
    ref = VAE(**cfg); ref.load_state_dict(ck['model_state_dict']); ref.eval()
    return X(split['forget_indices']), X(matched), ref


def main():
    out = {'ci': 'per checkpoint: percentile bootstrap over cells (10,000; groups resampled separately); per family: nested over seeds', 'families': {}}
    xf, xm, ref = cohort_pbmc(); rf, rm = neg_elbo_per_cell(ref, xf), neg_elbo_per_cell(ref, xm)
    families = {'ltu': BASE / 'outputs/p7/ltu', 'ltu_avg': BASE / 'outputs/p7/ltu_avg', 'ltu_disjoint': BASE / 'outputs/p7/ltu_disjoint', 'long_finetune_pbmc': BASE / 'outputs/p8/pbmc'}
    for fam, d in families.items():
        per = {}
        for sd in sorted(d.glob('seed*')):
            path = next((sd / f for f in ('final_model.pt', 'best_model.pt') if (sd / f).exists()), None)
            if path is None:
                continue
            m, _ = load_vae_model(str(path)); per[sd.name] = lr_auc_ci(rf, rm, neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm))
            print(f"  {fam:20s} {sd.name:9s} audit {per[sd.name]['auc']:.3f} [{per[sd.name]['ci95'][0]:.3f}, {per[sd.name]['ci95'][1]:.3f}]", flush=True)
        if per:
            out['families'][fam] = {'per_seed': per, 'nested': nested_auc_ci(list(per.values())), 'reference': 'outputs/p1/retrain_clean/seed_42', 'n_forget': len(xf), 'n_matched': len(xm)}
    xf, xm, ref = cohort_tcga(); rf, rm = neg_elbo_per_cell(ref, xf), neg_elbo_per_cell(ref, xm)
    per = {}
    for sd in sorted((BASE / 'outputs/p8/tcga').glob('seed*')):
        path = sd / 'final_model.pt'
        if not path.exists():
            continue
        m, _ = load_vae_model(str(path)); per[sd.name] = lr_auc_ci(rf, rm, neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm))
        print(f"  {'long_finetune_tcga':20s} {sd.name:9s} audit {per[sd.name]['auc']:.3f} [{per[sd.name]['ci95'][0]:.3f}, {per[sd.name]['ci95'][1]:.3f}]", flush=True)
    if per:
        out['families']['long_finetune_tcga'] = {'per_seed': per, 'nested': nested_auc_ci(list(per.values())), 'reference': 'outputs/tcga_brca/retrain_structured', 'n_forget': len(xf), 'n_matched': len(xm)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
