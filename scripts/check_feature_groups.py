#!/usr/bin/env python3
"""Which feature groups carry the separability found by the standardized refit?

Out-of-fold logistic regression (standardized, stratified 5-fold, 10 fold splits) on subsets of the
70 attack features: likelihood terms (recon_nll, kl, elbo), latent mean (32), latent log-variance (32),
the two norms, the kNN distance to the retain set, and all 70. Run on PBMC-33k cluster 13 against its
cross-type negatives (baseline, clean retrain seed 42) and on the PBMC-68k within-type sets (baseline,
clean retrain seed 42). Writes outputs/feature_group_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, get_features, get_retain_latent_codes

GROUPS = {'likelihood terms (3)': list(range(0, 3)), 'latent mean (32)': list(range(3, 35)), 'latent logvar (32)': list(range(35, 67)),
          'norms (2)': [67, 68], 'kNN distance (1)': [69], 'recon_nll only': [0], 'all 70': list(range(70))}


def oof_auc(x, y, cols, n_splits=10):
    aucs = []
    for s in range(n_splits):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=s).split(x, y):
            sc_ = StandardScaler().fit(x[tr][:, cols])
            oof[te] = LogisticRegression(C=1.0, max_iter=2000).fit(sc_.transform(x[tr][:, cols]), y[tr]).predict_proba(sc_.transform(x[te][:, cols]))[:, 1]
        aucs.append(float(roc_auc_score(y, oof)))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def run(tag, model, adata, forget, neg, retain, out):
    rz = get_retain_latent_codes(model, adata, retain)
    x = torch.cat([get_features(model, adata, forget, retain_z=rz), get_features(model, adata, neg, retain_z=rz)]).numpy()
    y = np.r_[np.ones(len(forget)), np.zeros(len(neg))]
    out[tag] = {g: oof_auc(x, y, cols) for g, cols in GROUPS.items()}
    print(f"{tag:36s} " + '  '.join(f"{g.split(' (')[0]} {m:.3f}" for g, (m, sd) in out[tag].items()), flush=True)


def main():
    out = {}
    adata = sc.read_h5ad(BASE / 'data/adata_processed.h5ad'); split = json.load(open(BASE / 'outputs/p1/split_structured.json'))
    forget, retain = np.array(split['forget_indices']), np.array(split['retain_indices'])
    cross = np.array(json.load(open(BASE / 'outputs/p1.5/s1_matched_negatives.json'))['matched_indices'])
    run('pbmc33k/c13_cross/baseline', load_vae_model(str(BASE / 'outputs/p1/baseline/best_model.pt'))[0], adata, forget, cross, retain, out)
    run('pbmc33k/c13_cross/retrain_42', load_vae_model(str(BASE / 'outputs/p1/retrain_clean/seed_42/best_model.pt'))[0], adata, forget, cross, retain, out)
    adata = sc.read_h5ad(BASE / 'data/pbmc68k_processed.h5ad')
    base = load_vae_model(str(BASE / 'outputs/pbmc68k/baseline/best_model.pt'))[0]
    for fs in ('mk', 'pdc'):
        split = json.load(open(BASE / f'outputs/pbmc68k/split_{fs}.json'))
        forget, retain, within = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'within_type_unseen_indices'))
        cross = np.array(json.load(open(BASE / f'outputs/pbmc68k/matched_negatives_{fs}.json'))['matched_indices'])
        retr = load_vae_model(str(BASE / f'outputs/pbmc68k/{fs}/retrain_clean_seed_42/best_model.pt'))[0]
        run(f'pbmc68k/{fs}_within/baseline', base, adata, forget, within, retain, out)
        run(f'pbmc68k/{fs}_within/retrain_42', retr, adata, forget, within, retain, out)
        run(f'pbmc68k/{fs}_cross/retrain_42', retr, adata, forget, cross, retain, out)
    json.dump(out, open(BASE / 'outputs/feature_group_check.json', 'w'), indent=1)
    print('saved outputs/feature_group_check.json')


if __name__ == '__main__':
    main()
