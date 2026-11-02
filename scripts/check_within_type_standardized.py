#!/usr/bin/env python3
"""Re-examine the within-type controls of the other cohorts with a well-conditioned refit attacker.

The paper's attacker is an MLP on raw 70-dimensional features (scale of hundreds). On PBMC-68k a refit
of that attacker scored the baseline's forget cells against held-out cells of the same type at 0.449 out
of fold, while a logistic regression on standardized features scored 0.68: the raw-feature MLP collapses
toward the class prior when the signal is weak. This script applies the standardized refit (logistic and
the MLP, out of fold, stratified 5-fold, repeated over fold splits) to the within-type controls used in
Section 4.1: PBMC-33k cluster 7 (50 and 100 forget cells vs 80 held-out), Tabula Muris cluster 28 (50 and
100 vs 67), TCGA-BRCA Basal (158 vs 30 held-out), and, for reference, PBMC-33k cluster 13 against its 194
cross-type negatives, on the baseline and the corresponding retrained model.

Writes outputs/within_type_standardized_check.json. Run: PYTHONPATH=src python scripts/check_within_type_standardized.py
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
import retrain_refit_attacker as RR
RR.DEVICE = 'cpu'
from retrain_refit_attacker import train_attacker
from utils import GLOBAL_SEED

OUT = BASE / 'outputs/within_type_standardized_check.json'


def refits(Ff, Fn, n_splits_log=20, n_splits_mlp=10):
    x = torch.cat([Ff, Fn]).numpy(); y = np.r_[np.ones(len(Ff)), np.zeros(len(Fn))]
    out = {'logistic_std': [], 'mlp_std': [], 'mlp_raw': []}
    for split_seed in range(n_splits_log):
        oof_l = np.zeros(len(y)); oof_m = np.zeros(len(y)); oof_r = np.zeros(len(y))
        for fi, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed).split(x, y)):
            sc_ = StandardScaler().fit(x[tr]); xtr, xte = sc_.transform(x[tr]), sc_.transform(x[te])
            oof_l[te] = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, y[tr]).predict_proba(xte)[:, 1]
            if split_seed < n_splits_mlp:
                att = train_attacker(torch.FloatTensor(xtr), torch.FloatTensor(y[tr]), seed=split_seed * 100 + fi)
                with torch.no_grad():
                    oof_m[te] = torch.sigmoid(att(torch.FloatTensor(xte)).squeeze()).numpy()
                att = train_attacker(torch.FloatTensor(x[tr]), torch.FloatTensor(y[tr]), seed=split_seed * 100 + fi)
                with torch.no_grad():
                    oof_r[te] = torch.sigmoid(att(torch.FloatTensor(x[te])).squeeze()).numpy()
        out['logistic_std'].append(float(roc_auc_score(y, oof_l)))
        if split_seed < n_splits_mlp:
            out['mlp_std'].append(float(roc_auc_score(y, oof_m))); out['mlp_raw'].append(float(roc_auc_score(y, oof_r)))
    return {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'per_split': v} for k, v in out.items()}


def run(tag, model, adata, forget, negatives, retain, results):
    rz = get_retain_latent_codes(model, adata, retain)
    Ff, Fn = get_features(model, adata, forget, retain_z=rz), get_features(model, adata, negatives, retain_z=rz)
    r = refits(Ff, Fn); r['n_forget'], r['n_negatives'] = int(len(forget)), int(len(negatives)); results[tag] = r
    print(f"  {tag:44s} n {len(forget):3d} vs {len(negatives):3d}  logistic-std {r['logistic_std']['mean']:.3f}±{r['logistic_std']['sd']:.3f}  "
          f"mlp-std {r['mlp_std']['mean']:.3f}±{r['mlp_std']['sd']:.3f}  mlp-raw {r['mlp_raw']['mean']:.3f}±{r['mlp_raw']['sd']:.3f}", flush=True)


def main():
    results = {}
    # ---- PBMC-33k: cluster 13 vs cross negatives (reference), cluster 7 within-type (NB39 forget sets)
    adata = sc.read_h5ad(BASE / 'data/adata_processed.h5ad'); leiden = adata.obs['leiden'].astype(str).values
    split = json.load(open(BASE / 'outputs/p1/split_structured.json'))
    forget13, retain, unseen = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'unseen_indices'))
    cross13 = np.array(json.load(open(BASE / 'outputs/p1.5/s1_matched_negatives.json'))['matched_indices'])
    base, _ = load_vae_model(str(BASE / 'outputs/p1/baseline/best_model.pt'))
    retr13, _ = load_vae_model(str(BASE / 'outputs/p1/retrain_clean/seed_42/best_model.pt'))
    print('PBMC-33k cluster 13 (194 cross-type negatives)', flush=True)
    run('pbmc33k/c13_cross/baseline', base, adata, forget13, cross13, retain, results)
    run('pbmc33k/c13_cross/retrain_seed42', retr13, adata, forget13, cross13, retain, results)
    train = np.array(split['forget_indices'] + split['retain_indices'])
    c7_train = [i for i in train if leiden[i] == '7']; c7_unseen = np.array([i for i in unseen if leiden[i] == '7'])
    np.random.seed(GLOBAL_SEED); shuffled = np.random.permutation(c7_train).tolist()
    print('PBMC-33k cluster 7 (within-type: 80 held-out cluster-7 cells)', flush=True)
    for n in (50, 100):
        forget = np.array(sorted(shuffled[:n])); retain_n = np.setdiff1d(train, forget)
        retr, _ = load_vae_model(str(BASE / f'outputs/larger_cluster_eval/pbmc_retrain_c7_{n}.pt'))
        run(f'pbmc33k/c7_within_n{n}/baseline', base, adata, forget, c7_unseen, retain_n, results)
        run(f'pbmc33k/c7_within_n{n}/retrain', retr, adata, forget, c7_unseen, retain_n, results)
    # ---- Tabula Muris cluster 28
    adata = sc.read_h5ad(BASE / 'data/tabula_muris_processed.h5ad'); leiden = adata.obs['leiden'].astype(str).values
    split = json.load(open(BASE / 'outputs/tabula_muris/split_structured.json'))
    train = np.array(split['forget_indices'] + split['retain_indices']); unseen = np.array(split['unseen_indices'])
    c28_train = [i for i in train if leiden[i] == '28']; c28_unseen = np.array([i for i in unseen if leiden[i] == '28'])
    np.random.seed(GLOBAL_SEED); shuffled = np.random.permutation(c28_train).tolist()
    base_tm, _ = load_vae_model(str(BASE / 'outputs/tabula_muris/baseline/best_model.pt'))
    print('Tabula Muris cluster 28 (within-type: 67 held-out cluster-28 cells)', flush=True)
    for n in (50, 100):
        forget = np.array(sorted(shuffled[:n])); retain_n = np.setdiff1d(train, forget)
        retr, _ = load_vae_model(str(BASE / f'outputs/larger_cluster_eval/tm_retrain_c28_{n}.pt'))
        run(f'tm/c28_within_n{n}/baseline', base_tm, adata, forget, c28_unseen, retain_n, results)
        run(f'tm/c28_within_n{n}/retrain', retr, adata, forget, c28_unseen, retain_n, results)
    # ---- TCGA-BRCA Basal within-subtype
    adata = sc.read_h5ad(BASE / 'data/tcga_brca/tcga_brca_processed.h5ad')
    split = json.load(open(BASE / 'outputs/tcga_brca/split_structured.json'))
    forget, retain, unseen = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'unseen_indices'))
    col = next(c for c in adata.obs.columns if adata.obs[c].astype(str).str.contains('Basal').any())
    within = np.array([i for i in unseen if str(adata.obs[col].iloc[i]) == 'Basal'])
    base_t, _ = load_vae_model(str(BASE / 'outputs/tcga_brca/baseline/best_model.pt'))
    retr_t, _ = load_vae_model(str(BASE / 'outputs/tcga_brca/retrain_structured/best_model.pt'))
    print(f'TCGA-BRCA Basal (within-subtype: {len(within)} held-out Basal patients; subtype column {col})', flush=True)
    run('tcga/basal_within/baseline', base_t, adata, forget, within, retain, results)
    run('tcga/basal_within/retrain', retr_t, adata, forget, within, retain, results)
    json.dump({'note': 'out-of-fold refit AUCs; logistic_std = logistic regression on standardized features (20 fold splits); mlp_std = the paper MLP on standardized features (10); mlp_raw = the paper MLP on raw features (10)',
               'results': results}, open(OUT, 'w'), indent=1)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
