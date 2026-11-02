#!/usr/bin/env python3
"""Symmetric, per-cell audit using every clean retrain as a shadow model (LiRA-style).

For each cell, the retrains give a distribution of the negative ELBO a model that never saw the forget
set assigns it. A target model's score for the cell is (mean over retrains - target) / sd over retrains,
positive when the target fits the cell better than never-seen models do. The audit AUC is forget cells
against matched negatives on that score. A retrain is scored leave-one-out (its own negELBO against the
other retrains' statistics), which gives a reference range that does not depend on which seed is the
reference. Also records per-cell medians of the forget fit, since Fisher scrubbing leaves heavy tails.

Writes outputs/audit_lira.json. Run: PYTHONPATH=src python scripts/audit_lira.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH
from save_persample_scores import checkpoint_map
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci


def lira_auc(R_f, R_m, t_f, t_m, exclude=None):
    """R_f, R_m: arrays (n_retrains, n_cells). exclude: index of the retrain to leave out."""
    keep = [i for i in range(len(R_f)) if i != exclude]
    mf, sf = R_f[keep].mean(0), R_f[keep].std(0, ddof=1) + 1e-6
    mm, sm = R_m[keep].mean(0), R_m[keep].std(0, ddof=1) + 1e-6
    s = np.r_[(mf - t_f) / sf, (mm - t_m) / sm]
    y = np.r_[np.ones(len(t_f)), np.zeros(len(t_m))]
    boot = []
    rng = np.random.default_rng(0); nf, nm = len(t_f), len(t_m)
    for _ in range(4000):
        i = np.r_[rng.integers(0, nf, nf), nf + rng.integers(0, nm, nm)]
        boot.append(roc_auc_score(y[i], s[i]))
    return {'auc': float(roc_auc_score(y, s)), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def run(cohort, X, forget, matched, retrain_paths, targets):
    tens = lambda idx: torch.FloatTensor(np.asarray(X[idx].toarray() if hasattr(X[idx], 'toarray') else X[idx]))
    xf, xm = tens(forget), tens(matched)
    R_f, R_m, names = [], [], []
    for name, p in retrain_paths:
        m, _ = load_vae_model(str(p)); R_f.append(neg_elbo_per_cell(m, xf)); R_m.append(neg_elbo_per_cell(m, xm)); names.append(name)
    R_f, R_m = np.array(R_f), np.array(R_m)
    out = {'n_forget': len(forget), 'n_matched': len(matched), 'retrains': names,
           'retrain_loo': {n: lira_auc(R_f, R_m, R_f[i], R_m[i], exclude=i) for i, n in enumerate(names)},
           'retrain_forget_median': {n: float(np.median(R_f[i])) for i, n in enumerate(names)}, 'models': {}}
    lo = min(v['auc'] for v in out['retrain_loo'].values()); hi = max(v['auc'] for v in out['retrain_loo'].values())
    print(f"[{cohort}] retrain leave-one-out LiRA audit: {lo:.3f} to {hi:.3f}  " + ' '.join(f"{n}={v['auc']:.2f}" for n, v in out['retrain_loo'].items()), flush=True)
    for method, seeds in targets.items():
        rows = []
        for sd, p in seeds:
            m, _ = load_vae_model(str(p)); t_f, t_m = neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm)
            r = lira_auc(R_f, R_m, t_f, t_m); r.update({'seed': str(sd), 'forget_mean': float(t_f.mean()), 'forget_median': float(np.median(t_f)), 'forget_p95': float(np.percentile(t_f, 95)), 'matched_mean': float(t_m.mean())})
            rows.append(r)
        out['models'][method] = rows
        print(f"[{cohort}] {method:22s} LiRA audit {np.mean([r['auc'] for r in rows]):.3f} (per seed {[round(r['auc'], 2) for r in rows]}) forget median {np.mean([r['forget_median'] for r in rows]):.0f} mean {np.mean([r['forget_mean'] for r in rows]):.0f}", flush=True)
    return out


def main():
    res = {'score': '(mean over retrains - target) / sd over retrains, per cell; retrains scored leave-one-out', 'cohorts': {}}
    adata = sc.read_h5ad(DATA_PATH); split = json.load(open(SPLIT_PATH)); matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    cm = checkpoint_map(); cm.pop('retrain_leaky', None)
    retrains = cm.pop('retrain')
    targets = {'baseline': cm.pop('baseline')}
    for m in ['retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'fisher_step_ablation', 'extragradient', 'extragradient_lambda1', 'contrastive', 'dp_sgd',
              'frozen_single_lambda5', 'frozen_single_lambda10', 'frozen_multi_lambda5', 'frozen_multi_lambda10']:
        if m in cm: targets[m] = cm[m]
    targets['long_finetune'] = [(p.parent.name, p) for p in sorted((BASE / 'outputs/p8/pbmc').glob('seed*/final_model.pt'))]
    targets['ltu_retrained_level'] = [(p.parent.name, p) for p in sorted((BASE / 'outputs/p7/ltu_avg').glob('seed*/best_model.pt'))]
    res['cohorts']['pbmc33k'] = run('pbmc33k', adata.X, split['forget_indices'], matched, retrains, targets)
    adata = sc.read_h5ad(BASE / 'data/pbmc68k_processed.h5ad'); OUT = BASE / 'outputs/pbmc68k'
    for fs in ('mk', 'pdc'):
        split = json.load(open(OUT / f'split_{fs}.json')); matched = json.load(open(OUT / f'matched_negatives_{fs}.json'))['matched_indices']
        retrains = [(f'seed_{s}', OUT / fs / f'retrain_clean_seed_{s}/best_model.pt') for s in range(42, 50)]
        targets = {'baseline': [('seed42', OUT / 'baseline/best_model.pt')]}
        for m in ['retain_finetune', 'gradient_ascent', 'fisher', 'ssd', 'scrub', 'contrastive', 'extragradient']:
            fn = 'unlearned_model.pt' if m == 'fisher' else 'best_model.pt'
            targets[m] = [(p.name.split('_seed')[-1], p / fn) for p in sorted((OUT / fs).glob(f'{m}_seed*')) if (p / fn).exists()]
        if fs == 'mk':
            targets['long_finetune'] = [(p.parent.name, p) for p in sorted((OUT / 'long_finetune').glob('seed*/final_model.pt'))]
        res['cohorts'][f'pbmc68k_{fs}'] = run(f'pbmc68k_{fs}', adata.X, split['forget_indices'], matched, retrains, targets)
    json.dump(res, open(BASE / 'outputs/audit_lira.json', 'w'), indent=1)
    print('saved outputs/audit_lira.json', flush=True)


if __name__ == '__main__':
    main()
