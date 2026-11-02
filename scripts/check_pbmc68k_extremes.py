#!/usr/bin/env python3
"""Examine every PBMC-68k number that sits within 0.02 of 0 or 1.

For each audit AUC near 0 or 1: the per-cell score distributions (reference minus target negative
ELBO) of forget cells and negatives, the separation margin, an independent AUC (Mann-Whitney), the
same audit under a different z seed with four times the samples, and against the other retrained
reference. For each attacker AUC near 0 or 1: the attacker's output distributions. For the
within-type "no per-cell signal" reading: an out-of-fold refit with standardized features, both the
paper's MLP and a logistic regression, on the baseline and the clean retrains.

Writes outputs/pbmc68k/extremes_check.json. Run: PYTHONPATH=src python scripts/check_pbmc68k_extremes.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, get_features, get_retain_latent_codes, train_fresh_attacker
from likelihood_audit import neg_elbo_per_cell, lr_auc
import retrain_refit_attacker as RR
RR.DEVICE = 'cpu'
from retrain_refit_attacker import train_attacker

OUT = BASE / 'outputs/pbmc68k'
EXTREME = 0.02


def q(a):
    return {k: float(v) for k, v in zip(('min', 'p5', 'median', 'p95', 'max', 'mean', 'sd'), (a.min(), np.percentile(a, 5), np.median(a), np.percentile(a, 95), a.max(), a.mean(), a.std()))}


def near_extreme(v):
    return v <= EXTREME or v >= 1 - EXTREME


def main():
    adata = sc.read_h5ad(BASE / 'data/pbmc68k_processed.h5ad'); X = adata.X
    tens = lambda idx: torch.FloatTensor(X[idx].toarray())
    report = {'threshold': EXTREME, 'audits': {}, 'attackers': {}, 'within_refit_standardized': {}}
    base, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    for fs in ('mk', 'pdc'):
        res = json.load(open(OUT / f'results_{fs}.json')); split = json.load(open(OUT / f'split_{fs}.json'))
        forget, retain, within = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'within_type_unseen_indices'))
        cross = np.array(json.load(open(OUT / f'matched_negatives_{fs}.json'))['matched_indices'])
        xf, xc, xw = tens(forget), tens(cross), tens(within)
        refs = {s: load_vae_model(str(OUT / fs / f'retrain_clean_seed_{s}/best_model.pt'))[0] for s in (42, 43)}
        ref_scores = {s: {'f': neg_elbo_per_cell(refs[s], xf), 'c': neg_elbo_per_cell(refs[s], xc)} for s in refs}
        ref_alt = {'f': neg_elbo_per_cell(refs[42], xf, n_samples=64, seed=1), 'c': neg_elbo_per_cell(refs[42], xc, n_samples=64, seed=1)}
        # ---- audits near 0 or 1
        for method, rec in res['per_checkpoint'].items():
            for r in rec['per_seed']:
                if not near_extreme(r['audit_cross']):
                    continue
                if method == 'baseline':
                    path = OUT / 'baseline/best_model.pt'
                elif method == 'retrain':
                    path = OUT / fs / f"retrain_clean_{r['seed']}/best_model.pt"
                elif method == 'long_finetune':
                    path = OUT / f"long_finetune/seed{r['seed']}/final_model.pt"
                else:
                    path = OUT / fs / f"{method}_seed{r['seed']}" / ('unlearned_model.pt' if method == 'fisher' else 'best_model.pt')
                m, _ = load_vae_model(str(path))
                nf, nc = neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xc)
                sf, sc_ = ref_scores[42]['f'] - nf, ref_scores[42]['c'] - nc
                labels = np.r_[np.ones(len(sf)), np.zeros(len(sc_))]
                u = mannwhitneyu(sf, sc_, alternative='two-sided')
                nf2, nc2 = neg_elbo_per_cell(m, xf, n_samples=64, seed=1), neg_elbo_per_cell(m, xc, n_samples=64, seed=1)
                entry = {'audit_reported': r['audit_cross'], 'audit_recomputed': lr_auc(ref_scores[42]['f'], ref_scores[42]['c'], nf, nc),
                         'auc_from_mann_whitney_U': float(u.statistic / (len(sf) * len(sc_))), 'mann_whitney_p': float(u.pvalue),
                         'audit_seed1_64samples': lr_auc(ref_alt['f'], ref_alt['c'], nf2, nc2),
                         'audit_vs_reference_seed43': lr_auc(ref_scores[43]['f'], ref_scores[43]['c'], nf, nc),
                         'forget_score': q(sf), 'negative_score': q(sc_),
                         'margin_min_forget_minus_max_negative': float(sf.min() - sc_.max()),
                         'n_forget_below_max_negative': int((sf < sc_.max()).sum()), 'n_negative_above_min_forget': int((sc_ > sf.min()).sum()),
                         'target_negelbo': {'forget': q(nf), 'cross': q(nc)}, 'reference_negelbo': {'forget': q(ref_scores[42]['f']), 'cross': q(ref_scores[42]['c'])}}
                report['audits'][f'{fs}/{method}/{r["seed"]}'] = entry
                print(f"[audit] {fs} {method:16s} {r['seed']:8s} reported {r['audit_cross']:.3f} recomputed {entry['audit_recomputed']:.3f} MWU {entry['auc_from_mann_whitney_U']:.3f} "
                      f"seed1x64 {entry['audit_seed1_64samples']:.3f} ref43 {entry['audit_vs_reference_seed43']:.3f} | forget score [{sf.min():.0f}, {np.median(sf):.0f}, {sf.max():.0f}] "
                      f"neg score [{sc_.min():.0f}, {np.median(sc_):.0f}, {sc_.max():.0f}] margin {entry['margin_min_forget_minus_max_negative']:.0f}", flush=True)
        # ---- attacker AUCs near 0 or 1 (transferred attacker outputs)
        per = json.load(open(OUT / f'persample_{fs}.json'))
        for kind in ('cross', 'within'):
            for method, recs in per[kind].items():
                for r in recs:
                    if not near_extreme(r['auc']):
                        continue
                    s, l = np.array(r['scores']), np.array(r['labels'])
                    entry = {'auc': r['auc'], 'forget_output': q(s[l == 1]), 'negative_output': q(s[l == 0]),
                             'auc_recomputed': float(roc_auc_score(l, s))}
                    report['attackers'][f'{fs}/{kind}/{method}/{r["seed"]}'] = entry
                    print(f"[attacker] {fs} {kind:6s} {method:16s} {r['seed']:8s} auc {r['auc']:.3f} | forget out [{s[l==1].min():.3f}, {np.median(s[l==1]):.3f}, {s[l==1].max():.3f}] "
                          f"neg out [{s[l==0].min():.3f}, {np.median(s[l==0]):.3f}, {s[l==0].max():.3f}]", flush=True)
        # ---- within-type refit with standardized features (baseline, retrains), MLP and logistic
        retain_z = get_retain_latent_codes(base, adata, retain)
        models = {'baseline': base, **{f'retrain_{s}': refs[s] for s in refs}}
        for name, m in models.items():
            rz = get_retain_latent_codes(m, adata, retain)
            for kind, neg in (('within', within), ('cross', cross)):
                Ff, Fn = get_features(m, adata, forget, retain_z=rz), get_features(m, adata, neg, retain_z=rz)
                x = torch.cat([Ff, Fn]).numpy(); y = np.r_[np.ones(len(Ff)), np.zeros(len(Fn))]
                aucs = {'logistic_std': [], 'mlp_std': []}
                for split_seed in range(10):
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
                    oof_l, oof_m = np.zeros(len(y)), np.zeros(len(y))
                    for fi, (tr, te) in enumerate(skf.split(x, y)):
                        sc_ = StandardScaler().fit(x[tr]); xtr, xte = sc_.transform(x[tr]), sc_.transform(x[te])
                        oof_l[te] = LogisticRegression(C=1.0, max_iter=2000).fit(xtr, y[tr]).predict_proba(xte)[:, 1]
                        att = train_attacker(torch.FloatTensor(xtr), torch.FloatTensor(y[tr]), seed=split_seed * 100 + fi)
                        with torch.no_grad():
                            oof_m[te] = torch.sigmoid(att(torch.FloatTensor(xte)).squeeze()).numpy()
                    aucs['logistic_std'].append(float(roc_auc_score(y, oof_l))); aucs['mlp_std'].append(float(roc_auc_score(y, oof_m)))
                entry = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'per_split': v} for k, v in aucs.items()}
                report['within_refit_standardized'][f'{fs}/{kind}/{name}'] = entry
                print(f"[refit-std] {fs} {kind:6s} {name:12s} logistic {entry['logistic_std']['mean']:.3f}±{entry['logistic_std']['sd']:.3f}  mlp {entry['mlp_std']['mean']:.3f}±{entry['mlp_std']['sd']:.3f}", flush=True)
    # ---- PBMC-33k frozen critics and contrastive (the 33k numbers near 0), from the canonical per-sample files
    p4 = BASE / 'outputs/p4/multiseed/persample'
    report['pbmc33k_attacker_extremes'] = {}
    for f in sorted(p4.glob('*.json')):
        for r in json.load(open(f)):
            if near_extreme(r['auc']):
                s, l = np.array(r['scores']), np.array(r['labels'])
                report['pbmc33k_attacker_extremes'][f'{f.stem}/{r["seed"]}'] = {'auc': r['auc'], 'forget_output': q(s[l == 1]), 'negative_output': q(s[l == 0])}
                print(f"[33k attacker] {f.stem:24s} {str(r['seed']):8s} auc {r['auc']:.3f} | forget out [{s[l==1].min():.3f}, {np.median(s[l==1]):.3f}, {s[l==1].max():.3f}] neg out [{s[l==0].min():.3f}, {np.median(s[l==0]):.3f}, {s[l==0].max():.3f}]", flush=True)
    lr = json.load(open(BASE / 'outputs/attack_lr_reference.json'))
    report['pbmc33k_audit_extremes'] = {k: v for k, v in lr['models'].items() if isinstance(v, dict) and 'auc_mean' in v and near_extreme(v['auc_mean'])}
    print('[33k audits near 0/1]', {k: round(v['auc_mean'], 3) for k, v in report['pbmc33k_audit_extremes'].items()})
    json.dump(report, open(OUT / 'extremes_check.json', 'w'), indent=1, default=float)
    print(f"saved {OUT / 'extremes_check.json'}")


if __name__ == '__main__':
    main()
