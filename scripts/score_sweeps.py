#!/usr/bin/env python3
"""Score the PBMC-33k step sweeps (scripts/run_sweeps.py): the audit against the mean of the five clean
retrains (reference-independent), the forget-cell median fit, and one fresh attacker trained on the
baseline and applied to every sweep checkpoint, the baseline and the retrains in this pass (its own
instance; not comparable across passes to two decimals). Writes outputs/p4/sweeps/sweep_scores.json.
Run: PYTHONPATH=src python scripts/score_sweeps.py
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
from eval_multiseed import load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT
from likelihood_audit import neg_elbo_per_cell

SW = BASE / 'outputs/p4/sweeps'


def main():
    adata = sc.read_h5ad(DATA_PATH); split = json.load(open(SPLIT_PATH)); matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    forget, retain = split['forget_indices'], split['retain_indices']
    X = adata.X; tens = lambda idx: torch.FloatTensor(np.asarray(X[idx].toarray() if hasattr(X[idx], 'toarray') else X[idx]))
    xf, xm = tens(forget), tens(matched)
    R = {}
    for sd in (42, 43, 44, 45, 46):
        m, _ = load_vae_model(str(BASE / f'outputs/p1/retrain_clean/seed_{sd}/best_model.pt')); R[sd] = (neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm), m)
    labels = np.r_[np.ones(len(forget)), np.zeros(len(matched))]
    def audit_vs_mean(t_f, t_m, exclude=None):
        keep = [k for k in R if k != exclude]; rf = np.mean([R[k][0] for k in keep], 0); rm = np.mean([R[k][1] for k in keep], 0)
        return float(roc_auc_score(labels, np.r_[rf - t_f, rm - t_m]))
    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    attacker = train_fresh_attacker(base, adata, forget, matched, retain); attacker.eval()
    def attack(model):
        torch.manual_seed(0); np.random.seed(0)
        rz = get_retain_latent_codes(model, adata, retain); F = torch.cat([get_features(model, adata, forget, retain_z=rz), get_features(model, adata, matched, retain_z=rz)])
        with torch.no_grad():
            return float(roc_auc_score(labels, torch.sigmoid(attacker(F).squeeze()).numpy()))
    out = {'reference': 'mean negative ELBO over the five clean retrains (leave-one-out for a retrain)', 'retrain_loo': {}, 'retrain_forget_median': {}, 'retrain_attacker': [], 'baseline': {}, 'configs': {}}
    for sd in R:
        out['retrain_loo'][str(sd)] = audit_vs_mean(R[sd][0], R[sd][1], exclude=sd); out['retrain_forget_median'][str(sd)] = float(np.median(R[sd][0])); out['retrain_attacker'].append(attack(R[sd][2]))
    bf, bm = neg_elbo_per_cell(base, xf), neg_elbo_per_cell(base, xm)
    out['baseline'] = {'audit': audit_vs_mean(bf, bm), 'forget_median': float(np.median(bf)), 'attacker_auc': attack(base)}
    print(f"retrains LOO audit {[round(v, 3) for v in out['retrain_loo'].values()]}; baseline audit {out['baseline']['audit']:.3f} attacker {out['baseline']['attacker_auc']:.3f}", flush=True)
    configs = {}
    for d in sorted(SW.glob('gradient_ascent_lr*_seed*')):
        lr = d.name.split('_lr')[1].split('_seed')[0]; configs.setdefault(f'ga_lr{lr}', {'label': f'Gradient ascent, ascent step {lr}', 'paths': []})['paths'].append((d.name.split('_seed')[-1], d / 'best_model.pt'))
    for d in sorted(SW.glob('fisher_step*_seed*')):
        st = d.name.split('_step')[1].split('_seed')[0]; configs.setdefault(f'fisher_step{st}', {'label': f'Fisher scrubbing, step cap {st}', 'paths': []})['paths'].append((d.name.split('_seed')[-1], d / 'unlearned_model.pt'))
    for key, cfg in configs.items():
        rows = []
        for sd, path in cfg['paths']:
            if not path.exists():
                continue
            m, _ = load_vae_model(str(path)); t_f, t_m = neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm)
            rows.append({'seed': sd, 'audit': audit_vs_mean(t_f, t_m), 'forget_median': float(np.median(t_f)), 'forget_mean': float(t_f.mean()), 'attacker_auc': attack(m)})
        out['configs'][key] = {'label': cfg['label'], 'per_seed': rows}
        print(f"{key:16s} audit {[round(r['audit'], 3) for r in rows]} median {[round(r['forget_median']) for r in rows]} attacker {[round(r['attacker_auc'], 3) for r in rows]}", flush=True)
    json.dump(out, open(SW / 'sweep_scores.json', 'w'), indent=1); print('saved', SW / 'sweep_scores.json', flush=True)


if __name__ == '__main__':
    main()
