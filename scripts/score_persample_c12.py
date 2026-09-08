#!/usr/bin/env python3
"""Cluster-12 forget set: score every checkpoint with one canonical CPU attacker.

Attacker: fresh MLP trained on the July-4 baseline's features, cluster-12 forget cells
(49) vs their k-NN matched negatives (57), the same recipe as the cluster-13 pipeline.
Scores baseline, the clean cluster-12 retrains, and the re-derived method checkpoints,
and writes per-sample records plus nested bootstrap CIs to
outputs/p1_c12/scores_c12_v2.json.

Run: PYTHONPATH=src python scripts/score_persample_c12.py
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
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features,
                            evaluate_utility, MARKER_GENES, DATA_PATH, BASELINE_CHECKPOINT)

SPLIT = BASE / 'outputs/p1_c12/split_cluster12.json'
MATCHED = BASE / 'outputs/p1_c12/matched_negatives_c12.json'
OUT = BASE / 'outputs/p1_c12/scores_c12_v2.json'
rng = np.random.RandomState(0)
B = 10000


def advantage(a):
    return 2 * abs(a - 0.5)


def nested(recs):
    arrs = [(np.array(r['scores']), np.array(r['labels'])) for r in recs]
    per_seed = np.array([advantage(roc_auc_score(l, s)) for s, l in arrs])
    S = len(arrs); samples = np.empty(B)
    for b in range(B):
        advs = []
        for di in (rng.randint(0, S, S) if S > 1 else [0]):
            s, l = arrs[di]; idx = rng.randint(0, len(s), len(s)); ls = l[idx]
            advs.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        samples[b] = np.mean(advs)
    return {'mean': float(per_seed.mean()), 'n_seeds': S, 'nested_ci_low': float(np.percentile(samples, 2.5)),
            'nested_ci_high': float(np.percentile(samples, 97.5)), 'per_seed_advantage': [float(a) for a in per_seed]}


def checkpoints():
    M = {'baseline': [('seed42', BASE / 'outputs/p1/baseline/best_model.pt')]}
    for sd in [42, 43, 44]:
        for key, path in [('retrain', BASE / f'outputs/p2_c12/retrain_clean_s{sd}/best_model.pt'),
                          ('retrain_leaky', BASE / f'outputs/p2_c12/retrain_s{sd}/best_model.pt'),
                          ('retain_finetune', BASE / f'outputs/p2_c12/retainft_s{sd}/best_model.pt'),
                          ('fisher', BASE / f'outputs/p2_c12/fisher_s{sd}/unlearned_model.pt'),
                          ('extragradient', BASE / f'outputs/p2_c12/eg_s{sd}/best_model.pt')]:
            if path.exists():
                M.setdefault(key, []).append((f'seed{sd}', path))
    return M


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT)); forget_idx = split['forget_indices']; retain_idx = split['retain_indices']
    unseen_idx = split['unseen_indices']
    matched_idx = json.load(open(MATCHED))['matched_negative_indices']
    labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_idx))])
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    X_holdout, labels_holdout = X[unseen_idx], adata.obs['leiden'].values[unseen_idx]
    gene_names = list(adata.var_names)
    marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]

    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    attacker = train_fresh_attacker(base, adata, forget_idx, matched_idx, retain_idx)
    attacker.eval()

    out = {'forget': 'cluster 12', 'n_forget': len(forget_idx), 'n_matched': len(matched_idx), 'device': 'cpu',
           'methods': {}}
    for method, seeds in checkpoints().items():
        recs, utils = [], []
        for sd, path in seeds:
            if method == 'retrain_leaky':
                import nested_bootstrap_retrain as NBR; NBR.DEVICE = 'cpu'
                model = NBR.load_retrain_model(str(path))
            else:
                model, _ = load_vae_model(str(path))
            rz = get_retain_latent_codes(model, adata, retain_idx)
            Xf = torch.cat([get_features(model, adata, forget_idx, retain_z=rz),
                            get_features(model, adata, matched_idx, retain_z=rz)], 0)
            with torch.no_grad():
                s = torch.sigmoid(attacker(Xf).squeeze()).numpy()
            auc = float(roc_auc_score(labels, s))
            u = evaluate_utility(model, X_holdout, labels_holdout, marker_idx, gene_names)
            recs.append({'seed': sd, 'auc': auc, 'scores': [float(v) for v in s], 'labels': [int(l) for l in labels]})
            utils.append({'elbo': u['elbo'], 'kl': u['kl'], 'marker_r': u['marker_r'], 'ari': u['ari']})
            print(f"  c12 {method} {sd}: AUC={auc:.4f} adv={advantage(auc):.3f} marker_r={u['marker_r']:.3f}", flush=True)
        st = nested(recs)
        out['methods'][method] = {**st, 'auc_mean': float(np.mean([r['auc'] for r in recs])),
                                  'per_seed_auc': {r['seed']: r['auc'] for r in recs},
                                  'utility_mean': {k: float(np.mean([u[k] for u in utils])) for k in utils[0]},
                                  'paths': [str(p.relative_to(BASE)) for _, p in seeds]}
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
