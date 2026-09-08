#!/usr/bin/env python3
"""Score the clean retrain seeds (scripts/retrain_clean.py) with the canonical CPU attacker.

Writes outputs/p4/multiseed/persample/retrain_clean.json in the same per-sample record
format as save_persample_scores.py, plus outputs/p1/retrain_clean/scores.json with the
per-seed AUC/advantage, the exact per-sample nested bootstrap CI, and the leaky-retrain
numbers side by side for comparison.

Run: PYTHONPATH=src python scripts/score_retrain_clean.py
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
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes,
                            get_features, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT)

CLEAN_DIR = BASE / 'outputs' / 'p1' / 'retrain_clean'
PERSAMPLE = BASE / 'outputs' / 'p4' / 'multiseed' / 'persample'
SEEDS = [42, 43, 44, 45, 46]
rng = np.random.RandomState(0)
B = 10000


def advantage(auc):
    return 2 * abs(auc - 0.5)


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
    return {'mean': float(per_seed.mean()), 'n_seeds': S,
            'nested_ci_low': float(np.percentile(samples, 2.5)),
            'nested_ci_high': float(np.percentile(samples, 97.5)),
            'per_seed_advantage': [float(a) for a in per_seed]}


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH)); forget_idx = split['forget_indices']; retain_idx = split['retain_indices']
    matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_neg_idx))])
    base_model, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    attacker = train_fresh_attacker(base_model, adata, forget_idx, matched_neg_idx, retain_idx)
    attacker.eval()

    recs = []
    for sd in SEEDS:
        p = CLEAN_DIR / f'seed_{sd}' / 'best_model.pt'
        if not p.exists():
            print(f"  [missing] {p}"); continue
        model, _ = load_vae_model(str(p))
        rz = get_retain_latent_codes(model, adata, retain_idx)
        X = torch.cat([get_features(model, adata, forget_idx, retain_z=rz),
                       get_features(model, adata, matched_neg_idx, retain_z=rz)], 0)
        with torch.no_grad():
            scores = torch.sigmoid(attacker(X).squeeze()).numpy()
        auc = float(roc_auc_score(labels, scores))
        recs.append({'seed': f'seed_{sd}', 'auc': auc, 'scores': [float(s) for s in scores],
                     'labels': [int(l) for l in labels]})
        print(f"  clean retrain seed_{sd}: AUC={auc:.4f} adv={advantage(auc):.3f}", flush=True)

    PERSAMPLE.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(PERSAMPLE / 'retrain_clean.json', 'w'))
    clean = nested(recs)
    leaky = json.load(open(BASE / 'outputs/p4/multiseed/nested_ci_all.json'))['retrain']
    out = {'clean_retrain': {'per_seed_auc': {r['seed']: r['auc'] for r in recs}, **clean,
                             'train_set': 'split retain_indices only', 'device': 'cpu'},
           'leaky_retrain_for_comparison': leaky,
           'note': ('leaky = outputs/p1/multiseed_retrain (retrain.py drew an 85/15 split from all '
                    'non-forget cells, so 159-170 of the 194 matched negatives were in its training set)')}
    json.dump(out, open(CLEAN_DIR / 'scores.json', 'w'), indent=2)
    print(f"\nclean retrain: mean adv={clean['mean']:.3f} nested95=[{clean['nested_ci_low']:.3f}, "
          f"{clean['nested_ci_high']:.3f}]  |  leaky: mean adv={leaky['mean']:.3f} "
          f"nested95=[{leaky['nested_ci_low']:.3f}, {leaky['nested_ci_high']:.3f}]")
    print(f"saved {CLEAN_DIR / 'scores.json'} and {PERSAMPLE / 'retrain_clean.json'}")


if __name__ == '__main__':
    main()
