#!/usr/bin/env python3
"""Score the two forget-set-size ablations with one CPU attacker per size.

For each size, a fresh canonical attacker is trained on the July-4 baseline (forget_n vs
that size's matched negatives) and applied to the baseline, the per-size clean retrains,
and every method checkpoint for that size. A reference likelihood-ratio AUC against the
per-size clean retrain seed 42 is reported too. Writes outputs/p4/size_ablations_v2.json.

  structured (extra-gradient): outputs/p4/extragradient_size/{n10,n50,n100}/seed*/best_model.pt
      retrains: outputs/p4/extragradient_size/retrain_clean/n*/seed_*/best_model.pt
  scattered (Fisher): outputs/p4/size_ablation/fisher_v2/n*/seed*/unlearned_model.pt
      retrains: outputs/p4/size_ablation/retrain_clean/n*/seed_*/best_model.pt

Run: PYTHONPATH=src python scripts/score_size_ablations.py
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
from likelihood_audit import neg_elbo_per_cell
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features,
                            DATA_PATH, BASELINE_CHECKPOINT)
from vae import vae_loss

OUT = BASE / 'outputs' / 'p4' / 'size_ablations_v2.json'
rng = np.random.RandomState(0)


def advantage(a):
    return 2 * abs(a - 0.5)


def neg_elbo(model, x):
    """Deterministic estimator shared by every likelihood instrument (src/likelihood_audit.py)."""
    return neg_elbo_per_cell(model.to('cpu'), x)


def nested(per_sample):
    """Exact nested bootstrap over seeds and samples for the advantage mean."""
    arrs = [(np.array(s), np.array(l)) for s, l in per_sample]
    S = len(arrs); samples = np.empty(4000)
    for b in range(4000):
        v = []
        for di in (rng.randint(0, S, S) if S > 1 else [0]):
            s, l = arrs[di]; idx = rng.randint(0, len(s), len(s)); ls = l[idx]
            v.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        samples[b] = np.mean(v)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def score_family(adata, X, family):
    if family == 'structured':
        sizes, split_dir = [10, 50, 100], BASE / 'outputs/p4/extragradient_size/splits'
        split_name = lambda n: split_dir / f'split_structured_n{n}.json'
        matched_name = lambda n: split_dir / f'matched_neg_structured_n{n}.json'
        method_dirs = lambda n: sorted((BASE / f'outputs/p4/extragradient_size/n{n}').glob('seed*/best_model.pt'))
        retrain_dirs = lambda n: sorted((BASE / f'outputs/p4/extragradient_size/retrain_clean/n{n}').glob('seed_*/best_model.pt'))
        method = 'extragradient'
    else:
        sizes, split_dir = [10, 30, 50, 100, 500], BASE / 'outputs/p4/size_ablation/splits'
        split_name = lambda n: split_dir / f'split_n{n}.json'
        matched_name = lambda n: split_dir / f'matched_neg_n{n}.json'
        method_dirs = lambda n: sorted((BASE / f'outputs/p4/size_ablation/fisher_v2/n{n}').glob('seed*/unlearned_model.pt'))
        retrain_dirs = lambda n: sorted((BASE / f'outputs/p4/size_ablation/retrain_clean/n{n}').glob('seed_*/best_model.pt'))
        method = 'fisher'
    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    out = {}
    for n in sizes:
        split = json.load(open(split_name(n))); matched = json.load(open(matched_name(n)))
        f_idx, r_idx, m_idx = split['forget_indices'], split['retain_indices'], matched['matched_indices']
        labels = np.r_[np.ones(len(f_idx)), np.zeros(len(m_idx))]
        attacker = train_fresh_attacker(base, adata, f_idx, m_idx, r_idx); attacker.eval()
        xf, xm = X[f_idx], X[m_idx]
        rets = retrain_dirs(n)
        ref_model = load_vae_model(str(rets[0]))[0] if rets else None
        ref_f, ref_m = (neg_elbo(ref_model, xf), neg_elbo(ref_model, xm)) if ref_model is not None else (None, None)

        def score(model):
            rz = get_retain_latent_codes(model, adata, r_idx)
            Xa = torch.cat([get_features(model, adata, f_idx, retain_z=rz), get_features(model, adata, m_idx, retain_z=rz)], 0)
            with torch.no_grad():
                s = torch.sigmoid(attacker(Xa).squeeze()).numpy()
            rec = {'auc': float(roc_auc_score(labels, s)), 'scores': s, 'labels': labels}
            if ref_model is not None:
                lr = np.r_[ref_f - neg_elbo(model, xf), ref_m - neg_elbo(model, xm)]
                rec['lr_auc'] = float(roc_auc_score(labels, lr))
            return rec

        groups = {'baseline': [('seed42', BASELINE_CHECKPOINT)],
                  'retrain': [(p.parent.name, p) for p in rets],
                  method: [(p.parent.name, p) for p in method_dirs(n)]}
        res = {'n_forget': len(f_idx), 'n_matched': len(m_idx)}
        for g, items in groups.items():
            recs = []
            for sd, p in items:
                m, _ = load_vae_model(str(p)); r = score(m); r['seed'] = sd; recs.append(r)
                print(f"  {family} n={n} {g} {sd}: AUC={r['auc']:.3f}" + (f" LR={r['lr_auc']:.3f}" if 'lr_auc' in r else ''), flush=True)
            if not recs:
                continue
            advs = [advantage(r['auc']) for r in recs]
            res[g] = {'auc_mean': float(np.mean([r['auc'] for r in recs])), 'adv_mean': float(np.mean(advs)),
                      'adv_ci': nested([(r['scores'], r['labels']) for r in recs]) if len(recs) else None,
                      'per_seed_auc': {r['seed']: r['auc'] for r in recs},
                      'lr_auc_mean': float(np.mean([r['lr_auc'] for r in recs])) if 'lr_auc' in recs[0] else None,
                      'per_seed_lr_auc': {r['seed']: r.get('lr_auc') for r in recs}, 'n_seeds': len(recs)}
        out[str(n)] = res
    return out


def main():
    adata = sc.read_h5ad(DATA_PATH)
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    result = {'device': 'cpu', 'attacker': 'canonical fresh attacker per size on the July-4 baseline',
              'structured_extragradient': score_family(adata, X, 'structured'),
              'scattered_fisher': score_family(adata, X, 'scattered')}
    json.dump(result, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
