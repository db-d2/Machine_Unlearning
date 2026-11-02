#!/usr/bin/env python3
"""Exact per-sample nested bootstrap for ALL methods (reviewer W4) + a permutation
null for the 2|AUC-0.5| advantage estimator (reviewer W6). Consumes the per-sample
score caches from save_persample_scores.py.

Outer bootstrap resamples seeds; inner resamples the ~224 per-sample scores; the
statistic is the mean advantage across drawn seeds. Welch t-tests vs retrain use the
per-seed advantages with Holm-Bonferroni. Output: outputs/p4/multiseed/nested_ci_all.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).parent.parent
PS = BASE / 'outputs/p4/multiseed/persample'
OUT = BASE / 'outputs/p4/multiseed/nested_ci_all.json'
rng = np.random.RandomState(0)
B = 10000

def load(m):
    p = PS / f'{m}.json'
    return json.load(open(p)) if p.exists() else None

def advantage(auc):
    return 2 * abs(auc - 0.5)

def nested(seed_records):
    """Two-level bootstrap. Returns (summary dict, per-seed advantage array)."""
    arrs = [(np.array(r['scores']), np.array(r['labels'])) for r in seed_records]
    per_seed_adv = np.array([advantage(roc_auc_score(l, s)) for s, l in arrs])
    S = len(arrs)
    samples = np.empty(B)
    for b in range(B):
        drawn = rng.randint(0, S, S) if S > 1 else [0]
        advs = []
        for di in drawn:
            s, l = arrs[di]; n = len(s); idx = rng.randint(0, n, n)
            ls = l[idx]
            advs.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        samples[b] = np.mean(advs)
    return ({'mean': float(per_seed_adv.mean()), 'n_seeds': S,
             'nested_ci_low': float(np.percentile(samples, 2.5)),
             'nested_ci_high': float(np.percentile(samples, 97.5)),
             'per_seed_advantage': [float(a) for a in per_seed_adv]}, per_seed_adv)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2)) if na+nb > 2 else 0.0
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float('inf')

def nested_diff(method_recs, retrain_recs, Bd=5000):
    """Nested bootstrap of (method advantage - retrain advantage): the correct test
    of separation, combining cross-seed and within-eval variance for both."""
    ma = [(np.array(r['scores']), np.array(r['labels'])) for r in method_recs]
    ra = [(np.array(r['scores']), np.array(r['labels'])) for r in retrain_recs]
    def madv(arrs):
        S = len(arrs); v = []
        for di in (rng.randint(0, S, S) if S > 1 else [0]):
            s, l = arrs[di]; idx = rng.randint(0, len(s), len(s)); ls = l[idx]
            v.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        return np.mean(v)
    diffs = np.array([madv(ma) - madv(ra) for _ in range(Bd)])
    return {'diff_mean': float(diffs.mean()),
            'diff_ci': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            'p_method_gt_retrain': float((diffs > 0).mean())}

METHODS = ['extragradient', 'extragradient_lambda1', 'gradient_ascent', 'retain_finetune', 'ssd', 'scrub', 'contrastive', 'dp_sgd', 'fisher', 'fisher_step_ablation', 'moon', 'retrain_leaky', 'frozen_single_lambda5', 'frozen_single_lambda10', 'frozen_multi_lambda5', 'frozen_multi_lambda10']
retrain = load('retrain')
if retrain is None:
    sys.exit("per-sample scores not found; run save_persample_scores.py first")
rt_stat, rt_advs = nested(retrain)
results = {'retrain': rt_stat, 'methods': {}}
tests = {}
for m in METHODS:
    recs = load(m)
    if not recs:
        print(f"  [missing] {m}"); continue
    st, advs = nested(recs)
    results['methods'][m] = st
    if len(advs) >= 2 and len(rt_advs) >= 2:
        t, p = sps.ttest_ind(advs, rt_advs, equal_var=False, alternative='greater')
        tests[m] = {'t': float(t), 'p_one_sided': float(p), 'cohens_d': cohens_d(advs, rt_advs),
                    'method_mean_adv': st['mean'], 'nested_ci': [st['nested_ci_low'], st['nested_ci_high']]}
    else:
        tests[m] = {'note': 'single seed, no cross-seed test', 'method_mean_adv': st['mean'],
                    'nested_ci': [st['nested_ci_low'], st['nested_ci_high']]}
    tests[m].update(nested_diff(recs, retrain))
    print(f"  {m}: adv={st['mean']:.3f} nested95=[{st['nested_ci_low']:.3f},{st['nested_ci_high']:.3f}] "
          f"diff={tests[m]['diff_mean']:.3f} diff95={[round(x,3) for x in tests[m]['diff_ci']]} "
          f"P(>retrain)={tests[m]['p_method_gt_retrain']:.3f} n={st['n_seeds']}")

# Holm-Bonferroni across methods with a valid test
keys = [k for k in tests if 'p_one_sided' in tests[k]]
if keys:
    rej, padj, _, _ = multipletests([tests[k]['p_one_sided'] for k in keys], alpha=0.05, method='holm')
    for k, r, pa in zip(keys, rej, padj):
        tests[k]['p_holm'] = float(pa); tests[k]['reject_holm'] = bool(r)

# Permutation null for the advantage estimator (folding floor) at the design n
base = load('baseline')[0]
s0 = np.array(base['scores']); l0 = np.array(base['labels'])
null = np.array([advantage(roc_auc_score(rng.permutation(l0), s0)) for _ in range(20000)])
results['advantage_null_floor'] = {
    'estimator': '2|AUC-0.5| under shuffled labels (H0: no membership signal)',
    'n_pos': int(l0.sum()), 'n_neg': int((l0 == 0).sum()),
    'mean': float(null.mean()), 'ci95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]}

# The step ablation holds two step caps at one seed each, so its pooled "nested" interval bootstraps two
# hyperparameters as if they were seeds and its mean describes no model (round-7 review). Each cap is also
# summarized on its own; this runs after everything above so that no earlier draw moves.
for rec in load('fisher_step_ablation') or []:
    key = 'fisher_' + rec['seed'].split('_seed')[0]          # fisher_step1e-2, fisher_step3e-3
    st, _ = nested([rec]); results['methods'][key] = st
    tests[key] = {'note': 'one step cap, single seed, no cross-seed test', 'method_mean_adv': st['mean'],
                  'nested_ci': [st['nested_ci_low'], st['nested_ci_high']], **nested_diff([rec], retrain)}
    print(f"  {key}: adv={st['mean']:.3f} ci95=[{st['nested_ci_low']:.3f},{st['nested_ci_high']:.3f}] "
          f"diff95={[round(x, 3) for x in tests[key]['diff_ci']]}")

results['hypothesis_tests'] = tests
json.dump(results, open(OUT, 'w'), indent=2)
print(f"\nretrain adv={rt_stat['mean']:.3f} nested95=[{rt_stat['nested_ci_low']:.3f},{rt_stat['nested_ci_high']:.3f}]")
print(f"advantage null floor: mean={results['advantage_null_floor']['mean']:.3f} "
      f"95%={results['advantage_null_floor']['ci95']}")
print("saved", OUT)
