#!/usr/bin/env python3
"""Paired non-inferiority test of the per-cell refit statistic against the clean-retrain arm (PBMC-68k).

Replaces the rule `per-cell AUC <= max(retrains' max, null p97.5)` of `make_tex_tables.status68k`, which
let the noise floor stand in for the retrained models and passed methods that sat above all eight of
them (round-7 review, B2).

Question (the user's criterion): can the adversary identify the forget cells among held-out cells of the
same type much better on this model than on a model retrained without them? Comparator: the arm of eight
clean retrains. Statistic: the out-of-fold standardized logistic AUC (mean over 20 fold splits) stored by
`scripts/score_pbmc68k.py` (v6) in `percell_oof_<fs>.npz`.

Inference: the difference, method arm mean minus retrain arm mean, is bootstrapped with an outer draw over
seeds within each arm and an inner stratified draw over the cells, shared by every checkpoint in that draw
(the same 107 + 71 or 302 + 201 cells are scored by every model, so the comparison is paired).

Decision rule, fixed on 2026-09-22 before the v6 scores existed (the reviewer's paired differences on the
v5 scores, whose retrain arm was mismeasured, had been seen):
- margin delta = one half of the baseline's excess over the retrain arm (point estimates, same instrument,
  same forget set). A method is AT THE RETRAINED LEVEL when the upper end of the two-sided 95 percent
  interval of its difference lies below delta, that is, when the data certify it removed more than half of
  the per-cell signal that training on the cells created. This is the fraction-of-effect construction of
  non-inferiority trials with an active control (the retrain) and a placebo (the baseline).
- MORE IDENTIFIABLE THAN RETRAINED when the lower end of the interval lies above zero.
- NOT RESOLVED otherwise. The design's resolution is reported with every row as `u95`, the upper end of
  the interval, which is the smallest margin at which the arm would pass.
- MORE IDENTIFIABLE THAN THE BASELINE, a further flag, when the paired interval of the method minus the
  baseline lies above zero.
Both one-sided tests are at 0.025, so each verdict can be read off the displayed interval. Holm-adjusted
p-values over the method arms of one forget set are reported alongside, with the verdict they give.

Writes outputs/pbmc68k/percell_paired_<fs>.json.
Run: PYTHONPATH=src python scripts/percell_paired_test.py --forget_set mk
"""
import argparse
import json
from pathlib import Path

import numpy as np

import percell_stats as PS

BASE = Path(__file__).parent.parent
OUT = BASE / 'outputs/pbmc68k'
B, BOOT_SEED, NEST_SEED = 10000, 0, 7
MARGIN_FRACTION = 0.5
ALPHA = 0.025
REFERENCE = 'retrain'
POSITIVE_CONTROL = 'baseline'


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def verdict(lo, hi, delta):
    """Four-way (round 8): above zero, below zero, inside plus or minus the margin, or neither."""
    if lo > 0:
        return 'more identifiable than retrained'
    if hi < 0:
        return 'less identifiable than retrained'
    if lo > -delta and hi < delta:
        return 'at the retrained level'
    return 'not resolved'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forget_set', required=True); args = ap.parse_args()
    fs = args.forget_set
    z = np.load(OUT / f'percell_oof_{fs}.npz'); res = json.load(open(OUT / f'results_{fs}.json'))
    assert res.get('version', '').startswith('v6'), 'percell_paired_test needs the v6 scores (round-7 B1 fix)'
    n_f = int(z['n_forget']); out = {'forget_set': fs, 'statistic': 'out-of-fold standardized logistic AUC, mean over 20 fold splits (refit_std)',
                                     'reference_arm': REFERENCE, 'margin_rule': f'delta = {MARGIN_FRACTION} x the lower 95 percent bound of the paired (baseline - retrain arm) difference (round 8, 2026-09-26; the 2026-09-22 rule used the point estimate, kept as percell_paired_*_v1_pointmargin.json); verdict two-sided equivalence',
                                     'interval': 'two-sided 95 percent; outer bootstrap over seeds within each arm, inner stratified cell bootstrap shared by every checkpoint',
                                     'alpha_one_sided': ALPHA, 'n_boot': B, 'boot_seed': BOOT_SEED, 'nest_seed': NEST_SEED, 'kinds': {}}
    for kind in ('within', 'cross'):
        n_neg = int(z[f'n_{kind}']); y = np.r_[np.ones(n_f), np.zeros(n_neg)]
        wp, wn = PS.boot_weights(n_f, n_neg, n_boot=B, seed=BOOT_SEED)
        arms = {}
        for key in z.files:
            if not key.startswith(f'{kind}__'):
                continue
            _, method, seed = key.split('__')
            oof = z[key].astype(np.float64)
            arms.setdefault(method, {'seeds': [], 'point': [], 'draws': []})
            arms[method]['seeds'].append(seed); arms[method]['point'].append(PS.auc_mean(oof, y)); arms[method]['draws'].append(PS.boot_auc(oof, y, wp, wn))
        ref = arms[REFERENCE]; ref_mean = float(np.mean(ref['point']))
        excess = PS.paired_nested_diff(np.array(arms[POSITIVE_CONTROL]['draws']), np.array(ref['draws']), seed=NEST_SEED + 1)
        excess_ci = [float(np.percentile(excess, 2.5)), float(np.percentile(excess, 97.5))]
        delta = MARGIN_FRACTION * excess_ci[0]   # round 8: half the LOWER bound of the baseline's excess, not half its point estimate
        # per-checkpoint label-permutation nulls, from the scorer
        nulls = {m: [r['refit_std_within']['perm_null'] for r in v['per_seed']] for m, v in res['per_checkpoint'].items()} if kind == 'within' else {}
        rows = {}
        for i, (method, a) in enumerate(sorted(arms.items())):
            arm_draws = PS.nested_arm(np.array(a['draws']), seed=NEST_SEED + 2 * i)
            row = {'seeds': a['seeds'], 'per_seed_auc': a['point'], 'mean': float(np.mean(a['point'])),
                   'nested_ci': [float(np.percentile(arm_draws, 2.5)), float(np.percentile(arm_draws, 97.5))]}
            if method != REFERENCE:
                d = PS.paired_nested_diff(np.array(a['draws']), np.array(ref['draws']), seed=NEST_SEED + 2 * i)
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                row.update({'diff': row['mean'] - ref_mean, 'diff_ci': [lo, hi], 'u95': hi,
                            'p_above': float((1 + (d <= 0).sum()) / (1 + len(d))), 'p_below': float((1 + (d >= 0).sum()) / (1 + len(d))),
                            'p_equivalent': float(max((1 + (d <= -delta).sum()) / (1 + len(d)), (1 + (d >= delta).sum()) / (1 + len(d)))),
                            'seeds_above_all_retrains': int(sum(p > max(ref['point']) for p in a['point'])),
                            'verdict': verdict(lo, hi, delta)})
            if method not in (REFERENCE, POSITIVE_CONTROL):   # against the baseline, paired the same way
                db = PS.paired_nested_diff(np.array(a['draws']), np.array(arms[POSITIVE_CONTROL]['draws']), seed=NEST_SEED + 2 * i)
                row.update({'diff_vs_baseline': row['mean'] - float(np.mean(arms[POSITIVE_CONTROL]['point'])),
                            'diff_vs_baseline_ci': [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]})
            if nulls:
                row['perm_null_p975'] = [n['p975'] for n in nulls[method]]
                row['perm_null_p_value'] = [n['p_value'] for n in nulls[method]]
            rows[method] = row
        tested = [m for m in rows if m not in (REFERENCE, POSITIVE_CONTROL)]
        for key, name in (('p_above', 'p_above_holm'), ('p_below', 'p_below_holm'), ('p_equivalent', 'p_equivalent_holm')):
            for m, p in zip(tested, holm([rows[m][key] for m in tested])):
                rows[m][name] = float(p)
        for m in tested:
            r = rows[m]
            r['verdict_holm'] = ('more identifiable than retrained' if r['p_above_holm'] < ALPHA else 'less identifiable than retrained' if r['p_below_holm'] < ALPHA else
                                 'at the retrained level' if r['p_equivalent_holm'] < ALPHA else 'not resolved')
            r['more_identifiable_than_baseline'] = bool(r['diff_vs_baseline_ci'][0] > 0)
        out['kinds'][kind] = {'n_forget': n_f, 'n_negatives': n_neg, 'retrain_mean': ref_mean, 'retrain_range': [min(ref['point']), max(ref['point'])],
                              'baseline_excess': float(np.mean(arms[POSITIVE_CONTROL]['point'])) - ref_mean, 'baseline_excess_ci': excess_ci, 'margin': delta, 'arms': rows}
        if kind == 'cross':
            out['kinds'][kind]['note'] = ('cross-type negatives are separated from the forget cells at 0.99 or above on every model, retrained '
                                          'ones included, so this kind measures the type, not membership; its verdicts are not membership verdicts')
        print(f'{fs} {kind}: retrain arm {ref_mean:.4f} ({min(ref["point"]):.4f} to {max(ref["point"]):.4f}), margin {delta:.4f}')
        for m in sorted(rows, key=lambda k: rows[k]['mean']):
            r = rows[m]
            extra = '' if m == REFERENCE else f"  diff {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]  {r['verdict']}" + ('' if m == POSITIVE_CONTROL else f" (Holm: {r['verdict_holm']})")
            print(f"  {m:18s} {r['mean']:.4f} [{r['nested_ci'][0]:.3f}, {r['nested_ci'][1]:.3f}]{extra}")
    json.dump(out, open(OUT / f'percell_paired_{fs}.json', 'w'), indent=1)
    print(f"saved {OUT / f'percell_paired_{fs}.json'}")


if __name__ == '__main__':
    main()
