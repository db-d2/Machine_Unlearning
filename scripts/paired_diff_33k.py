#!/usr/bin/env python3
"""Paired equivalence test (round 8, 2026-09-26: signed AUC, margin half the lower bound of the baseline's excess, four verdicts; the 2026-09-25 version tested the folded advantage against half the point excess and is kept as paired_diff_33k_v1_advantage.json)

Original description: Paired non-inferiority test of the PBMC-33k transferred attacker against the clean-retrain arm.

Replaces the rule "advantage at or below the retrained models' upper bound (0.182) with a difference interval
that includes zero" (`make_tex_tables.status`), which has the defect round 7 found on PBMC-68k (B2): it passes
a method whenever the data fail to separate it from the retrained models, so it grows more lenient as the noise
grows. The bound itself was the upper end of a retrain interval that drew fresh cells for every seed, although
all checkpoints are scored on the same 30 forget cells and 194 matched negatives (round-7 Q10); with the cells
shared it is 0.270, and under the old rule extra-gradient at lambda=10 would then pass.

Statistic: the folded advantage 2|AUC - 0.5| of the canonical transferred attacker (one attacker, one CPU pass,
`scripts/save_persample_scores.py`), mean over seeds. The advantage and not the AUC, because an over-corrected
model (AUC far below 0.5) identifies the forget cells as well as an intact one; the AUC is reported beside it.

Inference: one stratified cell draw shared by every checkpoint (percell_stats.boot_weights), seeds resampled
within each arm; the difference is method arm minus retrain arm, paired on the cells.

Decision rule: the PBMC-68k rule of 2026-09-22 (`scripts/percell_paired_test.py`) carried over unchanged, applied
on 2026-09-25 after the paired intervals of the published-setting methods had been seen (reported to the user
on 2026-09-22), so for this cohort the margin is not blind. Margin = one half of the baseline's excess over the
retrain arm. AT THE RETRAINED LEVEL when the upper end of the two-sided 95 percent interval of the difference lies
below the margin; SEPARATES when its lower end lies above zero; NOT RESOLVED otherwise. Holm-adjusted p-values
within each family (published settings; step ablations and sweeps; the two demonstrations) are reported alongside.

Writes outputs/p4/multiseed/paired_diff_33k.json.
Run: PYTHONPATH=src python scripts/paired_diff_33k.py
"""
import json
from pathlib import Path

import numpy as np

import percell_stats as PS

BASE = Path(__file__).parent.parent
D = BASE / 'outputs/p4/multiseed/persample'
OUT = BASE / 'outputs/p4/multiseed/paired_diff_33k.json'
B, BOOT_SEED, NEST_SEED = 10000, 0, 11
MARGIN_FRACTION, ALPHA = 0.5, 0.025
FAMILIES = {
    'published': ['retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'extragradient', 'extragradient_lambda1', 'contrastive',
                  'frozen_single_lambda5', 'frozen_single_lambda10', 'frozen_multi_lambda5', 'frozen_multi_lambda10', 'dp_sgd'],
    'ablation': ['fisher_step3e-3', 'fisher_step1e-2', 'sweep_fisher_step0.0015', 'sweep_fisher_step0.002', 'sweep_fisher_step0.0025', 'sweep_fisher_step0.003',
                 'sweep_gradient_ascent_lr0.0001', 'sweep_gradient_ascent_lr0.001', 'sweep_gradient_ascent_lr0.01'],
    'demonstration': ['ltu', 'ltu_avg', 'ltu_disjoint', 'long_finetune_200'],
}
STEP_SPLIT = {'fisher_step3e-3': 'step3e-3', 'fisher_step1e-2': 'step1e-2'}


def load(m):
    if m in STEP_SPLIT:
        return [r for r in json.load(open(D / 'fisher_step_ablation.json')) if r['seed'].startswith(STEP_SPLIT[m] + '_')]
    p = D / f'{m}.json'
    return json.load(open(p)) if p.exists() else None


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def verdict(lo, hi, margin):
    """Four-way, on the signed statistic: above zero, below zero, inside plus or minus the margin, or neither."""
    if lo > 0:
        return 'more identifiable than retrained'
    if hi < 0:
        return 'less identifiable than retrained'
    if lo > -margin and hi < margin:
        return 'at the retrained level'
    return 'not resolved'


def main():
    names = ['baseline', 'retrain', 'retrain_leaky'] + [m for f in FAMILIES.values() for m in f]
    arms = {m: r for m in names if (r := load(m))}
    lab = np.array(arms['retrain'][0]['labels'])
    assert all((np.array(x['labels']) == lab).all() for recs in arms.values() for x in recs), 'records must share one cell order'
    order = np.argsort(-lab, kind='stable'); y = lab[order]
    wp, wn = PS.boot_weights(int(y.sum()), int((y == 0).sum()), n_boot=B, seed=BOOT_SEED)
    aucd = {m: np.array([PS.boot_auc(np.array(x['scores'])[order], y, wp, wn) for x in recs]) for m, recs in arms.items()}   # signed AUC draws (seeds x B)
    adv = {m: 2 * np.abs(d - 0.5) for m, d in aucd.items()}   # folded advantage, reported beside the AUC, not tested
    point = {m: float(np.mean([2 * abs(x['auc'] - 0.5) for x in recs])) for m, recs in arms.items()}
    auc = {m: [float(x['auc']) for x in recs] for m, recs in arms.items()}
    excess = PS.paired_nested_diff(aucd['baseline'], aucd['retrain'], seed=NEST_SEED + 1)   # the baseline's excess AUC over the retrained arm
    excess_ci = [float(np.percentile(excess, 2.5)), float(np.percentile(excess, 97.5))]
    margin = MARGIN_FRACTION * excess_ci[0]   # half the LOWER bound of the excess (round 8), not half its point estimate
    out = {'statistic': 'signed AUC of the canonical transferred attacker, mean over seeds (round 8: the folded advantage of the 2026-09-25 test biased the difference toward zero; advantage is reported beside the AUC, not tested)',
           'baseline_excess_auc': float(np.mean(auc['baseline']) - np.mean(auc['retrain'])), 'baseline_excess_ci': excess_ci,
           'interval': 'two-sided 95 percent; outer bootstrap over seeds within each arm, inner stratified cell bootstrap shared by every checkpoint',
           'margin_rule': f'{MARGIN_FRACTION} x the lower 95 percent bound of the paired (baseline - retrain arm) AUC difference; two-sided equivalence, the whole interval within plus or minus the margin',
           'margin': margin, 'alpha_one_sided': ALPHA, 'n_boot': B, 'boot_seed': BOOT_SEED, 'nest_seed': NEST_SEED, 'arms': {}}
    for i, m in enumerate(arms):
        nd = PS.nested_arm(adv[m], seed=NEST_SEED + 100 + i); na = PS.nested_arm(aucd[m], seed=NEST_SEED + 100 + i)
        row = {'n_seeds': len(arms[m]), 'seeds': [x['seed'] for x in arms[m]], 'auc_per_seed': auc[m], 'auc': float(np.mean(auc[m])),
               'advantage': point[m], 'advantage_ci': [float(np.percentile(nd, 2.5)), float(np.percentile(nd, 97.5))], 'auc_ci': [float(np.percentile(na, 2.5)), float(np.percentile(na, 97.5))],
               'family': next((f for f, ms in FAMILIES.items() if m in ms), 'reference')}
        if m != 'retrain':
            d = PS.paired_nested_diff(aucd[m], aucd['retrain'], seed=NEST_SEED + 2 * i)
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            row.update({'diff': float(np.mean(auc[m]) - np.mean(auc['retrain'])), 'diff_ci': [lo, hi], 'u95': hi, 'verdict': verdict(lo, hi, margin),
                        'p_above': float((1 + (d <= 0).sum()) / (1 + len(d))), 'p_below': float((1 + (d >= 0).sum()) / (1 + len(d))),
                        'p_equivalent': float(max((1 + (d <= -margin).sum()) / (1 + len(d)), (1 + (d >= margin).sum()) / (1 + len(d))))})
        out['arms'][m] = row
    for fam, ms in FAMILIES.items():
        ms = [m for m in ms if m in out['arms']]
        for key in ('p_above', 'p_below', 'p_equivalent'):
            for m, p in zip(ms, holm([out['arms'][m][key] for m in ms])):
                out['arms'][m][key + '_holm'] = float(p)
        for m in ms:
            r = out['arms'][m]
            r['verdict_holm'] = ('more identifiable than retrained' if r['p_above_holm'] < ALPHA else 'less identifiable than retrained' if r['p_below_holm'] < ALPHA else
                                 'at the retrained level' if r['p_equivalent_holm'] < ALPHA else 'not resolved')
    rt = out['arms']['retrain']
    print(f"retrain arm AUC {rt['auc']:.3f} {rt['auc_ci']} (advantage {rt['advantage']:.3f}); baseline excess {out['baseline_excess_auc']:.3f} {excess_ci}; margin {margin:.3f}")
    for fam in ['reference'] + list(FAMILIES):
        print(f'-- {fam}')
        for m, r in out['arms'].items():
            if r['family'] != fam or m == 'retrain':
                continue
            print(f"  {m:32s} AUC {r['auc']:.3f} adv {r['advantage']:.3f} diff {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] "
                  f"{r['verdict']}" + (f" (Holm: {r['verdict_holm']})" if 'verdict_holm' in r else ''))
    json.dump(out, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
