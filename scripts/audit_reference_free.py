#!/usr/bin/env python3
"""Reference-free likelihood audit, paired type-level verdict, and per-cell z-scored audit (round-7 items).

Replaces three readings that depended on a choice made after the data were seen:
1. the audit against clean retrain seed 42, judged against a range built from the other seeds. Here the
   reference is the mean over every clean retrain, the retrained arm is scored leave-one-out, and the
   interval resamples the reference retrains as well as the target's seeds and the cells (`audit_stats`);
2. the switch to forget-cell medians on PBMC-68k cluster 9 when the audit range exceeded 0.8. Here the
   type-level reading is the paired gap, median (and mean) forget-cell negative ELBO of the target arm
   minus the retrain arm's, with the same nested interval, on every cohort;
   The gap is also given relative to the cross-type negatives' gap (`gap_rel`), which cancels a global change
   in fit such as DP-SGD's;
3. the per-cell z-score of `audit_lira.json`, computed there against cross-type negatives only. Here it
   is computed against the within-type negatives of PBMC-68k (the per-cell question), alongside the
   un-normalized within-type audit. PBMC-33k has five within-type cells and is not scored within type.

Inputs: per-cell negative ELBO of the forget cells and of the negatives under every checkpoint. PBMC-68k
arrays come from the v6 pass (`outputs/pbmc68k/persample_<fs>.json`, forget / cross / within); PBMC-33k
arrays (30 forget cells, 194 matched negatives) are computed here with the deterministic estimator for
every checkpoint of `save_persample_scores.checkpoint_map` + `extended_map` and cached in
`outputs/audit_percell_negelbo_pbmc33k.npz`.

Two-sided p-values from the draws are Holm-adjusted within the families of `paired_diff_33k.py` (published / ablation /
demonstration; on PBMC-68k every method arm is one family). Verdicts (round 8, four-way) read off the two-sided 95 percent interval of the paired difference against the retrain arm:
above zero = fitted better than retrained (signal intact), below zero = fitted worse (over-corrected), inside plus or
minus the margin = fitted as retrained, otherwise not resolved. The margin per instrument is half the 95 percent bound
nearer zero of the baseline's own difference from the retrained arm (the v1 file without margins is kept as
audit_reference_free_v1_nomargin.json). For the gap the signs are reversed (positive = fitted worse).

Writes outputs/audit_reference_free.json. Run: PYTHONPATH=src python scripts/audit_reference_free.py [--cohorts pbmc33k pbmc68k_mk pbmc68k_pdc] [--n_boot 2000]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import audit_stats as AS

OUT = BASE / 'outputs/audit_reference_free.json'
from paired_diff_33k import FAMILIES as FAMILIES_33K   # Holm within the same families as the 33k attacker test
CELL_SEED, REF_SEED = 0, 1


def load_pbmc33k():
    cache = BASE / 'outputs/audit_percell_negelbo_pbmc33k.npz'
    if not cache.exists():
        import scanpy as sc, torch
        import eval_multiseed as EM
        EM.DEVICE = 'cpu'
        from eval_multiseed import load_vae_model, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH
        from likelihood_audit import neg_elbo_per_cell
        import save_persample_scores as S
        import nested_bootstrap_retrain as NBR
        NBR.DEVICE = 'cpu'
        from nested_bootstrap_retrain import load_retrain_model
        adata = sc.read_h5ad(DATA_PATH); split = json.load(open(SPLIT_PATH)); matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
        tens = lambda idx: torch.FloatTensor(np.asarray(adata.X[idx].toarray() if hasattr(adata.X[idx], 'toarray') else adata.X[idx]))
        xf, xm = tens(split['forget_indices']), tens(matched)
        cm = S.checkpoint_map(); cm.update(S.extended_map()); arrays = {}
        for method, seeds in cm.items():
            for sd, path in seeds:
                model = load_retrain_model(str(path)) if method == 'retrain_leaky' else load_vae_model(str(path))[0]
                arrays[f'{method}__{sd}__forget'] = neg_elbo_per_cell(model, xf).astype(np.float64); arrays[f'{method}__{sd}__matched'] = neg_elbo_per_cell(model, xm).astype(np.float64)
            print(f'pbmc33k {method}: {len(seeds)} checkpoints', flush=True)
        np.savez_compressed(cache, **arrays)
    z = np.load(cache); arms = {}
    for key in z.files:
        method, sd, part = key.split('__')
        if method == 'fisher_step_ablation':   # one checkpoint per step cap; split as nested_ci_all.json does
            method = 'fisher_' + sd.split('_seed')[0]
        arms.setdefault(method, {}).setdefault(sd, {})[part] = z[key]
    return {m: {'seeds': list(v), 'forget': np.array([v[s]['forget'] for s in v]), 'cross': np.array([v[s]['matched'] for s in v])} for m, v in arms.items()}, ('cross',)


def load_pbmc68k(fs):
    ps = json.load(open(BASE / f'outputs/pbmc68k/persample_{fs}.json'))['neg_elbo']
    return {m: {'seeds': [r['seed'] for r in rows], **{k: np.array([r[k] for r in rows], dtype=np.float64) for k in ('forget', 'cross', 'within')}} for m, rows in ps.items()}, ('cross', 'within')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk', 'pbmc68k_pdc']); ap.add_argument('--n_boot', type=int, default=2000); args = ap.parse_args()
    B = args.n_boot
    res = json.load(open(OUT)) if OUT.exists() else {'cohorts': {}}
    res.update({'n_boot': B, 'cell_seed': CELL_SEED, 'reference_seed': REF_SEED,
                'method': 'reference = mean over all clean retrains; retrain arm leave-one-out; interval resamples target seeds (by arm name), reference retrains with replacement (shared sequence), and cells (multinomial weights shared by every arm); differences paired within draw',
                'audit_sign': 'above 0.5 / above the retrain arm = forget cells fitted better than by never-seen models', 'gap_sign': 'target minus retrain arm, negative ELBO; positive = fitted worse than retrained', 'gap_rel': 'forget gap minus the cross-type negatives\' gap, so a global shift in fit cancels'})
    for cohort in args.cohorts:
        arms, kinds = load_pbmc33k() if cohort == 'pbmc33k' else load_pbmc68k(cohort.split('_')[1])
        R = arms.pop('retrain'); k = len(R['seeds']); n_f = R['forget'].shape[1]
        wp = AS.cell_weights(n_f, B, CELL_SEED); wn = {kd: AS.cell_weights(R[kd].shape[1], B, CELL_SEED + 1 + i) for i, kd in enumerate(kinds)}
        ref_draws = np.random.default_rng(REF_SEED).integers(0, k, (B, k))
        out = {'n_forget': int(n_f), 'n_negatives': {kd: int(R[kd].shape[1]) for kd in kinds}, 'retrain_seeds': R['seeds'], 'retrain': {}, 'arms': {}}
        # the retrain arm on its own: leave-one-out points and the arm's draws
        for kd in kinds:
            for zs in (False, True):
                name = f'audit_{kd}' + ('_z' if zs else '')
                _, arm_draws = AS.nested_audit(R['forget'], R[kd], R['forget'], R[kd], wp, wn[kd], ref_draws, np.zeros(B, dtype=int), zscore=zs)
                out['retrain'][name] = {'loo_per_seed': AS.loo_audit(R['forget'], R[kd], zs), 'arm_mean': float(np.mean(AS.loo_audit(R['forget'], R[kd], zs))), 'arm_ci': AS.ci(arm_draws)}
        for stat in ('median', 'mean'):
            f = np.median if stat == 'median' else np.mean
            out['retrain'][f'forget_{stat}'] = {'per_seed': [float(f(r)) for r in R['forget']], 'arm_mean': float(np.mean([f(r) for r in R['forget']]))}
        print(f"[{cohort}] retrain arm (LOO): " + ', '.join(f"{n} {v['arm_mean']:.3f} {v['arm_ci']}" for n, v in out['retrain'].items() if 'arm_ci' in v)
              + f"; forget median {out['retrain']['forget_median']['arm_mean']:.1f} (per seed {[round(x) for x in out['retrain']['forget_median']['per_seed']]})", flush=True)
        # pass 1: draws for every arm and instrument (target seeds by arm name, shared reference draws and cells)
        D = {}
        for method, T in arms.items():
            tdraws = np.random.default_rng(zlib.crc32(method.encode())).integers(0, len(T['seeds']), B)
            d = {}
            for kd in kinds:
                for zs in (False, True):
                    name = f'audit_{kd}' + ('_z' if zs else '')
                    td, rd = AS.nested_audit(T['forget'], T[kd], R['forget'], R[kd], wp, wn[kd], ref_draws, tdraws, zscore=zs)
                    d[name] = (AS.point_audit(T['forget'], T[kd], R['forget'], R[kd], zs), td, td - rd)
            for stat in ('median', 'mean'):
                d[f'gap_{stat}'] = AS.nested_gap(T['forget'], R['forget'], wp, ref_draws, tdraws, stat)
                d[f'gap_rel_{stat}'] = AS.nested_gap_rel(T['forget'], T['cross'], R['forget'], R['cross'], wp, wn['cross'], ref_draws, tdraws, stat)
            D[method] = d
        # margins (round 8): half the bound of the baseline's difference from the retrained arm that lies nearer zero
        def half_inner(dr):
            lo, hi = AS.ci(dr); return 0.5 * min(abs(lo), abs(hi))
        margins = {inst: half_inner(D['baseline'][inst][2] if inst.startswith('audit') else D['baseline'][inst]) for inst in D['baseline']}
        out['margins'] = margins; out['margin_rule'] = 'half the 95 percent bound nearer zero of the baseline arm minus the retrained arm, per instrument (round 8)'
        print(f"[{cohort}] margins: " + ', '.join(f'{k} {v:.3f}' for k, v in margins.items()), flush=True)
        # pass 2: rows
        for method, T in arms.items():
            row = {'seeds': T['seeds'], 'n_seeds': len(T['seeds'])}
            for kd in kinds:
                for zs in (False, True):
                    name = f'audit_{kd}' + ('_z' if zs else '')
                    pts, td, d = D[method][name]; lo, hi = AS.ci(d)
                    row[name] = {'per_seed': pts, 'mean': float(np.mean(pts)), 'ci': AS.ci(td), 'diff_vs_retrain': float(np.mean(pts) - out['retrain'][name]['arm_mean']),
                                 'diff_ci': [lo, hi], 'margin': margins[name], 'verdict': AS.verdict_diff(lo, hi, margins[name]), 'p_two_sided': AS.p_two_sided(d)}
            for stat in ('median', 'mean'):
                pts = [float((np.median if stat == 'median' else np.mean)(t)) for t in T['forget']]
                d = D[method][f'gap_{stat}']; lo, hi = AS.ci(d)
                row[f'gap_{stat}'] = {'per_seed': pts, 'target_mean': float(np.mean(pts)), 'gap': AS.point_gap(T['forget'], R['forget'], stat), 'gap_ci': [lo, hi], 'margin': margins[f'gap_{stat}'],
                                      'verdict': AS.verdict_diff(lo, hi, margins[f'gap_{stat}'], better='fitted worse than retrained', worse='fitted better than retrained')}
                d = D[method][f'gap_rel_{stat}']; lo, hi = AS.ci(d)
                row[f'gap_rel_{stat}'] = {'gap': AS.point_gap_rel(T['forget'], T['cross'], R['forget'], R['cross'], stat), 'gap_ci': [lo, hi], 'margin': margins[f'gap_rel_{stat}'],
                                          'verdict': AS.verdict_diff(lo, hi, margins[f'gap_rel_{stat}'], better='fitted worse than retrained', worse='fitted better than retrained'), 'p_two_sided': AS.p_two_sided(d)}
            out['arms'][method] = row
            a = row['audit_cross']; g = row['gap_median']
            line = f"[{cohort}] {method:30s} audit {a['mean']:.3f} [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}] diff {a['diff_vs_retrain']:+.3f} [{a['diff_ci'][0]:+.2f}, {a['diff_ci'][1]:+.2f}] {a['verdict']:28s} gap(median) {g['gap']:+.1f} [{g['gap_ci'][0]:+.1f}, {g['gap_ci'][1]:+.1f}] {g['verdict']}"
            if 'within' in kinds:
                w, wz = row['audit_within'], row['audit_within_z']
                line += f" | within {w['mean']:.3f} [{w['diff_ci'][0]:+.2f}, {w['diff_ci'][1]:+.2f}] {w['verdict'].split()[0]}; z {wz['mean']:.3f} [{wz['diff_ci'][0]:+.2f}, {wz['diff_ci'][1]:+.2f}] {wz['verdict'].split()[0]}"
            print(line, flush=True)
        fams = FAMILIES_33K if cohort == 'pbmc33k' else {'published': [m for m in out['arms'] if m != 'baseline']}
        out['holm'] = {'families': fams, 'alpha': 0.05, 'instruments': ['audit_cross', 'gap_rel_median'] + (['audit_within', 'audit_within_z'] if 'within' in kinds else [])}
        for fam, ms in fams.items():
            ms = [m for m in ms if m in out['arms']]
            for inst in out['holm']['instruments']:
                for m, padj in zip(ms, AS.holm([out['arms'][m][inst]['p_two_sided'] for m in ms])):
                    out['arms'][m][inst]['p_holm'] = float(padj); out['arms'][m][inst]['distinguishable_holm'] = bool(padj < 0.05)
        res['cohorts'][cohort] = out
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'saved {cohort} to {OUT}', flush=True)


if __name__ == '__main__':
    main()
