#!/usr/bin/env python3
"""Generate every table body of Writeup.tex from the canonical result files.

Each table is written to tables/<name>.tex as a bare tabular (the paper supplies the float,
caption, and label), so a re-score regenerates the numbers and no value is typed by hand.
Run: PYTHONPATH=src python scripts/make_tex_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
import canonical as C
from likelihood_audit import nested_auc_ci

OUT = BASE / 'tables'
OUT.mkdir(exist_ok=True)


def J(rel):
    return json.load(open(BASE / rel))


def f3(x):
    return f'{x:.3f}'


def f2(x):
    return f'{x:.2f}'


def f1(x):
    return f'{x:.1f}'


def fb(x):
    """An interval bound at two decimals, except that a nonzero bound that would print as 0.00 keeps enough
    decimals (three, or four) to read as nonzero (round 9, R11: a directional verdict beside a '[0.00, ...]'
    display is the hair's-breadth situation the no-marginal-claims rule excludes)."""
    if x != 0 and abs(x) < 0.005:
        for d in (3, 4, 5, 6):
            if round(x, d) != 0:
                return f'{x:.{d}f}'
    return f'{x:.2f}'


def ci(lo, hi):
    return f'[{fb(lo)}, {fb(hi)}]'


def n(x):
    return f'{int(x):,}'


def opt_ci(rec, key):
    """' [lo, hi]' when rec carries the interval under key, else ''."""
    v = rec.get(key) if isinstance(rec, dict) else None
    return ' ' + ci(v[0], v[1]) if v and v[0] == v[0] else ''


SIDECAR = BASE / 'outputs/audit_ci_sidecar.json'


def sidecar(family):
    if not SIDECAR.exists():
        return None
    return json.load(open(SIDECAR))['families'].get(family)


def write(name, header, rows, align):
    # snippet writers append a trailing %% so that \input does not add a space before the next character (round 9, fixer)
    lines = [f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        if r == 'midrule':
            lines.append('\\midrule')
        else:
            lines.append(' & '.join(r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    (OUT / f'{name}.tex').write_text('\n'.join(lines))
    print(f'  tables/{name}.tex ({len([r for r in rows if r != "midrule"])} rows)')


def config_of(path):
    return torch.load(BASE / path, map_location='cpu', weights_only=False)['config']


# ---------------------------------------------------------------- datasets
def t_datasets():
    p = J('outputs/p1/split_structured.json'); t = J('outputs/tabula_muris/split_structured.json'); g = J('outputs/tcga_brca/split_structured.json')
    def sizes(s):
        nf = s.get('n_forget', len(s['forget_indices'])); nr = s.get('n_retain', len(s['retain_indices'])); nu = s.get('n_unseen', len(s['unseen_indices']))
        return nf, nr, nu
    pf, pr, pu = sizes(p); tf, tr, tu = sizes(t); gf, gr, gu = sizes(g)
    genes = [config_of(x)['input_dim'] for x in ('outputs/p1/baseline/best_model.pt', 'outputs/tabula_muris/baseline/best_model.pt', 'outputs/tcga_brca/baseline/best_model.pt')]
    m_p = len(J('outputs/p1.5/s1_matched_negatives.json')['matched_indices']); m_t = len(J('outputs/tabula_muris/matched_negatives.json')['matched_indices'])
    m_g = J('outputs/tcga_brca/matched_negatives.json')['n_matched']; w_g = J('outputs/tcga_brca/within_subtype_eval/within_subtype_results.json')['n_matched']
    k = {fs: C.pbmc68k_split(fs) for fs in ('mk', 'pdc')}
    k_genes = config_of('outputs/pbmc68k/baseline/best_model.pt')['input_dim']
    def k_cross(fs):
        f = BASE / f'outputs/pbmc68k/matched_negatives_{fs}.json'
        return str(J(f'outputs/pbmc68k/matched_negatives_{fs}.json')['n_matched']) if f.exists() else '--'
    km, kp = k['mk'], k['pdc']
    rows = [
        ['Type', 'scRNA-seq', 'scRNA-seq', 'scRNA-seq', 'scRNA-seq', 'bulk RNA-seq'],
        ['Samples', f'{n(pf + pr + pu)} cells', f'{n(km["n_forget"] + km["n_retain"] + km["n_unseen"])} cells', f'{n(kp["n_forget"] + kp["n_retain"] + kp["n_unseen"])} cells', f'{n(tf + tr + tu)} cells', f'{n(gf + gr + gu)} patients'],
        ['Genes', f'{n(genes[0])} HVG', f'{n(k_genes)} HVG', f'{n(k_genes)} HVG', f'{n(genes[1])} HVG', f'{n(genes[2])} L1000'],
        ['Retain / forget / held out', f'{n(pr)} / {n(pf)} / {n(pu)}', f'{n(km["n_retain"])} / {n(km["n_forget"])} / {n(km["n_unseen"])}', f'{n(kp["n_retain"])} / {n(kp["n_forget"])} / {n(kp["n_unseen"])}', f'{n(tr)} / {n(tf)} / {n(tu)}', f'{n(gr)} / {n(gf)} / {n(gu)}'],
        ['Forget set', f'cluster 13 ({pf} megakaryocytes)', f'cluster 9 ({km["n_forget"]} megakaryocytes)', f'cluster 8 ({kp["n_forget"]} pDC)', f'cluster 33 ({tf} cardiac muscle)', f'Basal subtype ({gf} patients)'],
        ['Forget fraction of training', f'{100 * pf / (pf + pr):.1f}\\%', f'{100 * km["n_forget"] / (km["n_forget"] + km["n_retain"]):.1f}\\%', f'{100 * kp["n_forget"] / (kp["n_forget"] + kp["n_retain"]):.1f}\\%', f'{100 * tf / (tf + tr):.1f}\\%', f'{100 * gf / (gf + gr):.0f}\\%'],
        ['Cross-type negatives', str(m_p), k_cross('mk'), k_cross('pdc'), str(m_t), str(m_g)],
        ['Within-type negatives', '5', str(km['n_within_type_unseen']), str(kp['n_within_type_unseen']), '--', str(w_g)],
    ]
    write('datasets', ['', 'PBMC-33k', 'PBMC-68k (cl.\\ 9)', 'PBMC-68k (cl.\\ 8)', 'Tabula Muris', 'TCGA-BRCA'], rows, 'lccccc')


# ---------------------------------------------------------------- reference and controls
def t_reference():
    nested = J('outputs/p4/multiseed/nested_ci_all.json'); refit = J('outputs/p4/multiseed/retrain_refit.json'); lr = J('outputs/attack_lr_reference.json')
    P = C.pbmc_paired()['arms']; rt, leaky = P['retrain'], P['retrain_leaky']; nf = nested['advantage_null_floor']   # shared-cell nested intervals
    per_auc = C.pbmc_per_seed_auc('retrain'); leaky_auc = C.pbmc_per_seed_auc('retrain_leaky')
    rows = [
        ['Clean retrain, five seeds (reference)', f'{min(per_auc):.3f} to {max(per_auc):.3f}', f3(rt['advantage']) + ' ' + ci(*rt['advantage_ci']), f'{C.audit_all_pairs_range("pbmc33k")[0]:.2f} to {C.audit_all_pairs_range("pbmc33k")[1]:.2f} (every ordered pair of the five retrains)'],
        ['Retrain that drew its own split (superseded)', f'{min(leaky_auc):.3f} to {max(leaky_auc):.3f}', f3(leaky['advantage']) + ' ' + ci(*leaky['advantage_ci']), f3(lr['models']['retrain_leaky']['auc_mean']) + (' ' + ci(*lr['models']['retrain_leaky']['nested_ci95']) if 'nested_ci95' in lr['models']['retrain_leaky'] else '')],
        ['Baseline', f3(C.pbmc_auc('baseline')), f3(C.pbmc_advantage('baseline')['mean']), f3(lr['models']['baseline']['auc_mean']) + (' ' + ci(*lr['models']['baseline']['nested_ci95']) if 'nested_ci95' in lr['models']['baseline'] else '')],
        'midrule',
        ['Refit attacker on the clean retrain (20 x 5-fold)', f3(refit['models']['retrain']['cv_auc_mean']), f3(refit['models']['retrain']['nested_advantage']['mean']) + ' ' + ci(refit['models']['retrain']['nested_advantage']['nested_ci_low'], refit['models']['retrain']['nested_advantage']['nested_ci_high']), '--'],
        ['Refit attacker on the baseline (positive control)', f3(refit['models']['baseline']['cv_auc']), f3(refit['models']['baseline']['cv_advantage']), '--'],
        ['Permutation null of the advantage estimator', '--', f3(nf['mean']) + ' ' + ci(nf['ci95'][0], nf['ci95'][1]), '--'],
    ]
    write('reference', ['Model', 'Fit-excess attacker, AUC', 'Advantage [nested 95\\% CI]', 'Audit AUC [95\\% CI]'], rows, 'lccc')


# ---------------------------------------------------------------- attack suite
def t_attacks():
    a = J('outputs/attack_suite_results.json')
    models = [('baseline', 'Baseline'), ('retrain', 'Clean retrain'), ('extra_grad_lambda10', 'Extra-gradient'), ('fisher_structured', 'Fisher scrubbing')]
    attacks = [('low_assumption', 'loss_recon', 'Reconstruction threshold'), ('low_assumption', 'loss_kl', 'KL threshold'), ('low_assumption', 'loss_elbo', 'ELBO threshold'), ('high_assumption', 'knn_latent', 'k-NN in latent space')]
    rows = []
    for grp, key, label in attacks:
        rows.append([label] + [f"{a[m][grp][key]['auc']:.3f} ({a[m][grp][key]['advantage']:.2f})" for m, _ in models])
    write('attacks', ['Attack'] + [lab for _, lab in models], rows, 'lcccc')


# ---------------------------------------------------------------- confound across cohorts
def t_confound():
    """Cross-type columns: the transferred attacker of each cohort's one-pass scoring. Within-type columns: the
    standardized logistic refit, out of fold (one instrument for every within-type number in the paper)."""
    c12 = C.c12_results(); tm = C.tm_results(); tc = C.tcga_cross_subtype(); wt = C.within_type_standardized()
    def aa(auc):   # AUC with its advantage 2|AUC - 0.5| (round 9, D10: every body table that reports an AUC level carries the advantage)
        return f'{auc:.3f} ({2 * abs(auc - 0.5):.2f})'
    def w(key):
        return aa(wt[key]['logistic_std']['mean'])
    rows = [
        ['PBMC-33k cluster 13 (30 cells; 194 cross)', aa(C.pbmc_auc('baseline')), aa(C.pbmc_auc('retrain')), '--', '--'],
        ['PBMC-33k cluster 12 (49 cells; 57 cross)', aa(c12['methods']['baseline']['auc_mean']), aa(c12['methods']['retrain']['auc_mean']), '--', '--'],
        ['PBMC-33k cluster 7 (50 cells; 80 held out within)', aa(C.larger_cluster()['pbmc_c7']['n=50']['Baseline_cross']['auc']), aa(C.larger_cluster()['pbmc_c7']['n=50']['Retrain_cross']['auc']), w('pbmc33k/c7_within_n50/baseline'), w('pbmc33k/c7_within_n50/retrain')],
    ]
    for fs, label in (('mk', 'PBMC-68k cluster 9'), ('pdc', 'PBMC-68k cluster 8')):
        if not (BASE / f'outputs/pbmc68k/results_{fs}.json').exists():
            continue
        sc_ = C.pbmc68k_summary(fs, 'cross'); pc = C.pbmc68k_refit_std(fs, 'within'); sp = C.pbmc68k_split(fs)
        rows.append([f'{label} ({sp["n_forget"]} cells; {C.pbmc68k_n_matched(fs)} cross, {sp["n_within_type_unseen"]} held out within)',
                     aa(sc_['baseline']['auc']), aa(sc_['retrain']['auc']), aa(pc['baseline']['auc']), aa(pc['retrain']['auc'])])
    rows += [
        ['Tabula Muris cluster 33 (82 cells; 135 cross)', aa(tm['structured']['baseline_auc']), aa(tm['structured']['retrain_auc']), '--', '--'],
        ['Tabula Muris cluster 28 (50 cells; 67 held out within)', aa(C.larger_cluster()['tm_c28']['n=50']['Baseline_cross']['auc']), aa(C.larger_cluster()['tm_c28']['n=50']['Retrain_cross']['auc']), w('tm/c28_within_n50/baseline'), w('tm/c28_within_n50/retrain')],
        ['TCGA-BRCA Basal (158 patients; 89 cross, 30 within)', aa(tc['auc']['baseline']), aa(tc['auc']['retrain']), w('tcga/basal_within/baseline'), w('tcga/basal_within/retrain')],
    ]
    # round 9 (fixer): short headers so that the table is legible under \resizebox; the caption says AUC (advantage) and
    # that the within-type columns are the standardized refit
    write('confound', ['Cohort and forget set', 'Baseline, cross', 'Retrained, cross', 'Baseline, within', 'Retrained, within'], rows, 'lcccc')


# ---------------------------------------------------------------- main results
NAMES = {'baseline': 'Baseline (no unlearning)', 'retrain': 'Clean retrain', 'retain_finetune': 'Retain-only fine-tune', 'gradient_ascent': 'Gradient ascent + fine-tune',
         'ssd': 'SSD', 'scrub': 'SCRUB', 'moon': 'Moon feature unlearning', 'fisher': 'Fisher scrubbing', 'fisher_step_ablation': 'Fisher, larger steps (2 runs)',
         'fisher_step3e-3': 'Fisher, step cap $3\\times 10^{-3}$', 'fisher_step1e-2': 'Fisher, step cap $10^{-2}$',
         'extragradient': 'Extra-gradient ($\\lambda=10$)', 'extragradient_lambda1': 'Extra-gradient ($\\lambda=1$)', 'contrastive': 'Contrastive latent',
         'frozen': 'Frozen critics (4 configurations)', 'dp_sgd': 'DP-SGD ($\\varepsilon=10$, from scratch)', 'retrain_leaky': 'Retrain, own split (superseded)'}
FROZEN = ['frozen_single_lambda5', 'frozen_single_lambda10', 'frozen_multi_lambda5', 'frozen_multi_lambda10']


LABEL = {'more identifiable than retrained': 'more identifiable', 'less identifiable than retrained': 'less identifiable', 'at the retrained level': 'at retrained level',
         'not resolved': 'not resolved', 'positive control': 'trained on the cells', 'reference': 'reference'}
AUDIT_LABEL = {'fitted better than retrained': 'fitted better', 'fitted worse than retrained': 'fitted worse', 'fitted as retrained': 'as retrained', 'not resolved': 'not resolved',
               'not distinguishable from retrained': 'not distinguishable'}


def label(v):
    """The one vocabulary (round 8): attacker verdicts from the paired equivalence tests, audit verdicts from the
    reference-free audit. No 'anchor', 'separates', 'in retrained range', 'known', 'overshot'."""
    return LABEL.get(v, AUDIT_LABEL.get(v, v))


def t_main():
    """PBMC-33k main table (round 8): the paired equivalence test of the transferred attacker's signed AUC against the
    five-retrain arm (margin half the lower bound of the baseline's excess) and, as the diagnostic, the reference-free
    audit against the mean of the retrains with its paired verdict. Utility from utility_intervals.json."""
    PD = C.pbmc_paired(); P = PD['arms']; margin = PD['margin']
    RF = C.audit_reference_free('pbmc33k'); RA = RF['arms']; RR = RF['retrain']['audit_cross']
    U = C.utility_intervals('pbmc33k')['arms']
    order = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'fisher_step3e-3', 'fisher_step1e-2', 'extragradient', 'extragradient_lambda1', 'contrastive', 'frozen', 'dp_sgd']
    def aud(m):
        a = RA[m]['audit_cross']; return f3(a['mean']) + ' ' + ci(*a['ci'])
    rows = []
    for m in order:
        if m == 'frozen':
            aucs = [P[k]['auc'] for k in FROZEN]; advs = [P[k]['advantage'] for k in FROZEN]; dfs = [P[k]['diff'] for k in FROZEN]; auds = [RA[k]['audit_cross']['mean'] for k in FROZEN]
            vs = {P[k]['verdict'] for k in FROZEN}; assert vs == {'less identifiable than retrained'}, vs
            avs = {RA[k]['audit_cross']['verdict'] for k in FROZEN}; assert avs == {'fitted worse than retrained'}, avs
            elbos = [U[k]['elbo'] for k in FROZEN]; mrs = [U[k]['marker_r'] for k in FROZEN]
            rows.append([NAMES[m], '3 each', f'{min(aucs):.3f} to {max(aucs):.3f}', f'{min(advs):.3f} to {max(advs):.3f}', f'{min(dfs):+.3f} to {max(dfs):+.3f}', label(vs.pop()),
                         f'{min(auds):.3f} to {max(auds):.3f}', label(avs.pop()), f'{min(elbos):.1f} to {max(elbos):.1f}', f'{min(mrs):.3f} to {max(mrs):.3f}'])
            continue
        pp = P[m]; u = U[m]
        auc = f3(pp['auc']) + ' ' + ci(*pp['auc_ci']); adv = f3(pp['advantage'])
        if m == 'retrain':
            rows.append([NAMES[m], str(pp['n_seeds']), auc, adv, '--', 'reference', f3(RR['arm_mean']) + ' ' + ci(*RR['arm_ci']), 'reference', f1(u['elbo']), f3(u['marker_r'])])
            rows.append('midrule'); continue
        dif = f"{pp['diff']:+.3f} " + ci(*pp['diff_ci'])
        rows.append([NAMES[m], str(pp['n_seeds']), auc, adv, dif, 'trained on the cells' if m == 'baseline' else label(pp['verdict']), aud(m), 'trained on the cells' if m == 'baseline' else label(RA[m]['audit_cross']['verdict']), f1(u['elbo']), f3(u['marker_r'])])
    write('main', ['Method', 'Seeds', 'Attacker AUC [nested 95\\% CI]', 'Advantage', 'Paired difference [95\\% CI]', 'Attacker', 'Reference-free audit [95\\% CI]', 'Audit', 'ELBO', 'Marker $r$'], rows, 'lccccccccc')
    (OUT / 'main_margin.tex').write_text(f'{margin:.3f}%')
    (OUT / 'main_excess.tex').write_text(f"{PD['baseline_excess_auc']:.3f} (interval {PD['baseline_excess_ci'][0]:.3f} to {PD['baseline_excess_ci'][1]:.3f})%")
    (OUT / 'main_audit_margin.tex').write_text(f"{RF['margins']['audit_cross']:.3f}%")
    band = C.audit_all_pairs_range('pbmc33k')
    (OUT / 'main_band.tex').write_text(f'{band[0]:.2f} to {band[1]:.2f}%')


# ---------------------------------------------------------------- PBMC-68k (notebook 44)
SHORT = {'retain_finetune': 'retain-FT', 'gradient_ascent': 'GA', 'ssd': 'SSD', 'scrub': 'SCRUB'}
K_ORDER = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'fisher', 'extragradient', 'contrastive', 'long_finetune']
NAMES['long_finetune'] = 'Retain-only fine-tune, 200 epochs'


def t_pbmc68k():
    """PBMC-68k table (round 8): per-cell paired equivalence test (within-type refit against the eight-retrain arm,
    margin half the lower bound of the baseline's excess), the reference-free cross-type audit and the paired median
    gap as the type-level diagnostic, utility from utility_intervals.json."""
    rows = []
    for fs, lab in (('mk', 'Cluster 9, megakaryocytes'), ('pdc', 'Cluster 8, plasmacytoid dendritic cells')):
        if not (BASE / f'outputs/pbmc68k/percell_paired_{fs}.json').exists():
            print(f'  (pbmc68k {fs} not tested yet; skipped)'); continue
        PD = C.pbmc68k_percell_paired(fs, 'within'); PP = PD['arms']; sp = C.pbmc68k_split(fs)
        RF = C.audit_reference_free(f'pbmc68k_{fs}'); RA = RF['arms']; RR = RF['retrain']; U = C.utility_intervals(f'pbmc68k_{fs}')['arms']
        rows.append([f'\\multicolumn{{11}}{{l}}{{\\emph{{{lab} ({sp["n_forget"]} forget cells, {sp["n_within_type_unseen"]} within-type and {C.pbmc68k_n_matched(fs)} cross-type negatives; margin {PD["margin"]:.3f})}}}}'])
        for m in K_ORDER:
            if m not in PP or (m != 'retrain' and m not in RA):
                continue
            pp = PP[m]; u = U[m]
            pc = (f'{min(pp["per_seed_auc"]):.3f} to {max(pp["per_seed_auc"]):.3f}' if len(pp['per_seed_auc']) > 1 else f3(pp['mean'])) + ' ' + ci(*pp['nested_ci'])
            if m == 'retrain':
                meds = RR['forget_median']['per_seed']; a = RR['audit_cross']
                rows.append([NAMES[m], str(len(pp['per_seed_auc'])), f'{min(meds):,.0f} to {max(meds):,.0f}', pc, '--', 'reference', f3(a['arm_mean']) + ' ' + ci(*a['arm_ci']), '--', 'reference', f1(u['elbo']), f3(u['marker_r'])])
                rows.append('midrule'); continue
            ra = RA[m]; a = ra['audit_cross']; g = ra['gap_median']; meds = g['per_seed']
            fit = f'{min(meds):,.0f} to {max(meds):,.0f}' if len(meds) > 1 else f'{meds[0]:,.0f}'
            base = m == 'baseline'
            rows.append([NAMES[m], str(len(pp['per_seed_auc'])), fit, pc, '--' if base else f"{pp['diff']:+.3f} " + ci(*pp['diff_ci']), 'trained on the cells' if base else label(pp['verdict']),
                         f3(a['mean']) + ' ' + ci(*a['ci']), f"{g['gap']:+.0f} [{g['gap_ci'][0]:+.0f}, {g['gap_ci'][1]:+.0f}]", 'trained on the cells' if base else label(g['verdict']), f1(u['elbo']), f3(u['marker_r'])])
        rows.append('midrule')
        (OUT / f'pbmc68k_percell_margin_{fs}.tex').write_text(f"{PD['margin']:.3f}%")
        (OUT / f'pbmc68k_excess_{fs}.tex').write_text(f"{PD['baseline_excess']:.3f} (interval {PD['baseline_excess_ci'][0]:.3f} to {PD['baseline_excess_ci'][1]:.3f})%")
        band_c = C.audit_all_pairs_range(f'pbmc68k_{fs}'); (OUT / f'pbmc68k_band_{fs}.tex').write_text(f'{band_c[0]:.2f} to {band_c[1]:.2f}%')
    if rows:
        rows.pop()
        write('pbmc68k', ['Method', 'Seeds', 'Forget median', 'Per-cell AUC, within [nested 95\\% CI]', 'Paired difference [95\\% CI]', 'Cells', 'Reference-free audit [95\\% CI]', 'Median gap [95\\% CI]', 'Type', 'ELBO', 'Marker $r$'], rows, 'lcccccccccc')


def t_longft68k():
    if not (BASE / 'outputs/pbmc68k/long_finetune.json').exists():
        print('  (pbmc68k long fine-tune not run yet; skipped)'); return
    d = C.pbmc68k_long_finetune(); H = d['history']; lv = d['levels']
    def at(epoch, key):
        vals = [next(r[key] for r in h if r['epoch'] == epoch) for h in H.values()]
        return min(vals), max(vals)
    def rng(v, f=f1):
        return f'{f(v[0])} to {f(v[1])}' if abs(v[0] - v[1]) > (0.05 if f is f1 else 0.0005) else f(v[0])
    rows = []
    res0 = C.pbmc68k_results('mk') if (BASE / 'outputs/pbmc68k/results_mk.json').exists() else None
    lfn = res0['per_checkpoint'].get('long_finetune', {}).get('audit_cross_nested') if res0 else None
    for e in (0, 10, 50, 100, 200):
        au = rng(at(e, 'lr_auc'), f3) + (' ' + ci(lfn['nested_ci_low'], lfn['nested_ci_high']) if e == 200 and lfn else '')
        rows.append([str(e), rng(at(e, 'forget_neg_elbo')), rng(at(e, 'within_neg_elbo')), rng(at(e, 'retain_neg_elbo')), rng(at(e, 'unseen_neg_elbo')), rng(at(e, 'mlp_auc'), f3), au])
    rows.append('midrule')
    rt = [v for k, v in lv.items() if k.startswith('retrain_clean')]
    rr = lambda key: rng((min(v[key] for v in rt), max(v[key] for v in rt)))
    res = C.pbmc68k_results('mk') if (BASE / 'outputs/pbmc68k/results_mk.json').exists() else None
    ra = rng((min(res['summary']['cross']['retrain']['per_seed_auc']), max(res['summary']['cross']['retrain']['per_seed_auc'])), f3) if res else '--'
    band = C.audit_all_pairs_range('pbmc68k_mk') if res else None
    rows.append(['Retrained models', rr('forget'), rr('within'), rr('retain_sample'), rr('unseen'), ra, f'{band[0]:.2f} to {band[1]:.2f} (every ordered pair)' if band else '--'])
    write('longft68k', ['Retain epochs', 'Forget cells, mean', 'Held-out megakaryocytes, mean', 'Retain sample, mean', 'Held-out cells, mean', 'Fit-excess attacker, cross', 'Audit vs.\\ retrain seed 42, cross [nested 95\\% CI at 200]'], rows, 'lcccccc')


# ---------------------------------------------------------------- Fisher overlap
def t_fisher():
    f = C.pbmc_fisher(); dc = J('outputs/p6/fisher_overlap_deep_clf.json'); z8 = J('outputs/p6/fisher_overlap_z8.json'); cv = J('outputs/p6/fisher_overlap_conditional.json'); tg = C.tcga_fisher()
    pc = f['per_category_full']
    rows = [
        ['VAE encoder', n(pc['Encoder']['n_params']), f3(pc['Encoder']['cosine'])],
        ['VAE bottleneck', n(pc['Bottleneck']['n_params']), f3(pc['Bottleneck']['cosine'])],
        ['VAE decoder hidden', n(pc['Decoder hidden']['n_params']), f3(pc['Decoder hidden']['cosine'])],
        ['VAE decoder output', n(pc['Decoder output']['n_params']), f3(pc['Decoder output']['cosine'])],
        ['VAE, all parameters', n(f['metadata'].get('total_params', 7349984) if 'total_params' in f['metadata'] else J('outputs/p6/fisher_overlap_results.json')['vae']['total_params']), f3(f['global_cosine'])],
        'midrule',
        ['Linear classifier, 14 classes (class-specific output weights)', n(J('outputs/p6/fisher_overlap_results.json')['classifier']['total_params']), f3(f['classifier_cosine'])],
        ['Deep MLP classifier, shared hidden layers', n(dc['per_category']['Shared hidden']['n_params']), f3(dc['per_category']['Shared hidden']['cosine'])],
        ['Deep MLP classifier, class-specific output', n(dc['per_category']['Class-specific output']['n_params']), f3(dc['per_category']['Class-specific output']['cosine'])],
        'midrule',
        ['VAE with $z=8$, all parameters', n(z8['total_params']), f3(z8['global_cosine'])],
        ['Conditional VAE, cluster-specific output columns', '--', f3(cv['fc_mean_decomposition']['cluster_cols_cosine'])],
        ['Conditional VAE, encoder', n(cv['per_category']['Encoder']['n_params']), f3(cv['per_category']['Encoder']['cosine'])],
        ['TCGA-BRCA VAE, all parameters', n(J('outputs/tcga_brca/fisher_overlap_results.json')['n_params']), f3(tg['global_cosine'])],
    ]
    write('fisher', ['Parameter group', 'Parameters', 'Forget--retain Fisher cosine'], rows, 'lrc')


# ---------------------------------------------------------------- phase audit
PHASE_HEADER = ['Stage', 'Forget, mean', 'Retain, mean', 'Held out, mean', 'Fit-excess attacker, AUC', 'Advantage', 'Audit AUC [95\\% CI]']


def phase_row(label, r):
    """One stage of the phase audit: mean negative ELBO per cell of the forget, retain and held-out cells, the attacker's
    AUC with its advantage (round 9, D10) and the seed-42 audit with its interval."""
    return [label, f1(r['forget_neg_elbo']), f1(r['retain_neg_elbo']), f1(r['unseen_neg_elbo']), f3(r['mlp_auc']), f2(2 * abs(r['mlp_auc'] - 0.5)), f3(r['lr_auc']) + opt_ci(r, 'lr_auc_ci')]


def t_phase():
    pa = J('outputs/p6/phase_audit.json')['runs']
    rows = [phase_row('Baseline', pa['baseline'])]
    for step, lab in (('fisher_step0.001', '$10^{-3}$'), ('fisher_step0.003', '$3\\times 10^{-3}$'), ('fisher_step0.01', '$10^{-2}$')):
        rows.append(phase_row(f'Fisher scrub, max step {lab}, after scrub', pa[step]['after_scrub']))
        rows.append(phase_row(f'Fisher scrub, max step {lab}, after retain fine-tune', pa[step]['after_finetune']))
    rows.append(phase_row('Gradient ascent, after ascent', pa['gradient_ascent']['after_ascent']))
    rows.append(phase_row('Gradient ascent, after retain fine-tune', pa['gradient_ascent']['after_finetune']))
    write('phase', PHASE_HEADER, rows, 'lcccccc')


# ---------------------------------------------------------------- likelihood-targeted unlearning
def t_ltu():
    """Likelihood-targeted unlearning (NB41, targeted_unlearning.json): the forget fit (mean negative ELBO per cell), the
    canonical attacker's paired test (paired_diff_33k.json, arms ltu / ltu_avg / ltu_disjoint), the reference-free audit
    with its paired verdict (audit_reference_free.json; round 9, R6 / D12: the seed-42 audit of the sidecar and the
    '0.24 to 0.52' band left this table) and held-out utility from utility_intervals.json, the source of tab:main
    (round 9, R12: one source for the baseline's ELBO)."""
    d = J('outputs/p7/targeted_unlearning.json'); lv = J('outputs/p7/calibration_levels.json')['mean_neg_elbo']
    P = C.pbmc_paired()['arms']; RF = C.audit_reference_free('pbmc33k'); RA = RF['arms']; RR = RF['retrain']['audit_cross']; U = C.utility_intervals('pbmc33k')['arms']
    def aud(m):
        a = RA[m]['audit_cross']; return f3(a['mean']) + ' ' + ci(*a['ci'])
    def agg(block, family):
        # attacker columns from the canonical pass (paired_diff_33k.json), not NB41's own attacker instance (2026-09-25)
        f = np.array([r['forget_neg_elbo'] for r in block]); pp = P[family]; u = U[family]
        return [f1(f.mean()), f3(pp['auc']), f3(pp['advantage']) + ' ' + ci(*pp['advantage_ci']), f"{pp['diff']:+.3f} " + ci(*pp['diff_ci']), label(pp['verdict']),
                aud(family), label(RA[family]['audit_cross']['verdict']), f1(u['elbo']), f3(u['marker_r'])]
    pb, pr, ub, ur = P['baseline'], P['retrain'], U['baseline'], U['retrain']
    rows = [
        ['Baseline', f1(lv['baseline']['forget']), f3(pb['auc']), f3(pb['advantage']), f"{pb['diff']:+.3f} " + ci(*pb['diff_ci']), 'trained on the cells', aud('baseline'), 'trained on the cells', f1(ub['elbo']), f3(ub['marker_r'])],
        ['Clean retrains (five seeds)', f1(np.mean([v['forget'] for k, v in lv.items() if k.startswith('retrain')])), f3(pr['auc']), f3(pr['advantage']) + ' ' + ci(*pr['advantage_ci']), '--', 'reference', f3(RR['arm_mean']) + ' ' + ci(*RR['arm_ci']), 'reference', f1(ur['elbo']), f3(ur['marker_r'])],
        'midrule',
        ['Target: ten nearest negatives'] + agg(d['per_seed'], 'ltu'),
        ['Target: mean over the 194 negatives (retrained level)'] + agg(d['cohort_average_target']['per_seed'], 'ltu_avg'),
        [f'Target: {d["disjoint_calibration_target"]["n_calibration_cells"]} disjoint held-out neighbors'] + agg(d['disjoint_calibration_target']['per_seed'], 'ltu_disjoint'),
    ]
    # round 9 (fixer): short headers and row labels for legibility under \resizebox; the caption defines each column
    write('ltu', ['Model', 'Forget fit', 'AUC', 'Advantage [95\\% CI]', 'Difference [95\\% CI]', 'Verdict', 'Audit [95\\% CI]', 'Verdict', 'ELBO', 'Marker $r$'], rows, 'lccccccccc')
    rt = {k: v for k, v in lv.items() if k.startswith('retrain')}
    rows = [['Baseline', f1(lv['baseline']['forget']), f1(lv['baseline']['matched_negatives']), f1(lv['baseline']['ten_nearest_matched_per_forget_cell']), f1(lv['baseline']['disjoint_lookalikes'])]]
    rows.append(['Clean retrains (range over five seeds)', f'{min(v["forget"] for v in rt.values()):.1f} to {max(v["forget"] for v in rt.values()):.1f}', f'{min(v["matched_negatives"] for v in rt.values()):.1f} to {max(v["matched_negatives"] for v in rt.values()):.1f}', f'{min(v["ten_nearest_matched_per_forget_cell"] for v in rt.values()):.1f} to {max(v["ten_nearest_matched_per_forget_cell"] for v in rt.values()):.1f}', f'{min(v["disjoint_lookalikes"] for v in rt.values()):.1f} to {max(v["disjoint_lookalikes"] for v in rt.values()):.1f}'])
    write('levels', ['Model', 'Forget cells, mean', 'Matched negatives, mean', 'Ten nearest negatives, mean', 'Disjoint neighbors, mean'], rows, 'lcccc')


# ---------------------------------------------------------------- long retain fine-tuning
def t_longft():
    d = J('outputs/p8/long_finetune.json')
    def at(hists, epoch, key):
        vals = [next(r[key] for r in h if r['epoch'] == epoch) for h in hists.values()]
        return min(vals), max(vals)
    def rng(v, f=f1):
        return f'{f(v[0])} to {f(v[1])}' if abs(v[0] - v[1]) > (0.05 if f is f1 else 0.0005) else f(v[0])
    hp, ht = d['pbmc']['histories'], d['tcga']['histories']
    rows = []
    sp, st = sidecar('long_finetune_pbmc'), sidecar('long_finetune_tcga')
    for e in (0, 10, 50, 100, 200):
        ap = rng(at(hp, e, 'lr_auc'), f3) + (' ' + ci(sp['nested']['nested_ci_low'], sp['nested']['nested_ci_high']) if e == 200 and sp else '')
        atc = rng(at(ht, e, 'lr_auc'), f3) + (' ' + ci(st['nested']['nested_ci_low'], st['nested']['nested_ci_high']) if e == 200 and st else '')
        rows.append([str(e), rng(at(hp, e, 'forget_neg_elbo')), rng(at(hp, e, 'mlp_auc'), f3), ap, rng(at(ht, e, 'forget_neg_elbo')), rng(at(ht, e, 'unseen_basal_neg_elbo')), atc])
    rows.append('midrule')
    # the per-epoch instruments are NB42's own: its attacker instance and the audit against retrain seed 42 (round 9, D12);
    # the retrained models' attacker level is the canonical instance's, and the cell says so
    rows.append(['Retrained models', f1(d['pbmc']['retrained_forget_level']), f3(C.pbmc_auc('retrain')) + ' (canonical instance)', f'{d["pbmc"]["lr_band"][0]:.2f} to {d["pbmc"]["lr_band"][1]:.2f}', f1(d['tcga']['retrained_forget_level']), f1(d['tcga']['retrained_unseen_basal_level']), '--'])
    write('longft', ['Retain epochs', 'PBMC forget negELBO, mean', 'PBMC attacker (the run\'s own instance)', 'PBMC audit vs.\\ retrain seed 42 [nested 95\\% CI at 200]', 'TCGA forget negELBO, mean', 'TCGA unseen Basal, mean', 'TCGA audit vs.\\ the TCGA retrain [95\\% CI at 200]'], rows, 'lcccccc')


# ---------------------------------------------------------------- simulation
def t_sim():
    d = J('outputs/p9/synthetic_support.json')
    rows = []
    for c in d['configurations']:
        t = {r['model']: r for r in c['table']}; e = {r['epoch']: r for r in c['erosion']}
        rows.append([f"{c['distance_sd']:.0f} sd, {'all' if c['fraction'] == 1.0 else 'half'}",
                     f"{t['baseline']['forget_neg_elbo']:.1f} / {c['oracle_level']:.1f}",
                     f"{t['baseline']['audit_cross']:.2f}" + opt_ci(t['baseline'], 'audit_cross_ci') + f" ({min(c.get('audit_band_all_pairs', c['audit_band'])):.2f} to {max(c.get('audit_band_all_pairs', c['audit_band'])):.2f})",
                     f"{t['baseline']['attacker_within']:.2f} / {t['retrain']['attacker_within']:.2f} / {t['retain fine-tune, 100 epochs']['attacker_within']:.2f}",
                     f"{e[10]['forget_neg_elbo']:.1f} / {e[100]['forget_neg_elbo']:.1f}",
                     f"{t['targeted, oracle level']['audit_cross']:.2f}" + opt_ci(t['targeted, oracle level'], 'audit_cross_ci') + f", {t['targeted, oracle level']['attacker_within']:.2f}",
                     f"{t['targeted, cross look-alikes']['forget_neg_elbo']:.1f}"])
    write('sim', ['Configuration', 'Forget fit (mean), base / retrained', 'Audit, base [95\\% CI] (retrained range)', 'Within attacker, base / retr. / FT100', 'FT forget (mean) at 10 / 100', 'Retrained level as target, audit [95\\% CI] / within', 'Neighbor target, forget (mean)'], rows, 'lcccccc')


# ---------------------------------------------------------------- appendix tables
def t_utility():
    util = J('outputs/p4/multiseed/utility_all.json')['methods']
    for k in C.STEP_SPLIT:
        util[k] = {'mean': C.pbmc_step_split(util['fisher_step_ablation']['per_seed'], k)}
    order = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'fisher_step3e-3', 'fisher_step1e-2', 'extragradient', 'extragradient_lambda1', 'contrastive'] + FROZEN + ['dp_sgd', 'retrain_leaky']
    names = dict(NAMES); names.update({'frozen_single_lambda5': 'Frozen critics, one critic, $\\lambda=5$', 'frozen_single_lambda10': 'Frozen critics, one critic, $\\lambda=10$', 'frozen_multi_lambda5': 'Frozen critics, three critics, $\\lambda=5$', 'frozen_multi_lambda10': 'Frozen critics, three critics, $\\lambda=10$'})
    rows = [[names[m], f1(util[m]['mean']['elbo']), f2(util[m]['mean']['kl']), f3(util[m]['mean']['marker_r']), f3(util[m]['mean']['ari'])] for m in order if m in util]
    write('utility', ['Model', 'ELBO', 'KL', 'Marker $r$', 'ARI'], rows, 'lcccc')


def t_sizes():
    d = J('outputs/p4/size_ablations_v2.json')
    rows = []
    for size, g in d['structured_extragradient'].items():
        rows.append([size, str(g['n_matched']), f"{g['baseline']['auc_mean']:.3f} ({g['baseline']['lr_auc_mean']:.2f})", f"{g['retrain']['auc_mean']:.3f}, {g['retrain']['adv_mean']:.3f} ({g['retrain']['lr_auc_mean']:.2f})",
                     f"{g['extragradient']['auc_mean']:.3f}, {g['extragradient']['adv_mean']:.3f} {ci(*g['extragradient']['adv_ci'])} ({g['extragradient']['lr_auc_mean']:.2f})"])
    write('size_structured', ['Forget cells', 'Negatives', 'Baseline AUC (audit)', 'Retrained AUC, advantage (audit)', 'Extra-gradient AUC, advantage [CI] (audit)'], rows, 'lcccc')
    rows = []
    for size, g in d['scattered_fisher'].items():
        rows.append([size, str(g['n_matched']), f"{g['baseline']['auc_mean']:.3f} ({g['baseline']['lr_auc_mean']:.2f})", f"{g['retrain']['auc_mean']:.3f} ({g['retrain']['lr_auc_mean']:.2f})", f"{g['fisher']['auc_mean']:.3f} {ci(*g['fisher']['adv_ci'])} ({g['fisher']['lr_auc_mean']:.2f})"])
    write('size_scattered', ['Forget cells', 'Negatives', 'Baseline AUC (audit)', 'Retrained AUC (audit)', 'Fisher AUC [advantage CI] (audit)'], rows, 'lcccc')


def t_crossdataset():
    tm = C.tm_results(); tc = C.tcga_cross_subtype(); tw = C.tcga_within_subtype(); mu = J('outputs/attack_lr_reference_multi.json')
    def au(cohort, key):
        r = mu[cohort]['models'][key]
        return f3(r['lr_auc']) + (' ' + ci(*r['ci95']) if 'ci95' in r else '')
    rows = [['Baseline', f3(tm['structured']['baseline_auc']), au('tabula_muris', 'baseline'), f3(tc['auc']['baseline']), f3(tw['structured']['baseline']['auc']), au('tcga_brca', 'baseline')],
            ['Retrained model', f3(tm['structured']['retrain_auc']), '--', f3(tc['auc']['retrain']), f3(tw['structured']['retrain']['auc']), '--']]
    rows.append('midrule')
    tm_methods = {'Extra-gradient': ('extra-gradient (structured)', 'extragradient'), 'Fisher scrubbing': ('fisher (structured)', 'fisher')}
    tc_methods = {'Fisher scrubbing': ('Fisher', 'fisher'), 'Retain-only fine-tune': ('Retain-FT', 'retain_finetune'), 'Gradient ascent + fine-tune': ('Grad. ascent', 'gradient_ascent'), 'SSD': ('SSD', 'ssd'), 'SCRUB': ('SCRUB', 'scrub'), 'Contrastive latent': ('Contrastive', 'contrastive')}
    for name in ['Retain-only fine-tune', 'Gradient ascent + fine-tune', 'SSD', 'SCRUB', 'Contrastive latent', 'Fisher scrubbing', 'Extra-gradient']:
        r = [name]
        if name in tm_methods:
            k, a = tm_methods[name]; r += [f3(tm['unlearning'][k]), au('tabula_muris', a)]
        else:
            r += ['--', '--']
        if name in tc_methods:
            k, a = tc_methods[name]; r += [f3(tc['auc'][k]), f3(tw['structured'][k]['auc']), au('tcga_brca', a)]
        else:
            r += ['--', '--', '--']
        rows.append(r)
    write('crossdataset', ['Model', 'TM attacker (cross)', 'TM audit [95\\% CI]', 'TCGA attacker (cross)', 'TCGA attacker (within)', 'TCGA audit [95\\% CI]'], rows, 'lccccc')


def t_mog08():
    d = J('outputs/p3/mog/mog_results.json')
    rows = []
    for name, s in d['scenarios'].items():
        rows.append('midrule') if rows else None
        for r in s['table']:
            rows.append([f"{name}: {r['model']}", f1(r['forget_neg_elbo']), f3(r['attacker_cross']), f3(r['attacker_within']), f3(r['audit_cross']) + opt_ci(r, 'audit_cross_ci'), f3(r['audit_within']) + opt_ci(r, 'audit_within_ci')])
    write('mog08', ['Scenario and model', 'Forget negELBO, mean', 'Attacker, cross', 'Attacker, within', 'Audit, cross [95\\% CI]', 'Audit, within [95\\% CI]'], rows, 'lccccc')


def t_sim_full():
    d = J('outputs/p9/synthetic_support.json')
    rows = []
    for c in d['configurations']:
        rows.append('midrule') if rows else None
        for r in c['table']:
            rows.append([f"{c['distance_sd']:.0f} sd, {'all' if c['fraction'] == 1.0 else 'half'}: {r['model']}", f1(r['forget_neg_elbo']), f3(r['attacker_cross']), f3(r['attacker_within']), f3(r['audit_cross']) + opt_ci(r, 'audit_cross_ci'), f3(r['audit_within']) + opt_ci(r, 'audit_within_ci')])
    write('sim_full', ['Configuration and model', 'Forget negELBO, mean', 'Attacker, cross', 'Attacker, within', 'Audit, cross [95\\% CI]', 'Audit, within [95\\% CI]'], rows, 'lccccc')


def t_refit():
    refit = J('outputs/p4/multiseed/retrain_refit.json')['models']
    rows = [['Baseline (positive control)', f3(refit['baseline']['cv_auc']), f3(refit['baseline']['cv_advantage']), f3(refit['baseline']['insample_auc'])]]
    rows.append('midrule')
    for i, r in enumerate(refit['retrain']['per_seed']):
        rows.append([f'Clean retrain, seed {42 + i}', f3(r['cv_auc']), f3(r['cv_advantage']), f3(r['insample_auc'])])
    rt = refit['retrain']
    rows.append(['Clean retrains, mean [nested 95\\% CI]', f3(rt['cv_auc_mean']) + ' ' + ci(rt['nested_auc']['nested_ci_low'], rt['nested_auc']['nested_ci_high']),
                 f3(rt['cv_advantage_mean']) + ' ' + ci(max(0.0, rt['nested_advantage']['nested_ci_low']), rt['nested_advantage']['nested_ci_high']), f3(rt['insample_auc_mean'])])
    write('refit', ['Model', 'Out-of-fold AUC', 'Out-of-fold advantage', 'In-sample AUC'], rows, 'lccc')


def t_refit_std():
    fg = C.feature_groups(); wt = C.within_type_standardized()
    groups = ['all 70', 'latent mean (32)', 'latent logvar (32)', 'likelihood terms (3)', 'recon_nll only', 'kNN distance (1)']
    heads = ['All 70', 'Latent mean', 'Latent log-var', 'Likelihood terms', 'Reconstruction', 'kNN distance']
    rows = []
    for key, label in (('pbmc33k/c13_cross/baseline', 'PBMC-33k cluster 13 vs 194 cross-type, baseline'), ('pbmc33k/c13_cross/retrain_42', 'PBMC-33k cluster 13 vs 194 cross-type, clean retrain'),
                       ('pbmc68k/mk_within/baseline', 'PBMC-68k cluster 9 vs 71 within-type, baseline'), ('pbmc68k/mk_within/retrain_42', 'PBMC-68k cluster 9 vs 71 within-type, clean retrain'),
                       ('pbmc68k/pdc_within/baseline', 'PBMC-68k cluster 8 vs 201 within-type, baseline'), ('pbmc68k/pdc_within/retrain_42', 'PBMC-68k cluster 8 vs 201 within-type, clean retrain')):
        rows.append([label] + [f3(fg[key][g][0]) for g in groups])
    write('feature_groups', ['Forget cells vs negatives, model'] + heads, rows, 'lcccccc')
    rows = []
    for key, label in (('pbmc33k/c13_cross/baseline', 'PBMC-33k cluster 13, 30 vs 194 cross-type, baseline'), ('pbmc33k/c13_cross/retrain_seed42', 'PBMC-33k cluster 13, 30 vs 194 cross-type, clean retrain'),
                       ('pbmc33k/c7_within_n50/baseline', 'PBMC-33k cluster 7, 50 vs 80 within, baseline'), ('pbmc33k/c7_within_n50/retrain', 'PBMC-33k cluster 7, 50 vs 80 within, retrained'),
                       ('pbmc33k/c7_within_n100/baseline', 'PBMC-33k cluster 7, 100 vs 80 within, baseline'), ('pbmc33k/c7_within_n100/retrain', 'PBMC-33k cluster 7, 100 vs 80 within, retrained'),
                       ('tm/c28_within_n50/baseline', 'Tabula Muris cluster 28, 50 vs 67 within, baseline'), ('tm/c28_within_n50/retrain', 'Tabula Muris cluster 28, 50 vs 67 within, retrained'),
                       ('tm/c28_within_n100/baseline', 'Tabula Muris cluster 28, 100 vs 67 within, baseline'), ('tm/c28_within_n100/retrain', 'Tabula Muris cluster 28, 100 vs 67 within, retrained'),
                       ('tcga/basal_within/baseline', 'TCGA-BRCA Basal, 158 vs 30 within, baseline'), ('tcga/basal_within/retrain', 'TCGA-BRCA Basal, 158 vs 30 within, retrained')):
        r = wt[key]
        rows.append([label, f"{r['logistic_std']['mean']:.3f} $\\pm$ {r['logistic_std']['sd']:.3f}", f"{r['mlp_std']['mean']:.3f} $\\pm$ {r['mlp_std']['sd']:.3f}", f"{r['mlp_raw']['mean']:.3f} $\\pm$ {r['mlp_raw']['sd']:.3f}"])
    for fs, label, n in (('mk', 'PBMC-68k cluster 9, 107 vs 71 within', 71), ('pdc', 'PBMC-68k cluster 8, 302 vs 201 within', 201)):
        if not (BASE / f'outputs/pbmc68k/results_{fs}.json').exists():
            continue
        rs = C.pbmc68k_refit_std(fs, 'within')
        if 'baseline' not in rs:
            continue
        raw = C.pbmc68k_refit(fs, 'within')
        rows.append([f'{label}, baseline', f"{rs['baseline']['auc']:.3f} $\\pm$ {rs['baseline']['split_sd']:.3f}", '--', f"{raw['baseline']['cv_auc_mean']:.3f} $\\pm$ {raw['baseline']['cv_auc_split_sd']:.3f}"])
        rows.append([f'{label}, clean retrains ({rs["retrain"]["n_seeds"]})', f"{min(rs['retrain']['auc_per_seed']):.3f} to {max(rs['retrain']['auc_per_seed']):.3f}", '--', f"{min(raw['retrain']['cv_auc_per_seed']):.3f} to {max(raw['retrain']['cv_auc_per_seed']):.3f}"])
    write('refit_std', ['Forget cells vs negatives, model', 'Logistic, standardized', 'MLP, standardized', 'MLP, raw (the paper\'s attacker)'], rows, 'lccc')


def t_methods():
    """Descriptive table of the methods (no result numbers). Budgets as run (round 9, R3 / D5): the retain-only fine-tune
    trains 50 epochs with patience 10 (src/train_retain_finetune.py defaults, both launchers; every published run on
    both cohorts completed all 50, best epoch 43 to 50), gradient ascent ten ascent steps then up to 30 retain epochs with
    patience 10 (src/train_gradient_ascent.py), Moon 20 decoder epochs (src/train_moon_unlearn.py), Fisher scrubbing, SSD,
    SCRUB and contrastive unlearning ten appended retain epochs, the frozen critics 50 epochs (tab:hparams)."""
    rows = [
        ['Retain-only fine-tuning', 'edit, then fine-tune', 'continues training on $\\mathcal{R}$', '50 epochs (patience 10)'],
        ['Gradient ascent', 'edit, then fine-tune', 'ten clipped ascent steps on the forget loss, then retain fine-tune', '10 ascent steps, then up to 30 retain epochs (patience 10)'],
        ['Fisher scrubbing', 'edit, then fine-tune', 'step against the forget gradient scaled by the inverse retain Fisher, capped, then retain fine-tune', '100 scrub steps, then 10 retain epochs'],
        ['SSD', 'edit, then fine-tune', 'multiplicative dampening by forget-set Fisher importance, then retain fine-tune', 'one dampening pass, then 10 retain epochs'],
        ['SCRUB', 'edit, then fine-tune', 'distills from the baseline, matching it on $\\mathcal{R}$ and diverging on $\\mathcal{F}$', '20 distillation epochs, then 10 retain epochs'],
        ['Moon feature unlearning', 'edit, then fine-tune', 'decoder fine-tune with a forget-versus-retain latent direction removed', '20 decoder epochs'],
        ['Contrastive latent', 'continuous objective', 'forget posteriors pushed to the prior, retain representations held', '20 epochs, then 10 retain epochs'],
        ['Frozen critics', 'continuous objective', 'VAE updated to defeat fixed pre-trained attackers (one or three, two weights)', '50 epochs'],
        ['Extra-gradient co-training', 'continuous objective', 'min-max game against three co-trained critics, extragradient updates', '50 passes, $\\lambda = 10$ (1)'],
        ['Synthetic augmentation', 'constructive', 'trains on $\\mathcal{R}$ plus synthetic forget-type cells from held-out seeds', 'from scratch'],
        ['Representation alignment', 'constructive', 'fine-tunes the baseline until its forget-cell posteriors match a retrained model\'s', '--'],
        ['DP-SGD', 'from scratch', 'trains on $\\mathcal{R}$ with $\\varepsilon = 10$', 'from scratch'],
        ['Likelihood-targeted fine-tune (Section~\\ref{sec:target})', 'demonstration', 'retain loss plus a penalty on the gap between each forget cell\'s negative ELBO and a target', 'until the gap stays small'],
    ]
    write('methods', ['Method', 'Family', 'What it does', 'Budget'], rows, 'llll')


def t_phase_short():
    pa = J('outputs/p6/phase_audit.json')['runs']
    rows = [phase_row('Baseline', pa['baseline']),
            phase_row('Fisher scrub, default step, after scrub', pa['fisher_step0.001']['after_scrub']),
            phase_row('Fisher scrub, default step, after retain fine-tune', pa['fisher_step0.001']['after_finetune']),
            phase_row('Fisher scrub, ten times the step, after scrub', pa['fisher_step0.01']['after_scrub']),
            phase_row('Fisher scrub, ten times the step, after retain fine-tune', pa['fisher_step0.01']['after_finetune']),
            phase_row('Gradient ascent, after retain fine-tune', pa['gradient_ascent']['after_finetune'])]
    write('phase_short', PHASE_HEADER, rows, 'lcccccc')


def t_pbmc68k_excerpt():
    """Body excerpt of the PBMC-68k table (round 8): the four edit-then-fine-tune methods pooled; verdicts from the
    paired equivalence test (cells) and the reference-free median gap (type)."""
    rows = []
    for fs, lab, keep in (('mk', 'Cluster 9, megakaryocytes (107 forget, 71 held out)', ['baseline', 'retrain', 'eft', 'fisher', 'contrastive', 'extragradient', 'long_finetune']),
                          ('pdc', 'Cluster 8, plasmacytoid dendritic cells (302 forget, 201 held out)', ['baseline', 'retrain', 'eft', 'fisher', 'contrastive', 'extragradient'])):
        if not (BASE / f'outputs/pbmc68k/percell_paired_{fs}.json').exists():
            continue
        PP = C.pbmc68k_percell_paired(fs, 'within')['arms']; RF = C.audit_reference_free(f'pbmc68k_{fs}'); RA = RF['arms']; RR = RF['retrain']
        rows.append([f'\\multicolumn{{8}}{{l}}{{\\emph{{{lab}}}}}'])
        def rng2(v, f):
            lo, hi = f(min(v)), f(max(v))
            return lo if lo == hi else f'{lo} to {hi}'
        names = {'baseline': 'Baseline (trained on the cells)', 'retrain': 'Clean retrains (8)', 'eft': 'Edit, then fine-tune (4 methods)', 'fisher': 'Fisher scrubbing', 'contrastive': 'Contrastive latent', 'extragradient': 'Extra-gradient ($\\lambda=10$)', 'long_finetune': 'Retain-only fine-tune, 200 epochs'}
        for k in keep:
            ms = ['retain_finetune', 'gradient_ascent', 'ssd', 'scrub'] if k == 'eft' else [k]
            if k == 'retrain':
                meds = RR['forget_median']['per_seed']; a = RR['audit_cross']
                rows.append([names[k], rng2(meds, lambda x: f'{x:,.0f}'), rng2(PP['retrain']['per_seed_auc'], lambda x: f'{x:.2f}') + ' ' + ci(*PP['retrain']['nested_ci']), adv_range(PP['retrain']['per_seed_auc']), '--', 'reference', f3(a['arm_mean']) + ' ' + ci(*a['arm_ci']), 'reference'])
                continue
            meds = [x for m in ms for x in RA[m]['gap_median']['per_seed']]
            aucs = [x for m in ms for x in PP[m]['per_seed_auc']]
            per = rng2(aucs, lambda x: f'{x:.2f}') + (' ' + ci(*PP[k]['nested_ci']) if len(ms) == 1 else '')
            dif = ('--' if k == 'baseline' else f"{PP[k]['diff']:+.3f} " + ci(*PP[k]['diff_ci'])) if len(ms) == 1 else rng2([PP[m]['diff'] for m in ms], lambda x: f'{x:+.3f}')
            if k == 'baseline':
                cells = typ = 'trained on the cells'
            elif len(ms) == 1:
                cells, typ = label(PP[k]['verdict']), label(RA[k]['gap_median']['verdict'])
            else:
                groups = {}
                for x in ms:
                    groups.setdefault(label(PP[x]['verdict']), []).append(SHORT[x])
                cells = '; '.join(f"{g} ({', '.join(v)})" for g, v in groups.items())
                tv = {label(RA[x]['gap_median']['verdict']) for x in ms}; typ = tv.pop() if len(tv) == 1 else 'mixed'
            aud = rng2([RA[m]['audit_cross']['mean'] for m in ms], f3) + (' ' + ci(*RA[k]['audit_cross']['ci']) if len(ms) == 1 else '')
            rows.append([names[k], rng2(meds, lambda x: f'{x:,.0f}'), per, adv_range(aucs), dif, cells, aud, typ])
        rows.append('midrule')
    rows.pop()
    write('pbmc68k_excerpt', ['Model', 'Forget median', 'Per-cell AUC [95\\% CI]', 'Advantage', 'Paired difference [95\\% CI]', 'Cells', 'Reference-free audit [95\\% CI]', 'Type'], rows, 'lcccc>{\\raggedright\\arraybackslash}p{2.8cm}cl')


def t_proposition():
    """Proposition 1 on the three forget sets with a within-cohort retain set (scripts/proposition_test.py).

    For an exactly rank-one matrix the marginal product equals the matrix cosine, so the 'Product' column
    measures rank-one structure (with the top-1 share); the direct product is the test of the proposition's
    independence assumption (round-7 review). Corollary 1's bound is in the appendix prose, not the table."""
    pt = J('outputs/p6/proposition_test.json')['cohorts']
    names = {'pbmc33k_c13': 'PBMC-33k cluster 13', 'pbmc68k_mk': 'PBMC-68k cluster 9', 'pbmc68k_pdc': 'PBMC-68k cluster 8'}
    rows = []
    for k, label in names.items():
        e = pt[k]; m = e['fisher_marginals']; d = e['direct_data']; t1 = e['svd_top1_share']; nu = e['nu_shared_support']
        rows.append([label, f3(e['measured_cosine']), f"{t1['forget']:.3f} / {t1['retain']:.3f}", f3(m['cos_sigma']), f3(m['cos_nu']), f3(m['product']),
                     f3(d['cos_sigma']), f3(d['cos_nu']), f3(d['product']), f"{nu['n_top90_forget']} / {nu['n_top90_retain']}"])
    write('proposition', ['Forget set', 'Measured', 'Top-1 share, F / R', '$\\cos\\sigma$', '$\\cos\\nu$', 'Product', '$\\cos\\sigma$, direct', '$\\cos\\nu$, direct', 'Product, direct', 'Units for 90\\%, F / R'], rows, 'lccccccccc')

def t_hparams():
    """Every setting used on PBMC-33k (scripts/stage1_rederive.py); PBMC-68k uses the same values with critics trained on its own baseline."""
    rows = [
        ['Baseline and clean retrains', 'Adam $10^{-4}$, 100 epochs, batch 256, KL warm-up 20 epochs, free bits 0.03, early stopping patience 10 (PBMC-68k: selection on the full-weight objective, patience counted after the warm-up)'],
        ['Retain-only fine-tuning', 'Adam $10^{-4}$, 50 epochs on $\\mathcal{R}$ with patience 10 (the methods\' own fine-tune phase: 10 epochs at $10^{-4}$)'],
        ['Gradient ascent', '10 ascent steps on $\\mathcal{F}$ at $10^{-5}$ with the gradient norm clipped to 1.0, then 30 retain epochs at $10^{-4}$ with patience 10 (sweep of the ascent step in Section~\\ref{sec:beyond})'],
        ['Fisher scrubbing', 'per-sample diagonal retain Fisher, relative damping 0.1, 100 scrub steps at $10^{-4}$ with the step capped at $10^{-3}$ per parameter (ablation $3 \\times 10^{-3}$ and $10^{-2}$; sweep in Section~\\ref{sec:beyond}), then 10 retain epochs at $10^{-4}$'],
        ['SSD', 'multiplicative dampening by forget-set Fisher importance, $\\alpha = 1$, threshold 0, damping $10^{-5}$, then 10 retain epochs at $10^{-4}$ (the published method has no fine-tune)'],
        ['SCRUB', '20 epochs of 5 forget steps (maximize the KL to the teacher, weight 1) and retain steps (match the teacher, weight 1, plus the ELBO) at $10^{-4}$, then 10 retain epochs at $10^{-4}$'],
        ['Moon feature unlearning', '20 decoder epochs with the forget-versus-retain latent direction removed, encoder frozen'],   # src/train_moon_unlearn.py n_epochs=20; every published run (outputs/p2/moon/seed*) has a 20-entry history (round 9, D5)
        ['Contrastive latent', '20 epochs, $\\gamma = 1$, $\\lambda = 1$, $10^{-4}$, then 10 retain epochs at $10^{-4}$'],
        ['Frozen critics', 'one or three pre-trained critics, $\\lambda \\in \\{5, 10\\}$, VAE at $10^{-4}$, 50 epochs'],
        ['Extra-gradient co-training', '$\\lambda \\in \\{10, 1\\}$, 50 passes over $\\mathcal{R}$, VAE at $10^{-4}$, three critics at $10^{-5}$ with two critic steps per VAE step, ExtraAdam; critics pre-trained on the cohort\'s baseline (forget versus retain, 100 epochs)'],
        ['DP-SGD', 'from scratch on $\\mathcal{R}$, $\\varepsilon = 10$, $\\delta = 10^{-5}$, gradient norm clipped to 1.0, noise calibrated by Opacus, 50 epochs at $10^{-3}$, patience 15'],
        ['Likelihood-targeted fine-tune', 'retain loss plus $\\lambda = 1$ times the mean absolute gap to the target, Adam $10^{-4}$, stop when the mean gap stays within 5 units for 3 epochs'],
        ['Attacker', 'MLP [256, 256], spectral normalization, dropout 0.3, 50 epochs, Adam $10^{-3}$, batch 128, 80/20 split, 70 features'],
    ]
    write('hparams', ['Method', 'Settings'], rows, 'lp{0.78\\textwidth}')


SWEEP_LABELS = {'ga_lr0.0001': 'Gradient ascent, ascent step $10^{-4}$', 'ga_lr0.001': 'Gradient ascent, ascent step $10^{-3}$', 'ga_lr0.01': 'Gradient ascent, ascent step $10^{-2}$',
                'fisher_step0.0015': 'Fisher scrubbing, step cap $1.5\\times 10^{-3}$', 'fisher_step0.002': 'Fisher scrubbing, step cap $2\\times 10^{-3}$',
                'fisher_step0.0025': 'Fisher scrubbing, step cap $2.5\\times 10^{-3}$', 'fisher_step0.003': 'Fisher scrubbing, step cap $3\\times 10^{-3}$'}


def adv_range(aucs):
    """Advantage 2|AUC - 0.5| ranged over seeds at two decimals (one value when the ends agree)."""
    a = sorted(2 * abs(x - 0.5) for x in aucs); lo, hi = f'{a[0]:.2f}', f'{a[-1]:.2f}'
    return lo if lo == hi else f'{lo} to {hi}'


def t_sweeps():
    """Step sweeps on PBMC-33k (sweep_scores.json for the audit and forget medians; paired_diff_33k.json for the canonical
    attacker's paired test; utility_intervals.json for held-out utility, the source of tab:main). Round 9 (R6, R12, D10):
    rows in the order of the knob, the audit-in-range column named for what it counts, the baseline labelled as in every
    other table, caps written as powers of ten, and an advantage column beside the AUC."""
    f = BASE / 'outputs/p4/sweeps/sweep_scores.json'
    if not f.exists():
        print('  (sweeps not scored yet; skipped)'); return
    d = J('outputs/p4/sweeps/sweep_scores.json'); band = C.audit_all_pairs_range('pbmc33k'); P = C.pbmc_paired()['arms']; U = C.utility_intervals('pbmc33k')['arms']
    def ut(arm):
        u = U.get(arm)
        return [f1(u['elbo']), f3(u['marker_r'])] if u else ['--', '--']
    def knob(k):   # gradient ascent first, then Fisher; each ordered by the value of its knob
        return (0, float(k.split('ga_lr')[1])) if k.startswith('ga_lr') else (1, float(k.split('fisher_step')[1]))
    rows = []
    for k in sorted(d['configs'], key=knob):
        v = d['configs'][k]
        aud = [r['audit'] for r in v['per_seed']]; mdn = [r['forget_median'] for r in v['per_seed']]
        inb = sum(band[0] <= a <= band[1] for a in aud)
        arm = 'sweep_' + ('gradient_ascent_lr' + k.split('ga_lr')[1] if k.startswith('ga_lr') else 'fisher_step' + k.split('fisher_step')[1])
        pp = P[arm]   # the canonical pass (save_persample_scores.py --extended), paired against the retrain arm
        rows.append([SWEEP_LABELS.get(k, v['label']), f'{min(mdn):.0f} to {max(mdn):.0f}', f'{min(aud):.3f} to {max(aud):.3f}', f'{inb} of {len(aud)}',
                     f"{min(pp['auc_per_seed']):.3f} to {max(pp['auc_per_seed']):.3f}", adv_range(pp['auc_per_seed']), f"{pp['diff']:+.3f} " + ci(*pp['diff_ci']), label(pp['verdict'])] + ut(arm))
    rows.append('midrule')
    rows.append(['Baseline', f"{d['baseline']['forget_median']:.0f}", f3(d['baseline']['audit']), '--', f3(P['baseline']['auc']), adv_range(P['baseline']['auc_per_seed']), f"{P['baseline']['diff']:+.3f} " + ci(*P['baseline']['diff_ci']), 'trained on the cells'] + ut('baseline'))
    rows.append(['Clean retrains (leave-one-out)', f"{min(d['retrain_forget_median'].values()):.0f} to {max(d['retrain_forget_median'].values()):.0f}", f"{min(d['retrain_loo'].values()):.3f} to {max(d['retrain_loo'].values()):.3f}", '--',
                 f"{min(P['retrain']['auc_per_seed']):.3f} to {max(P['retrain']['auc_per_seed']):.3f}", adv_range(P['retrain']['auc_per_seed']), '--', 'reference'] + ut('retrain'))
    # round 9 (fixer): the Seeds column (three per sweep setting, one baseline, five retrains; stated in the caption) is dropped
    # and the headers shortened so that the eleven-column table stays legible under \resizebox
    write('sweeps', ['Configuration', 'Forget median', 'Audit', 'In range', 'Attacker AUC', 'Advantage', 'Difference [95\\% CI]', 'Verdict', 'ELBO', 'Marker $r$'], rows, 'lccccccccc')


def t_utility_intervals():
    """Held-out utility with intervals and paired costs against the baseline (utility_intervals.json; round-7:
    utility claims carried no intervals and no margin)."""
    if not (BASE / 'outputs/utility_intervals.json').exists():
        print('  (utility_intervals.json missing; skipped)'); return
    names = dict(NAMES); names.update({'frozen_single_lambda5': 'Frozen critics, one critic, $\\lambda=5$', 'frozen_single_lambda10': 'Frozen critics, one critic, $\\lambda=10$',
                                       'frozen_multi_lambda5': 'Frozen critics, three critics, $\\lambda=5$', 'frozen_multi_lambda10': 'Frozen critics, three critics, $\\lambda=10$',
                                       'sweep_fisher_step0.0015': 'Fisher, step cap $1.5\\times 10^{-3}$', 'sweep_fisher_step0.002': 'Fisher, step cap $2\\times 10^{-3}$', 'sweep_fisher_step0.0025': 'Fisher, step cap $2.5\\times 10^{-3}$',
                                       'sweep_fisher_step0.003': 'Fisher, step cap $3\\times 10^{-3}$ (sweep)', 'sweep_gradient_ascent_lr0.0001': 'Gradient ascent, lr $10^{-4}$', 'sweep_gradient_ascent_lr0.001': 'Gradient ascent, lr $10^{-3}$',
                                       'sweep_gradient_ascent_lr0.01': 'Gradient ascent, lr $10^{-2}$', 'ltu': 'Likelihood-targeted, look-alike level', 'ltu_avg': 'Likelihood-targeted, retrained level', 'ltu_disjoint': 'Likelihood-targeted, disjoint look-alikes',
                                       'long_finetune_200': 'Retain-only fine-tune, 200 epochs', 'long_finetune': 'Retain-only fine-tune, 200 epochs', 'retrain_leaky': 'Leaky retrain (superseded)'})
    header = ['Model', 'Seeds', 'Held-out ELBO [95\\% CI]', 'ELBO cost [95\\% CI]', 'Marker $r$', 'Marker cost [95\\% CI]', 'ARI', 'Verdict']
    def row(m, a):
        base = 'elbo_cost' not in a
        return [names.get(m, m), str(len(a['seeds'])), f"{a['elbo']:.1f} " + f"[{a['elbo_ci'][0]:.1f}, {a['elbo_ci'][1]:.1f}]",
                '--' if base else f"{a['elbo_cost']:+.2f} [{a['elbo_cost_ci'][0]:+.2f}, {a['elbo_cost_ci'][1]:+.2f}]", f3(a['marker_r']),
                '--' if base else f"{a['marker_cost']:+.4f} [{a['marker_cost_ci'][0]:+.4f}, {a['marker_cost_ci'][1]:+.4f}]", f3(a['ari']), C.utility_verdict(a)]
    U = C.utility_intervals('pbmc33k'); A = U['arms']
    order = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'fisher_step3e-3', 'fisher_step1e-2', 'extragradient', 'extragradient_lambda1', 'contrastive'] + FROZEN + ['dp_sgd',
             'sweep_fisher_step0.0015', 'sweep_fisher_step0.002', 'sweep_fisher_step0.0025', 'sweep_fisher_step0.003', 'sweep_gradient_ascent_lr0.0001', 'sweep_gradient_ascent_lr0.001', 'sweep_gradient_ascent_lr0.01',
             'ltu', 'ltu_avg', 'ltu_disjoint', 'long_finetune_200', 'retrain_leaky']
    rows = [row(m, A[m]) for m in order if m in A]
    rows.insert(rows.index(row('dp_sgd', A['dp_sgd'])) + 1, 'midrule')
    write('utility_intervals_33k', header, rows, 'lccccccc')
    (OUT / 'utility_margin_33k.tex').write_text(f"{U['margin_elbo']:.2f}%")
    rows = []
    for fs, label in (('mk', 'Cluster 9, megakaryocytes'), ('pdc', 'Cluster 8, plasmacytoid dendritic cells')):
        try:
            U = C.utility_intervals(f'pbmc68k_{fs}')
        except KeyError:
            print(f'  (utility intervals for pbmc68k_{fs} not run; skipped)'); continue
        A = U['arms']
        rows.append([f'\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}}'])
        rows += [row(m, A[m]) for m in K_ORDER if m in A]   # K_ORDER already ends in long_finetune (round 9, R12: the 200-epoch row printed twice)
        rows.append('midrule')
    if rows:
        rows.pop(); write('utility_intervals_68k', header, rows, 'lccccccc')


EXT_NAMES = dict(NAMES)
EXT_NAMES.update({'frozen_single_lambda5': 'Frozen critics, one critic, $\\lambda=5$', 'frozen_single_lambda10': 'Frozen critics, one critic, $\\lambda=10$',
                  'frozen_multi_lambda5': 'Frozen critics, three critics, $\\lambda=5$', 'frozen_multi_lambda10': 'Frozen critics, three critics, $\\lambda=10$',
                  'sweep_fisher_step0.0015': 'Fisher, step cap $1.5\\times 10^{-3}$', 'sweep_fisher_step0.002': 'Fisher, step cap $2\\times 10^{-3}$', 'sweep_fisher_step0.0025': 'Fisher, step cap $2.5\\times 10^{-3}$',
                  'sweep_fisher_step0.003': 'Fisher, step cap $3\\times 10^{-3}$ (sweep)', 'sweep_gradient_ascent_lr0.0001': 'Gradient ascent, lr $10^{-4}$', 'sweep_gradient_ascent_lr0.001': 'Gradient ascent, lr $10^{-3}$',
                  'sweep_gradient_ascent_lr0.01': 'Gradient ascent, lr $10^{-2}$', 'ltu': 'Likelihood-targeted, look-alike level', 'ltu_avg': 'Likelihood-targeted, retrained level', 'ltu_disjoint': 'Likelihood-targeted, disjoint look-alikes',
                  'long_finetune_200': 'Retain-only fine-tune, 200 epochs', 'long_finetune': 'Retain-only fine-tune, 200 epochs', 'retrain_leaky': 'Leaky retrain (superseded)'})
ORDER_33K = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'fisher', 'fisher_step3e-3', 'fisher_step1e-2', 'extragradient', 'extragradient_lambda1', 'contrastive'] + FROZEN + ['dp_sgd',
             'sweep_fisher_step0.0015', 'sweep_fisher_step0.002', 'sweep_fisher_step0.0025', 'sweep_fisher_step0.003', 'sweep_gradient_ascent_lr0.0001', 'sweep_gradient_ascent_lr0.001', 'sweep_gradient_ascent_lr0.01',
             'ltu', 'ltu_avg', 'ltu_disjoint', 'long_finetune_200', 'retrain_leaky']
RF_VERDICT = {'fitted better than retrained': 'fitted better', 'fitted worse than retrained': 'fitted worse', 'fitted as retrained': 'as retrained', 'not resolved': 'not resolved', 'not distinguishable from retrained': 'not distinguishable'}


def t_audit_reference_free():
    """Reference-free audit (mean of all retrains, leave-one-out arm, reference choice inside the interval) and the
    paired forget-fit gap (audit_reference_free.json; round-7: audit conditional on seed 42, median switch post hoc)."""
    if not (BASE / 'outputs/audit_reference_free.json').exists():
        print('  (audit_reference_free.json missing; skipped)'); return
    def sd(v):
        return f"{v['diff_vs_retrain']:+.3f} " + ci(*v['diff_ci'])
    def fg(x):   # two decimals below 10 so that verify_numbers.py (tolerance 0.0055 there) can place the value
        return f'{x:+.2f}' if abs(x) < 10 else f'{x:+.1f}'
    def gp(v):
        return f"{fg(v['gap'])} [{fg(v['gap_ci'][0])}, {fg(v['gap_ci'][1])}]"
    def ref_row(R, ncol):
        a = R['audit_cross']; fm = R['forget_median']
        return ['Clean retrains (leave-one-out)', str(len(a['loo_per_seed'])), f"{a['arm_mean']:.3f} {ci(*a['arm_ci'])}; per seed {min(a['loo_per_seed']):.2f} to {max(a['loo_per_seed']):.2f}", '--', 'reference',
                f"{fm['arm_mean']:.0f} ({min(fm['per_seed']):.0f} to {max(fm['per_seed']):.0f})", '--', '--', '--'][:ncol]
    def type_row(m, a):
        return [EXT_NAMES.get(m, m), str(a['n_seeds']), f"{a['audit_cross']['mean']:.3f} {ci(*a['audit_cross']['ci'])}", sd(a['audit_cross']), RF_VERDICT[a['audit_cross']['verdict']],
                f"{a['gap_median']['target_mean']:.0f}", gp(a['gap_median']), gp(a['gap_rel_median']), RF_VERDICT[a['gap_rel_median']['verdict']]]
    header = ['Model', 'Seeds', 'Audit, mean-of-retrains reference [95\\% CI]', 'Difference from the retrained arm [95\\% CI]', 'Audit verdict', 'Forget median', 'Gap, median [95\\% CI]', 'Gap relative to the negatives [95\\% CI]', 'Fit verdict']
    D = C.audit_reference_free('pbmc33k')
    rows = [ref_row(D['retrain'], 9)] + [type_row(m, D['arms'][m]) for m in ORDER_33K if m in D['arms']]
    rows.insert(rows.index(type_row('dp_sgd', D['arms']['dp_sgd'])) + 1, 'midrule')
    write('audit_reference_free_33k', header, rows, 'lcccccccc')
    rows, wrows = [], []
    for fs, label in (('mk', 'Cluster 9, megakaryocytes'), ('pdc', 'Cluster 8, plasmacytoid dendritic cells')):
        try:
            D = C.audit_reference_free(f'pbmc68k_{fs}')
        except KeyError:
            continue
        rows.append([f'\\multicolumn{{9}}{{l}}{{\\emph{{{label}}}}}']); wrows.append([f'\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}}'])
        rows.append(ref_row(D['retrain'], 9))
        R = D['retrain']
        wrows.append(['Clean retrains (leave-one-out)', str(len(R['audit_within']['loo_per_seed'])), f"{R['audit_within']['arm_mean']:.3f} {ci(*R['audit_within']['arm_ci'])}", '--', 'reference',
                      f"{R['audit_within_z']['arm_mean']:.3f} {ci(*R['audit_within_z']['arm_ci'])}", '--', 'reference'])
        for m in K_ORDER:
            if m in D['arms']:
                a = D['arms'][m]; rows.append(type_row(m, a))
                wrows.append([EXT_NAMES.get(m, m), str(a['n_seeds']), f"{a['audit_within']['mean']:.3f} {ci(*a['audit_within']['ci'])}", sd(a['audit_within']), RF_VERDICT[a['audit_within']['verdict']],
                              f"{a['audit_within_z']['mean']:.3f} {ci(*a['audit_within_z']['ci'])}", sd(a['audit_within_z']), RF_VERDICT[a['audit_within_z']['verdict']]])
        rows.append('midrule'); wrows.append('midrule')
    if rows:
        rows.pop(); wrows.pop()
        write('audit_reference_free_68k', header, rows, 'lcccccccc')
        write('audit_reference_free_68k_within', ['Model', 'Seeds', 'Within-type audit [95\\% CI]', 'Difference [95\\% CI]', 'Verdict', 'Per-cell z-scored audit [95\\% CI]', 'Difference [95\\% CI]', 'Verdict'], wrows, 'lccccccc')


PROBE_NAMES = {'baseline': 'Baseline', 'fisher_cap0.001': 'Fisher, cap $10^{-3}$ (published)', 'fisher_cap0.002': 'Fisher, cap $2\\times 10^{-3}$', 'fisher_cap0.003': 'Fisher, cap $3\\times 10^{-3}$',
               'fisher_cap0.005': 'Fisher, cap $5\\times 10^{-3}$', 'fisher_cap0.01': 'Fisher, cap $10^{-2}$',
               # the published arm is outputs/pbmc68k/<fs>/retain_finetune_seed*/best_model.pt, trained by run_pbmc68k.py with
               # --epochs 50 --patience 10; every one of the six history.json files has 50 entries with the best epoch at 43 to 50,
               # so early stopping never triggered (round 9, R3 / D5)
               'retain_finetune_published': 'Retain fine-tune, 50 epochs (published)'}
PROBE_NAMES.update({f'finetune_epoch{e}': f'Retain fine-tune, {e} epochs' for e in (10, 25, 50, 100, 150, 200)})
PROBE_VERDICT = {'at the retrained level': 'at retrained level', 'more identifiable than retrained': 'more identifiable', 'less identifiable than retrained': 'less identifiable', 'not resolved': 'not resolved', 'positive control': 'trained on the cells'}
PROBE_TYPE = {'fitted better than retrained': 'better', 'fitted worse than retrained': 'worse', 'at the retrained level': 'at level'}


def t_stopping_probe():
    """Stopping probe (stopping_probe_{mk,pdc}.json): the knob grid with the retrain-based verdicts beside the
    quantities a controller can observe without a retrain."""
    rows = []
    for fs, label in (('mk', 'Cluster 9, megakaryocytes'), ('pdc', 'Cluster 8, plasmacytoid dendritic cells')):
        if not (BASE / f'outputs/pbmc68k/stopping_probe_{fs}.json').exists():
            print(f'  (stopping_probe_{fs}.json missing; skipped)'); continue
        D = C.stopping_probe(fs); rm = list(D['retrain_forget_medians'].values())
        rows.append([f'\\multicolumn{{12}}{{l}}{{\\emph{{{label}: retrained per-cell arm {D["retrain_percell_arm"]["mean"]:.3f}, margin {D["margin"]["within"]:.3f}, retrained forget medians {min(rm):,.0f} to {max(rm):,.0f}}}}}'])
        for m in ['baseline'] + [f'fisher_cap{c}' for c in ('0.001', '0.002', '0.003', '0.005', '0.01')] + ['retain_finetune_published'] + [f'finetune_epoch{e}' for e in (10, 25, 50, 100, 150, 200)]:
            if m not in D['arms']:
                continue
            a = D['arms'][m]; pc = a['refit_std_within']; au = a['audit_cross_nested']
            rows.append([PROBE_NAMES[m], f"{pc['mean']:.3f} {ci(*pc['nested_ci'])}", '--' if m == 'baseline' else f"{pc['diff_vs_retrain']:+.3f} " + ci(*pc['diff_ci']), PROBE_VERDICT[pc['verdict']],
                         f"{au['mean']:.3f}", (f"{a['forget_median_range'][0]:,.0f} to {a['forget_median_range'][1]:,.0f}" if a['n_seeds'] > 1 else f"{a['forget_median']:,.0f}"), PROBE_TYPE[a['type_verdict_median']],
                         f"{a['lookalike_mean']:.0f}", f"{a['retain_sample_mean']:.0f}", f"{a['cross_mean']:.0f}", f1(a['heldout_elbo']), f3(a['marker_r'])])
        rows.append('midrule')
    if rows:
        rows.pop()
        write('stopping_probe', ['Setting', 'Per-cell AUC [nested 95\\% CI]', 'Paired difference [95\\% CI]', 'Per-cell verdict', 'Audit', 'Forget median', 'Type', 'Look-alikes, mean', 'Retain sample, mean', 'Cross negatives, mean', 'Held-out ELBO', 'Marker $r$'], rows, 'lccccccccccc')


PROBE80_NAMES = {'baseline': 'Baseline', 'fisher_cap0.001': 'Fisher, cap $10^{-3}$', 'fisher_cap0.002': 'Fisher, cap $2\\times 10^{-3}$'}
PROBE80_NAMES.update({f'finetune_epoch{e}': f'Retain fine-tune, {e} epochs' for e in (10, 25, 50, 100, 150, 200)})


def t_stopping_probe80():
    """Full stopping probe (stopping_probe80_{mk,pdc}.json): 80 percent deleted, 20 percent held back as known members."""
    rows = []
    for fs, label in (('mk', 'Cluster 9, megakaryocytes'), ('pdc', 'Cluster 8, plasmacytoid dendritic cells')):
        if not (BASE / f'outputs/pbmc68k/stopping_probe80_{fs}.json').exists():
            print(f'  (stopping_probe80_{fs}.json missing; skipped)'); continue
        D = C.stopping_probe80(fs); sp = D['split']; R = D['arms']['retrain']
        rows.append([f'\\multicolumn{{11}}{{l}}{{\\emph{{{label}: {sp["n_forget"]} deleted, {sp["n_heldback"]} held back; retrained per-cell arm {D["retrain_percell_arm"]["within"]:.3f}, margin {D["margin"]:.3f}}}}}'])
        rows.append(['Clean retrains (8)', f"{R['refit_within']['mean']:.3f} {ci(*R['refit_within']['nested_ci'])}", '--', 'reference', f"{R['refit_heldback']['mean']:.3f} {ci(*R['refit_heldback']['nested_ci'])}", f"{R['refit_members_vs_within']['mean']:.3f}",
                     f"{R['deleted_median']:,.0f}", f"{R['heldback_median']:,.0f}", f"{R['gap_deleted_minus_members']:+.0f}", f1(R['heldout_elbo']), f3(R['marker_r'])])
        for m in ['baseline', 'fisher_cap0.001', 'fisher_cap0.002'] + [f'finetune_epoch{e}' for e in (10, 25, 50, 100, 150, 200)]:
            if m not in D['arms']:
                continue
            a = D['arms'][m]; v = a['refit_within']
            rows.append([PROBE80_NAMES[m], f"{v['mean']:.3f} {ci(*v['nested_ci'])}", '--' if m == 'baseline' else f"{v['diff_vs_retrain']:+.3f} " + ci(*v['diff_ci']), PROBE_VERDICT[v['verdict']],
                         f"{a['refit_heldback']['mean']:.3f} {ci(*a['refit_heldback']['nested_ci'])}", f"{a['refit_members_vs_within']['mean']:.3f}",
                         f"{a['deleted_median']:,.0f}", f"{a['heldback_median']:,.0f}", f"{a['gap_deleted_minus_members']:+.0f}", f1(a['heldout_elbo']), f3(a['marker_r'])])
        rows.append('midrule')
    if rows:
        rows.pop()
        write('stopping_probe80', ['Setting', 'Deleted vs held-out, per-cell AUC [nested 95\\% CI]', 'Paired difference [95\\% CI]', 'Verdict', 'Deleted vs kept members (observable) [95\\% CI]', 'Kept members vs held-out', 'Deleted median', 'Kept-member median', 'Gap', 'Held-out ELBO', 'Marker $r$'], rows, 'lcccccccccc')


def t_retain_null():
    """Retain-null mass and survival by setting and parameter group (retain_null_mass.json)."""
    if not (BASE / 'outputs/p6/retain_null_mass.json').exists():
        print('  (retain_null_mass.json missing; skipped)'); return
    D = C.retain_null_mass()['settings']
    names = {'pbmc33k': 'PBMC-33k cluster 13', 'pbmc68k_pdc': 'PBMC-68k cluster 8', 'pbmc68k_mk': 'PBMC-68k cluster 9', 'classifier_linear': 'Linear classifier (33k latents)'}
    groups = ['Encoder', 'Bottleneck', 'Decoder hidden', 'Decoder output', 'global']
    rows = []
    for st in ('pbmc33k', 'pbmc68k_pdc', 'pbmc68k_mk'):
        rows.append([f'\\multicolumn{{8}}{{l}}{{\\emph{{{names[st]}}}}}'])
        for g in groups:
            v = D[st]['groups'][g]
            rows.append([g if g != 'global' else 'Whole model', f3(v['forget_mass_share']), f3(v['cosine']), f3(v['survival']['1.0']), f3(v['survival']['10.0']), f3(v['survival']['100.0']), f3(v['retain_null_mass']['0.01']), f3(v['retain_null_mass']['0.1'])])
        rows.append('midrule')
    rows.append([f'\\multicolumn{{8}}{{l}}{{\\emph{{{names["classifier_linear"]}}}}}'])
    for g, lab in (('forget_class_row', 'Forget-class output row'), ('other_rows', 'Other output rows'), ('global', 'Whole classifier')):
        v = D['classifier_linear']['groups'][g]
        rows.append([lab, f3(v['forget_mass_share']), f3(v['cosine']), f3(v['survival']['1.0']), f3(v['survival']['10.0']), f3(v['survival']['100.0']), f3(v['retain_null_mass']['0.01']), f3(v['retain_null_mass']['0.1'])])
    write('retain_null', ['Parameter group', 'Share of forget mass', 'Cosine', '$S(1)$', '$S(10)$', '$S(100)$', '$N(10^{-2})$', '$N(10^{-1})$'], rows, 'lccccccc')
    # retain_null_columns.tex (the bottom-5-percent column shares) was written here but never input by the paper; dropped in round 9 (R12)


def t_restricted_scrub():
    """The restoration intervention (restricted_scrub_test.json), body table. Round 9 (D6): the Adam 1e-4 rerun from the
    SAVED edits (tags edits_opt = the published scrub's damage, edits_x2_opt = moderate damage, twice the starting forget
    loss), three edits each, after the edit and after 10, 50 and 200 retain epochs, with the no-edit fine-tune of the same
    arm as the trajectory an edit is restored onto. The legacy single-run arms (tags edits, edits_x2; 10 and 50 epochs
    only, no no-edit arm) stay in the JSON and leave the paper: at the published damage (49,000 to 61,000 nats) the two
    MPS runs differ by hundreds to thousands of nats, at moderate damage by 1 to 17 (troubleshooting 17). Fits are the mean
    negative ELBO per cell; the audit is the reference-free audit point (mean of the retrains as the reference)."""
    if not (BASE / 'outputs/p6/restricted_scrub_test.json').exists():
        print('  (restricted_scrub_test.json missing; skipped)'); return
    D = C.restricted_scrub_test()['cohorts']
    names = {'pbmc33k': 'PBMC-33k cluster 13', 'pbmc68k_mk': 'PBMC-68k cluster 9'}
    enames = {'unrestricted': 'Published scrub', 'active_only': 'Retain-active directions', 'null_only': 'Retain-null directions'}
    cfgs = {'edits_opt': 'published damage', 'edits_x2_opt': 'moderate damage'}   # round 9 (fixer): short labels; the caption defines both
    arm = 'adam:1e-4'; E = (10, 50, 200)
    def ft(r, e):
        return r['finetune'][arm][f'after_ft{e}']
    def row(lab, r):
        return ([lab, str(r['scrub_steps']), f"{r['after_edit']['forget']:,.0f}", f"{r['after_edit']['heldout']:,.0f}"] + [f"{ft(r, e)['forget']:,.0f}" for e in E]
                + [f"{ft(r, 200)['heldout']:,.0f}", f3(ft(r, 50)['audit_meanref']), f3(ft(r, 200)['audit_meanref'])])
    rows = []
    for c in ('pbmc33k', 'pbmc68k_mk'):
        tags = [t for t in cfgs if t in D.get(c, {})]
        if not tags:
            continue
        b = D[c]['baseline']
        rows.append([f'\\multicolumn{{10}}{{l}}{{\\emph{{{names[c]}: baseline forget {b["forget"]:.0f}, retain {b["retain_sample"]:.0f}, held-out {b["heldout"]:.0f} (mean negative ELBO per cell); {D[c]["n_null_coords"]:,} retain-null of {D[c]["n_coords"]:,} parameters}}}}'])
        none = next((D[c][t]['none'] for t in tags if 'none' in D[c][t]), None)
        if none is not None:
            rows.append(row('No edit', none))
        for t in tags:
            for e in ('unrestricted', 'active_only', 'null_only'):
                if e in D[c][t] and arm in D[c][t][e].get('finetune', {}):
                    rows.append(row(enames[e] + (f', {cfgs[t]}' if e == 'unrestricted' else ''), D[c][t][e]))
        rows.append('midrule')
    if not rows:
        print('  (no Adam arms from the saved edits; skipped)'); return
    rows.pop()
    write('restricted_scrub', ['Edit', 'Steps', 'Forget, edit', 'Held-out, edit', 'Forget 10', 'Forget 50', 'Forget 200', 'Held-out 200', 'Audit 50', 'Audit 200'], rows, 'lccccccccc')


def t_holm():
    """Multiplicity table (round-8 review: Holm-adjusted p-values quoted in prose appeared in no table). For every arm
    of the attacker test, the three one-sided bootstrap p-values of the paired difference (above zero, below zero,
    equivalence within the margin), raw and Holm-adjusted within the arm's family, with the verdict each set gives,
    beside the two-sided p-value of the type-level instrument, raw and Holm-adjusted within the same family: the
    reference-free cross-type audit on PBMC-33k and the paired median gap relative to the negatives on PBMC-68k."""
    def p(x):
        return '$<$0.001' if x < 0.001 else f'{x:.3f}'
    def arm_row(name, r, a):
        return [name, f"{r['diff']:+.3f} " + ci(*r['diff_ci']), p(r['p_above']), p(r['p_below']), p(r['p_equivalent']), label(r['verdict']),
                p(r['p_above_holm']), p(r['p_below_holm']), p(r['p_equivalent_holm']), label(r['verdict_holm']), f"{p(a['p_two_sided'])} / {p(a['p_holm'])}", label(a['verdict'])]
    rows = []
    PD = C.pbmc_paired(); P = PD['arms']; RA = C.audit_reference_free('pbmc33k')['arms']
    for fam, flab in (('published', 'published settings'), ('ablation', 'step and cap ablations'), ('demonstration', 'targeted and long fine-tunes')):
        ms = [m for m in ORDER_33K if m in P and P[m].get('family') == fam and 'verdict_holm' in P[m]]
        rows.append([f'\\multicolumn{{12}}{{l}}{{\\emph{{PBMC-33k cluster 13, transferred attacker, {flab} ({len(ms)} arms), margin {PD["margin"]:.3f}}}}}'])
        rows += [arm_row(EXT_NAMES.get(m, m), P[m], RA[m]['audit_cross']) for m in ms]
        rows.append('midrule')
    for fs, flab in (('mk', 'cluster 9, megakaryocytes'), ('pdc', 'cluster 8, plasmacytoid dendritic cells')):
        if not (BASE / f'outputs/pbmc68k/percell_paired_{fs}.json').exists():
            continue
        PP = C.pbmc68k_percell_paired(fs, 'within'); A = PP['arms']; RA = C.audit_reference_free(f'pbmc68k_{fs}')['arms']
        ms = [m for m in K_ORDER if m in A and 'verdict_holm' in A[m]]
        rows.append([f'\\multicolumn{{12}}{{l}}{{\\emph{{PBMC-68k {flab}, within-type refit ({len(ms)} arms), margin {PP["margin"]:.3f}}}}}'])
        rows += [arm_row(EXT_NAMES.get(m, m), A[m], RA[m]['gap_rel_median']) for m in ms]
        rows.append('midrule')
    rows.pop()
    write('holm', ['Method', 'Paired difference [95\\% CI]', '$p_{>}$', '$p_{<}$', '$p_{\\approx}$', 'Verdict', 'Holm $p_{>}$', 'Holm $p_{<}$', 'Holm $p_{\\approx}$', 'Holm verdict', 'Type $p$ (raw / Holm)', 'Type verdict'], rows, 'lccccccccccc')


def t_restricted_scrub_opt():
    """Optimizer arms of the restoration intervention (restricted_scrub_test.json, tags edits_x2_opt and edits_opt):
    from the same saved edit, the methods' Adam fine-tune (1e-4) and plain gradient descent (SGD 1e-3, the largest
    step one retain epoch leaves without a held-out cost), measured at 10, 50 and 200 retain epochs; the no-edit arm
    is the trajectory an edit is restored onto. Forget columns are the forget cells' negative ELBO per cell."""
    if not (BASE / 'outputs/p6/restricted_scrub_test.json').exists():
        print('  (restricted_scrub_test.json missing; skipped)'); return
    D = C.restricted_scrub_test()['cohorts']
    names = {'pbmc33k': 'PBMC-33k cluster 13', 'pbmc68k_mk': 'PBMC-68k cluster 9'}
    enames = {'none': 'No edit', 'unrestricted': 'Published scrub', 'active_only': 'Retain-active directions', 'null_only': 'Retain-null directions'}
    cfgs = {'edits_x2_opt': 'moderate damage, twice the starting forget loss', 'edits_opt': 'damage of the published scrub'}
    arms = ('adam:1e-4', 'sgd:1e-3'); E = (10, 50, 200)
    def cell(r, arm, e, key='forget'):
        v = r.get('finetune', {}).get(arm, {}).get(f'after_ft{e}')
        return f"{v[key]:,.0f}" if v else '--'
    def row(lab, r):
        return [lab, str(r['scrub_steps']), f"{r['after_edit']['forget']:,.0f}", f"{r['after_edit']['heldout']:,.0f}"] + [cell(r, a, e) for a in arms for e in E] + [cell(r, a, 200, 'heldout') for a in arms]
    rows = []
    for c in ('pbmc33k', 'pbmc68k_mk'):
        tags = [t for t in cfgs if t in D.get(c, {})]
        if not tags:
            continue
        b = D[c]['baseline']
        rows.append([f'\\multicolumn{{12}}{{l}}{{\\emph{{{names[c]}: baseline forget {b["forget"]:.0f}, held-out {b["heldout"]:.0f} (mean negative ELBO per cell)}}}}'])
        none = next((D[c][t]['none'] for t in tags if 'none' in D[c][t]), None)
        if none is not None:
            rows.append(row(enames['none'], none))
        for t in tags:
            for e in ('unrestricted', 'active_only', 'null_only'):
                if e in D[c][t] and 'finetune' in D[c][t][e]:
                    rows.append(row(enames[e] + (f' ({cfgs[t]})' if e == 'unrestricted' else ''), D[c][t][e]))
        rows.append('midrule')
    if not rows:
        print('  (no optimizer arms yet; skipped)'); return
    rows.pop()
    write('restricted_scrub_opt', ['Edit', 'Scrub steps', 'Forget after edit', 'Held-out after edit', 'Adam, 10', 'Adam, 50', 'Adam, 200', 'SGD, 10', 'SGD, 50', 'SGD, 200', 'Held-out, Adam 200', 'Held-out, SGD 200'], rows, 'lccccccccccc')


def t_restricted_scrub_rho():
    """Parameter-space reading of the optimizer arms (restricted_scrub_test.json, edit diagnostics). rho is the share of
    the edit vector delta still in the parameters, <theta - theta_base, delta> / |delta|^2, read relative to the no-edit
    fine-tune of the same optimizer at the same epoch (whose own drift projects onto delta; that projection is
    subtracted), overall and on the retain-null and retain-active coordinates, with the proposition's gradient-descent
    prediction for rho from the edit itself and the retain Fisher, beside the forget fit and the no-edit forget fit."""
    if not (BASE / 'outputs/p6/restricted_scrub_test.json').exists():
        print('  (restricted_scrub_test.json missing; skipped)'); return
    D = C.restricted_scrub_test()['cohorts']
    names = {'pbmc33k': 'PBMC-33k cluster 13', 'pbmc68k_mk': 'PBMC-68k cluster 9'}
    enames = {'unrestricted': 'Published scrub', 'active_only': 'Retain-active', 'null_only': 'Retain-null'}
    cfgs = {'edits_x2_opt': 'moderate damage', 'edits_opt': 'published damage'}
    onames = {'adam:1e-4': 'Adam $10^{-4}$', 'sgd:1e-3': 'SGD $10^{-3}$'}; E = (10, 50, 200)
    rows = []
    for c in ('pbmc33k', 'pbmc68k_mk'):
        tags = [t for t in cfgs if t in D.get(c, {})]
        none = next((D[c][t]['none'] for t in tags if 'none' in D[c][t]), None)
        if not tags or none is None:
            continue
        rows.append([f'\\multicolumn{{11}}{{l}}{{\\emph{{{names[c]} (fits are the mean negative ELBO per cell)}}}}'])
        for t in tags:
            for e in ('unrestricted', 'active_only', 'null_only'):
                r = D[c][t].get(e)
                if not r or 'finetune' not in r:
                    continue
                for o in onames:
                    a = r['finetune'].get(o); na = none['finetune'].get(o)
                    if not a or not na or not all(f'after_ft{k}' in a and 'edit' in a[f'after_ft{k}'] for k in E):
                        continue
                    def rho(k, g):
                        raw = a[f'after_ft{k}']['edit'][g]['rho']; dp = na[f'after_ft{k}'].get('drift_projection', {}).get(f'{t}/{e}')
                        return raw - dp[g] if (raw is not None and dp and dp.get(g) is not None) else float('nan')
                    last = a['after_ft200']['edit']
                    pred = f"{last['all']['pred_rho']:.3f}" if last['all'].get('pred_rho') is not None else '--'
                    f3n = lambda x: '--' if x != x else f'{x:.3f}'
                    rows.append([f'{enames[e]} ({cfgs[t]})', onames[o], f"{r['after_edit']['forget']:,.0f}"] + [f3n(rho(k, 'all')) for k in E]
                                + [f3n(rho(200, 'null')), f3n(rho(200, 'active')), pred, f"{a['after_ft200']['forget']:,.0f}", f"{na['after_ft200']['forget']:,.0f}"])
        rows.append('midrule')
    if not rows:
        print('  (no edit diagnostics yet; skipped)'); return
    rows.pop()
    write('restricted_scrub_rho', ['Edit', 'Fine-tune', 'Forget after edit', '$\\rho$, 10', '$\\rho$, 50', '$\\rho$, 200', '$\\rho$ null, 200', '$\\rho$ active, 200', 'Predicted $\\rho$, 200', 'Forget, 200', 'No edit, 200'], rows, 'llccccccccc')


def _sci(x):
    """Three significant figures; powers of ten below 0.01."""
    if x is None or x != x:
        return '--'
    if x == 0:
        return '0'
    if abs(x) >= 1000:
        return f'{x:,.0f}'
    if abs(x) >= 0.01:
        return f'{x:.3g}'
    m, e = f'{x:.2e}'.split('e'); return f'${m} \\times 10^{{{int(e)}}}$'


def _br(lo, hi, nd=2):
    """A Gauss / Radau pair as one cell: the value alone when the bounds agree at nd decimals, else [lo, hi]."""
    if lo is None or lo != lo:
        return '--'
    a, b = f'{lo:.{nd}f}', (None if hi is None or hi != hi else f'{hi:.{nd}f}')
    return a if b is None or b == a else f'[{a}, {b}]'


_COMP_NAMES = {'pbmc33k': 'PBMC-33k cluster 13', 'pbmc68k_mk': 'PBMC-68k cluster 9'}
_COMP_EDITS = {'unrestricted': 'Published scrub', 'active_only': 'Retain-active', 'null_only': 'Retain-null'}
_COMP_CFGS = {'edits_x2_opt': 'moderate damage'}   # the published-damage edits (5,000 to 49,000 nats) are outside any quadratic regime and stay in the JSON only


def t_edit_curvature():
    """Curvature along each saved edit (edit_compensation.json): the retain and forget quadratic forms delta' H delta by
    the diagonal the proposition uses, the empirical Fisher with its off-diagonal part, and the Hessian (central
    differences), beside the share of the edit left in the parameters under gradient descent, measured and as the
    quadratic model with the full Fisher predicts it (Lanczos; the 200-epoch prediction shows the Gauss / Radau bracket
    where the two differ, since 2026-10-30)."""
    if not (BASE / 'outputs/p6/edit_compensation.json').exists():
        print('  (edit_compensation.json missing; skipped)'); return
    D = C.edit_compensation()['cohorts']; E = ('10', '50', '200'); rows = []
    for c in ('pbmc33k', 'pbmc68k_mk'):
        if c not in D:
            continue
        rows.append([f'\\multicolumn{{15}}{{l}}{{\\emph{{{_COMP_NAMES[c]}}}}}'])
        for t in _COMP_CFGS:
            for e in _COMP_EDITS:
                k = f'{t}/{e}'; r = D[c]['edits'].get(k)
                if not r:
                    continue
                eps = [x for x in r['retain'] if x.startswith('eps')][0]; R, F = r['retain'][eps], r['forget'][eps]
                arm = r['arms'].get('sgd:1e-3', {}); lz = D[c].get('lanczos', {}).get(k, {}).get('epochs', {}); lg = D[c].get('lanczos_gn', {}).get(k, {}).get('epochs', {})
                rows.append([f'{_COMP_EDITS[e]} ({_COMP_CFGS[t]})', _sci(r['delta_sq']), _sci(r['retain']['diag_form']), _sci(R['fisher_form']), _sci(R['hess_form']),
                             _sci(r['forget']['diag_form']), _sci(F['fisher_form']), _sci(F['hess_form'])]
                            + [f"{arm[x]['rho']:.2f}" if x in arm else '--' for x in E] + [_br(lz[x]['rho_pred'], lz[x].get('rho_pred_upper')) if x in lz else '--' for x in ('10', '200')] + [_br(lg[x]['rho_pred'], lg[x].get('rho_pred_upper')) if x in lg else '--' for x in ('10', '200')])
        rows.append('midrule')
    if not rows:
        print('  (edit_compensation.json empty; skipped)'); return
    rows.pop()
    write('edit_curvature', ['Edit', '$\\|\\delta\\|^2$', 'Retain diag.', 'Retain Fisher', 'Retain Hessian', 'Forget diag.', 'Forget Fisher', 'Forget Hessian',
                             '$\\rho_{10}$', '$\\rho_{50}$', '$\\rho_{200}$', '$\\hat\\rho^{F}_{10}$', '$\\hat\\rho^{F}_{200}$', '$\\hat\\rho^{G}_{10}$', '$\\hat\\rho^{G}_{200}$'], rows, 'lcccccccccccccc')


def t_compensation():
    """Where the forget damage of an edit goes under gradient descent (edit_compensation.json): the share of the
    damage left in the forget fit after 10 and 200 retain epochs, measured; carried by the no-edit state plus the
    surviving share of the edit (superposition); carried by the surviving edit alone at the baseline; carried by the
    residual r - Delta alone at the baseline (the fine-tune's displacement with the no-edit path subtracted); and
    carried by the quadratic model's own residual (I - eta K)^T delta (Lanczos, empirical Fisher F and Gauss-Newton G); and
    the Gauss-Newton residual added to the no-edit state ('Quad. G sup.'), which is the model's prediction of 'Left' (2026-10-30).
    Brackets give the Gauss and Radau values where they differ."""
    if not (BASE / 'outputs/p6/edit_compensation.json').exists():
        print('  (edit_compensation.json missing; skipped)'); return
    D = C.edit_compensation()['cohorts']; E = ('10', '200'); rows = []
    sh = lambda x: '--' if x is None or x != x else f'{x:.2f}'
    for c in ('pbmc33k', 'pbmc68k_mk'):
        if c not in D:
            continue
        base = D[c]['baseline']['forget']
        rows.append([f'\\multicolumn{{16}}{{l}}{{\\emph{{{_COMP_NAMES[c]}}}}}'])
        for t in _COMP_CFGS:
            for e in _COMP_EDITS:
                k = f'{t}/{e}'; r = D[c]['edits'].get(k)
                if not r:
                    continue
                arm = r['arms'].get('sgd:1e-3', {}); st = D[c].get('states', {}).get(k, {}).get('epochs', {}); lz = D[c].get('lanczos', {}).get(k, {}).get('epochs', {}); lg = D[c].get('lanczos_gn', {}).get(k, {}).get('epochs', {})
                row = [f'{_COMP_EDITS[e]} ({_COMP_CFGS[t]})', f"{r['damage']:,.0f}"]
                for x in E:
                    a = arm.get(x, {}); s_ = st.get(x, {}); l_ = lz.get(x, {}); g_ = lg.get(x, {})
                    rd = g_.get('radau') or {}
                    row += [sh(a.get('share_meas')), sh(a.get('share_sup_rho')), sh(a.get('share_rev_only')),
                            sh((s_['fit_residual']['forget'] - base) / r['damage']) if s_ else '--', sh(l_.get('damage_share_pred')),
                            _br(g_.get('damage_share_pred'), rd.get('damage_share_pred')), _br(g_.get('share_sup_pred'), rd.get('share_sup_pred'))]
                rows.append(row)
        rows.append('midrule')
    if not rows:
        print('  (edit_compensation.json empty; skipped)'); return
    rows.pop()
    write('compensation', ['Edit', 'Damage', 'Left (10)', 'Superposed (10)', 'Edit alone (10)', 'Residual (10)', 'Quad.\\ $F$ (10)', 'Quad.\\ $G$ (10)', 'Quad.\\ $G$ sup.\\ (10)',
                           'Left (200)', 'Superposed (200)', 'Edit alone (200)', 'Residual (200)', 'Quad.\\ $F$ (200)', 'Quad.\\ $G$ (200)', 'Quad.\\ $G$ sup.\\ (200)'], rows, 'lccccccccccccccc')


def t_lanczos_convergence():
    """Convergence of the Lanczos prediction with the Krylov depth (edit_compensation.json, 'convergence' of the published
    scrub at moderate damage): per cohort and curvature, at every truncation m, the lowest Ritz value and its weight, and
    rho_hat by the Gauss rule (lower bound) and the Gauss-Radau rule with a node at 0 (upper bound) at 10, 50, 200 epochs."""
    if not (BASE / 'outputs/p6/edit_compensation.json').exists():
        print('  (edit_compensation.json missing; skipped)'); return
    D = C.edit_compensation()['cohorts']; E = ('10', '50', '200'); rows = []; arm = 'edits_x2_opt/unrestricted'
    for c in ('pbmc33k', 'pbmc68k_mk'):
        for key, name in (('lanczos', 'empirical Fisher'), ('lanczos_gn', 'Gauss--Newton')):
            conv = D.get(c, {}).get(key, {}).get(arm, {}).get('convergence')
            if not conv:
                continue
            rows.append([f'\\multicolumn{{6}}{{l}}{{\\emph{{{_COMP_NAMES[c]}, {name}}}}}'])
            for m in sorted(conv, key=int):
                v = conv[m]; rd = v.get('rho_radau')
                rows.append([m, _sci(v['lowest_ritz']), f"{v['weight_lowest']:.3f}"] + [_br(v['rho_gauss'][e], rd[e] if rd else None, 3) for e in E])
            rows.append('midrule')
    if not rows:
        print('  (no convergence record; skipped)'); return
    rows.pop()
    write('lanczos_convergence', ['$m$', '$\\lambda_{\\min}$', '$w_{\\min}$', '$\\hat\\rho_{10}$', '$\\hat\\rho_{50}$', '$\\hat\\rho_{200}$'], rows, 'lccccc')


if __name__ == '__main__':
    for fn in (t_datasets, t_reference, t_attacks, t_confound, t_main, t_pbmc68k, t_longft68k, t_fisher, t_phase, t_ltu, t_longft, t_sim, t_utility, t_sizes, t_crossdataset, t_mog08, t_sim_full, t_refit, t_refit_std, t_methods, t_phase_short, t_pbmc68k_excerpt, t_hparams, t_sweeps, t_proposition, t_utility_intervals, t_audit_reference_free, t_stopping_probe, t_stopping_probe80, t_retain_null, t_restricted_scrub, t_holm, t_restricted_scrub_opt, t_restricted_scrub_rho, t_edit_curvature, t_compensation, t_lanczos_convergence):
        fn()
    # round 9 (coordinator step 4): the reachability snippets (tables/reach_*.tex, tables/audit_margin_*.tex) and
    # outputs/equivalence_reachability.json are produced by their own script, run here with the same interpreter so
    # that one command regenerates every table and snippet
    import subprocess
    subprocess.run([sys.executable, str(BASE / 'scripts' / 'equivalence_reachability.py')], check=True)
    print('done')
