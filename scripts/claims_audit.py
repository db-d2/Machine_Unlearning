#!/usr/bin/env python3
"""Claim-by-claim audit: the paper's numbers (pre-Stage-1, superseded files) against the
re-derived numbers (Stage 1). Prints Markdown tables and writes
outputs/claims_audit.json so the analysis document can cite one file.

Run after scripts/stage1_rederive.py --phase C:
    PYTHONPATH=src python scripts/claims_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
import canonical as C

SUP = BASE / 'outputs' / '_superseded_20260908'
OLD_NESTED = SUP / 'nested_ci_all__pre_stage1.json'
OLD_PERSAMPLE = SUP / 'persample__pre_stage1'
OLD_REFIT = SUP / 'retrain_refit__leaky.json'
OLD_TM = SUP / 'tm_unlearning_results__jan.json'
OLD_ATTACKS = SUP / 'attack_suite_results__pre_stage1.json'
OLD_C12 = BASE / 'outputs' / 'p1_c12' / 'scores_c12_multiseed.json'
OLD_UTIL = BASE / 'outputs' / 'utility_suite_results.json'
OLD_CONS = BASE / 'outputs' / 'consolidated_method_comparison.json'

LABEL = {'baseline': 'Baseline', 'retrain': 'Retrain (clean)', 'retrain_leaky': 'Retrain (April, leaky)',
         'retain_finetune': 'Retain-only FT', 'gradient_ascent': 'Gradient ascent', 'ssd': 'SSD', 'scrub': 'SCRUB',
         'moon': 'Moon', 'contrastive': 'Contrastive', 'fisher': 'Fisher scrubbing', 'fisher_step_ablation': 'Fisher (step ablation)',
         'extragradient': 'Extra-gradient lambda=10', 'extragradient_lambda1': 'Extra-gradient lambda=1', 'dp_sgd': 'DP-SGD'}


def old_auc(method):
    p = OLD_PERSAMPLE / f'{method}.json'
    if not p.exists():
        return None
    return float(np.mean([r['auc'] for r in json.load(open(p))]))


def fmt(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


def main():
    old_nested = json.load(open(OLD_NESTED))
    new = C.pbmc_summary()
    thr_old, thr_new = old_nested['retrain']['nested_ci_high'], C.pbmc_success_threshold()
    audit = {'threshold': {'old': thr_old, 'new': thr_new}, 'pbmc': {}, 'utility': {}, 'tm': {}, 'c12': {}, 'attacks': {}}

    print("## PBMC-33k cluster 13: privacy (canonical CPU attacker)\n")
    print("| Method | AUC old | AUC new | Adv old [CI] | Adv new [CI] | seeds new | verdict old | verdict new |")
    print("|---|---|---|---|---|---|---|---|")
    for m in ['baseline', 'retrain_leaky', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon',
              'contrastive', 'fisher', 'fisher_step_ablation', 'extragradient', 'extragradient_lambda1', 'dp_sgd']:
        if m not in new:
            continue
        o = old_nested['methods'].get(m) or (old_nested['retrain'] if m == 'retrain_leaky' else None)
        oa = old_auc('retrain' if m == 'retrain_leaky' else m)
        n = new[m]
        oadv = None if o is None else f"{o['mean']:.3f} [{o['nested_ci_low']:.2f}, {o['nested_ci_high']:.2f}]"
        nadv = f"{n['mean']:.3f}" + (f" [{n['ci_low']:.2f}, {n['ci_high']:.2f}]" if n['ci_low'] is not None else '')
        vo = '' if o is None or m in ('baseline',) else ('pass' if o['mean'] <= thr_old else 'fail')
        vn = '' if m in ('baseline', 'retrain') else ('pass' if n['mean'] <= thr_new else 'fail')
        print(f"| {LABEL[m]} | {fmt(oa)} | {fmt(n['auc'])} | {oadv or ''} | {nadv} | {n['n_seeds']} | {vo} | {vn} |")
        audit['pbmc'][m] = {'auc_old': oa, 'auc_new': n['auc'], 'adv_old': None if o is None else o['mean'],
                            'adv_new': n['mean'], 'ci_new': [n['ci_low'], n['ci_high']], 'n_seeds_new': n['n_seeds']}
    print(f"\nSuccess threshold (retrain advantage nested-CI upper): old {thr_old:.3f} (leaky retrain), new {thr_new:.3f} (clean retrain)")
    null = C.pbmc_null_floor()
    print(f"Advantage estimator null floor: {null['mean']:.3f} [{null['ci_low']:.3f}, {null['ci_high']:.3f}]")

    print("\n## PBMC-33k: utility (held-out, ELBO / KL / marker r / ARI)\n")
    print("| Method | ELBO old | ELBO new | KL old | KL new | marker r old | marker r new | ARI new |")
    print("|---|---|---|---|---|---|---|---|")
    old_suite = json.load(open(OLD_UTIL))['metrics']
    old_cons = json.load(open(OLD_CONS))
    old_map = {'baseline': old_suite['Baseline'], 'retrain': old_suite['Retrain'], 'fisher': old_suite['Fisher'],
               'retain_finetune': old_suite['Retain-FT'], 'gradient_ascent': old_suite['Grad-Ascent'],
               'extragradient': old_suite['Extra-grad']}
    for m, pref in [('ssd', 'alpha1.0'), ('scrub', 'af1.0'), ('contrastive', 'gamma1.0'), ('dp_sgd', 'eps10.0')]:
        recs = [v['utility'] for k, v in old_cons['new_methods'][m]['results'].items() if k.startswith(pref)]
        old_map[m] = {f: float(np.mean([r[f] for r in recs])) for f in ['elbo', 'kl', 'marker_r', 'ari']}
    ua = C.pbmc_utility_all()['methods']
    for m in ['baseline', 'retrain', 'retrain_leaky', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'contrastive',
              'fisher', 'fisher_step_ablation', 'extragradient', 'extragradient_lambda1', 'dp_sgd']:
        if m not in ua:
            continue
        o = old_map.get(m); n = ua[m]['mean']
        print(f"| {LABEL[m]} | {fmt(o['elbo'],1) if o else ''} | {n['elbo']:.1f} | {fmt(o['kl'],2) if o else ''} | {n['kl']:.2f} | "
              f"{fmt(o['marker_r']) if o else ''} | {n['marker_r']:.3f} | {n['ari']:.3f} |")
        audit['utility'][m] = {'old': o, 'new': n}

    print("\n## Refit attacker control (clean retrains)\n")
    r = C.pbmc_refit()['models']
    ro = json.load(open(OLD_REFIT))['models']
    print(f"baseline refit CV AUC old {ro['baseline']['cv_auc']:.3f} new {r['baseline']['cv_auc']:.3f}; "
          f"retrain refit adv old {ro['retrain']['cv_advantage_mean']:.3f} [{ro['retrain']['nested_advantage']['nested_ci_low']:.2f}, {ro['retrain']['nested_advantage']['nested_ci_high']:.2f}] "
          f"new {r['retrain']['cv_advantage_mean']:.3f} [{r['retrain']['nested_advantage']['nested_ci_low']:.2f}, {r['retrain']['nested_advantage']['nested_ci_high']:.2f}]")
    audit['refit'] = {'old': ro['retrain']['cv_advantage_mean'], 'new': r['retrain']['cv_advantage_mean']}

    print("\n## Tabula Muris (structured, cluster 33)\n")
    tm = C.tm_results(); tmo = json.load(open(OLD_TM))
    print("| Model | AUC old (Jan, NB12 attacker for baseline/retrain) | AUC new (one attacker, July TM baseline) |")
    print("|---|---|---|")
    rows = [('Baseline', tmo['baseline']['structured']['baseline_auc'], tm['structured']['baseline_auc']),
            ('Retrain', tmo['baseline']['structured']['retrain_floor'], tm['structured']['retrain_auc']),
            ('Extra-gradient', tmo['unlearning']['extra-gradient (structured)']['auc'], tm['unlearning'].get('extra-gradient (structured)')),
            ('Fisher (structured)', tmo['unlearning']['fisher (structured)']['auc'], tm['unlearning'].get('fisher (structured)')),
            ('Fisher (scattered)', tmo['unlearning']['fisher (scattered)']['auc'], tm['unlearning'].get('fisher (scattered)'))]
    for name, o, n in rows:
        print(f"| {name} | {fmt(o)} | {fmt(n)} |"); audit['tm'][name] = {'old': o, 'new': n}

    print("\n## PBMC cluster 12\n")
    c = C.c12_results(); co = json.load(open(OLD_C12))
    print("| Model | Adv old | Adv new [CI] | AUC new | marker r new |")
    print("|---|---|---|---|---|")
    omap = {'baseline': 'Baseline', 'retrain_leaky': 'Retrain (c12-excl)', 'retain_finetune': 'Retain-only FT', 'fisher': 'Fisher', 'extragradient': 'Extra-grad (lam=10)'}
    for m, v in c['methods'].items():
        o = co.get(omap.get(m, ''), {}).get('adv_mean')
        print(f"| {LABEL.get(m, m)} | {fmt(o)} | {v['mean']:.3f} [{v['nested_ci_low']:.2f}, {v['nested_ci_high']:.2f}] | {v['auc_mean']:.3f} | {v['utility_mean']['marker_r']:.3f} |")
        audit['c12'][m] = {'old': o, 'new': v['mean'], 'auc_new': v['auc_mean']}

    print("\n## Attack suite (PBMC, library size restored; retrain = clean seed 42)\n")
    a = json.load(open(BASE / 'outputs' / 'attack_suite_results.json')); ao = json.load(open(OLD_ATTACKS))
    print("| Attack | Baseline old | Baseline new | Retrain old | Retrain new | EG old | EG new | Fisher old | Fisher new |")
    print("|---|---|---|---|---|---|---|---|---|")
    def get(d, model, grp, att):
        try:
            return d[model][grp][att]['advantage']
        except KeyError:
            return None
    for grp, att in [('low_assumption', 'loss_recon'), ('low_assumption', 'loss_kl'), ('low_assumption', 'loss_elbo'),
                     ('high_assumption', 'likelihood_ratio'), ('high_assumption', 'knn_latent')]:
        vals = []
        for model in ['baseline', 'retrain', 'extra_grad_lambda10', 'fisher_structured']:
            vals += [get(ao, model, grp, att), get(a, model, grp, att)]
        print(f"| {att} | " + " | ".join(fmt(v) for v in vals) + " |")
        audit['attacks'][att] = vals
    # ---- Stage 1b families ----
    print("\n## Frozen critics (re-derived, 3 seeds; old = December single seed)\n")
    print("| Config | AUC old | AUC new | Adv new [CI] | LR new | ELBO new | marker r new |")
    print("|---|---|---|---|---|---|---|")
    sweep_old = json.load(open(BASE / 'outputs' / 'p2' / 'adversarial_sweep' / 'sweep_summary.json'))['experiments']
    lr_all = json.load(open(BASE / 'outputs' / 'attack_lr_reference.json'))['models']
    for cfg in ['frozen_single_lambda5', 'frozen_single_lambda10', 'frozen_multi_lambda5', 'frozen_multi_lambda10']:
        if cfg not in new:
            continue
        o = sweep_old.get(cfg, {}).get('posthoc_auc'); nn_ = new[cfg]; u_ = ua.get(cfg, {}).get('mean', {})
        lrk = 'frozen_' + cfg.replace('frozen_', '')
        print(f"| {cfg} | {fmt(o)} | {nn_['auc']:.3f} | {nn_['mean']:.3f} [{nn_['ci_low']:.2f}, {nn_['ci_high']:.2f}] | {fmt(lr_all.get(lrk, {}).get('auc_mean'))} | {fmt(u_.get('elbo'), 1)} | {fmt(u_.get('marker_r'))} |")
        audit.setdefault('frozen', {})[cfg] = {'old_auc': o, 'new_auc': nn_['auc'], 'adv_new': nn_['mean'], 'lr_new': lr_all.get(lrk, {}).get('auc_mean')}
    sz = json.load(open(BASE / 'outputs' / 'p4' / 'size_ablations_v2.json'))
    print("\n## Structured size ablation (extra-gradient lambda=10, 3 seeds; per-size clean retrains)\n")
    print("| n | negatives | Baseline AUC (LR) | Retrain AUC, adv (LR) | Extra-gradient AUC, adv [CI] (LR) | old EG AUC |")
    print("|---|---|---|---|---|---|")
    old_eg = {'10': 0.263, '50': 0.503, '100': 0.611}   # paper Table 7 (December code)
    for n, r in sz['structured_extragradient'].items():
        b, rt, eg = r['baseline'], r.get('retrain', {}), r.get('extragradient', {})
        print(f"| {n} | {r['n_matched']} | {b['auc_mean']:.3f} ({fmt(b['lr_auc_mean'])}) | {fmt(rt.get('auc_mean'))}, {fmt(rt.get('adv_mean'))} ({fmt(rt.get('lr_auc_mean'))}) | {fmt(eg.get('auc_mean'))}, {fmt(eg.get('adv_mean'))} [{fmt(eg.get('adv_ci', [None, None])[0], 2)}, {fmt(eg.get('adv_ci', [None, None])[1], 2)}] ({fmt(eg.get('lr_auc_mean'))}) | {old_eg.get(n, '')} |")
    audit['size_structured'] = sz['structured_extragradient']
    print("\n## Scattered size ablation (Fisher, 10 seeds; per-size clean retrains)\n")
    print("| n | negatives | Baseline AUC (LR) | Retrain AUC, adv (LR) | Fisher AUC, adv [CI] (LR) | old Fisher AUC | old baseline |")
    print("|---|---|---|---|---|---|---|")
    old_sum = json.load(open(SUP / 'size_ablation_summary__dec.json'))['results_by_size']
    old_base = json.load(open(BASE / 'outputs' / 'p4' / 'size_ablation' / 'scattered_baselines.json'))
    for n, r in sz['scattered_fisher'].items():
        b, rt, f = r['baseline'], r.get('retrain', {}), r.get('fisher', {})
        print(f"| {n} | {r['n_matched']} | {b['auc_mean']:.3f} ({fmt(b['lr_auc_mean'])}) | {fmt(rt.get('auc_mean'))}, {fmt(rt.get('adv_mean'))} ({fmt(rt.get('lr_auc_mean'))}) | {fmt(f.get('auc_mean'))}, {fmt(f.get('adv_mean'))} [{fmt(f.get('adv_ci', [None, None])[0], 2)}, {fmt(f.get('adv_ci', [None, None])[1], 2)}] ({fmt(f.get('lr_auc_mean'))}) | {fmt(old_sum.get(n, {}).get('mean'))} | {fmt(old_base.get(n, {}).get('baseline_auc'))} |")
    audit['size_scattered'] = sz['scattered_fisher']
    pa = json.load(open(BASE / 'outputs' / 'p6' / 'phase_audit.json'))['runs']
    print("\n## Phase audit (negative ELBO per cell; MLP and reference-LR AUC)\n")
    print("| Stage | forget | matched | retain | unseen | MLP AUC | LR AUC |")
    print("|---|---|---|---|---|---|---|")
    def prow(tag, r): print(f"| {tag} | {r['forget_neg_elbo']:.1f} | {r['matched_neg_elbo']:.1f} | {r['retain_neg_elbo']:.1f} | {r['unseen_neg_elbo']:.1f} | {r['mlp_auc']:.3f} | {r['lr_auc']:.3f} |")
    prow('baseline', pa['baseline'])
    for k, v in pa.items():
        if k == 'baseline':
            continue
        for ph, r in v.items():
            prow(f"{k}, {ph.replace('_', ' ')}", r)
    audit['phase_audit'] = pa
    for fam, path in [('rau', BASE / 'outputs' / 'rau' / 'rau_results.json'), ('augmentation', BASE / 'outputs' / 'augmentation' / 'augmentation_results.json')]:
        if path.exists():
            audit[fam] = json.load(open(path))
    audit['lr_reference'] = {k: v['auc_mean'] for k, v in lr_all.items()}
    json.dump(audit, open(BASE / 'outputs' / 'claims_audit.json', 'w'), indent=2)
    print(f"\nsaved outputs/claims_audit.json")


if __name__ == '__main__':
    main()
