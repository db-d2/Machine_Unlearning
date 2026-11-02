#!/usr/bin/env python3
"""Scoring for the retrain-free stopping probe on PBMC-68k (round-7 review, "the best missing experiment").

The checkpoints of `scripts/stopping_probe_68k_train.py` form a knob grid on both forget sets: the Fisher scrub's step
cap (2e-3, 3e-3, 5e-3, 1e-2; the published 1e-3 checkpoints are scored alongside) and the length of the retain-only
fine-tune (epochs 10 to 200), three seeds each, all trained on the standard split with the whole forget set deleted.
The question the grid can answer: does any quantity a controller can observe WITHOUT a retrain change where the
retrain-based verdict changes along the knob.

Retrain-based verdicts (the paper's instruments, from the v6 pass so that the comparison is paired on the shared cells):
- per-cell: the standardized out-of-fold logistic AUC of the forget cells against the held-out cells of their type
  (`percell_stats`), paired against the eight-retrain arm exactly as `scripts/percell_paired_test.py` does, with that
  test's margin (half the baseline's excess, fixed 2026-09-22) and the retrains' out-of-fold matrices stored in
  `percell_oof_<fs>.npz`; nested over the three seeds of a knob setting;
- type level: the cross-type audit against clean retrain seed 42 (`likelihood_audit.lr_auc`, reference fits stored in
  `persample_<fs>.json`) read against the all-pairs retrained range (`audit_band_symmetric.json`), and the forget-cell
  median against the eight retrains' medians.

Controller-observable quantities (no retrain, no held-out cells of the forget type): the forget cells' fit (mean and
median negative ELBO), the fit of a fixed seeded sample of 2,000 retain cells, the fit of the forget cells' look-alikes
(their ten nearest retain cells in the baseline's latent space, over a seeded pool of 10,000 retain cells; the
substitute target of NB41), the fit of the cross-type matched negatives (held-out cells of other types), the held-out
utility (deterministic negative ELBO on the unseen cells, marker r, ARI), and the fine-tune's training loss where the
checkpoint stores it. The fit of the held-out cells of the forget type is recorded as the unobservable target.

Sanity check: the baseline is scored through the same code and must reproduce the v6 per-cell statistic and audit.

Writes outputs/pbmc68k/stopping_probe_<fs>.json. Run: PYTHONPATH=src python scripts/stopping_probe_68k.py --forget_set mk
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, get_retain_latent_codes, get_features, MARKER_GENES
from likelihood_audit import neg_elbo_per_cell, lr_auc, lr_auc_ci, nested_auc_ci
from utility_metrics import compute_latent_metrics, compute_marker_correlation
import percell_stats as PS
from percell_paired_test import verdict, B as PAIRED_B, BOOT_SEED, NEST_SEED
from stopping_probe_68k_train import CAPS, EPOCHS, SEEDS

OUT = BASE / 'outputs/pbmc68k'
DATA = BASE / 'data/pbmc68k_processed.h5ad'
N_RETAIN_SAMPLE, N_POOL, K_LOOKALIKE, SAMPLE_SEED = 2000, 10000, 10, 0


def checkpoints(fs):
    d = OUT / fs
    cm = {'baseline': [('seed42', OUT / 'baseline/best_model.pt')],
          'fisher_cap0.001': [(f'{s}', d / f'fisher_seed{s}/unlearned_model.pt') for s in SEEDS],
          'retain_finetune_published': [(f'{s}', d / f'retain_finetune_seed{s}/best_model.pt') for s in SEEDS]}
    for cap in CAPS:
        cm[f'fisher_cap{cap}'] = [(f'{s}', d / f'fisher_cap{cap}_seed{s}/unlearned_model.pt') for s in SEEDS]
    for ep in EPOCHS:
        cm[f'finetune_epoch{ep}'] = [(f'{s}', d / f'finetune_traj_seed{s}/epoch{ep}.pt') for s in SEEDS]
    return {k: [(s, p) for s, p in v if p.exists()] for k, v in cm.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forget_set', required=True); args = ap.parse_args()
    fs = args.forget_set
    adata = sc.read_h5ad(DATA); split = json.load(open(OUT / f'split_{fs}.json'))
    forget, retain, unseen, within = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'unseen_indices', 'within_type_unseen_indices'))
    cross = np.array(json.load(open(OUT / f'matched_negatives_{fs}.json'))['matched_indices'])
    X = adata.X
    def tens(idx):
        x = X[idx]; x = x.toarray() if hasattr(x, 'toarray') else x
        return torch.FloatTensor(np.asarray(x))
    # reference quantities from the v6 pass
    ps = json.load(open(OUT / f'persample_{fs}.json')); res = json.load(open(OUT / f'results_{fs}.json'))
    assert res.get('version', '').startswith('v6')
    ref = {k: np.array(v) for k, v in ps['reference_neg_elbo'].items()}
    retrain_medians = {r['seed']: float(np.median(r['forget'])) for r in ps['neg_elbo']['retrain']}
    retrain_within_medians = {r['seed']: float(np.median(r['within'])) for r in ps['neg_elbo']['retrain']}
    band = json.load(open(BASE / 'outputs/audit_band_symmetric.json'))[f'pbmc68k_{fs}']['all_pairs_range']
    band42 = res['summary']['cross']['audit_range_retrained']
    paired = json.load(open(OUT / f'percell_paired_{fs}.json'))
    z = np.load(OUT / f'percell_oof_{fs}.npz')
    kinds = {'cross': cross, 'within': within}   # cross before within: the order of score_pbmc68k v6 (the sampled z consumes torch's RNG; troubleshooting 13, 15)
    y = {k: np.r_[np.ones(len(forget)), np.zeros(len(v))] for k, v in kinds.items()}
    w = {k: PS.boot_weights(len(forget), len(v), n_boot=PAIRED_B, seed=BOOT_SEED) for k, v in kinds.items()}
    ref_draws, base_draws, ref_point = {}, {}, {}
    for k in kinds:
        keys = sorted(q for q in z.files if q.startswith(f'{k}__retrain__'))
        ref_draws[k] = np.array([PS.boot_auc(z[q].astype(np.float64), y[k], *w[k]) for q in keys])
        ref_point[k] = [PS.auc_mean(z[q].astype(np.float64), y[k]) for q in keys]
        base_draws[k] = PS.boot_auc(z[f'{k}__baseline__seed42'].astype(np.float64), y[k], *w[k])
    margin = {k: paired['kinds'][k]['margin'] for k in kinds}
    print(f'{fs}: forget {len(forget)}, within {len(within)}, cross {len(cross)}; retrain per-cell arm {np.mean(ref_point["within"]):.4f}, margin {margin["within"]:.4f}; audit band {band}', flush=True)
    # observable sets, fixed by the baseline and one seed
    base, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    rng = np.random.default_rng(SAMPLE_SEED)
    retain_sample = np.sort(rng.choice(retain, N_RETAIN_SAMPLE, replace=False)); pool = np.sort(rng.choice(retain, N_POOL, replace=False))
    with torch.no_grad():
        mu_f, _ = base.encode(tens(forget)); mu_p, _ = base.encode(tens(pool))
    _, nn_idx = NearestNeighbors(n_neighbors=K_LOOKALIKE).fit(mu_p.numpy()).kneighbors(mu_f.numpy())
    lookalike = pool[np.unique(nn_idx.flatten())]
    labels_h = adata.obs['leiden'].values[unseen]; X_h = tens(unseen)
    gene_names = list(adata.var_names); marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]
    sets = {'forget': tens(forget), 'within': tens(within), 'cross': tens(cross), 'lookalike': tens(lookalike), 'retain_sample': tens(retain_sample)}
    results = {}
    for method, seeds in checkpoints(fs).items():
        rows, draws = [], {k: [] for k in kinds}
        for sd, path in seeds:
            model, _ = load_vae_model(str(path))
            ck = torch.load(str(path), map_location='cpu', weights_only=False)
            torch.manual_seed(0); np.random.seed(0)
            retain_z = get_retain_latent_codes(model, adata, retain)
            Ff = get_features(model, adata, forget, retain_z=retain_z)
            Fn = {k: get_features(model, adata, v, retain_z=retain_z) for k, v in kinds.items()}
            row = {'seed': sd, 'path': str(path.relative_to(BASE))}
            for k in kinds:
                x = torch.cat([Ff, Fn[k]]).numpy(); oof = PS.oof_splits(x, y[k])
                row[f'refit_std_{k}'] = PS.auc_mean(oof, y[k]); draws[k].append(PS.boot_auc(oof, y[k], *w[k]))
            ne = {k: neg_elbo_per_cell(model, v) for k, v in sets.items()}
            row.update({f'{k}_mean': float(v.mean()) for k, v in ne.items()}); row.update({f'{k}_median': float(np.median(v)) for k, v in ne.items()})
            row.update({'audit_cross': lr_auc(ref['forget'], ref['cross'], ne['forget'], ne['cross']), 'audit_within': lr_auc(ref['forget'], ref['within'], ne['forget'], ne['within']),
                        'audit_cross_ci': lr_auc_ci(ref['forget'], ref['cross'], ne['forget'], ne['cross'])})
            heldout = neg_elbo_per_cell(model, X_h)
            row.update({'heldout_elbo': float(heldout.mean()), 'marker_r': compute_marker_correlation(model, X_h, marker_idx, gene_names, 'cpu')['mean_r'],
                        'ari': compute_latent_metrics(model, X_h, labels_h, 'cpu')['ari'], 'train_loss': ck.get('train_loss') if isinstance(ck, dict) else None})
            rows.append(row)
            print(f"  {method:26s} {sd:6s} per-cell {row['refit_std_within']:.4f} audit {row['audit_cross']:.3f} forget {row['forget_mean']:.1f}/{row['forget_median']:.1f} "
                  f"within {row['within_median']:.1f} lookalike {row['lookalike_mean']:.1f} retain {row['retain_sample_mean']:.1f} cross {row['cross_mean']:.1f} heldout {row['heldout_elbo']:.1f} marker {row['marker_r']:.3f}", flush=True)
        arm = {'per_seed': rows, 'n_seeds': len(rows)}
        for k in kinds:
            d = np.array(draws[k]); nd = PS.nested_arm(d, seed=NEST_SEED); diff = PS.paired_nested_diff(d, ref_draws[k], seed=NEST_SEED)
            lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            db = PS.nested_arm(d, seed=NEST_SEED) - base_draws[k]
            arm[f'refit_std_{k}'] = {'mean': float(np.mean([r[f'refit_std_{k}'] for r in rows])), 'nested_ci': [float(np.percentile(nd, 2.5)), float(np.percentile(nd, 97.5))],
                                    'diff_vs_retrain': float(np.mean([r[f'refit_std_{k}'] for r in rows]) - np.mean(ref_point[k])), 'diff_ci': [lo, hi],
                                    'diff_vs_baseline_ci': [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                                    'verdict': verdict(lo, hi, margin[k]) if method != 'baseline' else 'positive control'}
        arm['audit_cross_nested'] = nested_auc_ci([r['audit_cross_ci'] for r in rows])
        arm['audit_cross_in_all_pairs_range'] = bool(band[0] <= arm['audit_cross_nested']['mean'] <= band[1])
        fm = [r['forget_median'] for r in rows]; rm = list(retrain_medians.values())
        arm['forget_median_range'] = [min(fm), max(fm)]
        arm['type_verdict_median'] = ('fitted better than retrained' if max(fm) < min(rm) else 'fitted worse than retrained' if min(fm) > max(rm) else 'at the retrained level')
        for q in ('forget_mean', 'forget_median', 'within_median', 'lookalike_mean', 'retain_sample_mean', 'cross_mean', 'heldout_elbo', 'marker_r', 'ari'):
            arm[q] = float(np.mean([r[q] for r in rows]))
        results[method] = arm
        print(f"  => {method:26s} per-cell {arm['refit_std_within']['mean']:.4f} diff {arm['refit_std_within']['diff_vs_retrain']:+.3f} [{arm['refit_std_within']['diff_ci'][0]:+.3f}, {arm['refit_std_within']['diff_ci'][1]:+.3f}] {arm['refit_std_within']['verdict']}; "
              f"audit {arm['audit_cross_nested']['mean']:.3f} ({'in' if arm['audit_cross_in_all_pairs_range'] else 'outside'} range); median {arm['forget_median']:.0f} vs retrained {min(rm):.0f} to {max(rm):.0f}: {arm['type_verdict_median']}", flush=True)
    b0 = res['per_checkpoint']['baseline']['per_seed'][0]
    check = {'baseline_refit_std_within_v6': b0['refit_std_within']['cv_auc_mean'], 'baseline_refit_std_within_here': results['baseline']['per_seed'][0]['refit_std_within'],
             'baseline_audit_cross_v6': b0['audit_cross'], 'baseline_audit_cross_here': results['baseline']['per_seed'][0]['audit_cross']}
    check['reproduced'] = bool(abs(check['baseline_refit_std_within_v6'] - check['baseline_refit_std_within_here']) < 1e-6 and abs(check['baseline_audit_cross_v6'] - check['baseline_audit_cross_here']) < 1e-9)
    print(f"sanity: {check}", flush=True)
    json.dump({'forget_set': fs, 'split': res['split'], 'n_cross': int(len(cross)), 'n_lookalike': int(len(lookalike)), 'n_retain_sample': N_RETAIN_SAMPLE,
               'observable': ['forget_mean', 'forget_median', 'lookalike_mean', 'retain_sample_mean', 'cross_mean', 'heldout_elbo', 'marker_r', 'ari', 'train_loss'],
               'unobservable': ['within_mean', 'within_median', 'audit_cross', 'audit_within', 'refit_std_within'],
               'retrain_percell_arm': {'mean': float(np.mean(ref_point['within'])), 'per_seed': ref_point['within']}, 'margin': margin,
               'retrain_forget_medians': retrain_medians, 'retrain_within_medians': retrain_within_medians, 'audit_all_pairs_range': band, 'audit_range_seed42': band42,
               'estimator': 'likelihood_audit.neg_elbo_per_cell (16 seeded z samples); features with torch.manual_seed(0)/np.random.seed(0) per checkpoint as in score_pbmc68k v6',
               'paired_test': f'as scripts/percell_paired_test.py: B={PAIRED_B}, boot seed {BOOT_SEED}, nested seed {NEST_SEED}, retrain draws from percell_oof_{fs}.npz',
               'sanity': check, 'arms': results}, open(OUT / f'stopping_probe_{fs}.json', 'w'), indent=1)
    print(f"saved {OUT / f'stopping_probe_{fs}.json'}", flush=True)


if __name__ == '__main__':
    main()
