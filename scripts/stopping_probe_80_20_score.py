#!/usr/bin/env python3
"""Scoring for the full retrain-free stopping probe on PBMC-68k (checkpoints of `stopping_probe_80_20_train.py`).

Setting: 20 percent of the forget cells were HELD BACK as members the controller knows (they stay in the retain set)
and the other 80 percent were deleted. Along each knob (fine-tune length; Fisher cap) two things are measured per
checkpoint.

The retrain-based verdict on the deleted cells, which needs the eight clean retrains trained on the new retain set
(retain plus held-back cells) and the held-out cells of the forget type:
- per cell: the standardized out-of-fold logistic AUC of the deleted cells against the held-out cells of their type,
  paired against the eight-retrain arm on the shared cells (`percell_stats`), margin half the baseline's excess on the
  same cells (the rule of `percell_paired_test.py`), nested over the three seeds of a knob setting;
- type level: the reference-free cross-type audit and the paired median gap of `audit_stats` against the retrains.

What a controller can observe without a retrain and without held-out cells of the type:
- the calibration attacker, the same standardized refit of the deleted cells against the held-back members (both
  sets of known membership), out of fold; at the baseline the two groups are members alike and it sits near 0.5, and
  as the deletion takes effect the deleted cells separate from the members;
- the median fit gap, deleted cells minus held-back members (negative ELBO);
- the deleted cells' fit against their retain look-alikes and against the cross-type negatives, the retain sample's
  fit, and held-out utility (deterministic ELBO, marker r, ARI), as in `stopping_probe_68k.py`.
Diagnostic: the held-back members against the held-out cells of the type (are the kept members identifiable).

Cross-type negatives are the matched set of the original forget set (`matched_negatives_<fs>.json`), which contains
the deleted cells' neighbours. Features are extracted cross, within, held-back in that order under a pinned RNG.

Writes outputs/pbmc68k/stopping_probe80_<fs>.json. Run: PYTHONPATH=src python scripts/stopping_probe_80_20_score.py --forget_set mk
"""
import argparse
import json
import sys
import zlib
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
from likelihood_audit import neg_elbo_per_cell
from utility_metrics import compute_latent_metrics, compute_marker_correlation
import percell_stats as PS
import audit_stats as AS
from percell_paired_test import verdict, B as PAIRED_B, BOOT_SEED, NEST_SEED, MARGIN_FRACTION
from stopping_probe_68k_train import SEEDS, EPOCHS
from stopping_probe_80_20_train import RETRAIN_SEEDS, CAPS

OUT = BASE / 'outputs/pbmc68k'
DATA = BASE / 'data/pbmc68k_processed.h5ad'
N_RETAIN_SAMPLE, N_POOL, K_LOOKALIKE, SAMPLE_SEED, B_AUDIT = 2000, 10000, 10, 0, 2000


def checkpoints(fs):
    d = OUT / f'{fs}_probe80'
    cm = {'baseline': [('seed42', OUT / 'baseline/best_model.pt')],
          'retrain': [(f'seed_{s}', d / f'retrain_clean_seed_{s}/best_model.pt') for s in RETRAIN_SEEDS if (d / f'retrain_clean_seed_{s}/training_complete.json').exists()]}
    for cap in CAPS:
        cm[f'fisher_cap{cap}'] = [(f'{s}', d / f'fisher_cap{cap}_seed{s}/unlearned_model.pt') for s in SEEDS]
    for ep in EPOCHS:
        cm[f'finetune_epoch{ep}'] = [(f'{s}', d / f'finetune_traj_seed{s}/epoch{ep}.pt') for s in SEEDS]
    return {k: [(s, p) for s, p in v if p.exists()] for k, v in cm.items() if any(p.exists() for _, p in v)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forget_set', required=True); args = ap.parse_args()
    fs = args.forget_set
    adata = sc.read_h5ad(DATA); split = json.load(open(OUT / f'split_{fs}_probe80.json'))
    deleted, held, retain, unseen, within = (np.array(split[k]) for k in ('forget_indices', 'heldback_member_indices', 'retain_indices', 'unseen_indices', 'within_type_unseen_indices'))
    cross = np.array(json.load(open(OUT / f'matched_negatives_{fs}.json'))['matched_indices'])
    X = adata.X
    def tens(idx):
        x = X[idx]; x = x.toarray() if hasattr(x, 'toarray') else x
        return torch.FloatTensor(np.asarray(x))
    base, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    rng = np.random.default_rng(SAMPLE_SEED)
    retain_sample = np.sort(rng.choice(retain, N_RETAIN_SAMPLE, replace=False)); pool = np.sort(rng.choice(retain, N_POOL, replace=False))
    with torch.no_grad():
        mu_f, _ = base.encode(tens(deleted)); mu_p, _ = base.encode(tens(pool))
    _, nn_idx = NearestNeighbors(n_neighbors=K_LOOKALIKE).fit(mu_p.numpy()).kneighbors(mu_f.numpy())
    lookalike = pool[np.unique(nn_idx.flatten())]
    labels_h = adata.obs['leiden'].values[unseen]; X_h = tens(unseen)
    gene_names = list(adata.var_names); marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]
    sets = {'deleted': tens(deleted), 'within': tens(within), 'heldback': tens(held), 'cross': tens(cross), 'lookalike': tens(lookalike), 'retain_sample': tens(retain_sample)}
    kinds = {'cross': cross, 'within': within, 'heldback': held}   # extraction order (pinned RNG)
    pairs = {'within': ('deleted', 'within'), 'heldback': ('deleted', 'heldback'), 'members_vs_within': ('heldback', 'within')}
    y = {k: np.r_[np.ones(len(sets[a])), np.zeros(len(sets[b]))] for k, (a, b) in pairs.items()}
    w = {k: PS.boot_weights(len(sets[a]), len(sets[b]), n_boot=PAIRED_B, seed=BOOT_SEED) for k, (a, b) in pairs.items()}
    cm = checkpoints(fs); print(f'{fs} probe80: deleted {len(deleted)}, held back {len(held)}, within {len(within)}, cross {len(cross)}; arms {[(m, len(v)) for m, v in cm.items()]}', flush=True)
    rows, draws, ne_store = {}, {m: {k: [] for k in pairs} for m in cm}, {m: [] for m in cm}
    for method, seeds in cm.items():
        rows[method] = []
        for sd, path in seeds:
            model, _ = load_vae_model(str(path)); ck = torch.load(str(path), map_location='cpu', weights_only=False)
            torch.manual_seed(0); np.random.seed(0)
            retain_z = get_retain_latent_codes(model, adata, retain)
            F = {'deleted': get_features(model, adata, deleted, retain_z=retain_z)}
            F.update({k: get_features(model, adata, v, retain_z=retain_z) for k, v in kinds.items()})
            row = {'seed': sd, 'path': str(path.relative_to(BASE))}
            for k, (a, b) in pairs.items():
                x = torch.cat([F[a], F[b]]).numpy(); oof = PS.oof_splits(x, y[k])
                row[f'refit_{k}'] = PS.auc_mean(oof, y[k]); draws[method][k].append(PS.boot_auc(oof, y[k], *w[k]))
            ne = {k: neg_elbo_per_cell(model, v) for k, v in sets.items()}; ne_store[method].append(ne)
            row.update({f'{k}_mean': float(v.mean()) for k, v in ne.items()}); row.update({f'{k}_median': float(np.median(v)) for k, v in ne.items()})
            heldout = neg_elbo_per_cell(model, X_h)
            row.update({'heldout_elbo': float(heldout.mean()), 'marker_r': compute_marker_correlation(model, X_h, marker_idx, gene_names, 'cpu')['mean_r'],
                        'ari': compute_latent_metrics(model, X_h, labels_h, 'cpu')['ari'], 'train_loss': ck.get('train_loss') if isinstance(ck, dict) else None})
            rows[method].append(row)
            print(f"  {method:20s} {sd:8s} deleted-vs-within {row['refit_within']:.4f} deleted-vs-members {row['refit_heldback']:.4f} members-vs-within {row['refit_members_vs_within']:.4f} "
                  f"deleted {row['deleted_median']:.0f} members {row['heldback_median']:.0f} within {row['within_median']:.0f} lookalike {row['lookalike_mean']:.0f} retain {row['retain_sample_mean']:.0f} heldout {row['heldout_elbo']:.1f} marker {row['marker_r']:.3f}", flush=True)
    # reference arms
    R_del = np.array([ne['deleted'] for ne in ne_store['retrain']]); R_cross = np.array([ne['cross'] for ne in ne_store['retrain']])
    ref_mean = {k: float(np.mean([r[f'refit_{k}'] for r in rows['retrain']])) for k in pairs}
    base_mean = {k: rows['baseline'][0][f'refit_{k}'] for k in pairs}
    excess = PS.paired_nested_diff(np.array(draws['baseline']['within']), np.array(draws['retrain']['within']), seed=NEST_SEED + 1)
    excess_ci = [float(np.percentile(excess, 2.5)), float(np.percentile(excess, 97.5))]
    margin = MARGIN_FRACTION * excess_ci[0]   # round 8: half the LOWER bound of the baseline's excess
    n_del = len(deleted); wp = AS.cell_weights(n_del, B_AUDIT, 0); wn = AS.cell_weights(len(cross), B_AUDIT, 1)
    ref_draws = np.random.default_rng(1).integers(0, len(R_del), (B_AUDIT, len(R_del)))
    print(f'retrain per-cell arm (deleted vs within) {ref_mean["within"]:.4f}; baseline {base_mean["within"]:.4f}; margin {margin:.4f}', flush=True)
    arms = {}
    for method, rs in rows.items():
        arm = {'per_seed': rs, 'n_seeds': len(rs)}
        for k in pairs:
            d = np.array(draws[method][k]); nd = PS.nested_arm(d, seed=NEST_SEED); mean = float(np.mean([r[f'refit_{k}'] for r in rs]))
            arm[f'refit_{k}'] = {'mean': mean, 'nested_ci': AS.ci(nd)}
            if method != 'retrain':
                diff = PS.paired_nested_diff(d, np.array(draws['retrain'][k]), seed=NEST_SEED); lo, hi = AS.ci(diff)
                arm[f'refit_{k}'].update({'diff_vs_retrain': mean - ref_mean[k], 'diff_ci': [lo, hi]})
                if k == 'within':
                    arm[f'refit_{k}']['verdict'] = verdict(lo, hi, margin) if method != 'baseline' else 'positive control'
        T_del = np.array([ne['deleted'] for ne in ne_store[method]]); T_cross = np.array([ne['cross'] for ne in ne_store[method]])
        tdraws = np.random.default_rng(zlib.crc32(method.encode())).integers(0, len(T_del), B_AUDIT)
        if method != 'retrain':
            pts = AS.point_audit(T_del, T_cross, R_del, R_cross); td, rd = AS.nested_audit(T_del, T_cross, R_del, R_cross, wp, wn, ref_draws, tdraws)
            lo, hi = AS.ci(td - rd); arm['audit_cross'] = {'mean': float(np.mean(pts)), 'ci': AS.ci(td), 'diff_ci': [lo, hi], 'verdict': AS.verdict_diff(lo, hi, None)}
            g = AS.nested_gap(T_del, R_del, wp, ref_draws, tdraws, 'median'); lo, hi = AS.ci(g)
            arm['gap_median'] = {'gap': AS.point_gap(T_del, R_del, 'median'), 'gap_ci': [lo, hi], 'verdict': AS.verdict_diff(lo, hi, better='fitted worse than retrained', worse='fitted better than retrained')}
        for q in ('deleted_median', 'heldback_median', 'within_median', 'lookalike_mean', 'retain_sample_mean', 'cross_mean', 'heldout_elbo', 'marker_r', 'ari'):
            arm[q] = float(np.mean([r[q] for r in rs]))
        arm['gap_deleted_minus_members'] = arm['deleted_median'] - arm['heldback_median']
        arms[method] = arm
        if method != 'retrain':
            v = arm['refit_within']
            print(f"  => {method:20s} deleted-vs-within {v['mean']:.4f} diff {v.get('diff_vs_retrain', 0):+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}] {v['verdict']}; "
                  f"CALIBRATION deleted-vs-members {arm['refit_heldback']['mean']:.4f} {arm['refit_heldback']['nested_ci']}; members-vs-within {arm['refit_members_vs_within']['mean']:.4f}; "
                  f"audit {arm['audit_cross']['mean']:.3f} {arm['audit_cross']['verdict']}; gap {arm['gap_median']['gap']:+.0f}; deleted-members gap {arm['gap_deleted_minus_members']:+.0f}", flush=True)
    json.dump({'forget_set': fs, 'split': {k: split[k] for k in ('n_forget', 'n_heldback', 'n_retain', 'n_unseen', 'n_within_type_unseen', 'forget_cluster', 'forget_type', 'holdback_fraction', 'holdback_seed')},
               'n_cross': int(len(cross)), 'n_lookalike': int(len(lookalike)), 'retrain_seeds': [s for s, _ in cm['retrain']],
               'retrain_percell_arm': {k: ref_mean[k] for k in pairs}, 'baseline_percell': base_mean, 'baseline_excess_ci': excess_ci, 'margin': margin, 'margin_rule': 'half the lower 95 percent bound of the baseline excess (round 8)',
               'observable': ['refit_heldback', 'gap_deleted_minus_members', 'deleted_median', 'heldback_median', 'lookalike_mean', 'retain_sample_mean', 'cross_mean', 'heldout_elbo', 'marker_r', 'ari', 'train_loss'],
               'retrain_based': ['refit_within', 'audit_cross', 'gap_median', 'within_median', 'refit_members_vs_within'],
               'estimator': 'likelihood_audit.neg_elbo_per_cell; features with torch.manual_seed(0)/np.random.seed(0) per checkpoint, extracted deleted, cross, within, held-back in that order',
               'paired_test': f'as scripts/percell_paired_test.py: B={PAIRED_B}, boot seed {BOOT_SEED}, nest seed {NEST_SEED}, margin {MARGIN_FRACTION} x (baseline - retrain arm) on the deleted cells',
               'audit': f'reference-free (audit_stats), B={B_AUDIT}, reference = mean over the probe80 retrains', 'arms': arms},
              open(OUT / f'stopping_probe80_{fs}.json', 'w'), indent=1)
    print(f"saved {OUT / f'stopping_probe80_{fs}.json'}", flush=True)


if __name__ == '__main__':
    main()
