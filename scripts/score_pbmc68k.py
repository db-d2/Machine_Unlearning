#!/usr/bin/env python3
"""Score the PBMC-68k cohort with the paper's instruments, one forget set at a time.

For a forget set: cross-type negatives are the ten nearest held-out cells of other clusters of each
forget cell in the baseline's latent space, deduplicated; within-type negatives are the held-out cells
of the forget cluster. One attacker per negative set is trained on the baseline (canonical recipe) and
applied to every checkpoint in one CPU pass; advantages carry nested bootstrap intervals and a nested
difference from the clean retrains; the audit is the deterministic likelihood ratio against clean
retrain seed 42 with the other retrain seeds as the range; utility is held-out ELBO, marker correlation, ARI.

The per-cell question uses a logistic refit on standardized features against the within-type negatives
(`src/percell_stats.py`): its interval is a cell bootstrap of the mean-over-splits statistic with one draw
shared by every checkpoint, and each checkpoint carries a label-permutation null on its own features.
`scripts/percell_paired_test.py` compares each arm with the clean retrains, paired on the shared cells.

v6 (2026-09-22, round-7 B1): every checkpoint's features (forget, cross, within) are extracted before any
refit, and the MLP refit, which reseeds torch's global RNG, runs last. Before this the MLP refit ran only
for the reference models and between the cross and the within extraction, so the reference arm's
within-type negatives came from a different z draw than every method's.

Writes outputs/pbmc68k/results_<fs>.json, persample_<fs>.json and percell_oof_<fs>.npz.
Run: PYTHONPATH=src python scripts/score_pbmc68k.py --forget_set mk
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features, evaluate_utility, MARKER_GENES
from likelihood_audit import neg_elbo_per_cell, lr_auc, lr_auc_ci, nested_auc_ci
import retrain_refit_attacker as RR
RR.DEVICE = 'cpu'
from retrain_refit_attacker import refit_cv_once, SPLIT_SEEDS
import percell_stats as PS

OUT = BASE / 'outputs/pbmc68k'
DATA = BASE / 'data/pbmc68k_processed.h5ad'
METHODS = ['retain_finetune', 'gradient_ascent', 'fisher', 'ssd', 'scrub', 'contrastive', 'extragradient']
REFIT = ['baseline', 'retrain', 'long_finetune']   # out-of-fold refit attacker (retrain_refit_attacker recipe) on these
B, BD = 10000, 5000
BOOT_SEED = 0        # one stratified cell bootstrap per negative set, shared by every checkpoint (paired draws)
N_PERM = 1000        # label permutations per checkpoint for the per-cell null (within-type negatives)
PERM_SEED = 20260922
rng = np.random.RandomState(0)


def advantage(a):
    return 2 * abs(a - 0.5)


def nested(recs):
    arrs = [(np.array(r['scores']), np.array(r['labels'])) for r in recs]
    per_seed = np.array([advantage(roc_auc_score(l, s)) for s, l in arrs]); S = len(arrs); samples = np.empty(B)
    for b in range(B):
        v = []
        for di in (rng.randint(0, S, S) if S > 1 else [0]):
            s, l = arrs[di]; idx = rng.randint(0, len(s), len(s)); ls = l[idx]
            v.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        samples[b] = np.mean(v)
    return {'mean': float(per_seed.mean()), 'n_seeds': S, 'nested_ci_low': float(np.percentile(samples, 2.5)),
            'nested_ci_high': float(np.percentile(samples, 97.5)), 'per_seed_advantage': per_seed.tolist(),
            'per_seed_auc': [float(roc_auc_score(l, s)) for s, l in arrs]}


def nested_diff(mrecs, rrecs):
    ma = [(np.array(r['scores']), np.array(r['labels'])) for r in mrecs]; ra = [(np.array(r['scores']), np.array(r['labels'])) for r in rrecs]
    def madv(arrs):
        S = len(arrs); v = []
        for di in (rng.randint(0, S, S) if S > 1 else [0]):
            s, l = arrs[di]; idx = rng.randint(0, len(s), len(s)); ls = l[idx]
            v.append(advantage(roc_auc_score(ls, s[idx])) if ls.min() != ls.max() else 0.0)
        return np.mean(v)
    diffs = np.array([madv(ma) - madv(ra) for _ in range(BD)])
    return {'diff_mean': float(diffs.mean()), 'diff_ci': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], 'p_method_gt_retrain': float((diffs > 0).mean())}


def checkpoints(fs):
    d = OUT / fs
    cm = {'baseline': [('seed42', OUT / 'baseline/best_model.pt')],
          'retrain': [(q.name.replace('retrain_clean_', ''), q / 'best_model.pt') for q in sorted(d.glob('retrain_clean_seed_*')) if (q / 'best_model.pt').exists() and (q / 'training_complete.json').exists()]}
    for m in METHODS:
        fn = 'unlearned_model.pt' if m == 'fisher' else 'best_model.pt'
        cm[m] = [(p.name.split('_seed')[-1], p / fn) for p in sorted(d.glob(f'{m}_seed*')) if (p / fn).exists()]
    if fs == 'mk':   # 200-epoch retain-only fine-tune (scripts/long_finetune_pbmc68k.py), megakaryocytes only
        cm['long_finetune'] = [(p.name.replace('seed', ''), p / 'final_model.pt') for p in sorted((OUT / 'long_finetune').glob('seed*')) if (p / 'final_model.pt').exists()]
    return {k: v for k, v in cm.items() if v}


def refit_std(Ff, Fn, with_null=False):
    """Out-of-fold AUC of a logistic regression on standardized features (scaler fit on the training folds),
    stratified 5-fold, averaged over 20 fold splits. The paper's raw-feature MLP collapses toward the class
    prior when the per-cell signal is weak (2026-09-19 check: 0.449 in-sample on the baseline where this gives
    0.68), so this is the instrument for the per-cell question. The interval bootstraps the cells and
    recomputes the mean over all 20 splits; the draw is shared by every checkpoint (BOOT_SEED). Returns the
    summary, the out-of-fold matrix (splits x cells) and the bootstrap draws."""
    x = torch.cat([Ff, Fn]).numpy(); y = np.r_[np.ones(len(Ff)), np.zeros(len(Fn))]
    oof = PS.oof_splits(x, y); aucs = [float(roc_auc_score(y, o)) for o in oof]
    wp, wn = PS.boot_weights(len(Ff), len(Fn), n_boot=B, seed=BOOT_SEED)
    draws = PS.boot_auc(oof, y, wp, wn); adv = 2 * np.abs(draws - 0.5)
    out = {'cv_auc_mean': float(np.mean(aucs)), 'cv_auc_sd': float(np.std(aucs, ddof=1)), 'cv_advantage_mean': float(np.mean([advantage(a) for a in aucs])),
           'auc_boot_ci': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
           'advantage_boot_ci': [float(np.percentile(adv, 2.5)), float(np.percentile(adv, 97.5))], 'per_split_auc': aucs,
           'boot': f'stratified cell bootstrap of the mean over {len(aucs)} fold splits, {B} draws, seed {BOOT_SEED}, shared by every checkpoint'}
    if with_null:
        out['perm_null'] = PS.null_summary(PS.perm_null(x, y, n_perm=N_PERM, seed=PERM_SEED), out['cv_auc_mean'])
    return out, oof, draws


def refit(Ff, Fn):
    """Repeated stratified 5-fold out-of-fold AUC of an attacker refit on this checkpoint's own features."""
    x = torch.cat([Ff, Fn]); y = torch.cat([torch.ones(len(Ff)), torch.zeros(len(Fn))]); y_np = y.numpy()
    aucs = [float(roc_auc_score(y_np, refit_cv_once(x, y, y_np, sd))) for sd in SPLIT_SEEDS]
    return {'cv_auc_mean': float(np.mean(aucs)), 'cv_auc_sd': float(np.std(aucs, ddof=1)), 'cv_advantage_mean': float(np.mean([advantage(a) for a in aucs])), 'per_split_auc': aucs}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forget_set', required=True); args = ap.parse_args()
    fs = args.forget_set
    adata = sc.read_h5ad(DATA); split = json.load(open(OUT / f'split_{fs}.json'))
    forget, retain, unseen, within = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'unseen_indices', 'within_type_unseen_indices'))
    base, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    X = adata.X
    def tens(idx):
        x = X[idx]; x = x.toarray() if hasattr(x, 'toarray') else x
        return torch.FloatTensor(np.asarray(x))
    # cross-type matched negatives: k nearest held-out cells of other clusters in the baseline's latent space
    pool = np.setdiff1d(unseen, within)
    with torch.no_grad():
        mu_f, _ = base.encode(tens(forget)); mu_p, _ = base.encode(tens(pool))
    nn = NearestNeighbors(n_neighbors=10).fit(mu_p.numpy()); _, idx = nn.kneighbors(mu_f.numpy())
    cross = pool[np.unique(idx.flatten())]
    json.dump({'matched_indices': cross.tolist(), 'method': 'latent_knn', 'k': 10, 'n_forget': int(len(forget)), 'n_matched': int(len(cross))},
              open(OUT / f'matched_negatives_{fs}.json', 'w'))
    print(f"{fs}: forget {len(forget)}, cross negatives {len(cross)}, within-type negatives {len(within)}", flush=True)
    attackers = {'cross': train_fresh_attacker(base, adata, forget, cross, retain), 'within': train_fresh_attacker(base, adata, forget, within, retain)}
    negs = {'cross': cross, 'within': within}
    retrain_seeds = [int(q.name.split('_')[-1]) for q in sorted((OUT / fs).glob('retrain_clean_seed_*')) if (q / 'training_complete.json').exists()]
    refs = {s: load_vae_model(str(OUT / fs / f'retrain_clean_seed_{s}/best_model.pt'))[0] for s in retrain_seeds}
    print(f'clean retrain seeds: {retrain_seeds} (reference 42; the others give the retrained range)', flush=True)
    ref = refs[42]
    ref_f = neg_elbo_per_cell(ref, tens(forget)); ref_c = neg_elbo_per_cell(ref, tens(cross)); ref_w = neg_elbo_per_cell(ref, tens(within))
    labels_h = adata.obs['leiden'].values[unseen]; X_h = tens(unseen)
    gene_names = list(adata.var_names); marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]
    cm = checkpoints(fs); persample = {k: {} for k in negs}; results = {}
    oof_store, draw_store = {}, {k: {} for k in negs}
    ref_cells = {'forget': ref_f.tolist(), 'cross': ref_c.tolist(), 'within': ref_w.tolist()}
    for method, seeds in cm.items():
        rows = []
        for sd, path in seeds:
            model, _ = load_vae_model(str(path))
            # the 70 features include a sampled z (torch's global RNG); pin the state per checkpoint so a row does not
            # depend on how many models were constructed before it (troubleshooting 13), and extract every feature
            # set before anything that may consume or reseed that RNG (round-7 B1: the MLP refit reseeds it)
            torch.manual_seed(0); np.random.seed(0)
            retain_z = get_retain_latent_codes(model, adata, retain)
            Ff = get_features(model, adata, forget, retain_z=retain_z)
            Fn = {kind: get_features(model, adata, neg, retain_z=retain_z) for kind, neg in negs.items()}
            row = {'seed': sd}
            for kind, neg in negs.items():
                with torch.no_grad():
                    s = torch.sigmoid(attackers[kind](torch.cat([Ff, Fn[kind]])).squeeze()).numpy()
                lab = np.r_[np.ones(len(forget)), np.zeros(len(neg))]
                persample[kind].setdefault(method, []).append({'seed': sd, 'auc': float(roc_auc_score(lab, s)), 'scores': s.tolist(), 'labels': lab.astype(int).tolist()})
                row[f'attacker_{kind}'] = float(roc_auc_score(lab, s))
                row[f'refit_std_{kind}'], oof, draws = refit_std(Ff, Fn[kind], with_null=(kind == 'within'))
                oof_store[f'{kind}/{method}/{sd}'] = oof.astype(np.float32); draw_store[kind].setdefault(method, []).append(draws)
            nf = neg_elbo_per_cell(model, tens(forget)); nc = neg_elbo_per_cell(model, tens(cross)); nw = neg_elbo_per_cell(model, tens(within))
            row.update({'forget_neg_elbo': float(nf.mean()), 'cross_neg_elbo': float(nc.mean()), 'within_neg_elbo': float(nw.mean()),
                        'audit_cross': lr_auc(ref_f, ref_c, nf, nc), 'audit_within': lr_auc(ref_f, ref_w, nf, nw),
                        'audit_cross_ci': lr_auc_ci(ref_f, ref_c, nf, nc), 'audit_within_ci': lr_auc_ci(ref_f, ref_w, nf, nw)})
            persample.setdefault('neg_elbo', {}).setdefault(method, []).append({'seed': sd, 'forget': nf.tolist(), 'cross': nc.tolist(), 'within': nw.tolist()})
            u = evaluate_utility(model, X_h, labels_h, marker_idx, gene_names)
            row.update({'elbo': u['elbo'], 'kl': u['kl'], 'marker_r': u['marker_r'], 'ari': u['ari']})
            if method in REFIT:   # last: refit() reseeds torch's global RNG
                for kind in negs:
                    row[f'refit_{kind}'] = refit(Ff, Fn[kind])
            rows.append(row)
            pn = row['refit_std_within']['perm_null']
            print(f"  {method:18s} {sd:8s} cross {row['attacker_cross']:.3f} within {row['attacker_within']:.3f}" + (f" refit {row['refit_cross']['cv_auc_mean']:.3f}/{row['refit_within']['cv_auc_mean']:.3f}" if 'refit_cross' in row else '') + f" refit-std {row['refit_std_cross']['cv_auc_mean']:.3f}/{row['refit_std_within']['cv_auc_mean']:.4f} (null p97.5 {pn['p975']:.3f}, p {pn['p_value']:.3f})" + f" audit {row['audit_cross']:.3f}/{row['audit_within']:.3f} "
                  f"forget {row['forget_neg_elbo']:.1f} elbo {row['elbo']:.1f} marker {row['marker_r']:.3f}", flush=True)
        # seed 42 is the audit reference, so its own audit row (0.5 by construction) stays out of the retrain arm's interval
        arows = [r for r in rows if not (method == 'retrain' and r['seed'] == 'seed_42')]
        results[method] = {'per_seed': rows,
                           'audit_cross_nested': nested_auc_ci([r['audit_cross_ci'] for r in arows]),
                           'audit_within_nested': nested_auc_ci([r['audit_within_ci'] for r in arows])}
        for kind in negs:   # outer bootstrap over seeds, inner over the cells every checkpoint shares
            nd = PS.nested_arm(np.array(draw_store[kind][method]), seed=1)
            results[method][f'refit_std_{kind}_nested'] = {'mean': float(np.mean([r[f'refit_std_{kind}']['cv_auc_mean'] for r in rows])),
                                                           'nested_ci_low': float(np.percentile(nd, 2.5)), 'nested_ci_high': float(np.percentile(nd, 97.5)),
                                                           'n_seeds': len(rows), 'per_seed_auc': [r[f'refit_std_{kind}']['cv_auc_mean'] for r in rows],
                                                           'estimator': 'outer bootstrap over seeds, inner stratified cell bootstrap shared across checkpoints'}
    # audit range between retrained models (every other retrain seed against 42)
    band = {}
    for kind, neg, rn in (('cross', cross, ref_c), ('within', within, ref_w)):
        band[kind] = [lr_auc(ref_f, rn, neg_elbo_per_cell(refs[s], tens(forget)), neg_elbo_per_cell(refs[s], tens(neg))) for s in retrain_seeds if s != 42]
    summary = {}
    for kind in negs:
        rt = nested(persample[kind]['retrain']); summary[kind] = {'retrain': rt, 'threshold': rt['nested_ci_high'], 'audit_range_retrained': band[kind], 'methods': {}}
        for method, recs in persample[kind].items():
            if method == 'retrain': continue
            st = nested(recs)
            if method != 'baseline': st.update(nested_diff(recs, persample[kind]['retrain']))
            summary[kind]['methods'][method] = st
        # permutation null
        s0, l0 = np.array(persample[kind]['baseline'][0]['scores']), np.array(persample[kind]['baseline'][0]['labels'])
        null = np.array([advantage(roc_auc_score(rng.permutation(l0), s0)) for _ in range(5000)])
        summary[kind]['null_floor'] = {'mean': float(null.mean()), 'ci95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]}
    json.dump({'forget_set': fs, 'split': {k: split[k] for k in ('n_forget', 'n_retain', 'n_unseen', 'n_within_type_unseen', 'forget_cluster', 'forget_type')},
               'n_cross_negatives': int(len(cross)), 'device': 'cpu', 'version': 'v6 (2026-09-22): features extracted before any refit (round-7 B1), per-cell null by label permutation (B3), cell bootstrap of the mean-over-splits statistic shared across checkpoints',
               'rng': 'torch.manual_seed(0) and np.random.seed(0) before each checkpoint\'s feature extraction; forget, cross and within features extracted before any refit; the MLP refit (which reseeds torch) runs last', 'estimator': 'likelihood_audit.neg_elbo_per_cell (16 seeded z samples)',
               'reference': f'outputs/pbmc68k/{fs}/retrain_clean_seed_42', 'retrain_seeds': retrain_seeds, 'refit': f'out-of-fold stratified 5-fold, {len(SPLIT_SEEDS)} fold splits, on {REFIT}',
               'refit_std': f'logistic regression on standardized features, out of fold, stratified 5-fold, {PS.N_SPLITS} fold splits, every checkpoint; auc_boot_ci = stratified cell bootstrap of the mean over splits ({B} draws, seed {BOOT_SEED}, shared by every checkpoint); perm_null (within only) = {N_PERM} label permutations on the checkpoint\'s own features, whole out-of-fold procedure refit per permutation, seed [{PERM_SEED}, p]',
               'retrain_audit_nested': 'seed 42 (the audit reference, 0.5 against itself) excluded', 'per_checkpoint': results, 'summary': summary},
              open(OUT / f'results_{fs}.json', 'w'), indent=1)
    persample['reference_neg_elbo'] = ref_cells
    json.dump(persample, open(OUT / f'persample_{fs}.json', 'w'))
    np.savez_compressed(OUT / f'percell_oof_{fs}.npz', **{k.replace('/', '__'): v for k, v in oof_store.items()},
                        n_forget=np.array(len(forget)), n_cross=np.array(len(cross)), n_within=np.array(len(within)))
    print(f"saved {OUT / f'results_{fs}.json'}", flush=True)


if __name__ == '__main__':
    main()
