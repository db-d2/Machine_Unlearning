#!/usr/bin/env python3
"""Held-out utility with intervals and a paired comparison against the baseline (round-7 review: "utility claims carry
no intervals at all and no equivalence margin, while the title asks about utility").

Per checkpoint, on the held-out cells of the cohort (PBMC-33k: the 4,964 unseen cells; PBMC-68k: the 10,447 unseen
cells, as in `score_utility_all.py` and `score_pbmc68k.py`):
- negative ELBO per cell with the deterministic estimator (`likelihood_audit.neg_elbo_per_cell`, eval mode, 16 seeded
  z draws). The earlier utility numbers came from `utility_metrics.compute_held_out_elbo`, the VAE's unseeded forward
  pass, which troubleshooting rule 14 excludes from tables; the two agree to about 0.05 per cell on the baseline;
- the marker-gene reconstruction at z = mu (the existing, deterministic `compute_marker_correlation` recipe), kept per
  cell so the mean Pearson r over the eight markers can be resampled;
- ARI of k-means on the latent means against the Leiden labels (deterministic, no interval).

Intervals: one multinomial draw of the held-out cells shared by every checkpoint of a cohort (B = 2,000), seeds resampled
within an arm. Costs are paired against the baseline: held-out negative ELBO minus the baseline's (positive is worse) and
the baseline's marker r minus the method's (positive is worse). Verdict at a margin of 1 percent of the baseline's
held-out negative ELBO and 0.01 in marker r: NO COST when the upper end of the cost's 95 percent interval lies below the
margin, COSTS UTILITY when its lower end lies above the margin, NOT RESOLVED otherwise. The margins are conventions set
after the point values were known; every row carries the interval, so any other margin can be read off.

The pooled `fisher_step_ablation` checkpoint map (steps 3e-3 and 1e-2, one checkpoint each) is split into the arms
`fisher_step3e-3` and `fisher_step1e-2`, as `nested_ci_all.json` splits it. On PBMC-68k the checkpoint map also picks up
the stopping-probe grid (`scripts/stopping_probe_68k_train.py`): `fisher_cap<cap>` arms and `finetune_epoch<E>` arms.

Writes outputs/utility_intervals.json (+ utility_percell_<cohort>.npz). Run: PYTHONPATH=src python scripts/utility_intervals.py [--cohorts pbmc33k pbmc68k_mk pbmc68k_pdc]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, MARKER_GENES, DATA_PATH, SPLIT_PATH
from likelihood_audit import neg_elbo_per_cell
from utility_metrics import compute_latent_metrics
import nested_bootstrap_retrain as NBR
NBR.DEVICE = 'cpu'
from nested_bootstrap_retrain import load_retrain_model
import percell_stats as PS

OUT = BASE / 'outputs/utility_intervals.json'
B, BOOT_SEED, NEST_SEED = 2000, 0, 5
MARGIN_ELBO_FRAC, MARGIN_MARKER = 0.01, 0.01


def cohort_checkpoints(cohort):
    if cohort == 'pbmc33k':
        import save_persample_scores as S
        cm = S.checkpoint_map(); cm.update(S.extended_map())
        return DATA_PATH, json.load(open(SPLIT_PATH))['unseen_indices'], cm
    fs = cohort.split('_')[1]
    import score_pbmc68k as SP
    cm = SP.checkpoints(fs)
    for pat, key, fname in (('fisher_cap*_seed*', 'fisher_cap', 'unlearned_model.pt'),):
        for d in sorted((SP.OUT / fs).glob(pat)):
            if (d / fname).exists():
                cap, sd = d.name[len('fisher_cap'):].split('_seed')
                cm.setdefault(f'fisher_cap{cap}', []).append((sd, d / fname))
    for d in sorted((SP.OUT / fs).glob('finetune_traj_seed*')):
        for f in sorted(d.glob('epoch*.pt')):
            cm.setdefault(f'finetune_{f.stem}', []).append((d.name.split('seed')[-1], f))
    return SP.DATA, json.load(open(SP.OUT / f'split_{fs}.json'))['unseen_indices'], cm


def arm_seed(name):
    """Seed of an arm's nested (over-seeds) resampling, derived from the arm's name so that adding or splitting
    an arm leaves every other arm's draws unchanged."""
    return NEST_SEED * 1000003 + zlib.crc32(name.encode())


def marker_recon(model, x, marker_idx):
    with torch.no_grad():
        mu, _ = model.encode(x)
        out = model.decode(mu)
        rec = out[0] if isinstance(out, tuple) else out
    return rec[:, marker_idx].numpy()


def weighted_r(w, a, b):
    """Pearson r of columns of a and b under per-cell count weights w (B x n); returns (B, genes)."""
    s = w.sum(1, keepdims=True)
    ma, mb = (w @ a) / s, (w @ b) / s
    cov = (w @ (a * b)) / s - ma * mb
    va, vb = (w @ (a * a)) / s - ma ** 2, (w @ (b * b)) / s - mb ** 2
    return cov / np.sqrt(va * vb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk', 'pbmc68k_pdc']); args = ap.parse_args()
    res = json.load(open(OUT)) if OUT.exists() else {'margins': {'elbo_fraction_of_baseline': MARGIN_ELBO_FRAC, 'marker_r': MARGIN_MARKER}, 'cohorts': {}}
    for cohort in args.cohorts:
        data, held, cm = cohort_checkpoints(cohort)
        adata = sc.read_h5ad(data); held = np.array(held)
        xh = adata.X[held]; x = torch.FloatTensor(xh.toarray() if hasattr(xh, 'toarray') else np.asarray(xh)); labels = adata.obs['leiden'].values[held]
        genes = list(adata.var_names); midx = [genes.index(g) for g in MARKER_GENES if g in genes]; obs = x[:, midx].numpy()
        per, ari, keys = {}, {}, []
        for method, seeds in cm.items():
            for sd, path in seeds:
                model = load_retrain_model(str(path)) if method == 'retrain_leaky' else load_vae_model(str(path))[0]
                k = f'{method}__{sd}'; keys.append(k)
                per[k] = (neg_elbo_per_cell(model, x).astype(np.float32), marker_recon(model, x, midx).astype(np.float32))
                ari[k] = compute_latent_metrics(model, x, labels, 'cpu')['ari']
            print(f'{cohort} {method}: {len(seeds)} checkpoints', flush=True)
        np.savez_compressed(BASE / f'outputs/utility_percell_{cohort}.npz', observed_markers=obs.astype(np.float32),
                            **{f'{k}__negelbo': v[0] for k, v in per.items()}, **{f'{k}__markers': v[1] for k, v in per.items()})
        n = len(held); w = np.random.default_rng(BOOT_SEED).multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
        draws, point = {}, {}
        for k, (ne, rec) in per.items():
            draws[k] = (w @ ne.astype(np.float64)) / n, weighted_r(w, obs.astype(np.float64), rec.astype(np.float64)).mean(1)
            point[k] = (float(ne.mean()), float(np.mean([np.corrcoef(obs[:, g], rec[:, g])[0, 1] for g in range(obs.shape[1])])))
        arms = {}
        for k in keys:
            m, sd = k.split('__')
            if m == 'fisher_step_ablation':   # two step caps, one checkpoint each; split as nested_ci_all.json does
                m = 'fisher_' + sd.split('_seed')[0]      # fisher_step3e-3, fisher_step1e-2
            arms.setdefault(m, []).append(k)
        base = arms['baseline'][0]; margin_e = MARGIN_ELBO_FRAC * point[base][0]
        out = {'n_heldout': int(n), 'baseline_elbo': point[base][0], 'margin_elbo': margin_e, 'margin_marker': MARGIN_MARKER, 'arms': {}}
        for m, ks in arms.items():
            E = np.array([draws[k][0] for k in ks]); R = np.array([draws[k][1] for k in ks])
            e_arm, r_arm = PS.nested_arm(E, seed=arm_seed(m)), PS.nested_arm(R, seed=arm_seed(m))
            row = {'seeds': [k.split('__')[1] for k in ks], 'elbo_per_seed': [point[k][0] for k in ks], 'marker_r_per_seed': [point[k][1] for k in ks],
                   'ari_per_seed': [ari[k] for k in ks], 'elbo': float(np.mean([point[k][0] for k in ks])), 'marker_r': float(np.mean([point[k][1] for k in ks])),
                   'ari': float(np.mean([ari[k] for k in ks])),
                   'elbo_ci': [float(np.percentile(e_arm, 2.5)), float(np.percentile(e_arm, 97.5))], 'marker_r_ci': [float(np.percentile(r_arm, 2.5)), float(np.percentile(r_arm, 97.5))]}
            if m != 'baseline':
                ce = e_arm - draws[base][0]; cr = draws[base][1] - r_arm
                for name, c, mg in (('elbo', ce, margin_e), ('marker', cr, MARGIN_MARKER)):
                    lo, hi = float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))
                    row[f'{name}_cost'] = float((row['elbo'] - point[base][0]) if name == 'elbo' else (point[base][1] - row['marker_r']))
                    row[f'{name}_cost_ci'] = [lo, hi]
                    row[f'{name}_verdict'] = 'no cost' if hi < mg else ('costs utility' if lo > mg else 'not resolved')
            out['arms'][m] = row
            if m != 'baseline':
                print(f"  {m:32s} ELBO {row['elbo']:.1f} cost {row['elbo_cost']:+.2f} [{row['elbo_cost_ci'][0]:+.2f}, {row['elbo_cost_ci'][1]:+.2f}] {row['elbo_verdict']:13s} "
                      f"marker {row['marker_r']:.3f} cost {row['marker_cost']:+.4f} [{row['marker_cost_ci'][0]:+.4f}, {row['marker_cost_ci'][1]:+.4f}] {row['marker_verdict']}", flush=True)
        res['cohorts'][cohort] = out
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'saved {cohort} to {OUT}', flush=True)


if __name__ == '__main__':
    main()
