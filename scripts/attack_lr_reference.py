#!/usr/bin/env python3
"""Likelihood-ratio membership attack with a clean retrain as the reference model.

Score(x) = negELBO_reference(x) - negELBO_target(x): a member of the target's training
set has a lower negative ELBO under the target than under a model that never saw it.
The reference is the clean retain-split retrain seed 42 for every target; the retrain
row itself uses retrain seed 43 as the target, so a retrain-versus-retrain control shows
what the attack reports when there is no membership difference at all. Forget cells vs
the 194 matched negatives, CPU, library size passed to the decoder. The negative ELBO is the
deterministic estimator of src/likelihood_audit.py (eval mode, 16 seeded z samples).

Writes outputs/attack_lr_reference.json. Run: PYTHONPATH=src python scripts/attack_lr_reference.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT
from save_persample_scores import checkpoint_map
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci, nested_auc_ci, N_SAMPLES, SEED

REF = BASE / 'outputs/p1/retrain_clean/seed_42/best_model.pt'
OUT = BASE / 'outputs/attack_lr_reference.json'


def neg_elbo(model, x):
    """Deterministic estimator shared by every likelihood instrument (src/likelihood_audit.py)."""
    return neg_elbo_per_cell(model.to('cpu'), x)


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH))
    matched = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    def X(idx):
        x = adata.X[idx]; x = x.toarray() if hasattr(x, 'toarray') else x
        return torch.FloatTensor(np.asarray(x))
    xf, xm = X(split['forget_indices']), X(matched)
    labels = np.r_[np.ones(len(xf)), np.zeros(len(xm))]
    ref, _ = load_vae_model(str(REF))
    ref_f, ref_m = neg_elbo(ref, xf), neg_elbo(ref, xm)

    targets = {'baseline': [('seed42', BASELINE_CHECKPOINT)]}
    cm = checkpoint_map()
    targets['retrain_vs_retrain'] = [(sd, p) for sd, p in cm['retrain'] if sd != 'seed_42']
    for m in ['retain_finetune', 'gradient_ascent', 'ssd', 'scrub', 'moon', 'contrastive', 'fisher',
              'fisher_step_ablation', 'extragradient', 'extragradient_lambda1', 'dp_sgd', 'retrain_leaky']:
        if m in cm:
            targets[m] = cm[m]
    for cfg_dir in sorted((BASE / 'outputs/p2/frozen_critics').glob('*_seed*')):
        if (cfg_dir / 'best_model.pt').exists():
            key = 'frozen_' + cfg_dir.name.rsplit('_seed', 1)[0].replace('frozen_', '')
            targets.setdefault(key, []).append((cfg_dir.name.rsplit('_', 1)[-1], cfg_dir / 'best_model.pt'))
    for p in sorted((BASE / 'outputs/rau').glob('rau_lambda*_model.pt')):
        targets.setdefault('rau_' + p.stem.split('_')[1], []).append(('seed42', p))
    for p in sorted((BASE / 'outputs/augmentation').glob('pbmc_aug*_model.pt')):
        targets.setdefault('augmentation_' + p.stem.split('_')[1], []).append(('seed42', p))
    out = {'reference': str(REF.relative_to(BASE)), 'score': 'negELBO_ref - negELBO_target (member => positive)',
           'estimator': f'likelihood_audit.neg_elbo_per_cell: eval mode, {N_SAMPLES} seeded z samples (seed {SEED}), CPU',
           'n_forget': len(xf), 'n_matched': len(xm), 'ci': 'per seed: percentile bootstrap over cells (10,000, groups resampled separately); per model: nested over seeds and cells; null95: permutation null of the AUC', 'models': {}}
    for m, seeds in targets.items():
        aucs, per = [], []
        for sd, p in seeds:
            if m == 'retrain_leaky':
                import nested_bootstrap_retrain as NBR; NBR.DEVICE = 'cpu'
                model = NBR.load_retrain_model(str(p))
            else:
                model, _ = load_vae_model(str(p))
            tf, tm = neg_elbo(model, xf), neg_elbo(model, xm)
            sf, sm = ref_f - tf, ref_m - tm
            aucs.append(float(roc_auc_score(labels, np.r_[sf, sm])))
            per.append(lr_auc_ci(ref_f, ref_m, tf, tm))
        nest = nested_auc_ci(per)
        out['models'][m] = {'auc_mean': float(np.mean(aucs)), 'advantage_mean': float(np.mean([2 * abs(a - 0.5) for a in aucs])),
                            'per_seed_auc': dict(zip([str(s) for s, _ in seeds], aucs)), 'n_seeds': len(aucs),
                            'per_seed_ci95': dict(zip([str(s) for s, _ in seeds], [r['ci95'] for r in per])),
                            'nested_ci95': [nest['nested_ci_low'], nest['nested_ci_high']],
                            'null95': per[0]['null95']}
        print(f"  {m:22s} LR-vs-clean-retrain AUC={np.mean(aucs):.3f} [{nest['nested_ci_low']:.3f}, {nest['nested_ci_high']:.3f}] adv={np.mean([2*abs(a-0.5) for a in aucs]):.3f} (n={len(aucs)})", flush=True)
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
