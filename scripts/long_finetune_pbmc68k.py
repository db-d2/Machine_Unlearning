#!/usr/bin/env python3
"""Long retain-only fine-tuning on PBMC-68k cluster 9 (megakaryocytes), the notebook 42 experiment
on the second cohort: 200 epochs of the methods' own retain fine-tune, three seeds, the forget cells,
their held-out siblings, the cross-type negatives, a retain sample and a held-out sample measured
every five epochs with the deterministic negative ELBO, the cross-type attacker and the reference audit.

Writes outputs/pbmc68k/long_finetune/seed<sd>/final_model.pt and outputs/pbmc68k/long_finetune.json.
Run: PYTHONPATH=src python scripts/long_finetune_pbmc68k.py [--seeds 42 123 456] [--epochs 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model, train_fresh_attacker
from train_targeted_unlearn import Evaluator
from likelihood_audit import neg_elbo_per_cell, lr_auc
from long_finetune import long_retain_finetune

OUT = BASE / 'outputs/pbmc68k'
FS = 'mk'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456]); ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=5); ap.add_argument('--lr', type=float, default=1e-4); args = ap.parse_args()
    train_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    adata = sc.read_h5ad(BASE / 'data/pbmc68k_processed.h5ad'); split = json.load(open(OUT / f'split_{FS}.json'))
    forget, retain, unseen, within = (np.array(split[k]) for k in ('forget_indices', 'retain_indices', 'unseen_indices', 'within_type_unseen_indices'))
    cross = np.array(json.load(open(OUT / f'matched_negatives_{FS}.json'))['matched_indices'])
    X = adata.X
    def tens(idx):
        return torch.FloatTensor(X[idx].toarray())
    X_forget, X_cross, X_within, X_retain = tens(forget), tens(cross), tens(within), tens(retain)
    rs = np.random.RandomState(42)
    X_retain_sample = tens(rs.choice(retain, 5000, replace=False)); X_unseen = tens(rs.choice(np.setdiff1d(unseen, within), 2000, replace=False))
    baseline, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    reference, _ = load_vae_model(str(OUT / FS / 'retrain_clean_seed_42/best_model.pt'))
    attacker = train_fresh_attacker(baseline, adata, forget, cross, retain); attacker.eval()
    evaluator = Evaluator(attacker, reference, X_forget, X_cross, X_retain_sample, 'cpu', extra_sets={'within': X_within, 'unseen': X_unseen})
    neighbors_all = torch.arange(len(X_cross)).repeat(len(X_forget), 1)
    # the retrained models' levels for the same sets
    levels = {}
    for sd in sorted(int(q.name.split('_')[-1]) for q in (OUT / FS).glob('retrain_clean_seed_*') if (q / 'training_complete.json').exists()):
        m, _ = load_vae_model(str(OUT / FS / f'retrain_clean_seed_{sd}/best_model.pt'))
        levels[f'retrain_clean_seed_{sd}'] = {k: float(neg_elbo_per_cell(m, x).mean()) for k, x in (('forget', X_forget), ('within', X_within), ('cross', X_cross), ('retain_sample', X_retain_sample), ('unseen', X_unseen))}
    levels['baseline'] = {k: float(neg_elbo_per_cell(baseline, x).mean()) for k, x in (('forget', X_forget), ('within', X_within), ('cross', X_cross), ('retain_sample', X_retain_sample), ('unseen', X_unseen))}
    print('levels', json.dumps(levels, indent=1), flush=True)
    out_dir = OUT / 'long_finetune'; out_dir.mkdir(exist_ok=True)
    cfg = torch.load(OUT / 'baseline/best_model.pt', map_location='cpu', weights_only=False)['config']
    hist = {}
    for sd in args.seeds:
        d = out_dir / f'seed{sd}'; d.mkdir(exist_ok=True)
        if (d / 'final_model.pt').exists():
            hist[sd] = torch.load(d / 'final_model.pt', map_location='cpu', weights_only=False)['history']; print(f'seed {sd}: loaded', flush=True); continue
        print(f'=== seed {sd} ===', flush=True)
        model, h = long_retain_finetune(baseline, X_retain, evaluator, neighbors_all, train_device, epochs=args.epochs, lr=args.lr, batch_size=256,
                                        eval_every=args.eval_every, seed=sd, log=lambda s: print(s, flush=True))
        torch.save({'model_state_dict': model.to('cpu').state_dict(), 'config': cfg, 'history': h, 'method': 'long_retain_finetune', 'epochs': args.epochs, 'lr': args.lr, 'seed': sd}, d / 'final_model.pt')
        hist[sd] = h
    json.dump({'forget_set': FS, 'epochs': args.epochs, 'lr': args.lr, 'eval_every': args.eval_every, 'attacker': 'cross-type (canonical recipe, trained on the baseline)',
               'reference': f'outputs/pbmc68k/{FS}/retrain_clean_seed_42', 'levels': levels, 'history': {str(k): v for k, v in hist.items()},
               'estimator': 'likelihood_audit.neg_elbo_per_cell (16 seeded z samples), CPU'}, open(OUT / 'long_finetune.json', 'w'), indent=1)
    print(f'saved {OUT / "long_finetune.json"}', flush=True)


if __name__ == '__main__':
    main()
