#!/usr/bin/env python3
"""Checkpoints for the full retrain-free stopping probe on PBMC-68k (round-7 review, "the best missing experiment":
hold back 20 percent of the forget set, delete the rest, calibrate the stopping point on the held-back cells).

Per forget set (mk, pdc), from `split_<fs>.json`:
- a new split `split_<fs>_probe80.json`: a seeded 20 percent of the forget cells are HELD BACK (they stay in the
  training set and are members the controller knows), the other 80 percent are the deleted forget set;
  `retain_indices` = the original retain cells plus the held-back cells; unseen and within-type cells unchanged;
- eight clean retrains on the new retain set (`scripts/train_cohort.py --subset retain`, seeds 42 to 49), the
  retrain-based reference for the deleted cells, into outputs/pbmc68k/<fs>_probe80/retrain_clean_seed_<s>/;
- the retain-only fine-tune of the 68k baseline on the new retain set, saved at epochs 10, 25, 50, 100, 150, 200
  (the loop of `stopping_probe_68k_train.py`), three seeds, into <fs>_probe80/finetune_traj_seed<sd>/;
- Fisher scrubbing of the deleted cells at caps 1e-3 (published) and 2e-3, three seeds, with a Fisher cache of the
  new retain set, into <fs>_probe80/fisher_cap<cap>_seed<sd>/ (larger caps wreck the model on this cohort,
  `stopping_probe_<fs>.json`).
The baseline is unchanged (it trained on all forget cells). Scoring: `scripts/stopping_probe_80_20_score.py`.
Resumable. Run: PYTHONPATH=src python scripts/stopping_probe_80_20_train.py
"""
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
from vae import vae_loss
import eval_multiseed as EM
from stopping_probe_68k_train import SEEDS, EPOCHS

OUT = BASE / 'outputs/pbmc68k'
DATA = BASE / 'data/pbmc68k_processed.h5ad'
BASELINE = OUT / 'baseline/best_model.pt'
PY = sys.executable
HOLDBACK_FRACTION, SPLIT_SEED = 0.2, 0
RETRAIN_SEEDS = list(range(42, 50))
CAPS = ['0.001', '0.002']


def make_split(fs):
    path = OUT / f'split_{fs}_probe80.json'
    if path.exists():
        return path
    s = json.load(open(OUT / f'split_{fs}.json')); forget = np.array(s['forget_indices'])
    rng = np.random.default_rng(SPLIT_SEED); perm = rng.permutation(len(forget)); n_hold = int(round(HOLDBACK_FRACTION * len(forget)))
    held = np.sort(forget[perm[:n_hold]]); deleted = np.sort(forget[perm[n_hold:]])
    retain = np.sort(np.concatenate([np.array(s['retain_indices']), held]))
    new = {'forget_indices': deleted.tolist(), 'heldback_member_indices': held.tolist(), 'retain_indices': retain.tolist(),
           'unseen_indices': s['unseen_indices'], 'within_type_unseen_indices': s['within_type_unseen_indices'],
           'n_forget': int(len(deleted)), 'n_heldback': int(len(held)), 'n_retain': int(len(retain)), 'n_unseen': s['n_unseen'], 'n_within_type_unseen': s['n_within_type_unseen'],
           'forget_cluster': s['forget_cluster'], 'forget_type': s['forget_type'], 'parent_split': f'split_{fs}.json', 'holdback_fraction': HOLDBACK_FRACTION, 'holdback_seed': SPLIT_SEED,
           'description': f"{s['description']}; stopping probe 80/20: {len(held)} forget cells held back as known members (kept in the retain set), {len(deleted)} deleted"}
    json.dump(new, open(path, 'w')); print(f'{fs}: {len(deleted)} deleted, {len(held)} held back, retain {len(retain)}', flush=True)
    return path


def retrains(fs, split_path):
    for sd in RETRAIN_SEEDS:
        out = OUT / f'{fs}_probe80' / f'retrain_clean_seed_{sd}'
        if (out / 'training_complete.json').exists():
            continue
        print(f'{fs} probe80 retrain seed {sd}', flush=True)
        subprocess.run([PY, str(BASE / 'scripts/train_cohort.py'), '--data_path', str(DATA), '--split_path', str(split_path), '--subset', 'retain', '--seed', str(sd), '--output_dir', str(out)],
                       check=True, cwd=str(BASE), stdout=open(OUT / 'logs' / f'probe80_{fs}_retrain_seed{sd}.log', 'w'), stderr=subprocess.STDOUT)


def fisher(fs, split_path):
    for cap in CAPS:
        for sd in SEEDS:
            out = OUT / f'{fs}_probe80' / f'fisher_cap{cap}_seed{sd}'
            if (out / 'unlearned_model.pt').exists():
                continue
            print(f'{fs} probe80 fisher cap {cap} seed {sd}', flush=True)
            subprocess.run([PY, str(BASE / 'src/train_fisher_unlearn.py'), '--data_path', str(DATA), '--split_path', str(split_path),
                            '--baseline_checkpoint', str(BASELINE), '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', cap,
                            '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1',
                            '--fisher_cache', str(OUT / f'{fs}_probe80' / 'fisher_retain_persample.pt'), '--batch_size', '256',
                            '--output_dir', str(out), '--seed', str(sd)], check=True, cwd=str(BASE),
                           stdout=open(OUT / 'logs' / f'probe80_{fs}_fisher_cap{cap}_seed{sd}.log', 'w'), stderr=subprocess.STDOUT)


def finetune_trajectory(fs, split_path, adata, device):
    split = json.load(open(split_path)); retain = np.array(split['retain_indices'])
    x = torch.FloatTensor(adata.X[retain].toarray())
    ckpt = torch.load(BASELINE, map_location='cpu', weights_only=False)
    base, _ = EM.load_vae_model(str(BASELINE))
    for sd in SEEDS:
        d = OUT / f'{fs}_probe80' / f'finetune_traj_seed{sd}'; d.mkdir(parents=True, exist_ok=True)
        if (d / f'epoch{EPOCHS[-1]}.pt').exists():
            continue
        print(f'{fs} probe80 fine-tune trajectory seed {sd}', flush=True)
        torch.manual_seed(sd); np.random.seed(sd)
        model = deepcopy(base).to(device); opt = optim.Adam(model.parameters(), lr=1e-4)
        loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=256, shuffle=True)
        for epoch in range(1, EPOCHS[-1] + 1):
            model.train(); total, n = 0.0, 0
            for xb, lb in loader:
                xb, lb = xb.to(device), lb.to(device)
                opt.zero_grad()
                loss, _, _ = vae_loss(xb, model(xb, library_size=lb), likelihood=model.likelihood, library_size=lb, beta=1.0)
                loss.backward(); opt.step(); total += loss.item(); n += 1
            if epoch in EPOCHS:
                torch.save({'model_state_dict': deepcopy(model).to('cpu').state_dict(), 'config': ckpt['config'], 'method': 'retain_finetune_trajectory_probe80',
                            'epoch': epoch, 'lr': 1e-4, 'seed': sd, 'train_loss': total / n, 'split_path': str(split_path)}, d / f'epoch{epoch}.pt')
                print(f'  seed {sd} epoch {epoch}: train loss {total / n:.2f}', flush=True)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    EM.DEVICE = 'cpu'; (OUT / 'logs').mkdir(exist_ok=True)
    splits = {fs: make_split(fs) for fs in ('mk', 'pdc')}
    for fs in ('mk', 'pdc'):
        (OUT / f'{fs}_probe80').mkdir(exist_ok=True); retrains(fs, splits[fs])
    adata = sc.read_h5ad(DATA)
    for fs in ('mk', 'pdc'):
        finetune_trajectory(fs, splits[fs], adata, device)
    for fs in ('mk', 'pdc'):
        fisher(fs, splits[fs])
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
