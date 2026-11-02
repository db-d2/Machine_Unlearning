#!/usr/bin/env python3
"""Checkpoints for the retrain-free stopping probe on PBMC-68k (round-7 review, item "the best missing experiment").

Two knobs a controller would tune, on both forget sets, three seeds each, all descending from the 68k baseline:
- the Fisher scrub's step cap, 2e-3, 3e-3, 5e-3 and 1e-2 (the published 1e-3 checkpoints already exist), with the
  settings of `scripts/run_pbmc68k.py` otherwise unchanged, into outputs/pbmc68k/<fs>/fisher_cap<cap>_seed<sd>/;
- the length of the retain-only fine-tune, with checkpoints saved at epochs 10, 25, 50, 100, 150 and 200 (Adam 1e-4,
  batch 256, the loop of `src/long_finetune.py` without its per-epoch evaluation), into
  outputs/pbmc68k/<fs>/finetune_traj_seed<sd>/epoch<E>.pt.
Scoring is `scripts/stopping_probe_68k.py`. Resumable: finished checkpoints are skipped.

Run: PYTHONPATH=src python scripts/stopping_probe_68k_train.py
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

OUT = BASE / 'outputs/pbmc68k'
DATA = BASE / 'data/pbmc68k_processed.h5ad'
BASELINE = OUT / 'baseline/best_model.pt'
PY = sys.executable
SEEDS = [42, 123, 456]
CAPS = ['0.002', '0.003', '0.005', '0.01']
EPOCHS = [10, 25, 50, 100, 150, 200]


def fisher_sweep(fs):
    for cap in CAPS:
        for sd in SEEDS:
            out = OUT / fs / f'fisher_cap{cap}_seed{sd}'
            if (out / 'unlearned_model.pt').exists():
                continue
            print(f'{fs} fisher cap {cap} seed {sd}', flush=True)
            subprocess.run([PY, str(BASE / 'src/train_fisher_unlearn.py'), '--data_path', str(DATA), '--split_path', str(OUT / f'split_{fs}.json'),
                            '--baseline_checkpoint', str(BASELINE), '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', cap,
                            '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1',
                            '--fisher_cache', str(OUT / fs / 'fisher_retain_persample.pt'), '--batch_size', '256',
                            '--output_dir', str(out), '--seed', str(sd)], check=True, cwd=str(BASE),
                           stdout=open(OUT / 'logs' / f'probe_{fs}_fisher_cap{cap}_seed{sd}.log', 'w'), stderr=subprocess.STDOUT)


def finetune_trajectory(fs, adata, device):
    split = json.load(open(OUT / f'split_{fs}.json')); retain = np.array(split['retain_indices'])
    x = torch.FloatTensor(adata.X[retain].toarray())
    ckpt = torch.load(BASELINE, map_location='cpu', weights_only=False)
    base, _ = EM.load_vae_model(str(BASELINE))
    for sd in SEEDS:
        d = OUT / fs / f'finetune_traj_seed{sd}'; d.mkdir(exist_ok=True)
        if (d / f'epoch{EPOCHS[-1]}.pt').exists():
            continue
        print(f'{fs} fine-tune trajectory seed {sd}', flush=True)
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
                torch.save({'model_state_dict': deepcopy(model).to('cpu').state_dict(), 'config': ckpt['config'], 'method': 'retain_finetune_trajectory',
                            'epoch': epoch, 'lr': 1e-4, 'seed': sd, 'train_loss': total / n}, d / f'epoch{epoch}.pt')
                print(f'  seed {sd} epoch {epoch}: train loss {total / n:.2f}', flush=True)


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    EM.DEVICE = 'cpu'
    for fs in ('mk', 'pdc'):
        fisher_sweep(fs)
    adata = sc.read_h5ad(DATA)
    for fs in ('mk', 'pdc'):
        finetune_trajectory(fs, adata, device)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
