#!/usr/bin/env python3
"""Train a VAE with the PBMC-33k baseline recipe (notebook 02, scripts/retrain_clean.py) on a chosen subset.

--subset training : the split's retain plus forget cells (a baseline that saw the forget set)
--subset retain   : the split's retain cells only (a clean retrained reference)

Run: PYTHONPATH=src python scripts/train_cohort.py --data_path ... --split_path ... --subset retain --seed 42 --output_dir ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.optim as optim

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
from utils import set_global_seed, DEVICE
from vae import VAE
from train import create_dataloader, train_epoch, get_kl_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True); ap.add_argument('--split_path', required=True)
    ap.add_argument('--subset', choices=['training', 'retain'], required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--output_dir', required=True); ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    adata = sc.read_h5ad(args.data_path); split = json.load(open(args.split_path))
    retain = np.array(split['retain_indices']); forget = np.array(split['forget_indices']); unseen = set(split['unseen_indices'])
    idx = np.sort(np.concatenate([retain, forget])) if args.subset == 'training' else retain
    assert not (set(idx.tolist()) & unseen)
    config = {'input_dim': adata.n_vars, 'hidden_dims': [1024, 512, 128], 'latent_dim': 32, 'likelihood': 'nb', 'dropout': 0.1,
              'use_layer_norm': True, 'lr': 1e-4, 'epochs': args.epochs, 'batch_size': 256, 'kl_warmup_epochs': 20, 'free_bits': 0.03,
              'early_stopping_patience': 10, 'selection': 'recon + kl at beta 1, patience counted after warm-up', 'seed': args.seed, 'train_set': f'split {args.subset}', 'n_train': int(len(idx)), 'split_path': str(args.split_path)}
    loader = create_dataloader(adata, idx, batch_size=config['batch_size'], shuffle=True)
    model = VAE(input_dim=config['input_dim'], hidden_dims=config['hidden_dims'], latent_dim=config['latent_dim'], likelihood='nb',
                dropout=config['dropout'], use_layer_norm=True).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    print(f"subset={args.subset} seed={args.seed} device={DEVICE} n_train={len(idx)}", flush=True)
    history = {'train_loss': [], 'train_recon': [], 'train_kl': [], 'kl_weight': []}
    best, patience, t0 = float('inf'), 0, time.time()
    for epoch in range(config['epochs']):
        beta = get_kl_weight(epoch, config['kl_warmup_epochs'])
        m = train_epoch(model, loader, opt, DEVICE, likelihood='nb', beta=beta, free_bits=config['free_bits'])
        for k in ('loss', 'recon', 'kl'):
            history[f'train_{k}'].append(m[k])
        history['kl_weight'].append(beta)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d} loss={m['loss']:.1f} kl={m['kl']:.2f} beta={beta:.2f} ({time.time() - t0:.0f}s)", flush=True)
        # Model selection on the full-weight objective (reconstruction + KL at beta 1), and no early
        # stopping during the KL warm-up: on this larger cohort the beta-weighted loss bottoms out
        # while beta is still rising, which stopped a first run at epoch 22 with beta 0.55.
        full = m['recon'] + m['kl']
        if full < best:
            best, patience = full, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'train_loss': m['loss'], 'train_loss_beta1': full, 'config': config}, out / 'best_model.pt')
        elif epoch >= config['kl_warmup_epochs']:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print(f"  early stop at epoch {epoch + 1}", flush=True); break
    json.dump(history, open(out / 'training_history.json', 'w'), indent=1)
    json.dump({'seed': args.seed, 'config': config, 'best_train_loss': best, 'wall_seconds': time.time() - t0}, open(out / 'metadata.json', 'w'), indent=1)
    # best_model.pt is written during training, so a driver must not treat its existence as completion
    json.dump({'complete': True, 'epochs_run': len(history['train_loss']), 'seed': args.seed}, open(out / 'training_complete.json', 'w'))
    print(f"done subset={args.subset} seed={args.seed} best_loss={best:.2f} wall={time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
