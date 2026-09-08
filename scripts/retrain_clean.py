#!/usr/bin/env python3
"""Clean retrain reference: train on exactly the retain split, with notebook 2's recipe.

Why this exists. src/retrain.py draws its own 85/15 split from every non-forget cell,
so its training set includes ~85% of the baseline's unseen split and therefore 159 to
170 of the 194 matched negatives (verified 2026-09-08 by reconstructing the shuffle for
seeds 42 to 46). A retrain reference that trained on its own non-member negatives cannot
serve as the membership floor. This script trains on split['retain_indices'] only, which
is the baseline's training set minus the forget set, and mirrors notebook 2 exactly:
100 epochs, KL warm-up over 20 epochs, free bits 0.03, Adam 1e-4, batch 256, LayerNorm,
dropout 0.1, checkpoint selected by training loss with patience 10.

Usage:
    PYTHONPATH=src python scripts/retrain_clean.py --seed 42 --output_dir outputs/p1/retrain_clean/seed_42
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
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_path', default=str(BASE / 'data' / 'adata_processed.h5ad'))
    ap.add_argument('--split_path', default=str(BASE / 'outputs' / 'p1' / 'split_structured.json'))
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    adata = sc.read_h5ad(args.data_path)
    split = json.load(open(args.split_path))
    retain = np.array(split['retain_indices'])
    forget = set(split['forget_indices']); unseen = set(split['unseen_indices'])
    assert not (set(retain.tolist()) & forget) and not (set(retain.tolist()) & unseen)

    config = {'input_dim': adata.n_vars, 'hidden_dims': [1024, 512, 128], 'latent_dim': 32,
              'likelihood': 'nb', 'dropout': 0.1, 'use_layer_norm': True, 'lr': 1e-4,
              'epochs': args.epochs, 'batch_size': 256, 'kl_warmup_epochs': 20, 'free_bits': 0.03,
              'early_stopping_patience': 10, 'seed': args.seed,
              'train_set': 'split retain_indices only (no unseen, no forget)', 'n_train': int(len(retain))}
    loader = create_dataloader(adata, retain, batch_size=config['batch_size'], shuffle=True)
    model = VAE(input_dim=config['input_dim'], hidden_dims=config['hidden_dims'],
                latent_dim=config['latent_dim'], likelihood=config['likelihood'],
                dropout=config['dropout'], use_layer_norm=config['use_layer_norm']).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=config['lr'])
    print(f"seed={args.seed} device={DEVICE} n_train={len(retain)} (forget/unseen excluded)", flush=True)

    history = {'train_loss': [], 'train_recon': [], 'train_kl': [], 'kl_weight': []}
    best, patience = float('inf'), 0
    t0 = time.time()
    for epoch in range(config['epochs']):
        beta = get_kl_weight(epoch, config['kl_warmup_epochs'])
        m = train_epoch(model, loader, opt, DEVICE, likelihood=config['likelihood'],
                        beta=beta, free_bits=config['free_bits'])
        for k in ('loss', 'recon', 'kl'):
            history[f'train_{k}'].append(m[k])
        history['kl_weight'].append(beta)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d} loss={m['loss']:.1f} kl={m['kl']:.2f} beta={beta:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if m['loss'] < best:
            best, patience = m['loss'], 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'train_loss': m['loss'], 'config': config}, out / 'best_model.pt')
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print(f"  early stop at epoch {epoch+1}", flush=True)
                break
    json.dump(history, open(out / 'training_history.json', 'w'), indent=1)
    json.dump({'seed': args.seed, 'config': config, 'best_train_loss': best,
               'wall_seconds': time.time() - t0}, open(out / 'metadata.json', 'w'), indent=1)
    print(f"done seed={args.seed} best_loss={best:.2f} wall={time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
