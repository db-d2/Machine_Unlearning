#!/usr/bin/env python3
"""Frozen-critic adversarial unlearning (notebook 4, methods 1 and 2), re-derived.

Pre-trained critics stay fixed; the VAE minimizes the critics' membership confidence on
the forget set plus lambda times the retain ELBO (per gene). Configurations as in
notebook 4: single critic (v1) and three critics (v1, v2, v3) at lambda 5 and 10, 30
epochs of one pass over the retain set each. Revision 2026-09-08: July-4 baseline and
critics, utility term with the training objective's reconstruction/KL ratio, three seeds.

Usage: PYTHONPATH=src python scripts/run_frozen_critics.py [--seeds 42 123 456]
Output: outputs/p2/frozen_critics/<config>_seed<s>/best_model.pt
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
from vae import VAE
from train_unlearn import create_dataloader, load_attackers, train_vae_multi_step

DATA = BASE / 'data' / 'adata_processed.h5ad'
SPLIT = BASE / 'outputs' / 'p1' / 'split_structured.json'
BASELINE = BASE / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
CRITICS = [BASE / 'outputs/p2/attackers' / f'attacker_{v}_seed{s}.pt' for v, s in [('v1', 42), ('v2', 43), ('v3', 44)]]
OUT = BASE / 'outputs' / 'p2' / 'frozen_critics'
CONFIGS = {'frozen_single_lambda5': (1, 5.0), 'frozen_single_lambda10': (1, 10.0),
           'frozen_multi_lambda5': (3, 5.0), 'frozen_multi_lambda10': (3, 10.0)}


class Args:
    def __init__(self, lam):
        self.lambda_retain = lam
        self.batch_size = 256
        self.steps_per_epoch = None
        self.privacy_repeats_k = 4
        self.balance_mode = 'none'
        self.use_max_privacy = False
        self.mmd_gamma = 0.0
        self.alpha_f_kl = 0.0
        self.grad_clip = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    adata = sc.read_h5ad(DATA)
    split = json.load(open(SPLIT))
    forget_idx, retain_idx = np.array(split['forget_indices']), np.array(split['retain_indices'])
    ckpt = torch.load(BASELINE, map_location=device, weights_only=False)
    config = ckpt['config']
    critics = load_attackers([str(p) for p in CRITICS], device)

    for name, (n_critics, lam) in CONFIGS.items():
        for seed in args.seeds:
            out = OUT / f'{name}_seed{seed}'
            if (out / 'best_model.pt').exists():
                print(f"skip {out.name}"); continue
            torch.manual_seed(seed); np.random.seed(seed)
            vae = VAE(input_dim=config['input_dim'], latent_dim=config['latent_dim'], hidden_dims=config['hidden_dims'],
                      likelihood=config['likelihood'], dropout=config.get('dropout', 0.1),
                      use_layer_norm=config.get('use_layer_norm', True)).to(device)
            vae.load_state_dict(ckpt['model_state_dict'])
            opt = optim.Adam(vae.parameters(), lr=1e-4)
            forget_loader = create_dataloader(adata, forget_idx, batch_size=min(len(forget_idx), 32), shuffle=True)
            retain_loader = create_dataloader(adata, retain_idx, batch_size=256, shuffle=True)
            attacker_list = critics[:n_critics]
            a = Args(lam)
            history = {'j_privacy': [], 'j_utility': []}
            ema_p, ema_u = None, None
            t0 = time.time()
            for epoch in range(args.epochs):
                avg_p, avg_u, ema_p, ema_u, _, _ = train_vae_multi_step(
                    vae, attacker_list, retain_loader, forget_loader, None, opt, device, a, ema_p, ema_u, None, epoch)
                history['j_privacy'].append(avg_p); history['j_utility'].append(avg_u)
                if (epoch + 1) % 10 == 0:
                    print(f"  {name} seed{seed} epoch {epoch+1}/{args.epochs} privacy={avg_p:.4f} utility/gene={avg_u:.4f} ({time.time()-t0:.0f}s)", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': vae.state_dict(), 'config': config, 'history': history,
                        'method': name, 'lambda_retain': lam, 'n_critics': n_critics, 'seed': seed}, out / 'best_model.pt')
            json.dump({'method': name, 'lambda_retain': lam, 'n_critics': n_critics, 'epochs': args.epochs, 'seed': seed,
                       'baseline_checkpoint': str(BASELINE), 'critics': [str(p) for p in CRITICS[:n_critics]],
                       'revision': '2026-09-08 July baseline/critics, per-gene training objective as utility',
                       'wall_seconds': time.time() - t0}, open(out / 'metadata.json', 'w'), indent=2)
            print(f"saved {out}", flush=True)


if __name__ == '__main__':
    main()
