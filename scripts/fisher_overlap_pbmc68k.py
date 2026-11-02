#!/usr/bin/env python3
"""Forget-retain Fisher overlap on the PBMC-68k baseline, per parameter category, with the notebook 26
recipe (per-sample diagonal empirical Fisher of the negative ELBO, eval mode, z = mu, damping 0), for each
of the two forget sets against the split's retain set. Writes outputs/pbmc68k/fisher_overlap_<fs>.json.
Run: PYTHONPATH=src python scripts/fisher_overlap_pbmc68k.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model
from fisher_utils import AnnDataDataset, compute_vae_fisher, cosine_sim, vae_layer_category

OUT = BASE / 'outputs/pbmc68k'


def main():
    adata = sc.read_h5ad(BASE / 'data/pbmc68k_processed.h5ad')
    model, _ = load_vae_model(str(OUT / 'baseline/best_model.pt'))
    for fs in ('mk', 'pdc'):
        split = json.load(open(OUT / f'split_{fs}.json'))
        f_idx, r_idx = split['forget_indices'], split['retain_indices']
        Ff, nf = compute_vae_fisher(model, DataLoader(AnnDataDataset(adata, f_idx), batch_size=128, shuffle=False), 'cpu', damping=0.0)
        print(f'[{fs}] forget Fisher over {nf} cells', flush=True)
        Fr, nr = compute_vae_fisher(model, DataLoader(AnnDataDataset(adata, r_idx), batch_size=256, shuffle=False), 'cpu', damping=0.0)
        print(f'[{fs}] retain Fisher over {nr} cells', flush=True)
        cats = {}
        for name in Ff:
            if 'fc_dispersion' in name:
                continue   # vestigial head under the MSE loss
            c = vae_layer_category(name); cats.setdefault(c, {'f': [], 'r': []}); cats[c]['f'].append(Ff[name].flatten()); cats[c]['r'].append(Fr[name].flatten())
        res = {'n_forget': nf, 'n_retain': nr, 'damping': 0.0, 'per_category': {}, 'global_cosine': None}
        for c, v in cats.items():
            ff, fr = torch.cat(v['f']), torch.cat(v['r']); res['per_category'][c] = {'cosine': float(cosine_sim(ff, fr)), 'n_params': int(len(ff))}
        allf = torch.cat([torch.cat(v['f']) for v in cats.values()]); allr = torch.cat([torch.cat(v['r']) for v in cats.values()])
        res['global_cosine'] = float(cosine_sim(allf, allr))
        json.dump(res, open(OUT / f'fisher_overlap_{fs}.json', 'w'), indent=1)
        print(f'[{fs}] ' + ', '.join(f"{c} {v['cosine']:.3f}" for c, v in res['per_category'].items()) + f", global {res['global_cosine']:.3f}", flush=True)


if __name__ == '__main__':
    main()
