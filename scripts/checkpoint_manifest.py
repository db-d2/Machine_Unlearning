#!/usr/bin/env python3
"""Checkpoint manifest: hash, size, time, parent baseline, and training set of every
checkpoint the paper's numbers rest on. Written to outputs/checkpoint_manifest.json.

The parent of a fine-tuned checkpoint is identified by the mean absolute weight
distance to each candidate baseline (the nearer one, reported with both distances), and
by whether the encoder weights are bit-identical to a candidate (decoder-only methods).
Retrain and DP-SGD checkpoints are trained from scratch and have no parent.

Run: PYTHONPATH=src python scripts/checkpoint_manifest.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
from save_persample_scores import checkpoint_map

OUT = BASE / 'outputs' / 'checkpoint_manifest.json'
CANDIDATES = {
    'baseline_jul4': BASE / 'outputs/p1/baseline/best_model.pt',
    'baseline_dec7_snapshot': BASE.parent / 'final_proj_staging_ckpt_snapshot_20260704/p1/baseline/best_model.pt',
    'baseline_pbmc68k': BASE / 'outputs/pbmc68k/baseline/best_model.pt',
}
FROM_SCRATCH = {'retrain', 'retrain_leaky', 'dp_sgd', 'baseline'}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def state(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck


def dist(a, b):
    keys = [k for k in a if k in b and a[k].dtype.is_floating_point and a[k].shape == b[k].shape]
    return float(torch.cat([(a[k] - b[k]).flatten().abs() for k in keys]).mean()) if keys else None


def enc_equal(a, b):
    keys = [k for k in a if k.startswith('encoder') and k in b]
    return bool(keys) and all(torch.equal(a[k], b[k]) for k in keys)


def pbmc68k_map():
    """PBMC-68k cohort (notebook 44): baseline, clean retrains per forget set, methods, long fine-tune."""
    root = BASE / 'outputs/pbmc68k'
    M = {}
    if (root / 'baseline/best_model.pt').exists():
        M['pbmc68k/baseline'] = [('seed42', root / 'baseline/best_model.pt')]
    for fs in ('mk', 'pdc'):
        if not (root / fs).exists():
            continue
        for d in sorted((root / fs).glob('retrain_clean_seed_*')):
            if (d / 'best_model.pt').exists() and (d / 'training_complete.json').exists():
                M.setdefault(f'pbmc68k/{fs}/retrain', []).append((d.name.replace('retrain_clean_', ''), d / 'best_model.pt'))
        for m in ('retain_finetune', 'gradient_ascent', 'fisher', 'ssd', 'scrub', 'contrastive', 'extragradient'):
            fn = 'unlearned_model.pt' if m == 'fisher' else 'best_model.pt'
            for d in sorted((root / fs).glob(f'{m}_seed*')):
                if (d / fn).exists():
                    M.setdefault(f'pbmc68k/{fs}/{m}', []).append((d.name.split('_seed')[-1], d / fn))
    if (root / 'long_finetune').exists():
        for d in sorted((root / 'long_finetune').glob('seed*')):
            if (d / 'final_model.pt').exists():
                M.setdefault('pbmc68k/long_finetune', []).append((d.name, d / 'final_model.pt'))
    return M


def main():
    cands = {k: state(p) for k, p in CANDIDATES.items() if p.exists()}
    manifest = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'candidates': {k: sha256(p) for k, p in CANDIDATES.items() if p.exists()},
                'checkpoints': {}}
    for method, seeds in {**checkpoint_map(), **pbmc68k_map()}.items():
        for sd, path in seeds:
            rec = {'method': method, 'seed': sd, 'path': str(path.relative_to(BASE)), 'sha256': sha256(path),
                   'bytes': os.path.getsize(path), 'mtime': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(path)))}
            if method.split('/')[-1] not in FROM_SCRATCH:
                sd_state = state(path)
                rec['distance_to'] = {k: dist(sd_state, c) for k, c in cands.items()}
                rec['encoder_identical_to'] = [k for k, c in cands.items() if enc_equal(sd_state, c)]
                rec['parent'] = min(rec['distance_to'], key=lambda k: rec['distance_to'][k] if rec['distance_to'][k] is not None else 1e9)
            else:
                rec['parent'] = None
            manifest['checkpoints'][f'{method}/{sd}'] = rec
            print(f"  {method:22s} {str(sd):9s} parent={rec['parent']}  {rec['sha256'][:12]}", flush=True)
    json.dump(manifest, open(OUT, 'w'), indent=2)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
