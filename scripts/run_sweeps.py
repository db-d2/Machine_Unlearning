#!/usr/bin/env python3
"""Step sweeps on PBMC-33k cluster 13, three seeds each: gradient ascent's ascent learning rate (the
paper's 1e-5 leaves the forget loss unchanged) and Fisher scrubbing's step cap between the default
1e-3 (undone by the fine-tune) and 3e-3 (over-corrected). Every other setting is the paper's.
Resumable. Run: PYTHONPATH=src python scripts/run_sweeps.py
"""
import os, subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent.parent; PY = sys.executable
DATA = BASE / 'data/adata_processed.h5ad'; SPLIT = BASE / 'outputs/p1/split_structured.json'; BASELINE = BASE / 'outputs/p1/baseline/best_model.pt'
OUT = BASE / 'outputs/p4/sweeps'; LOGS = OUT / 'logs'; FCACHE = BASE / 'outputs/p2/fisher_unlearn/fisher_retain_persample.pt'


def run(name, cmd, done):
    if Path(done).exists():
        print(f'[skip] {name}', flush=True); return
    t0 = time.time(); print(f'[run ] {name}', flush=True)
    with open(LOGS / f'{name}.log', 'w') as f:
        r = subprocess.run([str(c) for c in cmd], env=dict(os.environ, PYTHONPATH=str(BASE / 'src')), cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{'done' if r.returncode == 0 else 'FAIL'}] {name} ({time.time() - t0:.0f}s)", flush=True)


for lr in ('0.0001', '0.001', '0.01'):
    for sd in ('42', '123', '456'):
        out = OUT / f'gradient_ascent_lr{lr}_seed{sd}'
        run(f'ga_lr{lr}_s{sd}', [PY, BASE / 'src/train_gradient_ascent.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA, '--split_path', SPLIT,
             '--output_dir', out, '--ascent_steps', '10', '--ascent_lr', lr, '--max_grad_norm', '1.0', '--finetune_epochs', '30', '--finetune_lr', '0.0001',
             '--patience', '10', '--batch_size', '256', '--seed', sd], out / 'best_model.pt')
for step in ('0.0015', '0.002', '0.0025', '0.003'):
    for sd in ('42', '123', '456'):
        out = OUT / f'fisher_step{step}_seed{sd}'
        run(f'fisher_step{step}_s{sd}', [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', SPLIT, '--baseline_checkpoint', BASELINE,
             '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', step, '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1',
             '--fisher_cache', FCACHE, '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'unlearned_model.pt')
print('all done', flush=True)
