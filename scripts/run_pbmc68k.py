#!/usr/bin/env python3
"""PBMC-68k cohort driver: baseline, clean retrains, and the post-hoc methods for two forget sets.

Resumable: every step is skipped when its output checkpoint exists. Same settings as the PBMC-33k
re-derivation (scripts/stage1_rederive.py). Logs go to outputs/pbmc68k/logs/.
Run: PYTHONPATH=src python scripts/run_pbmc68k.py [--forget_sets mk pdc] [--seeds 42 123 456]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent.parent
PY = sys.executable
DATA = BASE / 'data/pbmc68k_processed.h5ad'
OUT = BASE / 'outputs/pbmc68k'
LOGS = OUT / 'logs'
# extra-gradient critics are trained on THIS cohort's baseline (forget vs retain, src/train_multi_attackers.py);
# the PBMC-33k critics collapsed to chance within five epochs here (critic BCE 0.69, privacy term 0.50) and left the method inert
def critics(fs):
    return [str(OUT / fs / 'attackers' / f'attacker_v{i}_seed{s}.pt') for i, s in ((1, 42), (2, 43), (3, 44))]


def run(name, cmd, done):
    if done is not None and Path(done).exists():
        print(f"[skip] {name}", flush=True); return
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); print(f"[run ] {name}", flush=True)
    env = dict(os.environ, PYTHONPATH=str(BASE / 'src'))
    with open(LOGS / f"{name.replace(' ', '_').replace('/', '_')}.log", 'w') as f:
        r = subprocess.run([str(c) for c in cmd], env=env, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{'done' if r.returncode == 0 else 'FAIL'}] {name} ({time.time() - t0:.0f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{name} failed; see {LOGS}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forget_sets', nargs='+', default=['mk', 'pdc']); ap.add_argument('--seeds', nargs='+', default=['42', '123', '456'])
    args = ap.parse_args()
    base_dir = OUT / 'baseline'
    run('baseline', [PY, BASE / 'scripts/train_cohort.py', '--data_path', DATA, '--split_path', OUT / 'split_mk.json', '--subset', 'training',
                     '--seed', '42', '--output_dir', base_dir], base_dir / 'training_complete.json')
    BASELINE = base_dir / 'best_model.pt'
    for fs in args.forget_sets:
        SPLIT = OUT / f'split_{fs}.json'
        for sd in ('42', '43', '44', '45', '46', '47', '48', '49'):   # 42 is the audit reference; the other seven give the retrained range (2026-09-21: five added)
            out = OUT / fs / f'retrain_clean_seed_{sd}'
            run(f'{fs} retrain_clean seed{sd}', [PY, BASE / 'scripts/train_cohort.py', '--data_path', DATA, '--split_path', SPLIT, '--subset', 'retain',
                                                 '--seed', sd, '--output_dir', out], out / 'training_complete.json')
        att_dir = OUT / fs / 'attackers'
        run(f'{fs} critics', [PY, BASE / 'src/train_multi_attackers.py', '--vae_path', BASELINE, '--data_path', DATA, '--split_path', SPLIT,
                              '--output_dir', att_dir], att_dir / 'attacker_v3_seed44.pt')
        CRITICS = critics(fs)
        fcache = OUT / fs / 'fisher_retain_persample.pt'
        for sd in args.seeds:
            out = OUT / fs / f'retain_finetune_seed{sd}'
            run(f'{fs} retain_finetune seed{sd}', [PY, BASE / 'src/train_retain_finetune.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA,
                 '--split_path', SPLIT, '--output_dir', out, '--lr', '0.0001', '--epochs', '50', '--patience', '10', '--batch_size', '256', '--seed', sd], out / 'best_model.pt')
            out = OUT / fs / f'gradient_ascent_seed{sd}'
            run(f'{fs} gradient_ascent seed{sd}', [PY, BASE / 'src/train_gradient_ascent.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA,
                 '--split_path', SPLIT, '--output_dir', out, '--ascent_steps', '10', '--ascent_lr', '0.00001', '--max_grad_norm', '1.0',
                 '--finetune_epochs', '30', '--finetune_lr', '0.0001', '--patience', '10', '--batch_size', '256', '--seed', sd], out / 'best_model.pt')
            out = OUT / fs / f'fisher_seed{sd}'
            run(f'{fs} fisher seed{sd}', [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', SPLIT, '--baseline_checkpoint', BASELINE,
                 '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', '0.001', '--finetune_epochs', '10', '--finetune_lr', '0.0001',
                 '--fisher_damping', '0.1', '--fisher_cache', fcache, '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'unlearned_model.pt')
            out = OUT / fs / f'ssd_seed{sd}'
            run(f'{fs} ssd seed{sd}', [PY, BASE / 'src/train_ssd.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA, '--split_path', SPLIT,
                 '--output_dir', out, '--alpha', '1.0', '--seed', sd], out / 'best_model.pt')
            out = OUT / fs / f'scrub_seed{sd}'
            run(f'{fs} scrub seed{sd}', [PY, BASE / 'src/train_scrub.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA, '--split_path', SPLIT,
                 '--output_dir', out, '--alpha_forget', '1.0', '--seed', sd], out / 'best_model.pt')
            out = OUT / fs / f'contrastive_seed{sd}'
            run(f'{fs} contrastive seed{sd}', [PY, BASE / 'src/train_contrastive_unlearn.py', '--baseline_checkpoint', BASELINE, '--data_path', DATA,
                 '--split_path', SPLIT, '--output_dir', out, '--gamma', '1.0', '--seed', sd], out / 'best_model.pt')
            out = OUT / fs / f'extragradient_seed{sd}'
            run(f'{fs} extragradient seed{sd}', [PY, BASE / 'src/train_unlearn_extragradient.py', '--data_path', DATA, '--split_path', SPLIT,
                 '--baseline_checkpoint', BASELINE, '--attacker_paths', *CRITICS, '--lambda_privacy', '10', '--epochs', '50', '--lr_vae', '0.0001',
                 '--lr_critic', '0.00001', '--critic_steps', '2', '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'best_model.pt')
    print("all done", flush=True)


if __name__ == '__main__':
    main()
