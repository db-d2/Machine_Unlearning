#!/usr/bin/env python3
"""Stage 1 of the round-5 repair: re-derive every method checkpoint from the July-4
baseline with the fixed code, then re-score everything with one CPU attacker.

Phases (each step is skipped when its output already exists, so the script can be
re-run after an interruption):

  A  PBMC retain-only fine-tune (5 seeds), gradient ascent (5 seeds), Fisher
     (3 seeds + step-size ablation), cluster-12 clean retrains + Fisher + extra-gradient,
     Tabula Muris critics + extra-gradient + Fisher + evaluation, TCGA notebook 33.
  B  PBMC extra-gradient at lambda_privacy in EG_LAMBDAS (10 seeds each). Slowest phase.
  C  Scoring: per-sample scores, nested CIs, refit control, utility, cluster-12,
     notebooks 15/34b/36/40, figures, checkpoint manifest.

Usage: PYTHONPATH=src python scripts/stage1_rederive.py --phase A B C
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent.parent
PY = sys.executable
DATA = BASE / 'data' / 'adata_processed.h5ad'
SPLIT = BASE / 'outputs' / 'p1' / 'split_structured.json'
SPLIT_C12 = BASE / 'outputs' / 'p1_c12' / 'split_cluster12.json'
BASELINE = BASE / 'outputs' / 'p1' / 'baseline' / 'best_model.pt'
CRITICS = [BASE / 'outputs' / 'p2' / 'attackers' / f'attacker_{v}_seed{s}.pt' for v, s in [('v1', 42), ('v2', 43), ('v3', 44)]]
SIMPLE_SEEDS = [42, 123, 456, 789, 1011]
EG_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
FISHER_SEEDS = [42, 123, 456]
C12_SEEDS = [42, 43, 44]
EG_LAMBDAS = {'10': 'extragradient', '1': 'extragradient_lambda1'}
LOG = BASE / 'outputs' / 'stage1_rederive.log'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def run(name, cmd, done, env=None, cwd=None):
    if done is not None and done.exists():
        log(f"skip {name} (exists: {done.relative_to(BASE) if str(done).startswith(str(BASE)) else done})")
        return True
    log(f"start {name}")
    e = os.environ.copy()
    e['PYTHONPATH'] = str(BASE / 'src')
    if env:
        e.update(env)
    t0 = time.time()
    with open(BASE / 'outputs' / 'stage1_steps.log', 'a') as f:
        f.write(f"\n===== {name} =====\n")
        r = subprocess.run([str(c) for c in cmd], env=e, cwd=cwd or BASE, stdout=f, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and (done is None or done.exists())
    log(f"{'done' if ok else 'FAILED'} {name} ({time.time() - t0:.0f}s)")
    return ok


def nb(name):
    return run(f"notebook {name}", [PY, BASE / 'scripts' / 'run_notebook.py', name], None)


def phase_a():
    for sd in SIMPLE_SEEDS:
        out = BASE / 'outputs/p4/multiseed/retain_finetune' / f'seed{sd}'
        run(f"retain_finetune seed{sd}", [PY, BASE / 'src/train_retain_finetune.py', '--baseline_checkpoint', BASELINE,
             '--data_path', DATA, '--split_path', SPLIT, '--output_dir', out, '--lr', '0.0001', '--epochs', '50',
             '--patience', '10', '--batch_size', '256', '--seed', sd], out / 'best_model.pt')
    for sd in SIMPLE_SEEDS:
        out = BASE / 'outputs/p4/multiseed/gradient_ascent' / f'seed{sd}'
        run(f"gradient_ascent seed{sd}", [PY, BASE / 'src/train_gradient_ascent.py', '--baseline_checkpoint', BASELINE,
             '--data_path', DATA, '--split_path', SPLIT, '--output_dir', out, '--ascent_steps', '10', '--ascent_lr', '0.00001',
             '--max_grad_norm', '1.0', '--finetune_epochs', '30', '--finetune_lr', '0.0001', '--patience', '10',
             '--batch_size', '256', '--seed', sd], out / 'best_model.pt')
    fcache = BASE / 'outputs/p2/fisher_unlearn/fisher_retain_persample.pt'
    for sd in FISHER_SEEDS:
        out = BASE / 'outputs/p2/fisher_unlearn' / f'seed{sd}'
        run(f"fisher seed{sd}", [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', SPLIT,
             '--baseline_checkpoint', BASELINE, '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', '0.001',
             '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1', '--fisher_cache', fcache,
             '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'unlearned_model.pt')
    for step in ['3e-3', '1e-2']:
        out = BASE / 'outputs/p2/fisher_unlearn' / f'step{step}_seed42'
        run(f"fisher step ablation {step}", [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', SPLIT,
             '--baseline_checkpoint', BASELINE, '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', step,
             '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1', '--fisher_cache', fcache,
             '--batch_size', '256', '--output_dir', out, '--seed', '42'], out / 'unlearned_model.pt')
    # cluster 12
    for sd in C12_SEEDS:
        out = BASE / 'outputs/p2_c12' / f'retrain_clean_s{sd}'
        run(f"c12 retrain_clean s{sd}", [PY, BASE / 'scripts/retrain_clean.py', '--seed', sd, '--split_path', SPLIT_C12,
             '--output_dir', out], out / 'best_model.pt')
    c12cache = BASE / 'outputs/p2_c12/fisher_retain_persample.pt'
    for sd in C12_SEEDS:
        out = BASE / 'outputs/p2_c12' / f'fisher_s{sd}'
        run(f"c12 fisher s{sd}", [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', SPLIT_C12,
             '--baseline_checkpoint', BASELINE, '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', '0.001',
             '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1', '--fisher_cache', c12cache,
             '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'unlearned_model.pt')
    for sd in C12_SEEDS:
        out = BASE / 'outputs/p2_c12' / f'eg_s{sd}'
        run(f"c12 extragradient s{sd}", [PY, BASE / 'src/train_unlearn_extragradient.py', '--data_path', DATA,
             '--split_path', SPLIT_C12, '--baseline_checkpoint', BASELINE, '--attacker_paths', *CRITICS,
             '--lambda_privacy', '10', '--epochs', '50', '--lr_vae', '0.0001', '--lr_critic', '0.00001', '--critic_steps', '2',
             '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'best_model.pt')
    # Tabula Muris
    tm = BASE / 'scripts/run_tabula_muris_unlearning.py'
    tmo = BASE / 'outputs/tabula_muris'
    run("TM critics", [PY, tm, '--step', 'attackers'], tmo / 'attackers/attacker_v3_seed44.pt')
    run("TM extragradient", [PY, tm, '--step', 'extragradient'], tmo / 'extragradient_structured/best_model.pt', env={'EG_LAMBDA': '10'})
    run("TM fisher structured", [PY, tm, '--step', 'fisher-structured'], tmo / 'fisher_structured/unlearned_model.pt')
    run("TM fisher scattered", [PY, tm, '--step', 'fisher-scattered'], tmo / 'fisher_scattered/unlearned_model.pt')
    run("TM evaluate (CPU)", [PY, tm, '--step', 'evaluate'], None, env={'MIA_DEVICE': 'cpu'})
    # TCGA
    nb('33_tcga_brca_unlearning')


def phase_b():
    for lam, dirname in EG_LAMBDAS.items():
        for sd in EG_SEEDS:
            out = BASE / 'outputs/p4/multiseed' / dirname / f'seed{sd}'
            pilot = BASE / 'outputs/p4/multiseed/extragradient_pilot' / f'lambda{lam}_seed{sd}'
            if not (out / 'best_model.pt').exists() and (pilot / 'best_model.pt').exists():
                shutil.copytree(pilot, out)
                log(f"copied pilot {pilot.name} -> {dirname}/seed{sd}")
            run(f"extragradient lambda={lam} seed{sd}", [PY, BASE / 'src/train_unlearn_extragradient.py', '--data_path', DATA,
                 '--split_path', SPLIT, '--baseline_checkpoint', BASELINE, '--attacker_paths', *CRITICS,
                 '--lambda_privacy', lam, '--epochs', '50', '--lr_vae', '0.0001', '--lr_critic', '0.00001',
                 '--critic_steps', '2', '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'best_model.pt')


def phase_c():
    sc = BASE / 'scripts'
    run("save_persample_scores", [PY, sc / 'save_persample_scores.py'], None)
    run("nested_bootstrap_all", [PY, sc / 'nested_bootstrap_all.py'], None)
    run("retrain_refit_attacker", [PY, sc / 'retrain_refit_attacker.py'], None)
    run("score_utility_all", [PY, sc / 'score_utility_all.py'], None)
    run("score_persample_c12", [PY, sc / 'score_persample_c12.py'], None)
    for name in ['15_attack_suite', '34b_tcga_brca_within_subtype_eval', '36_cross_domain_comparison', '40_confound_fraction_cis']:
        nb(name)
    run("figures", [PY, BASE / 'figures/make_result_figures.py'], None)
    run("checkpoint_manifest", [PY, sc / 'checkpoint_manifest.py'], None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', nargs='+', default=['A', 'B', 'C'])
    args = ap.parse_args()
    log(f"stage1 phases {args.phase}")
    for ph in args.phase:
        {'A': phase_a, 'B': phase_b, 'C': phase_c}[ph]()
    log("stage1 finished")
