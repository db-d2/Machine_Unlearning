#!/usr/bin/env python3
"""Stage 1b: re-derive the December-era experiment families with the fixed code.

  D  frozen critics (4 configs x 3 seeds); extra-gradient size ablation (n = 10, 50, 100;
     3 seeds) plus per-size clean retrains (3 seeds); Fisher scattered size ablation
     (n = 10, 30, 50, 100, 500; 10 seeds) plus per-size clean retrains (3 seeds);
     notebooks 37 (synthetic augmentation) and 38 (RAU) against the clean retrain;
     Tabula Muris evaluation with the scattered baseline; the Fisher/gradient-ascent
     phase audit.
  E  scoring: per-sample scores (frozen critics added), nested CIs, reference-LR audit
     (frozen, RAU, augmentation added), size-ablation scorer, utility, manifest.

Usage: PYTHONPATH=src python scripts/stage1b_rederive.py --phase D E
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
from stage1_rederive import run, nb, log, PY, DATA, BASELINE, CRITICS

EG_SIZES, EG_SEEDS, RETRAIN_SEEDS = [10, 50, 100], [42, 123, 456], [42, 43, 44]
FISHER_SIZES = [10, 30, 50, 100, 500]
FISHER_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def phase_d():
    run("frozen critics (4 configs x 3 seeds)", [PY, BASE / 'scripts/run_frozen_critics.py'],
        BASE / 'outputs/p2/frozen_critics/frozen_multi_lambda10_seed456/best_model.pt')
    # extra-gradient size ablation: per-size clean retrains, then the method
    for n in EG_SIZES:
        split = BASE / f'outputs/p4/extragradient_size/splits/split_structured_n{n}.json'
        for sd in RETRAIN_SEEDS:
            out = BASE / f'outputs/p4/extragradient_size/retrain_clean/n{n}/seed_{sd}'
            run(f"EG-size n={n} retrain_clean seed{sd}", [PY, BASE / 'scripts/retrain_clean.py', '--seed', sd,
                 '--split_path', split, '--output_dir', out], out / 'best_model.pt')
        for sd in EG_SEEDS:
            out = BASE / f'outputs/p4/extragradient_size/n{n}/seed{sd}'
            run(f"EG-size n={n} extragradient seed{sd}", [PY, BASE / 'src/train_unlearn_extragradient.py', '--data_path', DATA,
                 '--split_path', split, '--baseline_checkpoint', BASELINE, '--attacker_paths', *CRITICS,
                 '--lambda_privacy', '10', '--epochs', '50', '--lr_vae', '0.0001', '--lr_critic', '0.00001',
                 '--critic_steps', '2', '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'best_model.pt')
    # Fisher scattered size ablation
    for n in FISHER_SIZES:
        split = BASE / f'outputs/p4/size_ablation/splits/split_n{n}.json'
        for sd in RETRAIN_SEEDS:
            out = BASE / f'outputs/p4/size_ablation/retrain_clean/n{n}/seed_{sd}'
            run(f"Fisher-size n={n} retrain_clean seed{sd}", [PY, BASE / 'scripts/retrain_clean.py', '--seed', sd,
                 '--split_path', split, '--output_dir', out], out / 'best_model.pt')
        cache = BASE / f'outputs/p4/size_ablation/fisher_v2/n{n}/fisher_retain_persample.pt'
        for sd in FISHER_SEEDS:
            out = BASE / f'outputs/p4/size_ablation/fisher_v2/n{n}/seed{sd}'
            run(f"Fisher-size n={n} seed{sd}", [PY, BASE / 'src/train_fisher_unlearn.py', '--data_path', DATA, '--split_path', split,
                 '--baseline_checkpoint', BASELINE, '--scrub_lr', '0.0001', '--scrub_steps', '100', '--max_step', '0.001',
                 '--finetune_epochs', '10', '--finetune_lr', '0.0001', '--fisher_damping', '0.1', '--fisher_cache', cache,
                 '--batch_size', '256', '--output_dir', out, '--seed', sd], out / 'unlearned_model.pt')
    nb('37_synthetic_augmentation')
    nb('38_representation_unlearning')
    run("TM evaluate (CPU, with scattered baseline)", [PY, BASE / 'scripts/run_tabula_muris_unlearning.py', '--step', 'evaluate'],
        None, env={'MIA_DEVICE': 'cpu'})
    run("phase audit (Fisher steps, gradient ascent)", [PY, BASE / 'scripts/fisher_phase_audit.py'], BASE / 'outputs/p6/phase_audit.json')


def phase_e():
    sc = BASE / 'scripts'
    run("save_persample_scores", [PY, sc / 'save_persample_scores.py'], None)
    run("nested_bootstrap_all", [PY, sc / 'nested_bootstrap_all.py'], None)
    run("attack_lr_reference", [PY, sc / 'attack_lr_reference.py'], None)
    run("score_size_ablations", [PY, sc / 'score_size_ablations.py'], BASE / 'outputs/p4/size_ablations_v2.json')
    run("score_utility_all", [PY, sc / 'score_utility_all.py'], None)
    run("checkpoint_manifest", [PY, sc / 'checkpoint_manifest.py'], None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', nargs='+', default=['D', 'E'])
    args = ap.parse_args()
    log(f"stage1b phases {args.phase}")
    for ph in args.phase:
        {'D': phase_d, 'E': phase_e}[ph]()
    log("stage1b finished")
