#!/usr/bin/env python3
"""Cache per-sample attacker scores (probabilities + labels) for every method x seed,
enabling an EXACT per-sample nested bootstrap (reviewer W4). Reuses the canonical eval
machinery (train one fresh attacker on baseline, apply to each method's features).

Usage: PYTHONPATH=src python scripts/save_persample_scores.py [--extended] [--out DIR]
Output: outputs/p4/multiseed/persample/<method>.json  (list of {seed, auc, scores, labels})

--extended (2026-09-25) appends, AFTER every canonical checkpoint, the Fisher-cap and gradient-ascent sweeps
(outputs/p4/sweeps), the three NB41 likelihood-targeted variants (outputs/p7/ltu*) and the NB42 200-epoch
retain fine-tunes (outputs/p8/pbmc). The pass is deterministic on CPU in a fixed order, so the canonical rows
come out bit-identical and the appended rows are scored by the same attacker instance in the same pass
(checked against the stored files before adoption). The paired test is scripts/paired_diff_33k.py.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import scanpy as sc
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
# Pin to CPU so the canonical attacker and all per-sample scores are reproducible
# across runs (MPS training is not bit-reproducible). All methods are scored by one
# attacker instance in a single pass, which is what makes the nested CIs comparable.
EM.DEVICE = 'cpu'
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes,
                            get_features, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH,
                            BASELINE_CHECKPOINT)
import nested_bootstrap_retrain as NBR
NBR.DEVICE = 'cpu'
from nested_bootstrap_retrain import load_retrain_model
DEVICE = 'cpu'

OUT = BASE / 'outputs' / 'p4' / 'multiseed' / 'persample'; OUT.mkdir(parents=True, exist_ok=True)

def checkpoint_map():
    """Every checkpoint scored by the single canonical attacker. Revision 2026-09-08:
    the reference 'retrain' is the clean retain-split retrain (scripts/retrain_clean.py);
    the April retrains that trained on the matched negatives are kept as 'retrain_leaky'
    for the record; Fisher and extra-gradient point at the re-derived July-baseline
    checkpoints, extra-gradient at both privacy weights."""
    M = {}
    for lam_dir, key in [('extragradient', 'extragradient'), ('extragradient_lambda1', 'extragradient_lambda1')]:
        for sd_dir in sorted((BASE / f'outputs/p4/multiseed/{lam_dir}').glob('seed*')):
            if (sd_dir / 'best_model.pt').exists():
                M.setdefault(key, []).append((sd_dir.name, sd_dir / 'best_model.pt'))
    for meth in ['gradient_ascent', 'retain_finetune']:
        for sd_dir in sorted((BASE / f'outputs/p4/multiseed/{meth}').glob('seed*')):
            if (sd_dir / 'best_model.pt').exists():
                M.setdefault(meth, []).append((sd_dir.name, sd_dir / 'best_model.pt'))
    for meth, pref in [('ssd', 'alpha1.0'), ('scrub', 'af1.0'), ('contrastive', 'gamma1.0'), ('dp_sgd', 'eps10.0')]:
        for sd_dir in sorted((BASE / f'outputs/p2/{meth}').glob(f'{pref}_seed*')):
            M.setdefault(meth, []).append((sd_dir.name.split('_')[-1], sd_dir / 'best_model.pt'))
    for sd_dir in sorted((BASE / 'outputs/p2/moon').glob('seed*')):
        M.setdefault('moon', []).append((sd_dir.name, sd_dir / 'best_model.pt'))
    for sd_dir in sorted((BASE / 'outputs/p2/fisher_unlearn').glob('seed*')):
        if (sd_dir / 'unlearned_model.pt').exists():
            M.setdefault('fisher', []).append((sd_dir.name, sd_dir / 'unlearned_model.pt'))
    for sd_dir in sorted((BASE / 'outputs/p2/fisher_unlearn').glob('step*_seed42')):
        if (sd_dir / 'unlearned_model.pt').exists():
            M.setdefault('fisher_step_ablation', []).append((sd_dir.name, sd_dir / 'unlearned_model.pt'))
    for sd_dir in sorted((BASE / 'outputs/p1/retrain_clean').glob('seed_*')):
        M.setdefault('retrain', []).append((sd_dir.name, sd_dir / 'best_model.pt'))
    for sd_dir in sorted((BASE / 'outputs/p1/multiseed_retrain').glob('seed*')):
        M.setdefault('retrain_leaky', []).append((sd_dir.name, sd_dir / 'best_model.pt'))
    for cfg_dir in sorted((BASE / 'outputs/p2/frozen_critics').glob('*_seed*')):
        if (cfg_dir / 'best_model.pt').exists():
            key = cfg_dir.name.rsplit('_seed', 1)[0]
            M.setdefault(key, []).append((cfg_dir.name.rsplit('_', 1)[-1], cfg_dir / 'best_model.pt'))
    M['baseline'] = [('seed42', BASE / 'outputs/p1/baseline/best_model.pt')]
    return M


def extended_map():
    """Checkpoints scored after the canonical ones in an --extended pass (never before them)."""
    M = {}
    for d in sorted((BASE / 'outputs/p4/sweeps').glob('*_seed*')):
        cfg, sd = d.name.rsplit('_seed', 1)
        f = d / ('unlearned_model.pt' if cfg.startswith('fisher') else 'best_model.pt')
        if f.exists():
            M.setdefault('sweep_' + cfg, []).append((f'seed{sd}', f))
    for v in ['ltu', 'ltu_avg', 'ltu_disjoint']:
        for d in sorted((BASE / f'outputs/p7/{v}').glob('seed*')):
            if (d / 'best_model.pt').exists():
                M.setdefault(v, []).append((d.name, d / 'best_model.pt'))
    for d in sorted((BASE / 'outputs/p8/pbmc').glob('seed*')):
        if (d / 'final_model.pt').exists():
            M.setdefault('long_finetune_200', []).append((d.name, d / 'final_model.pt'))
    return M


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--extended', action='store_true'); ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH)); forget_idx = split['forget_indices']; retain_idx = split['retain_indices']
    matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    print(f"forget={len(forget_idx)} matched_neg={len(matched_neg_idx)} retain={len(retain_idx)}", flush=True)
    base_model, _ = load_vae_model(BASELINE_CHECKPOINT)
    attacker = train_fresh_attacker(base_model, adata, forget_idx, matched_neg_idx, retain_idx)
    attacker.eval()
    labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_neg_idx))])
    cmap = checkpoint_map()
    if args.extended:
        cmap.update(extended_map())   # appended after 'baseline', the last canonical key
    for method, seeds in cmap.items():
        recs = []
        for sd, path in seeds:
            try:
                # retrain checkpoints lack input_dim in config and use BatchNorm
                model = load_retrain_model(str(path)) if method == 'retrain_leaky' else load_vae_model(str(path))[0]
            except Exception as e:
                print(f"  [skip] {method} {sd}: {type(e).__name__} {e}", flush=True); continue
            retain_z = get_retain_latent_codes(model, adata, retain_idx)
            ff = get_features(model, adata, forget_idx, retain_z=retain_z)
            nf = get_features(model, adata, matched_neg_idx, retain_z=retain_z)
            X = torch.cat([ff, nf], 0).to(DEVICE)
            with torch.no_grad():
                scores = torch.sigmoid(attacker(X).squeeze()).cpu().numpy()
            auc = float(roc_auc_score(labels, scores))
            recs.append({'seed': str(sd), 'auc': auc, 'scores': [float(s) for s in scores], 'labels': [int(l) for l in labels]})
            print(f"  {method} {sd}: AUC={auc:.4f}", flush=True)
        json.dump(recs, open(out_dir / f'{method}.json', 'w'))
        print(f"== saved {method}: {len(recs)} seeds ==", flush=True)
    print("DONE")

if __name__ == '__main__':
    main()
