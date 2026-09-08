#!/usr/bin/env python3
"""Score the Moon (2024) feature-unlearning checkpoints with the SAME canonical
attacker used for every other method, and append outputs/p4/multiseed/persample/moon.json.

The fresh attacker is deterministic (train_fresh_attacker seeds torch/np with 42),
so this reproduces the exact attacker used by save_persample_scores.py. As a guard,
it re-scores contrastive seed42 and checks it matches the cached AUC (0.1818) before
trusting the Moon scores. Run: PYTHONPATH=src python scripts/score_moon.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import scanpy as sc
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes,
                            get_features, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH,
                            BASELINE_CHECKPOINT, DEVICE)

OUT = BASE / 'outputs' / 'p4' / 'multiseed' / 'persample'


def score(model, adata, forget_idx, matched_neg_idx, retain_idx, attacker, labels):
    retain_z = get_retain_latent_codes(model, adata, retain_idx)
    ff = get_features(model, adata, forget_idx, retain_z=retain_z)
    nf = get_features(model, adata, matched_neg_idx, retain_z=retain_z)
    Xf = torch.cat([ff, nf], 0).to(DEVICE)
    with torch.no_grad():
        s = torch.sigmoid(attacker(Xf).squeeze()).cpu().numpy()
    return s, float(roc_auc_score(labels, s))


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH)); forget_idx = split['forget_indices']; retain_idx = split['retain_indices']
    matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_neg_idx))])
    base_model, _ = load_vae_model(BASELINE_CHECKPOINT)
    attacker = train_fresh_attacker(base_model, adata, forget_idx, matched_neg_idx, retain_idx)
    attacker.eval()

    # Determinism guard: contrastive seed42 must reproduce the cached AUC
    guard_ckpt = BASE / 'outputs/p2/contrastive/gamma1.0_seed42/best_model.pt'
    if guard_ckpt.exists():
        gm, _ = load_vae_model(str(guard_ckpt))
        _, gauc = score(gm, adata, forget_idx, matched_neg_idx, retain_idx, attacker, labels)
        cached = {r['seed']: r['auc'] for r in json.load(open(OUT / 'contrastive.json'))}.get('seed42')
        print(f"[guard] contrastive seed42: fresh={gauc:.4f} cached={cached}", flush=True)
        if cached is not None and abs(gauc - cached) > 5e-3:
            sys.exit(f"ATTACKER NOT REPRODUCIBLE (|{gauc:.4f}-{cached:.4f}|>0.005); re-score all methods instead.")

    recs = []
    for sd_dir in sorted((BASE / 'outputs/p2/moon').glob('seed*')):
        model, _ = load_vae_model(str(sd_dir / 'best_model.pt'))
        s, auc = score(model, adata, forget_idx, matched_neg_idx, retain_idx, attacker, labels)
        recs.append({'seed': sd_dir.name, 'auc': auc, 'scores': [float(x) for x in s], 'labels': [int(l) for l in labels]})
        print(f"  moon {sd_dir.name}: AUC={auc:.4f}", flush=True)
    json.dump(recs, open(OUT / 'moon.json', 'w'))
    print(f"== saved moon: {len(recs)} seeds ==", flush=True)


if __name__ == '__main__':
    main()
