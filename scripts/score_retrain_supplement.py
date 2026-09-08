#!/usr/bin/env python3
"""Supplement: score the 5 retrain checkpoints (which use the retrain loader, not the
standard config) into the per-sample cache. Same canonical attacker as the main pass."""
import json, sys
from pathlib import Path
import numpy as np, torch, scanpy as sc
from sklearn.metrics import roc_auc_score
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'  # reproducible scoring, consistent with save_persample_scores.py
from eval_multiseed import (train_fresh_attacker, get_retain_latent_codes, get_features,
                            DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT, load_vae_model)
from nested_bootstrap_retrain import load_retrain_model
DEVICE = 'cpu'
OUT = BASE / 'outputs/p4/multiseed/persample'
adata = sc.read_h5ad(DATA_PATH)
split = json.load(open(SPLIT_PATH)); forget_idx = split['forget_indices']; retain_idx = split['retain_indices']
matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
base_model, _ = load_vae_model(BASELINE_CHECKPOINT)
attacker = train_fresh_attacker(base_model, adata, forget_idx, matched_neg_idx, retain_idx); attacker.eval()
labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_neg_idx))])
recs = []
for sd in [42, 43, 44, 45, 46]:
    model = load_retrain_model(str(BASE / f'outputs/p1/multiseed_retrain/seed_{sd}/best_model.pt'))
    retain_z = get_retain_latent_codes(model, adata, retain_idx)
    ff = get_features(model, adata, forget_idx, retain_z=retain_z)
    nf = get_features(model, adata, matched_neg_idx, retain_z=retain_z)
    X = torch.cat([ff, nf], 0).to(DEVICE)
    with torch.no_grad():
        scores = torch.sigmoid(attacker(X).squeeze()).cpu().numpy()
    auc = float(roc_auc_score(labels, scores))
    recs.append({'seed': str(sd), 'auc': auc, 'scores': [float(s) for s in scores], 'labels': [int(l) for l in labels]})
    print(f"retrain seed{sd}: AUC={auc:.4f}", flush=True)
json.dump(recs, open(OUT / 'retrain.json', 'w'))
print("saved retrain", len(recs))
