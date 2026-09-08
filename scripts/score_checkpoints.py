#!/usr/bin/env python3
"""Score arbitrary checkpoints with the canonical CPU attacker and the utility metrics.

Trains the canonical fresh attacker once on the July-4 baseline (forget vs matched
negatives, 70-dim features, CPU, seed 42) and applies it to each checkpoint given on the
command line. Also reports held-out utility (ELBO, KL, marker r, ARI). Meant for pilots
and spot checks; the full re-score is save_persample_scores.py.

Usage: PYTHONPATH=src python scripts/score_checkpoints.py label=path [label=path ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
import nested_bootstrap_retrain as NBR
NBR.DEVICE = 'cpu'
from nested_bootstrap_retrain import load_retrain_model
from eval_multiseed import (load_vae_model, train_fresh_attacker, get_retain_latent_codes, get_features,
                            evaluate_utility, MARKER_GENES, DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH,
                            BASELINE_CHECKPOINT)


def load_any(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
    if 'input_dim' not in cfg:
        return load_retrain_model(path)
    return load_vae_model(path)[0]


def main():
    targets = dict(a.split('=', 1) for a in sys.argv[1:])
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH))
    forget_idx, retain_idx, unseen_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
    matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    labels = np.concatenate([np.ones(len(forget_idx)), np.zeros(len(matched_neg_idx))])
    X = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, dtype=torch.float32)
    X_holdout, labels_holdout = X[unseen_idx], adata.obs['leiden'].values[unseen_idx]
    gene_names = list(adata.var_names)
    marker_idx = [gene_names.index(g) for g in MARKER_GENES if g in gene_names]

    base, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    attacker = train_fresh_attacker(base, adata, forget_idx, matched_neg_idx, retain_idx)
    attacker.eval()

    rows = {}
    for label, path in targets.items():
        model = load_any(path)
        rz = get_retain_latent_codes(model, adata, retain_idx)
        Xf = torch.cat([get_features(model, adata, forget_idx, retain_z=rz),
                        get_features(model, adata, matched_neg_idx, retain_z=rz)], 0)
        with torch.no_grad():
            s = torch.sigmoid(attacker(Xf).squeeze()).numpy()
        auc = float(roc_auc_score(labels, s))
        util = evaluate_utility(model, X_holdout, labels_holdout, marker_idx, gene_names)
        rows[label] = {'auc': auc, 'advantage': 2 * abs(auc - 0.5), 'elbo': util['elbo'], 'kl': util['kl'],
                       'marker_r': util['marker_r'], 'ari': util['ari'], 'path': path}
        print(f"{label:34s} AUC={auc:.3f} adv={2*abs(auc-0.5):.3f} | ELBO={util['elbo']:.1f} "
              f"KL={util['kl']:.2f} marker_r={util['marker_r']:.3f} ARI={util['ari']:.3f}", flush=True)
    out = Path(sys.argv[0]).with_suffix('') if False else None
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
