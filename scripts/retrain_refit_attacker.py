#!/usr/bin/env python3
"""Retrain-refit attacker experiment (peer-review round-3, issue 11).

The canonical protocol trains one attacker on the BASELINE model's features and
transfers it to every other model, including retrain. A skeptic can then argue
that retrain's low AUC reflects a weak transferred attacker that simply saturates
at n=30, not a genuine absence of membership signal.

This script removes that inference. For each model it trains a FRESH attacker
DIRECTLY on that model's own features, giving a dedicated adversary its best
possible shot. To avoid the in-sample inflation that would occur if the attacker
were trained and scored on the same points, the reported AUC is out-of-fold from
stratified 5-fold cross-validation. A positive control on the baseline model
confirms the refit protocol recovers strong signal at n=30, ruling out attacker
saturation. An in-sample AUC (80/20 split, scored on the full set, matching the
canonical estimator) is also reported as an upper bound that grants the adversary
the in-sample advantage.

If even this dedicated, model-specific adversary lands at chance on retrain,
absence of signal is measured rather than assumed.

Usage:  PYTHONPATH=src python scripts/retrain_refit_attacker.py
Output: outputs/p4/multiseed/retrain_refit.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
sys.path.insert(0, str(BASE / 'scripts'))

# Pin to CPU: MPS is not bit-reproducible run to run, and every canonical MIA
# number in the paper is scored on CPU. Set the module DEVICE globals BEFORE the
# reused helpers read them.
import eval_multiseed as EM
EM.DEVICE = 'cpu'
import nested_bootstrap_retrain as NBR
NBR.DEVICE = 'cpu'

from eval_multiseed import (
    load_vae_model, get_features, get_retain_latent_codes,
    DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT,
)
from nested_bootstrap_retrain import load_retrain_model
from attacker import MLPAttacker
from attacker_eval import compute_advantage, compute_confidence_interval
from stats_utils import ci_to_se, nested_ci

DEVICE = 'cpu'
RETRAIN_DIR = BASE / 'outputs' / 'p1' / 'retrain_clean'   # 2026-09-08: clean retain-split retrains (the April ones trained on the negatives)
OUT = BASE / 'outputs' / 'p4' / 'multiseed' / 'retrain_refit.json'
RETRAIN_SEEDS = [42, 43, 44, 45, 46]

N_FOLDS = 5
N_EPOCHS = 50
LR = 1e-3
BATCH = 128
# Repeat the whole CV over many random fold splits so the reported AUC is not an
# artifact of one split (~6 positives per held-out fold at n=30). The attacker is
# re-seeded per (split, fold) so init variance is captured too. The split-induced
# SD is reported and folded into the nested CI (round-4 review, finding F1).
SPLIT_SEEDS = list(range(20))


def train_attacker(train_x, train_y, seed):
    """Train one MLP attacker matching the canonical NB03 architecture."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    att = MLPAttacker(
        input_dim=train_x.shape[1], hidden_dims=[256, 256],
        dropout=0.3, use_spectral_norm=True,
    ).to(DEVICE)
    opt = torch.optim.Adam(att.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH, shuffle=True)
    for _ in range(N_EPOCHS):
        att.train()
        for bx, by in loader:
            opt.zero_grad()
            logits = att(bx.to(DEVICE)).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, by.to(DEVICE))
            loss.backward()
            opt.step()
    att.train(False)
    return att


def refit_cv_once(x, y, y_np, split_seed):
    """One out-of-fold CV pass. The attacker never scores a point it trained on.

    The KFold split and each fold's attacker are seeded from split_seed, so
    repeating over split seeds captures both fold-split and init variance.
    """
    oof = np.full(len(x), np.nan)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x.numpy(), y_np)):
        att = train_attacker(x[train_idx], y[train_idx], seed=split_seed * 100 + fold_i)
        with torch.no_grad():
            oof[test_idx] = torch.sigmoid(att(x[test_idx].to(DEVICE)).squeeze()).cpu().numpy()
    return oof


def refit_cv(forget_x, neg_x):
    """Repeated out-of-fold refit AUC over SPLIT_SEEDS random fold splits.

    Returns the split-mean AUC/advantage, the split-induced SD, a within-eval
    bootstrap CI from one representative split, and a combined SE that folds the
    split SD in quadrature with the bootstrap SE for the nested aggregation.
    """
    x = torch.cat([forget_x, neg_x], dim=0)
    y = torch.cat([torch.ones(len(forget_x)), torch.zeros(len(neg_x))], dim=0)
    y_np = y.numpy()

    aucs, advs, first_oof = [], [], None
    for split_seed in SPLIT_SEEDS:
        oof = refit_cv_once(x, y, y_np, split_seed)
        if first_oof is None:
            first_oof = oof
        auc = float(roc_auc_score(y_np, oof))
        aucs.append(auc)
        advs.append(compute_advantage(auc))

    aucs = np.array(aucs)
    advs = np.array(advs)
    boot = compute_confidence_interval(first_oof, y_np)
    boot_se_auc = ci_to_se(boot['auc'][1], boot['auc'][2])
    boot_se_adv = ci_to_se(boot['advantage'][1], boot['advantage'][2])
    split_sd_auc = float(aucs.std(ddof=1))
    split_sd_adv = float(advs.std(ddof=1))
    return {
        'auc': float(aucs.mean()), 'advantage': float(advs.mean()),
        'auc_split_sd': split_sd_auc, 'adv_split_sd': split_sd_adv,
        'auc_min': float(aucs.min()), 'auc_max': float(aucs.max()),
        'n_splits': len(SPLIT_SEEDS),
        'boot_auc_ci': list(boot['auc']), 'boot_adv_ci': list(boot['advantage']),
        # combined SE = quadrature of split SD and within-eval bootstrap SE
        'combined_se_auc': float(np.hypot(split_sd_auc, boot_se_auc)),
        'combined_se_adv': float(np.hypot(split_sd_adv, boot_se_adv)),
    }


def refit_insample(forget_x, neg_x, seed=42):
    """In-sample analogue of the canonical estimator (80/20 train, full-set AUC).

    Grants the adversary the in-sample advantage, so it is an upper bound on what
    a refit attacker can extract.
    """
    x = torch.cat([forget_x, neg_x], dim=0)
    y = torch.cat([torch.ones(len(forget_x)), torch.zeros(len(neg_x))], dim=0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    perm = torch.randperm(len(x))
    n_train = int(0.8 * len(x))
    att = train_attacker(x[perm[:n_train]], y[perm[:n_train]], seed)
    with torch.no_grad():
        preds = torch.sigmoid(att(x.to(DEVICE)).squeeze()).cpu().numpy()
    auc = float(roc_auc_score(y.numpy(), preds))
    return {'auc': auc, 'advantage': compute_advantage(auc)}


def features_for(model, adata, forget_idx, matched_neg_idx, retain_idx):
    """Extract 70-dim attack features from THIS model (its own latent space)."""
    retain_z = get_retain_latent_codes(model, adata, retain_idx)
    forget_x = get_features(model, adata, forget_idx, retain_z=retain_z)
    neg_x = get_features(model, adata, matched_neg_idx, retain_z=retain_z)
    return forget_x, neg_x


def evaluate_model(tag, model, adata, forget_idx, matched_neg_idx, retain_idx):
    forget_x, neg_x = features_for(model, adata, forget_idx, matched_neg_idx, retain_idx)
    cv = refit_cv(forget_x, neg_x)
    ins = refit_insample(forget_x, neg_x)
    print(f"  [{tag}] refit CV AUC={cv['auc']:.4f} (split SD {cv['auc_split_sd']:.4f}, "
          f"range [{cv['auc_min']:.3f}, {cv['auc_max']:.3f}]) adv={cv['advantage']:.4f} | "
          f"in-sample AUC={ins['auc']:.4f}", flush=True)
    return {
        'cv_auc': cv['auc'], 'cv_advantage': cv['advantage'],
        'cv_auc_split_sd': cv['auc_split_sd'], 'cv_adv_split_sd': cv['adv_split_sd'],
        'cv_auc_range': [cv['auc_min'], cv['auc_max']], 'n_splits': cv['n_splits'],
        'cv_boot_auc_ci': cv['boot_auc_ci'], 'cv_boot_adv_ci': cv['boot_adv_ci'],
        'cv_combined_se_auc': cv['combined_se_auc'], 'cv_combined_se_adv': cv['combined_se_adv'],
        'insample_auc': ins['auc'], 'insample_advantage': ins['advantage'],
    }


def main():
    adata = sc.read_h5ad(DATA_PATH)
    split = json.load(open(SPLIT_PATH))
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    matched_neg_idx = json.load(open(MATCHED_NEG_PATH))['matched_indices']
    print(f"forget={len(forget_idx)} matched_neg={len(matched_neg_idx)} "
          f"retain={len(retain_idx)}  device={DEVICE}", flush=True)

    results = {
        'protocol': (
            'Refit attacker: fresh MLP ([256,256], spectral norm, dropout 0.3, '
            '50 epochs, Adam 1e-3) trained DIRECTLY on each model\'s own 70-dim '
            'features. cv_auc is the out-of-fold AUC averaged over %d random '
            'stratified 5-fold splits (no point scored in-sample; attacker '
            're-seeded per split and fold). auc_split_sd is the SD across those '
            'splits, folded into the nested CI via combined_se. insample_auc is '
            'the 80/20-train, full-set analogue of the canonical estimator '
            '(adversary upper bound). Baseline is the positive control.'
            % len(SPLIT_SEEDS)),
        'n_folds': N_FOLDS,
        'n_splits': len(SPLIT_SEEDS),
        'models': {},
    }

    print("\n[baseline] positive control (should recover signal)...", flush=True)
    base_model, _ = load_vae_model(str(BASELINE_CHECKPOINT))
    results['models']['baseline'] = evaluate_model(
        'baseline', base_model, adata, forget_idx, matched_neg_idx, retain_idx)

    print("\n[retrain] dedicated adversary on each retrain seed...", flush=True)
    seed_recs = []
    for sd in RETRAIN_SEEDS:
        ckpt = RETRAIN_DIR / f'seed_{sd}' / 'best_model.pt'
        model = load_vae_model(str(ckpt))[0]
        rec = evaluate_model(f'retrain seed_{sd}', model, adata, forget_idx,
                             matched_neg_idx, retain_idx)
        rec['seed'] = sd
        rec['source'] = str(ckpt)
        seed_recs.append(rec)

    cv_aucs = [r['cv_auc'] for r in seed_recs]
    cv_advs = [r['cv_advantage'] for r in seed_recs]
    # per-seed SE folds split-induced SD together with within-eval bootstrap SE
    auc_ses = [r['cv_combined_se_auc'] for r in seed_recs]
    adv_ses = [r['cv_combined_se_adv'] for r in seed_recs]

    results['models']['retrain'] = {
        'per_seed': seed_recs,
        'cv_auc_mean': float(np.mean(cv_aucs)),
        'cv_auc_std': float(np.std(cv_aucs, ddof=1)),
        'cv_auc_split_sd_mean': float(np.mean([r['cv_auc_split_sd'] for r in seed_recs])),
        'cv_advantage_mean': float(np.mean(cv_advs)),
        'cv_advantage_std': float(np.std(cv_advs, ddof=1)),
        'cv_adv_split_sd_mean': float(np.mean([r['cv_adv_split_sd'] for r in seed_recs])),
        'insample_auc_mean': float(np.mean([r['insample_auc'] for r in seed_recs])),
        'nested_auc': nested_ci(cv_aucs, auc_ses),
        'nested_advantage': nested_ci(cv_advs, adv_ses),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, 'w'), indent=2)

    b = results['models']['baseline']
    r = results['models']['retrain']
    print("\n" + "=" * 64)
    print("SUMMARY (refit / out-of-fold)")
    print(f"  baseline (control): CV AUC={b['cv_auc']:.4f}  adv={b['cv_advantage']:.4f}")
    print(f"  retrain (5 seeds):  CV AUC={r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f}")
    print(f"                      adv ={r['cv_advantage_mean']:.4f} +/- {r['cv_advantage_std']:.4f}")
    print(f"    nested AUC CI = [{r['nested_auc']['nested_ci_low']:.4f}, "
          f"{r['nested_auc']['nested_ci_high']:.4f}]")
    print(f"    nested adv CI = [{r['nested_advantage']['nested_ci_low']:.4f}, "
          f"{r['nested_advantage']['nested_ci_high']:.4f}]")
    print(f"  Saved -> {OUT}")
    print("=" * 64)


if __name__ == '__main__':
    main()
