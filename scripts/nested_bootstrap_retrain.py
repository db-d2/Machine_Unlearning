#!/usr/bin/env python3
"""Evaluate the multi-seed retrain checkpoints and compute nested CIs.

For each retrain seed in {42, 43, 44, 45, 46}:
1. Load checkpoint
2. Apply the canonical fresh attacker (trained on baseline F vs matched neg)
3. Compute AUC, advantage, and per-seed sample bootstrap CI

Then aggregate via nested bootstrap:
- Outer: resample seeds with replacement
- Inner: for each drawn seed, draw an AUC ~ Normal(seed_mean, seed_SE)
- Compute mean across drawn seeds; repeat 10000 times
- Report 95% CI

This combines seed-level and sample-level uncertainty into one CI that
is directly comparable to the t-CI used for methods.

Also computes nested CIs for the existing methods (extragradient,
retain_finetune, gradient_ascent) using the cached per-seed JSONs.

Usage:
    PYTHONPATH=src python scripts/nested_bootstrap_retrain.py
"""

import json
import sys
import numpy as np
import torch
import scanpy as sc
from pathlib import Path
from statsmodels.stats.multitest import multipletests

SRC_DIR = Path(__file__).parent.parent / 'src'
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR / 'scripts'))

from eval_multiseed import (
    load_vae_model, train_fresh_attacker, evaluate_privacy,
    DATA_PATH, SPLIT_PATH, MATCHED_NEG_PATH, BASELINE_CHECKPOINT, DEVICE
)
from stats_utils import (
    ci_to_se, cohens_d, nested_ci,
    welch_test_vs_retrain, welch_test_from_stats,
)

RETRAIN_DIR = BASE_DIR / 'outputs' / 'p1' / 'multiseed_retrain'
EVAL_DIR = BASE_DIR / 'outputs' / 'p4' / 'multiseed' / 'eval'
OUTPUT_PATH = BASE_DIR / 'outputs' / 'p4' / 'multiseed' / 'retrain_nested_ci.json'


def load_retrain_model(checkpoint_path, input_dim=2000):
    """Load retrain checkpoint. New retrain.py models use BatchNorm (VAE default).

    Canonical baseline/retrain use LayerNorm, but the retrain.py script as it
    currently exists does not expose a --use_layer_norm flag and defaults to
    BatchNorm. For the multiseed_retrain family (seeds 43-46), all 4 models
    are BatchNorm, which is consistent within the family.
    """
    from vae import VAE
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    use_ln = cfg.get('use_layer_norm', False)
    model = VAE(
        input_dim=input_dim,
        latent_dim=cfg['latent_dim'],
        hidden_dims=cfg['hidden_dims'],
        likelihood=cfg.get('likelihood', 'nb'),
        dropout=cfg.get('dropout', 0.1),
        use_layer_norm=use_ln,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train(False)
    return model


def evaluate_one(checkpoint_path, attacker, adata, forget_idx, matched_neg_idx, retain_idx):
    model = load_retrain_model(checkpoint_path)
    return evaluate_privacy(model, attacker, adata, forget_idx, matched_neg_idx, retain_idx)


# Statistical primitives (ci_to_se, cohens_d, nested_ci, welch_test_*) are
# imported from src/stats_utils.py above.


def main():
    # === Step 1: Evaluate the 4 new retrain checkpoints ===
    print("Loading data...")
    adata = sc.read_h5ad(DATA_PATH)
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    forget_idx = split['forget_indices']
    retain_idx = split['retain_indices']
    with open(MATCHED_NEG_PATH) as f:
        matched_data = json.load(f)
    matched_neg_idx = matched_data['matched_indices']

    print("Training fresh attacker on baseline...")
    baseline_model, _ = load_vae_model(BASELINE_CHECKPOINT)
    attacker = train_fresh_attacker(baseline_model, adata, forget_idx, matched_neg_idx, retain_idx)

    print("\nEvaluating retrain seeds (canonical LayerNorm, seeds 42-46)...")
    retrain_seeds = []

    for sd in [42, 43, 44, 45, 46]:
        ckpt = RETRAIN_DIR / f'seed_{sd}' / 'best_model.pt'
        out = evaluate_one(ckpt, attacker, adata, forget_idx, matched_neg_idx, retain_idx)
        seed_record = {
            'seed': sd,
            'mlp_auc': out['mlp_auc'],
            'mlp_advantage': out['mlp_advantage'],
            'auc_se': ci_to_se(out['ci_lower'], out['ci_upper']),
            'adv_se': ci_to_se(out['advantage_ci_lower'], out['advantage_ci_upper']),
            'auc_ci': [out['ci_lower'], out['ci_upper']],
            'adv_ci': [out['advantage_ci_lower'], out['advantage_ci_upper']],
            'source': str(ckpt),
        }
        retrain_seeds.append(seed_record)
        print(f"  seed={sd}: AUC={out['mlp_auc']:.4f}  adv={out['mlp_advantage']:.4f}  "
              f"sample_CI=[{out['ci_lower']:.3f}, {out['ci_upper']:.3f}]")
        # Save per-seed eval JSON
        per_seed_path = RETRAIN_DIR / f'seed_{sd}' / 'eval.json'
        with open(per_seed_path, 'w') as f:
            json.dump(seed_record, f, indent=2)

    # === Step 2: Nested CI for retrain ===
    auc_means = [s['mlp_auc'] for s in retrain_seeds]
    auc_ses = [s['auc_se'] for s in retrain_seeds]
    adv_means = [s['mlp_advantage'] for s in retrain_seeds]
    adv_ses = [s['adv_se'] for s in retrain_seeds]

    retrain_nested = {
        'mlp_auc': nested_ci(auc_means, auc_ses),
        'mlp_advantage': nested_ci(adv_means, adv_ses),
    }

    # === Step 3: Nested CIs for the methods (using cached per-seed JSONs) ===
    method_nested = {}
    for method in ['extragradient', 'retain_finetune', 'gradient_ascent']:
        method_dir = EVAL_DIR / method
        if not method_dir.exists():
            continue
        seed_files = sorted(method_dir.glob('seed*.json'))
        if not seed_files:
            continue
        m_auc, m_aucse, m_adv, m_advse = [], [], [], []
        per_seed_records = []
        for sf in seed_files:
            d = json.load(open(sf))
            p = d['privacy']
            m_auc.append(p['mlp_auc'])
            m_aucse.append(ci_to_se(p.get('ci_lower'), p.get('ci_upper')))
            m_adv.append(p['mlp_advantage'])
            m_advse.append(ci_to_se(p.get('advantage_ci_lower'), p.get('advantage_ci_upper')))
            per_seed_records.append({
                'seed': d['seed'],
                'mlp_auc': p['mlp_auc'],
                'mlp_advantage': p['mlp_advantage'],
                'auc_se': ci_to_se(p.get('ci_lower'), p.get('ci_upper')),
                'adv_se': ci_to_se(p.get('advantage_ci_lower'), p.get('advantage_ci_upper')),
            })
        method_nested[method] = {
            'mlp_auc': nested_ci(m_auc, m_aucse),
            'mlp_advantage': nested_ci(m_adv, m_advse),
            'per_seed': per_seed_records,
        }

    # === Step 3b: Welch's t-tests (method vs retrain) with Holm-Bonferroni ===
    retrain_advs = [s['mlp_advantage'] for s in retrain_seeds]
    r_mean = float(np.mean(retrain_advs))
    r_std = float(np.std(retrain_advs, ddof=1))
    r_n = len(retrain_advs)

    hypothesis_tests = {}

    # Methods with per-seed JSONs: use full values
    for m in sorted(method_nested.keys()):
        method_advs = [r['mlp_advantage'] for r in method_nested[m]['per_seed']]
        hypothesis_tests[m] = welch_test_vs_retrain(method_advs, retrain_advs)

    # Methods from consolidated_method_comparison.json: summary stats only
    consolidated_path = BASE_DIR / 'outputs' / 'consolidated_method_comparison.json'
    if consolidated_path.exists():
        cons = json.load(open(consolidated_path))
        # AUC_std in the table is the across-seed standard deviation.
        # Convert to advantage_std: for AUC i, advantage = 2|AUC - 0.5|. For seeds
        # where AUC is all on the same side of 0.5 (which is the case for all
        # methods here), advantage_std = 2 * AUC_std.
        for row in cons.get('table', []):
            if row.get('Category') not in ('existing', 'new'):
                continue
            # Normalize method name for matching
            name_map = {
                'Extra-gradient': 'extragradient',
                'Retain-only FT': 'retain_finetune',
                'Gradient ascent': 'gradient_ascent',
                'Fisher scrubbing': 'fisher_scrubbing',
                'SSD': 'ssd',
                'Contrastive latent': 'contrastive_latent',
                'SCRUB': 'scrub',
                'DP-SGD (eps=10)': 'dp_sgd_eps10',
            }
            key = name_map.get(row['Method'], row['Method'].lower().replace(' ', '_'))
            if key in hypothesis_tests:
                continue  # already have per-seed test
            m_mean = row['Advantage']
            m_std = 2.0 * row['AUC_std']  # AUCs are on one side of 0.5
            m_n = row['Seeds']
            if m_n < 2:
                continue  # can't do hypothesis test with n=1
            test = welch_test_from_stats(m_mean, m_std, m_n, r_mean, r_std, r_n)
            test['method_display_name'] = row['Method']
            test['auc_mean'] = row['AUC']
            test['auc_std'] = row['AUC_std']
            test['advantage_mean'] = row['Advantage']
            hypothesis_tests[key] = test

    # Holm-Bonferroni correction across ALL methods tested
    method_keys = sorted(hypothesis_tests.keys())
    p_values = [hypothesis_tests[m]['p_one_sided'] for m in method_keys]
    if p_values:
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        for m, r, p in zip(method_keys, reject, p_adj):
            hypothesis_tests[m]['p_holm'] = float(p)
            hypothesis_tests[m]['reject_holm'] = bool(r)

    # === Step 4: Save ===
    out = {
        'method_description': ('Nested bootstrap CI: outer resamples seeds (10000 iterations), '
                               'inner draws from Normal(seed_mean, seed_sample_SE). '
                               'Hypothesis tests: one-sided Welch t-test vs retrain on per-seed advantages, '
                               "Cohen's d effect size, Holm-Bonferroni correction across methods."),
        'retrain': {
            'per_seed': retrain_seeds,
            'nested': retrain_nested,
        },
        'methods': method_nested,
        'hypothesis_tests': hypothesis_tests,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved nested CIs to {OUTPUT_PATH}")

    # Print summary
    print("\n=== Retrain (5 seeds) ===")
    print(f"  AUC: mean={retrain_nested['mlp_auc']['mean']:.4f}  "
          f"nested CI=[{retrain_nested['mlp_auc']['nested_ci_low']:.4f}, {retrain_nested['mlp_auc']['nested_ci_high']:.4f}]  "
          f"naive t=[{retrain_nested['mlp_auc']['naive_t_low']:.4f}, {retrain_nested['mlp_auc']['naive_t_high']:.4f}]")
    print(f"  Adv: mean={retrain_nested['mlp_advantage']['mean']:.4f}  "
          f"nested CI=[{retrain_nested['mlp_advantage']['nested_ci_low']:.4f}, {retrain_nested['mlp_advantage']['nested_ci_high']:.4f}]")

    print("\n=== Methods (nested CIs) ===")
    for m, d in method_nested.items():
        n = d['mlp_advantage']['n_seeds']
        print(f"  {m} ({n} seeds):")
        print(f"    AUC: mean={d['mlp_auc']['mean']:.4f}  "
              f"nested CI=[{d['mlp_auc']['nested_ci_low']:.4f}, {d['mlp_auc']['nested_ci_high']:.4f}]")
        print(f"    Adv: mean={d['mlp_advantage']['mean']:.4f}  "
              f"nested CI=[{d['mlp_advantage']['nested_ci_low']:.4f}, {d['mlp_advantage']['nested_ci_high']:.4f}]")

    print("\n=== Hypothesis tests (method vs retrain, one-sided Welch) ===")
    for m in sorted(hypothesis_tests.keys()):
        h = hypothesis_tests[m]
        reject_str = 'REJECT H0' if h.get('reject_holm') else 'fail to reject'
        print(f"  {m}: t={h['t_stat']:.3f} df={h['df']:.1f} "
              f"p_raw={h['p_one_sided']:.4f} p_holm={h.get('p_holm', float('nan')):.4f} "
              f"d={h['cohens_d']:.2f} [{reject_str}]")


if __name__ == '__main__':
    main()
