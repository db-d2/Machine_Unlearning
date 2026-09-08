"""Canonical result loaders.

Every number quoted in a notebook summary cell, a figure script, or the paper must
trace to a JSON file under ``outputs/``. Notebooks and scripts import from this
module instead of typing values in, so a re-scored result propagates everywhere on
the next run instead of going stale in a dict literal.

Regimes
-------
* ``pbmc_*`` (no suffix): the CPU one-attacker re-score of 2026-07-05, the paper's
  canonical PBMC regime. Sources are ``outputs/p4/multiseed/persample/*.json``
  (per-sample attacker scores, one fresh attacker trained on the baseline model and
  applied to every checkpoint) and ``outputs/p4/multiseed/nested_ci_all.json``
  (nested bootstrap CIs, hypothesis tests, permutation null floor).
* ``pbmc_single_seed``: the original seed-42, MPS-era numbers from notebooks 03 to
  05 and 09 (``outputs/p2/retrain_floor.json``, ``adversarial_sweep/sweep_summary.json``,
  ``p4/ablation_results.json``). Superseded for headline claims, kept for the
  early notebooks that were built on them and for frozen-critic and lambda=5
  configurations that were never re-scored.

All paths are resolved relative to the repository root, so the loaders work from a
notebook (``notebooks/``), a script (``scripts/``), or the root.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'

PERSAMPLE_DIR = OUT / 'p4' / 'multiseed' / 'persample'
NESTED_PATH = OUT / 'p4' / 'multiseed' / 'nested_ci_all.json'
REFIT_PATH = OUT / 'p4' / 'multiseed' / 'retrain_refit.json'
PUBLICATION_PATH = OUT / 'p4' / 'multiseed' / 'publication_results.json'
CONSOLIDATED_PATH = OUT / 'consolidated_method_comparison.json'
UTILITY_SUITE_PATH = OUT / 'utility_suite_results.json'
MOON_UTILITY_PATH = OUT / 'p2' / 'moon' / 'utility.json'
MATCHED_VALIDATION_PATH = OUT / 'matched_negative_validation.json'
LARGER_CLUSTER_PATH = OUT / 'larger_cluster_eval' / 'results.json'
PBMC_FISHER_PATH = OUT / 'p6' / 'fisher_overlap_results.json'
TCGA_DIR = OUT / 'tcga_brca'
TM_DIR = OUT / 'tabula_muris'


@lru_cache(maxsize=None)
def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def advantage(auc: float) -> float:
    """Membership advantage, 2|AUC - 0.5| (Yeom et al. 2018), direction-agnostic."""
    return 2.0 * abs(auc - 0.5)


# ---------------------------------------------------------------------------
# PBMC-33k, canonical regime (CPU one-attacker re-score)
# ---------------------------------------------------------------------------

PBMC_METHODS = ['baseline', 'retrain', 'retrain_leaky', 'retain_finetune', 'gradient_ascent', 'ssd',
                'scrub', 'moon', 'contrastive', 'fisher', 'fisher_step_ablation', 'extragradient',
                'extragradient_lambda1', 'dp_sgd', 'frozen_single_lambda5', 'frozen_single_lambda10',
                'frozen_multi_lambda5', 'frozen_multi_lambda10']


def pbmc_persample(method: str) -> List[dict]:
    """Per-seed records ``{seed, auc, scores, labels}`` for one method."""
    return _load(PERSAMPLE_DIR / f'{method}.json')


def pbmc_per_seed_auc(method: str) -> List[float]:
    return [r['auc'] for r in pbmc_persample(method)]


def pbmc_auc(method: str) -> float:
    """Mean AUC across seeds under the canonical attacker."""
    return float(np.mean(pbmc_per_seed_auc(method)))


def pbmc_advantage(method: str) -> Dict[str, float]:
    """Mean per-seed advantage with the exact nested 95% CI where one exists.

    ``baseline`` has a single checkpoint and no nested CI, so only ``mean`` is set.
    """
    nested = _load(NESTED_PATH)
    if method == 'retrain':
        rec = nested['retrain']
    elif method in nested['methods']:
        rec = nested['methods'][method]
    else:
        aucs = pbmc_per_seed_auc(method)
        return {'mean': float(np.mean([advantage(a) for a in aucs])),
                'ci_low': None, 'ci_high': None, 'n_seeds': len(aucs)}
    return {'mean': rec['mean'], 'ci_low': rec['nested_ci_low'],
            'ci_high': rec['nested_ci_high'], 'n_seeds': rec['n_seeds']}


def pbmc_success_threshold() -> float:
    """Upper bound of the retrain advantage nested 95% CI (the paper's target)."""
    return _load(NESTED_PATH)['retrain']['nested_ci_high']


def pbmc_null_floor() -> Dict[str, float]:
    """Permutation null for the folded advantage estimator at n=30 vs 194."""
    rec = _load(NESTED_PATH)['advantage_null_floor']
    return {'mean': rec['mean'], 'ci_low': rec['ci95'][0], 'ci_high': rec['ci95'][1]}


def pbmc_hypothesis_test(method: str) -> dict:
    return _load(NESTED_PATH)['hypothesis_tests'][method]


def pbmc_summary() -> Dict[str, dict]:
    """One row per method: AUC, advantage, nested CI, seed count."""
    rows = {}
    for m in PBMC_METHODS:
        if not (PERSAMPLE_DIR / f'{m}.json').exists():
            continue
        adv = pbmc_advantage(m)
        rows[m] = {'auc': pbmc_auc(m), **adv}
    return rows


def pbmc_refit() -> dict:
    """Retrain-refit attacker control (out-of-fold, 20 fold splits)."""
    return _load(REFIT_PATH)


# ---------------------------------------------------------------------------
# PBMC-33k, single-seed (seed 42, MPS-era) regime from notebooks 03 to 05 and 09
# ---------------------------------------------------------------------------

def pbmc_single_seed() -> dict:
    """Seed-42 numbers behind the early notebooks. Superseded for headline claims."""
    floor = _load(OUT / 'p2' / 'retrain_floor.json')
    sweep = _load(OUT / 'p2' / 'adversarial_sweep' / 'sweep_summary.json')
    ablation = _load(OUT / 'p4' / 'ablation_results.json')
    return {
        'baseline_auc': floor['baseline_auc'],
        'retrain_auc': floor['auc_floor'],
        'retrain_ci': [floor['retrain_metrics']['auc_ci_lower'],
                       floor['retrain_metrics']['auc_ci_upper']],
        'target_band': floor['target_band'],
        'adversarial': {k: v['posthoc_auc'] for k, v in sweep['experiments'].items()},
        'fisher_structured': ablation['fisher_multiseed']['structured'],
        'fisher_scattered': ablation['fisher_multiseed']['scattered'],
    }


def pbmc_size_ablation() -> dict:
    """Fisher scattered-set size ablation (10 seeds per size)."""
    return _load(OUT / 'p4' / 'size_ablation' / 'summary.json')


def pbmc_scattered_baselines() -> dict:
    """Baseline AUC on each scattered forget set (scripts/eval_scattered_baselines.py).

    Keys are forget-set sizes as strings plus ``'35_original'``, the original scattered
    split used by notebooks 05 and 09 (the paper's scattered baseline of 0.525).
    """
    return _load(OUT / 'p4' / 'size_ablation' / 'scattered_baselines.json')


# ---------------------------------------------------------------------------
# PBMC-33k within-cluster controls and larger-cluster holdouts
# ---------------------------------------------------------------------------

def pbmc_within_cluster() -> dict:
    """NB19: forget vs the 5 unseen cluster-13 cells, and vs 194 matched negatives."""
    return _load(MATCHED_VALIDATION_PATH)


def larger_cluster() -> dict:
    """NB39: PBMC cluster 7 and Tabula Muris cluster 28 cross vs within holdouts."""
    return _load(LARGER_CLUSTER_PATH)


# ---------------------------------------------------------------------------
# Fisher overlap
# ---------------------------------------------------------------------------

def pbmc_fisher() -> dict:
    """NB26 per-sample Fisher overlap on PBMC (global, per category, gene level)."""
    d = _load(PBMC_FISHER_PATH)
    vae = d['vae']
    return {
        'global_cosine': vae['global_cosine'],
        'per_category': {k: v['cosine'] for k, v in vae['per_category'].items()},
        'per_category_full': vae['per_category'],
        'eff_rank_ratio_forget_global': vae['global_eff_rank_forget'] / vae['total_params'],
        'log_fisher_correlation': vae['log_fisher_correlation'],
        'gene_median_cosine': d['gene_analysis']['median_cosine'],
        'gene_frac_above_0.5': d['gene_analysis']['n_genes_cos_gt_0.5'] / d['gene_analysis']['n_genes_total'],
        'gene_n_above_0.5': d['gene_analysis']['n_genes_cos_gt_0.5'],
        'gene_n_total': d['gene_analysis']['n_genes_total'],
        'classifier_cosine': d['classifier']['global_cosine'],
        'metadata': d['metadata'],
    }


def tcga_fisher() -> dict:
    """NB35 Fisher overlap on TCGA-BRCA."""
    d = _load(TCGA_DIR / 'fisher_overlap_results.json')
    return {
        'global_cosine': d['global_cosine'],
        'per_category': {k: v['cosine'] for k, v in d['per_layer'].items()},
        'gene_median_cosine': d['gene_level']['median_cosine'],
        'log_fisher_correlation': d['log_fisher_correlation'],
    }


# ---------------------------------------------------------------------------
# PBMC-33k utility (held-out set)
# ---------------------------------------------------------------------------

def _mean_utility(records: Dict[str, dict], keys: List[str]) -> dict:
    """Average the utility block over the given result keys of a consolidated method."""
    fields = ['elbo', 'kl', 'marker_r', 'ari', 'silhouette']
    return {f: float(np.mean([records[k]['utility'][f] for k in keys])) for f in fields}


UTILITY_ALL_PATH = OUT / 'p4' / 'multiseed' / 'utility_all.json'


def pbmc_utility_all() -> dict:
    """One-pass CPU utility for every current checkpoint (scripts/score_utility_all.py)."""
    return _load(UTILITY_ALL_PATH)


def pbmc_utility(method: str) -> dict:
    """Held-out utility (ELBO, KL, marker r, ARI, silhouette) per method.

    If ``outputs/p4/multiseed/utility_all.json`` exists (Stage 1 re-score, 2026-09-08), it
    is the source for every method, so privacy and utility describe the same checkpoints.
    Otherwise the legacy per-notebook files below are used.

    Sources: ``utility_suite_results.json`` (NB18) for the single-checkpoint models,
    ``publication_results.json`` for the 10-seed extra-gradient mean,
    ``consolidated_method_comparison.json`` (NB25) for SSD/SCRUB/contrastive/DP-SGD
    at their canonical hyperparameter, and ``outputs/p2/moon/utility.json`` for Moon.
    """
    if UTILITY_ALL_PATH.exists():
        methods = _load(UTILITY_ALL_PATH)['methods']
        key = {'extragradient_seed42': 'extragradient'}.get(method, method)
        if key in methods:
            return dict(methods[key]['mean'])
    suite = _load(UTILITY_SUITE_PATH)['metrics']
    suite_key = {'baseline': 'Baseline', 'retrain': 'Retrain', 'fisher': 'Fisher',
                 'retain_finetune': 'Retain-FT', 'gradient_ascent': 'Grad-Ascent',
                 'extragradient_seed42': 'Extra-grad'}
    if method in suite_key:
        rec = suite[suite_key[method]]
        return {f: rec[f] for f in ['elbo', 'kl', 'marker_r', 'ari', 'silhouette']}
    if method == 'extragradient':
        rec = _load(PUBLICATION_PATH)['pbmc_utility']['extragradient']
        return {f: rec[f]['mean'] for f in ['elbo', 'kl', 'marker_r', 'ari', 'silhouette']}
    consolidated = _load(CONSOLIDATED_PATH)['new_methods']
    prefix = {'ssd': 'alpha1.0_', 'scrub': 'af1.0_', 'contrastive': 'gamma1.0_',
              'dp_sgd': 'eps10.0_'}
    if method in prefix:
        recs = consolidated[method]['results']
        keys = [k for k in recs if k.startswith(prefix[method])]
        return _mean_utility(recs, keys)
    if method == 'moon':
        return _load(MOON_UTILITY_PATH)['mean']
    raise KeyError(f'no utility source for {method!r}')


# ---------------------------------------------------------------------------
# TCGA-BRCA
# ---------------------------------------------------------------------------

def tcga_baseline() -> dict:
    """NB32 split sizes and cross-subtype baseline/retrain (89 matched negatives)."""
    return _load(TCGA_DIR / 'baseline_results.json')


def tcga_n_matched() -> int:
    return _load(TCGA_DIR / 'matched_negatives.json')['n_matched']


def tcga_cross_subtype() -> dict:
    """NB33 cross-subtype MIA AUCs for baseline, retrain, and the six methods.

    Keys use the display names shared with notebook 34b so the two matchings line up.
    """
    d = _load(TCGA_DIR / 'unlearning' / 'unlearning_results.json')
    names = {'fisher_scrubbing': 'Fisher', 'retain_finetune': 'Retain-FT',
             'gradient_ascent': 'Grad. ascent', 'ssd': 'SSD', 'scrub': 'SCRUB',
             'contrastive': 'Contrastive'}
    aucs = {'baseline': d['baseline']['auc'], 'retrain': d['retrain']['auc']}
    aucs.update({names[k]: v['auc'] for k, v in d['methods'].items()})
    return {'auc': aucs, 'n_forget': d['n_forget'], 'n_matched': tcga_n_matched()}


def tcga_within_subtype() -> dict:
    """NB34b within-subtype (30 unseen Basal) MIA results."""
    return _load(TCGA_DIR / 'within_subtype_eval' / 'within_subtype_results.json')


# ---------------------------------------------------------------------------
# PBMC cluster 12 (second structured forget set)
# ---------------------------------------------------------------------------

def c12_results() -> dict:
    """Cluster-12 scores. Prefers the Stage-1 one-attacker CPU re-score
    (``outputs/p1_c12/scores_c12_v2.json``); falls back to the July file."""
    v2 = OUT / 'p1_c12' / 'scores_c12_v2.json'
    if v2.exists():
        return _load(v2)
    return _load(OUT / 'p1_c12' / 'scores_c12_multiseed.json')


# ---------------------------------------------------------------------------
# Tabula Muris
# ---------------------------------------------------------------------------

def tm_results() -> dict:
    """NB12/13 baseline, retrain, and unlearning AUCs on Tabula Muris."""
    d = _load(TM_DIR / 'unlearning_results.json')
    s = d['baseline']['structured']
    sc = d['baseline']['scattered']
    unl = {k: v['auc'] for k, v in d['unlearning'].items()}
    # Stage 1 (2026-09-08): baseline and retrain are scored by the same attacker as the methods
    base_auc = unl.get('baseline (structured)', s['baseline_auc'])
    retr_auc = unl.get('retrain (structured)', s['retrain_floor'])
    return {
        'structured': {'baseline_auc': base_auc, 'retrain_auc': retr_auc,
                       'baseline_auc_nb12_attacker': s['baseline_auc'], 'retrain_auc_nb12_attacker': s['retrain_floor'],
                       'n_forget': s['forget_set']['size'], 'n_matched': s['matched_negatives']},
        'scattered': {'baseline_auc': sc['baseline_auc'], 'retrain_auc': sc['retrain_floor'],
                      'n_forget': sc['forget_size'], 'n_matched': sc['matched_negatives']},
        'unlearning': unl,
        'attacker_device': d.get('attacker_device'),
    }
