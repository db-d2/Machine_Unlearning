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
                'frozen_multi_lambda5', 'frozen_multi_lambda10', 'fisher_step3e-3', 'fisher_step1e-2']
# fisher_step_ablation holds two step caps at one seed each; pooled, its mean describes no model (round-7 review),
# so each cap is also a row of its own (nested_ci_all.json carries both since 2026-09-22)
STEP_SPLIT = {'fisher_step3e-3': 'step3e-3', 'fisher_step1e-2': 'step1e-2'}


def pbmc_persample(method: str) -> List[dict]:
    """Per-seed records ``{seed, auc, scores, labels}`` for one method."""
    if method in STEP_SPLIT:
        return [r for r in _load(PERSAMPLE_DIR / 'fisher_step_ablation.json') if r['seed'].startswith(STEP_SPLIT[method] + '_')]
    return _load(PERSAMPLE_DIR / f'{method}.json')


def pbmc_step_split(records: dict, method: str):
    """The entry of one step cap from a per-seed dict keyed like ``step1e-2_seed42`` (audit, utility files)."""
    return records[f'{STEP_SPLIT[method]}_seed42']


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


def pbmc_paired(method: str = None) -> dict:
    """Paired non-inferiority test of the PBMC-33k transferred attacker against the clean-retrain arm on the shared
    cells (``scripts/paired_diff_33k.py`` -> ``paired_diff_33k.json``). The criterion since 2026-09-25; it replaces
    the 0.182 bar of :func:`pbmc_success_threshold`. Returns one arm, or the whole file when ``method`` is None."""
    d = _load(OUT / 'p4' / 'multiseed' / 'paired_diff_33k.json')
    return d if method is None else d['arms'][method]


def pbmc_success_threshold() -> float:
    """Upper bound of the retrain advantage nested 95% CI under independent cell draws per seed (0.182). SUPERSEDED as
    a criterion on 2026-09-25 by :func:`pbmc_paired` (shared cells give 0.269, and a bar of this kind passes any
    method the data cannot separate); kept for the historical record."""
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
        if m not in STEP_SPLIT and not (PERSAMPLE_DIR / f'{m}.json').exists():
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


# ---------------------------------------------------------------------------
# PBMC-68k (second scRNA-seq cohort, two rare-cluster forget sets; notebook 44)
# ---------------------------------------------------------------------------

PBMC68K_DIR = OUT / 'pbmc68k'
PBMC68K_FORGET_SETS = ['mk', 'pdc']
PBMC68K_METHODS = ['baseline', 'retrain', 'retain_finetune', 'gradient_ascent', 'ssd', 'scrub',
                   'contrastive', 'fisher', 'extragradient', 'long_finetune']


def pbmc68k_results(forget_set: str) -> dict:
    """Full ``results_<fs>.json`` of ``scripts/score_pbmc68k.py`` (one CPU attacker pass per negative set)."""
    return _load(PBMC68K_DIR / f'results_{forget_set}.json')


def pbmc68k_split(forget_set: str) -> dict:
    return _load(PBMC68K_DIR / f'split_{forget_set}.json')


def pbmc68k_n_matched(forget_set: str) -> int:
    return _load(PBMC68K_DIR / f'matched_negatives_{forget_set}.json')['n_matched']


def pbmc68k_summary(forget_set: str, kind: str = 'cross') -> Dict[str, dict]:
    """One row per method under the ``cross`` or ``within`` attacker, in the shape of
    :func:`pbmc_summary` (``auc``, ``mean``, ``ci_low``, ``ci_high``, ``n_seeds``) plus the
    nested difference from the clean retrains (``diff_ci``), the audit AUC against clean retrain
    seed 42 (mean over the method's seeds), the forget-cell negative ELBO, and utility means."""
    d = pbmc68k_results(forget_set)
    s = d['summary'][kind]
    rows = {}
    for m in PBMC68K_METHODS:
        if m not in d['per_checkpoint']:
            continue
        per = d['per_checkpoint'][m]['per_seed']
        rec = s['retrain'] if m == 'retrain' else s['methods'][m]
        rows[m] = {'auc': float(np.mean([r[f'attacker_{kind}'] for r in per])),
                   'mean': rec['mean'], 'ci_low': rec['nested_ci_low'], 'ci_high': rec['nested_ci_high'],
                   'n_seeds': rec['n_seeds'], 'diff_ci': rec.get('diff_ci'),
                   'p_method_gt_retrain': rec.get('p_method_gt_retrain'),
                   'audit': float(np.mean([r[f'audit_{kind}'] for r in per])),
                   'audit_per_seed': [r[f'audit_{kind}'] for r in per],
                   'forget_neg_elbo': float(np.mean([r['forget_neg_elbo'] for r in per])),
                   'within_neg_elbo': float(np.mean([r['within_neg_elbo'] for r in per])),
                   'cross_neg_elbo': float(np.mean([r['cross_neg_elbo'] for r in per])),
                   'elbo': float(np.mean([r['elbo'] for r in per])),
                   'marker_r': float(np.mean([r['marker_r'] for r in per])),
                   'ari': float(np.mean([r['ari'] for r in per]))}
    return rows


def pbmc68k_threshold(forget_set: str, kind: str = 'cross') -> float:
    """Upper bound of the clean retrains' nested 95% CI under the given attacker."""
    return pbmc68k_results(forget_set)['summary'][kind]['threshold']


def pbmc68k_audit_band(forget_set: str, kind: str = 'cross') -> tuple:
    """Audit AUC range of clean retrain seeds 43 and 44 scored against seed 42."""
    b = pbmc68k_results(forget_set)['summary'][kind]['audit_range_retrained']
    return (min(b), max(b))


def pbmc68k_null_floor(forget_set: str, kind: str = 'cross') -> Dict[str, float]:
    rec = pbmc68k_results(forget_set)['summary'][kind]['null_floor']
    return {'mean': rec['mean'], 'ci_low': rec['ci95'][0], 'ci_high': rec['ci95'][1]}


def pbmc68k_refit(forget_set: str, kind: str = 'within') -> Dict[str, dict]:
    """Out-of-fold refit attacker (per checkpoint) for the methods that have one: mean and sd over seeds of the split-mean AUC."""
    d = pbmc68k_results(forget_set)
    rows = {}
    for m, v in d['per_checkpoint'].items():
        per = [r[f'refit_{kind}'] for r in v['per_seed'] if f'refit_{kind}' in r]
        if per:
            rows[m] = {'cv_auc_mean': float(np.mean([r['cv_auc_mean'] for r in per])), 'cv_auc_per_seed': [r['cv_auc_mean'] for r in per],
                       'cv_auc_split_sd': float(np.mean([r['cv_auc_sd'] for r in per])), 'n_seeds': len(per)}
    return rows


def pbmc68k_long_finetune() -> dict:
    """Notebook 44 long retain-only fine-tune on PBMC-68k cluster 9 (three seeds, 200 epochs)."""
    return _load(PBMC68K_DIR / 'long_finetune.json')


# ---------------------------------------------------------------------------
# Attacker-conditioning checks (2026-09-19): standardized refit and feature groups
# ---------------------------------------------------------------------------

def within_type_standardized() -> dict:
    """Out-of-fold refits (logistic on standardized features, the MLP on standardized and on raw features)
    on the within-type controls of PBMC-33k cluster 7, Tabula Muris cluster 28, TCGA Basal, and on PBMC-33k
    cluster 13 against its cross-type negatives (``scripts/check_within_type_standardized.py``)."""
    return _load(OUT / 'within_type_standardized_check.json')['results']


def feature_groups() -> dict:
    """Out-of-fold logistic AUC by feature group (``scripts/check_feature_groups.py``)."""
    return _load(OUT / 'feature_group_check.json')


def pbmc68k_refit_std(forget_set: str, kind: str = 'within') -> Dict[str, dict]:
    """Per-cell instrument on PBMC-68k: logistic regression on standardized features, out of fold, for every
    checkpoint; mean over seeds of the split-mean AUC and advantage, with the per-seed values."""
    d = pbmc68k_results(forget_set)
    rows = {}
    for m, v in d['per_checkpoint'].items():
        per = [r[f'refit_std_{kind}'] for r in v['per_seed'] if f'refit_std_{kind}' in r]
        if per:
            rows[m] = {'auc': float(np.mean([r['cv_auc_mean'] for r in per])), 'auc_per_seed': [r['cv_auc_mean'] for r in per],
                       'advantage': float(np.mean([r['cv_advantage_mean'] for r in per])), 'advantage_per_seed': [r['cv_advantage_mean'] for r in per],
                       'split_sd': float(np.mean([r['cv_auc_sd'] for r in per])), 'boot_ci_per_seed': [r['advantage_boot_ci'] for r in per], 'n_seeds': len(per)}
    return rows


# ---------------------------------------------------------------------------
# Reference-independent audit ranges, per-cell medians, per-cell nulls (2026-09-21 verification)
# ---------------------------------------------------------------------------

def audit_band_symmetric(cohort: str) -> dict:
    """All ordered pairs of clean retrains scored against one another (``scripts/`` inline, 2026-09-21).
    ``cohort`` in {'pbmc33k', 'pbmc68k_mk', 'pbmc68k_pdc'}; keys pairs, band_by_reference, all_pairs_range, forget_fit_per_seed."""
    return _load(OUT / 'audit_band_symmetric.json')[cohort]


def audit_all_pairs_range(cohort: str) -> tuple:
    r = audit_band_symmetric(cohort)['all_pairs_range']
    return (r[0], r[1])


def forget_fit_quantiles(cohort: str) -> dict:
    """Per-checkpoint forget-fit mean, median and 95th percentile, and the retrains' medians, from
    ``outputs/audit_lira.json`` (``scripts/audit_lira.py``). ``cohort`` as in :func:`audit_band_symmetric`."""
    d = _load(OUT / 'audit_lira.json')['cohorts'][cohort]
    return {'retrain_median': d['retrain_forget_median'], 'models': {m: [{k: r[k] for k in ('seed', 'forget_mean', 'forget_median', 'forget_p95', 'matched_mean')} for r in rows] for m, rows in d['models'].items()}}


def percell_refit_null(forget_set: str) -> dict:
    """Label-permutation null of the per-cell refit on each checkpoint's own features (v6 scores, 1,000
    permutations each). Replaces `percell_refit_null.json`, a pure-noise-feature null that no script wrote
    and the text called a permutation null (round-7 B3). The null states the instrument's resolution; it is
    never an acceptance threshold. Returns the per-checkpoint summaries and the range of their 97.5th percentiles."""
    per = {f'{m}/{r["seed"]}': r['refit_std_within']['perm_null'] for m, v in pbmc68k_results(forget_set)['per_checkpoint'].items() for r in v['per_seed']}
    p975 = [v['p975'] for v in per.values()]
    return {'per_checkpoint': per, 'p975_range': [min(p975), max(p975)]}


def pbmc68k_percell_paired(forget_set: str, kind: str = 'within') -> dict:
    """Paired comparison of the per-cell refit with the clean-retrain arm on the shared cells, with the
    non-inferiority margin and verdicts (``scripts/percell_paired_test.py`` -> ``percell_paired_<fs>.json``)."""
    return _load(OUT / 'pbmc68k' / f'percell_paired_{forget_set}.json')['kinds'][kind]


UTILITY_INTERVALS_PATH = OUT / 'utility_intervals.json'


def utility_intervals(cohort: str = 'pbmc33k') -> dict:
    """Held-out utility with intervals and paired costs against the baseline (scripts/utility_intervals.py).

    ``cohort`` is ``pbmc33k``, ``pbmc68k_mk`` or ``pbmc68k_pdc``. Returns the cohort block: ``n_heldout``,
    ``baseline_elbo``, ``margin_elbo`` (1 percent of the baseline's held-out negative ELBO), ``margin_marker``
    (0.01) and ``arms`` keyed by method, each with ``elbo``, ``elbo_ci``, ``marker_r``, ``marker_r_ci``, ``ari`` and,
    for every arm but the baseline, ``elbo_cost``, ``elbo_cost_ci``, ``elbo_verdict``, ``marker_cost``,
    ``marker_cost_ci``, ``marker_verdict`` (no cost / costs utility / not resolved). Deterministic held-out ELBO
    (``likelihood_audit.neg_elbo_per_cell``), one multinomial cell draw shared by every checkpoint, seeds resampled
    within an arm. The pooled Fisher step ablation is split into ``fisher_step3e-3`` and ``fisher_step1e-2``."""
    return _load(UTILITY_INTERVALS_PATH)['cohorts'][cohort]


def utility_verdict(arm: dict) -> str:
    """One label from the ELBO and marker verdicts of a ``utility_intervals`` arm."""
    e, m = arm.get('elbo_verdict'), arm.get('marker_verdict')
    if e is None:
        return 'reference'
    if e == 'no cost' and m == 'no cost':
        return 'no cost'
    hit = [k for k, v in (('ELBO', e), ('marker', m)) if v == 'costs utility']
    return 'costs utility (' + ' and '.join(hit) + ')' if hit else 'not resolved'


AUDIT_REFERENCE_FREE_PATH = OUT / 'audit_reference_free.json'


def audit_reference_free(cohort: str = 'pbmc33k') -> dict:
    """Reference-free likelihood audit (scripts/audit_reference_free.py): reference = mean over every clean retrain,
    retrain arm leave-one-out, intervals over target seeds, reference retrains and shared cells. Cohort block with
    ``retrain`` (``audit_cross``, ``audit_cross_z``, and on PBMC-68k ``audit_within``, ``audit_within_z``, each with
    ``loo_per_seed``, ``arm_mean``, ``arm_ci``; ``forget_median`` / ``forget_mean``) and ``arms`` keyed by method with
    the same audit keys (``mean``, ``ci``, ``diff_vs_retrain``, ``diff_ci``, ``verdict``), ``gap_median`` / ``gap_mean``
    (target minus retrain arm on the forget cells; positive = fitted worse) and ``gap_rel_median`` / ``gap_rel_mean``
    (the forget gap minus the cross-type negatives' gap). The pooled Fisher step ablation is split by cap."""
    return _load(AUDIT_REFERENCE_FREE_PATH)['cohorts'][cohort]


def stopping_probe(forget_set: str) -> dict:
    """Retrain-free stopping probe on PBMC-68k (scripts/stopping_probe_68k.py): the Fisher cap and fine-tune length
    grids of `stopping_probe_68k_train.py`, each arm with the paired per-cell verdict against the eight-retrain arm
    (``refit_std_within``), the cross-type audit, the forget-cell median against the retrains' medians
    (``type_verdict_median``) and the controller-observable fits (``forget_median``, ``lookalike_mean``,
    ``retain_sample_mean``, ``cross_mean``, ``heldout_elbo``, ``marker_r``)."""
    return _load(OUT / 'pbmc68k' / f'stopping_probe_{forget_set}.json')


def stopping_probe80(forget_set: str) -> dict:
    """Full stopping probe on PBMC-68k (scripts/stopping_probe_80_20_score.py): 20 percent of the forget cells held back
    as known members, 80 percent deleted, eight retrains on the new retain set. Per arm: ``refit_within`` (deleted cells
    against held-out cells of the type, paired verdict against the retrains), ``refit_heldback`` (the observable
    calibration attacker, deleted cells against held-back members), ``refit_members_vs_within``, ``audit_cross`` and
    ``gap_median`` (reference-free), ``gap_deleted_minus_members`` and the observable fits."""
    return _load(OUT / 'pbmc68k' / f'stopping_probe80_{forget_set}.json')


def retain_null_mass() -> dict:
    """Retain-null mass and survival of the forget Fisher mass under retain fine-tuning (scripts/retain_null_mass.py):
    per setting (pbmc33k, pbmc68k_mk, pbmc68k_pdc, classifier_linear) and parameter group, ``survival`` S(t) at
    t = 1, 10, 100, 1000, ``retain_null_mass`` N(tau) at tau = 1e-3, 1e-2, 1e-1, the cosine, and for the VAEs the
    decoder-output column shares."""
    return _load(OUT / 'p6' / 'retain_null_mass.json')


def restricted_scrub_test() -> dict:
    """The restoration intervention (scripts/restricted_scrub_test.py): unrestricted, active-only and null-only Fisher
    edits of matched damage, measured after the edit and after 10 and 50 retain epochs (tags ``edits``, ``edits_x2``).
    Tags ``edits_x2_opt`` and ``edits_opt`` (2026-10-13) hold the optimizer arms from the same saved edit: per edit
    ``finetune/<name:lr>`` with ``after_ft{10,50,200}`` and, under ``edit``, ``rho`` (share of the edit vector left,
    overall / ``null`` / ``active`` / per layer group), ``pred_rho`` (the proposition's gradient-descent prediction, SGD
    arms only), ``orth``, ``fisher_survival``; the ``none`` edit's arms carry ``drift_projection`` onto every saved edit,
    which ``make_tex_tables.t_restricted_scrub_rho`` subtracts from ``rho``."""
    return _load(OUT / 'p6' / 'restricted_scrub_test.json')


def edit_compensation() -> dict:
    """Linear response of the retain fine-tune to the saved edits of the restoration intervention
    (scripts/edit_compensation_test.py, edit_compensation_states.py, edit_lanczos_prediction.py; 2026-10-16). Per cohort:
    ``edits/<tag>/<edit>`` with the retain and forget quadratic forms along the edit (``diag_form``, ``fisher_form``,
    ``hess_form`` per finite-difference step), the no-edit fine-tune's displacement under ``none``, and per optimizer
    arm and epoch the superposition checks (``sup_rho`` = the no-edit state plus the surviving share of the edit,
    ``sup_full``, ``rev_only``, the shares of the damage ``share_meas`` / ``share_sup_rho`` / ``share_rev_only``);
    ``states/<tag>/<edit>/epochs/<e>`` the saved SGD states' path decomposition (``rho``, ``path_diff_sq`` by group,
    ``fit_residual`` = the residual r - Delta alone at the baseline, ``effective`` = |delta W a(x) + delta b|^2 per
    edited layer); ``lanczos/<tag>/<edit>`` the quadratic model's own prediction (I - eta H_R)^T delta with the full
    retain curvature: ``lanczos`` with the empirical Fisher and ``lanczos_gn`` with the generalized Gauss-Newton matrix
    (``rho_pred`` = the m-point Gauss rule, a lower bound; ``rho_pred_upper`` = the Gauss-Radau rule with a node at 0, an
    upper bound; ``convergence/<m>`` both rules at every truncation; ``fit_residual_pred`` / ``damage_share_pred`` the
    predicted residual at the baseline; ``fit_superposed_pred`` / ``share_sup_pred`` the predicted residual added to the
    saved no-edit state, the model's prediction of the measured excess; ``radau`` the same for the Radau residual;
    ``alphas`` / ``betas`` / ``beta_next`` the recurrence; m = 120 since 2026-10-30, 40 before)."""
    return _load(OUT / 'p6' / 'edit_compensation.json')


def sgd_survival_prediction() -> dict:
    """Proposition 1's survival S(t) at the gradient-descent arms' actual budget (scripts/sgd_survival_prediction.py,
    2026-10-13): per cohort under ``settings`` the mean and maximum retain Fisher, batches per epoch, the retain-null mass,
    and per epoch (10, 50, 200 at step 1e-3) the proposition's time ``t`` = 2 eta T mean(F_R) with the survival overall /
    retain-active / retain-null and a curvature-calibrated variant. Cited since 2026-10-30 for the conversion between the
    methods' epochs and the proposition's units (ten epochs are t = 0.065 on PBMC-33k, 0.12 on cluster 9)."""
    return _load(OUT / 'p6' / 'sgd_survival_prediction.json')


def biology_facts() -> dict:
    return _load(OUT / 'biology_facts.json')['cohorts']


def proposition_test() -> dict:
    """Proposition 1 and Corollary 1 tested on PBMC-33k cluster 13 and both PBMC-68k forget sets (``scripts/proposition_test.py``).

    Per cohort: the measured cosine of the decoder output weight matrix's forget and retain Fisher, the two factors from the
    Fisher marginals and from a direct forward pass, the rank-one share, the corollary's shared-support bound, and the profile vectors.
    """
    return _load(ROOT / 'outputs/p6/proposition_test.json')
