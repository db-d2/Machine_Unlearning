"""Reusable statistical helpers for CI computation and hypothesis testing.

These are generic primitives used by notebooks and scripts.

Functions:
    ci_to_se          - convert a 95% CI to a standard error via normal approx
    cohens_d          - Cohen's d for two independent samples (pooled SD)
    nested_ci         - nested bootstrap aggregation of per-seed (mean, SE) pairs
    welch_test_vs_retrain - one-sided Welch's t-test from raw per-seed values
    welch_test_from_stats - one-sided Welch's t-test from summary statistics
"""
from typing import Sequence, Optional

import numpy as np
from scipy import stats


def ci_to_se(ci_low: Optional[float], ci_high: Optional[float]) -> float:
    """Convert a 95% confidence interval to a standard error.

    Assumes the CI is approximately symmetric and Normal. Returns 0 if either
    endpoint is None (signals absence of CI data).
    """
    if ci_low is None or ci_high is None:
        return 0.0
    return (ci_high - ci_low) / (2.0 * 1.96)


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d for two independent samples using pooled standard deviation.

    Returns NaN if either sample has fewer than 2 observations. Returns +/- inf
    if pooled SD is zero but the means differ.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    pooled_sd = float(np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)))
    if pooled_sd == 0:
        return float('inf') if np.mean(x) != np.mean(y) else 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def nested_ci(means: Sequence[float],
              ses: Sequence[float],
              n_boot: int = 10000,
              seed: int = 42) -> dict:
    """Nested bootstrap aggregation of per-seed means with within-seed SEs.

    Outer resampling: draw seeds with replacement.
    Inner draw:       for each drawn seed, sample an observation from
                      Normal(seed_mean, seed_SE).
    Returns mean, nested 95% CI, and naive t-CI for comparison.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(means)
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        drawn_means = means[idx]
        drawn_ses = ses[idx]
        sampled = rng.normal(drawn_means, drawn_ses)
        boot_means[i] = float(np.mean(sampled))
    result = {
        'mean': float(np.mean(means)),
        'nested_ci_low': float(np.percentile(boot_means, 2.5)),
        'nested_ci_high': float(np.percentile(boot_means, 97.5)),
        'n_seeds': n_seeds,
    }
    if n_seeds > 1:
        se_seeds = float(np.std(means, ddof=1) / np.sqrt(n_seeds))
        t_crit = stats.t.ppf(0.975, n_seeds - 1)
        result['naive_t_low'] = float(np.mean(means) - t_crit * se_seeds)
        result['naive_t_high'] = float(np.mean(means) + t_crit * se_seeds)
    else:
        result['naive_t_low'] = None
        result['naive_t_high'] = None
    return result


def welch_test_vs_retrain(method_values: Sequence[float],
                          retrain_values: Sequence[float]) -> dict:
    """One-sided Welch's t-test: H0 method = retrain, H1 method > retrain.

    Uses raw per-seed values for both method and retrain. Returns t-statistic,
    Welch-Satterthwaite degrees of freedom, one-sided p-value, Cohen's d, and
    bookkeeping metadata.
    """
    method_values = np.asarray(method_values)
    retrain_values = np.asarray(retrain_values)
    t_stat, p_two_sided = stats.ttest_ind(method_values, retrain_values, equal_var=False)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    nx, ny = len(method_values), len(retrain_values)
    var_x = float(np.var(method_values, ddof=1))
    var_y = float(np.var(retrain_values, ddof=1))
    df_num = (var_x / nx + var_y / ny) ** 2
    df_den = (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
    df_welch = df_num / df_den if df_den > 0 else float('nan')
    return {
        't_stat': float(t_stat),
        'df': float(df_welch),
        'p_one_sided': float(p_one_sided),
        'cohens_d': cohens_d(method_values, retrain_values),
        'n_method': int(nx),
        'n_retrain': int(ny),
        'mean_diff': float(np.mean(method_values) - np.mean(retrain_values)),
        'source': 'per_seed_values',
    }


def welch_test_from_stats(m_mean: float, m_std: float, m_n: int,
                          r_mean: float, r_std: float, r_n: int) -> dict:
    """One-sided Welch's t-test from summary statistics only.

    Used when per-seed values are unavailable. Cohen's d is computed from
    (means, stds) via pooled SD.
    """
    t_stat, p_two_sided = stats.ttest_ind_from_stats(
        mean1=m_mean, std1=m_std, nobs1=m_n,
        mean2=r_mean, std2=r_std, nobs2=r_n,
        equal_var=False,
    )
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    var_x, var_y = m_std ** 2, r_std ** 2
    df_num = (var_x / m_n + var_y / r_n) ** 2
    df_den = (var_x / m_n) ** 2 / max(m_n - 1, 1) + (var_y / r_n) ** 2 / max(r_n - 1, 1)
    df_welch = df_num / df_den if df_den > 0 else float('nan')
    pooled_sd = float(np.sqrt(((m_n - 1) * var_x + (r_n - 1) * var_y) / max(m_n + r_n - 2, 1)))
    d = (m_mean - r_mean) / pooled_sd if pooled_sd > 0 else float('inf')
    return {
        't_stat': float(t_stat),
        'df': float(df_welch),
        'p_one_sided': float(p_one_sided),
        'cohens_d': float(d),
        'n_method': int(m_n),
        'n_retrain': int(r_n),
        'mean_diff': float(m_mean - r_mean),
        'source': 'summary_stats',
    }
