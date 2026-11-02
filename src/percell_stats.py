"""Per-cell refit statistic for the within-type question, with its resampling machinery.

The statistic (PBMC-68k, `scripts/score_pbmc68k.py`): for one checkpoint, the forget cells (label 1) and
the held-out cells of the same type (label 0) are scored by a logistic regression on standardized
features, out of fold, stratified 5-fold, and the AUC is averaged over ``N_SPLITS`` fold splits.

Everything here resamples that same statistic, which the 2026-09-21 code did not do (round-7 review):
- ``boot_auc``: cell bootstrap of the mean-over-splits AUC (the old interval bootstrapped fold split 0
  alone, so its centre could sit 0.05 from the point estimate). Forget cells and negatives are resampled
  within their groups through multinomial count weights, so one draw can be applied to every checkpoint
  and the comparison of two checkpoints is paired on the shared cells.
- ``perm_null``: label-permutation null of the statistic on a checkpoint's own features, refitting the
  whole out-of-fold procedure per permutation. It replaces the pure-noise-feature null of
  `percell_refit_null.json`, which no script wrote.
- ``paired_nested_diff``: difference between a method arm and the clean-retrain arm, outer bootstrap over
  seeds within each arm, inner bootstrap over cells shared by every checkpoint in a draw.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS = 20
N_FOLDS = 5


def oof_splits(x, y, n_splits=N_SPLITS):
    """Out-of-fold probabilities, one row per fold split (scaler fit on the training folds only)."""
    x = np.asarray(x); y = np.asarray(y)
    out = np.zeros((n_splits, len(y)))
    for split_seed in range(n_splits):
        for tr, te in StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed).split(x, y):
            sc = StandardScaler().fit(x[tr])
            out[split_seed, te] = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(x[tr]), y[tr]).predict_proba(sc.transform(x[te]))[:, 1]
    return out


def auc_mean(oof, y):
    """The statistic: AUC per fold split, averaged."""
    return float(np.mean([roc_auc_score(y, o) for o in oof]))


def boot_weights(n_pos, n_neg, n_boot=10000, seed=0):
    """Multinomial count weights for a stratified cell bootstrap (both group sizes kept)."""
    rng = np.random.default_rng(seed)
    return (rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot).astype(np.float64),
            rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot).astype(np.float64))


def boot_auc(oof, y, wp, wn):
    """Bootstrap draws of the mean-over-splits AUC. ``y`` must list the positives first.

    For one split the resampled AUC is wp C wn / (n_pos n_neg), with C[i, j] = 1 if positive i outscores
    negative j and 0.5 on a tie; this is the Mann-Whitney AUC of the resampled cells, exactly."""
    y = np.asarray(y); n_pos = int(y.sum()); assert (y[:n_pos] == 1).all() and (y[n_pos:] == 0).all()
    n_neg = len(y) - n_pos; draws = np.zeros(wp.shape[0])
    for o in np.atleast_2d(oof):
        p, q = o[:n_pos, None], o[None, n_pos:]
        c = (p > q).astype(np.float64) + 0.5 * (p == q)
        draws += np.einsum('bi,ij,bj->b', wp, c, wn, optimize=True) / (n_pos * n_neg)
    return draws / len(np.atleast_2d(oof))


def _perm_stat(x, y, seed, n_splits):
    yp = np.random.default_rng(seed).permutation(y)
    return auc_mean(oof_splits(x, yp, n_splits), yp)


def perm_null(x, y, n_perm=1000, seed=20260922, n_splits=N_SPLITS, n_jobs=-1):
    """Label-permutation null of the statistic on these features. Permutation p uses seed [seed, p], so
    every checkpoint sees the same label permutations. Returns the null draws."""
    x = np.asarray(x); y = np.asarray(y)
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_perm_stat)(x, y, [seed, p], n_splits) for p in range(n_perm)))


def null_summary(null, observed):
    return {'n_perm': int(len(null)), 'mean': float(null.mean()), 'p025': float(np.percentile(null, 2.5)),
            'p975': float(np.percentile(null, 97.5)), 'p_value': float((1 + (null >= observed).sum()) / (1 + len(null)))}


def nested_arm(draws, seed=0):
    """Nested draws of an arm's mean: ``draws`` is (seeds, B) cell-bootstrap draws on shared cells; each
    draw b resamples the seeds with replacement and averages their b-th cell draw."""
    draws = np.atleast_2d(draws); s, b = draws.shape
    if s == 1:
        return draws[0]
    idx = np.random.default_rng(seed).integers(0, s, (b, s))
    return draws[idx, np.arange(b)[:, None]].mean(1)


def paired_nested_diff(method_draws, ref_draws, seed=0):
    """Method arm minus reference arm, per draw. The cell draw b is shared by both arms (paired); the
    seeds are resampled independently within each arm."""
    return nested_arm(method_draws, seed=seed) - nested_arm(ref_draws, seed=seed + 1)
