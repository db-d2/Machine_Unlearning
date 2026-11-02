"""Reference-free likelihood audit statistics (round-7 review: the audit is conditional on retrain seed 42
while judged against a range built from every other seed; the median switch on cluster 9 was post hoc;
the per-cell z-score of `audit_lira.json` was unused for the per-cell question).

Every statistic here takes the whole clean-retrain arm as the reference and puts the reference choice
inside the interval. A draw b resamples (i) the target arm's seeds, (ii) the retrain seeds entering the
reference, with replacement, and (iii) the cells, through multinomial count weights shared by every arm
of a cohort, so that the difference between a target arm and the retrain arm is paired on the cells and
on the reference draw. The retrain arm is scored leave-one-out inside each draw: each drawn retrain is
audited against the mean of the other drawn retrains.

Statistics, all on per-cell negative ELBO arrays (rows = seeds, columns = cells):
- ``audit``: AUC of the score reference - target, forget cells against negatives; reference = mean over
  the retrains (``zscore=False``) or (mean - target) / sd over the retrains (``zscore=True``, LiRA-style).
  Above 0.5 the target fits the forget cells better than never-seen models do (signal intact); below,
  worse (over-corrected).
- ``gap``: median (or mean) over forget cells of the target's negative ELBO minus the retrain arm's,
  the type-level reading that replaces the post-hoc median switch; positive = fitted worse than retrained.
"""
import numpy as np

EPS = 1e-6


def weighted_auc(sf, sm, wp, wn):
    """Mann-Whitney AUC of positives ``sf`` against negatives ``sm`` under count weights."""
    c = (sf[:, None] > sm[None, :]).astype(np.float64) + 0.5 * (sf[:, None] == sm[None, :])
    return float(wp @ c @ wn / (wp.sum() * wn.sum()))


def weighted_median(v, w):
    o = np.argsort(v); cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def weighted_mean(v, w):
    return float(w @ v / w.sum())


def cell_weights(n, n_boot, seed=0):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def _score(ref_rows, t, zscore):
    """Reference rows (k x n) against one target row (n,)."""
    m = ref_rows.mean(0)
    if not zscore:
        return m - t
    sd = ref_rows.std(0, ddof=1) + EPS if len(ref_rows) > 1 else np.ones_like(m)
    return (m - t) / sd


def point_audit(T_f, T_m, R_f, R_m, zscore=False):
    """Per target seed, the audit against the mean of every retrain (no resampling)."""
    w_f, w_m = np.ones(T_f.shape[1]), np.ones(T_m.shape[1])
    return [weighted_auc(_score(R_f, tf, zscore), _score(R_m, tm, zscore), w_f, w_m) for tf, tm in zip(T_f, T_m)]


def loo_audit(R_f, R_m, zscore=False):
    """Each retrain against the mean of the others (the retrained arm, reference-free)."""
    k = len(R_f); out = []
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        out.append(weighted_auc(_score(R_f[keep], R_f[j], zscore), _score(R_m[keep], R_m[j], zscore), np.ones(R_f.shape[1]), np.ones(R_m.shape[1])))
    return out


def nested_audit(T_f, T_m, R_f, R_m, wp, wn, ref_draws, target_draws, zscore=False):
    """Draws of (target arm audit, retrain arm audit). ``ref_draws`` (B x k) indexes the retrains with
    replacement; ``target_draws`` (B,) indexes the target's seeds; ``wp``/``wn`` (B x cells) are shared."""
    B = len(target_draws); k = R_f.shape[0]; out_t, out_r = np.empty(B), np.empty(B)
    for b in range(B):
        S = ref_draws[b]; t = target_draws[b]
        out_t[b] = weighted_auc(_score(R_f[S], T_f[t], zscore), _score(R_m[S], T_m[t], zscore), wp[b], wn[b])
        vals = []
        for i in range(k):
            rest = np.r_[S[:i], S[i + 1:]]
            vals.append(weighted_auc(_score(R_f[rest], R_f[S[i]], zscore), _score(R_m[rest], R_m[S[i]], zscore), wp[b], wn[b]))
        out_r[b] = np.mean(vals)
    return out_t, out_r


def point_gap(T_f, R_f, stat='median'):
    f = np.median if stat == 'median' else np.mean
    return float(np.mean([f(t) for t in T_f]) - np.mean([f(r) for r in R_f]))


def nested_gap(T_f, R_f, wp, ref_draws, target_draws, stat='median'):
    f = weighted_median if stat == 'median' else weighted_mean
    B = len(target_draws); out = np.empty(B)
    for b in range(B):
        out[b] = f(T_f[target_draws[b]], wp[b]) - np.mean([f(R_f[j], wp[b]) for j in ref_draws[b]])
    return out


def point_gap_rel(T_f, T_m, R_f, R_m, stat='median'):
    """Forget-cell gap minus the same gap on the negatives (a global shift in fit cancels)."""
    return point_gap(T_f, R_f, stat) - point_gap(T_m, R_m, stat)


def nested_gap_rel(T_f, T_m, R_f, R_m, wp, wn, ref_draws, target_draws, stat='median'):
    return nested_gap(T_f, R_f, wp, ref_draws, target_draws, stat) - nested_gap(T_m, R_m, wn, ref_draws, target_draws, stat)


def p_two_sided(d):
    """Two-sided bootstrap p-value of a paired difference: twice the smaller tail, with the +1 correction."""
    d = np.asarray(d); n = len(d)
    return float(min(1.0, 2 * min((1 + (d <= 0).sum()) / (1 + n), (1 + (d >= 0).sum()) / (1 + n))))


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def verdict_diff(lo, hi, margin=None, better='fitted better than retrained', worse='fitted worse than retrained', same='fitted as retrained', unresolved='not resolved'):
    """Four-way (round 8): above zero, below zero, inside plus or minus ``margin`` (equivalence), or neither.
    With ``margin=None`` the third outcome is 'not distinguishable from retrained' (no equivalence claim)."""
    if lo > 0:
        return better
    if hi < 0:
        return worse
    if margin is None:
        return 'not distinguishable from retrained'
    return same if (lo > -margin and hi < margin) else unresolved
