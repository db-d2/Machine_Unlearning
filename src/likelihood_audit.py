"""Deterministic per-cell negative ELBO and the reference-model likelihood-ratio audit.

One estimator for every instrument that reports a likelihood: ``scripts/attack_lr_reference.py``,
``scripts/attack_lr_reference_multi.py``, ``scripts/fisher_phase_audit.py``, and notebook 41.
Eval mode (no dropout), K Monte-Carlo samples of z drawn from a CPU generator with a fixed
seed, reconstruction summed over features (MSE against softmax x library size for the
gene-expression models; the Gaussian negative log-likelihood for the simulation VAEs), KL
summed over latent dimensions. The earlier scripts used the VAE's single-sample
forward pass with no seed, and a 30-cell audit AUC moved by up to 0.14 between two runs
(2026-09-08). With this estimator two runs on CPU agree exactly.

The audit score of a cell is negELBO under a reference model that never saw it minus
negELBO under the target model. A member of the target's training set scores positive.
AUC of forget cells against matched negatives near 1 means the forget cells are still
fitted better than the reference fits them (signal intact); near 0 means fitted worse
(over-corrected); a second retrain seed scored against the reference gives the floor.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

N_SAMPLES = 16
SEED = 0


def neg_elbo_per_cell(model, x, lib=None, n_samples=N_SAMPLES, seed=SEED):
    """Per-cell negative ELBO, averaged over ``n_samples`` seeded z draws. Returns a numpy array."""
    model.eval()
    dev = next(model.parameters()).device
    x = x.to(dev)
    lib = x.sum(1, keepdim=True) if lib is None else lib.to(dev)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        std = (0.5 * logvar).exp()
        recon = torch.zeros(x.shape[0], device=dev)
        gaussian = getattr(model, 'likelihood', 'nb') == 'gaussian'
        for _ in range(n_samples):
            eps = torch.randn(mu.shape, generator=gen).to(dev)
            if gaussian:
                recon_mu, recon_logvar = model.decode(mu + eps * std)
                recon += 0.5 * (recon_logvar + (x - recon_mu) ** 2 / recon_logvar.exp() + np.log(2 * np.pi)).sum(1)
            else:
                mean, _ = model.decode(mu + eps * std, lib)
                recon += ((x - mean) ** 2).sum(1)
        recon /= n_samples
    return (recon + kl).cpu().numpy()


def lr_auc(ref_forget, ref_matched, target_forget, target_matched):
    """AUC of the reference-minus-target score, forget cells (1) against matched negatives (0)."""
    labels = np.r_[np.ones(len(ref_forget)), np.zeros(len(ref_matched))]
    scores = np.r_[ref_forget - target_forget, ref_matched - target_matched]
    return float(roc_auc_score(labels, scores))


def lr_auc_ci(ref_forget, ref_matched, target_forget, target_matched, n_boot=10000, seed=0, n_perm=2000):
    """Audit AUC with a percentile bootstrap over cells (forget cells and negatives resampled within their
    groups, so both group sizes are kept) and a permutation null of the AUC under shuffled labels.
    Returns a dict with auc, ci95, and the null's 2.5th and 97.5th percentiles."""
    rng = np.random.default_rng(seed)
    sf = np.asarray(ref_forget - target_forget); sm = np.asarray(ref_matched - target_matched)
    nf, nm = len(sf), len(sm)
    labels = np.r_[np.ones(nf), np.zeros(nm)]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = np.r_[sf[rng.integers(0, nf, nf)], sm[rng.integers(0, nm, nm)]]
        boot[b] = roc_auc_score(labels, s)
    scores = np.r_[sf, sm]
    null = np.array([roc_auc_score(rng.permutation(labels), scores) for _ in range(n_perm)]) if n_perm else None
    return {'auc': float(roc_auc_score(labels, scores)), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'null95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))] if n_perm else None, 'n_boot': n_boot}


def nested_auc_ci(per_seed, n_boot=10000, seed=0):
    """Nested bootstrap of a per-seed list of ``lr_auc_ci`` dicts: outer over seeds with replacement, inner
    from each drawn seed's cell bootstrap (approximated as normal with the SE implied by its ci95).
    Returns mean, nested 95 percent interval, and the per-seed AUCs."""
    rng = np.random.default_rng(seed)
    means = np.array([r['auc'] for r in per_seed]); ses = np.array([(r['ci95'][1] - r['ci95'][0]) / 3.92 for r in per_seed])
    k = len(means)
    if k == 1:
        return {'mean': float(means[0]), 'nested_ci_low': per_seed[0]['ci95'][0], 'nested_ci_high': per_seed[0]['ci95'][1], 'n_seeds': 1, 'per_seed_auc': means.tolist()}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        draws[b] = np.mean(np.clip(rng.normal(means[idx], ses[idx]), 0, 1))
    return {'mean': float(means.mean()), 'nested_ci_low': float(np.percentile(draws, 2.5)), 'nested_ci_high': float(np.percentile(draws, 97.5)),
            'n_seeds': k, 'per_seed_auc': means.tolist()}
