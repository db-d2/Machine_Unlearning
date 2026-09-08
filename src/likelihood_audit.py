"""Deterministic per-cell negative ELBO and the reference-model likelihood-ratio audit.

One estimator for every instrument that reports a likelihood: ``scripts/attack_lr_reference.py``,
``scripts/attack_lr_reference_multi.py``, ``scripts/fisher_phase_audit.py``, and notebook 41.
Eval mode (no dropout), K Monte-Carlo samples of z drawn from a CPU generator with a fixed
seed, reconstruction summed over genes (MSE against softmax x library size, the training
loss), KL summed over latent dimensions. The earlier scripts used the VAE's single-sample
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
        for _ in range(n_samples):
            eps = torch.randn(mu.shape, generator=gen).to(dev)
            mean, _ = model.decode(mu + eps * std, lib)
            recon += ((x - mean) ** 2).sum(1)
        recon /= n_samples
    return (recon + kl).cpu().numpy()


def lr_auc(ref_forget, ref_matched, target_forget, target_matched):
    """AUC of the reference-minus-target score, forget cells (1) against matched negatives (0)."""
    labels = np.r_[np.ones(len(ref_forget)), np.zeros(len(ref_matched))]
    scores = np.r_[ref_forget - target_forget, ref_matched - target_matched]
    return float(roc_auc_score(labels, scores))
