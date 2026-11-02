"""Mixture-of-Gaussians simulations for VAE unlearning (rewritten 2026-09-09).

The earlier version of this module (December 2025) scored forget points against negatives from
other components and against the retain set, trained its attacker in sample, sampled z without a
seed, and had no rare component, so it could not show the phenomenon the paper is about. This
version uses the project's instruments and exposes the two knobs the mechanism needs.

Data. K common components on a grid in the first two coordinates of a d-dimensional space with
isotropic unit variance, plus an optional rare component placed ``distance_sd`` standard
deviations away from component 0 along the third coordinate, so its distance to every other
common component is larger. ``distance_sd`` is the support knob: at 1 the rare points sit inside
component 0's mass, at 8 nothing in the common data resembles them.

Splits. A stratified held-out fraction per component gives within-component negatives for every
component including the rare one. Cross-component negatives are the k nearest held-out common
points of each forget point in the baseline's latent space, deduplicated, as in the real protocol.

Instruments. The deterministic negative ELBO of ``likelihood_audit`` (Gaussian branch), one MLP
attacker trained on the baseline with the canonical recipe (features standardized once with the
baseline's statistics) and applied to every model (one for
cross-component negatives, one for within-component negatives), and the reference audit against a
clean retrain with a retrain-versus-retrain floor. Every model is trained on CPU from explicit
seeds; every loss goes through ``vae_loss``.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from attacker import MLPAttacker, extract_vae_features, build_attack_features, compute_knn_distances
from likelihood_audit import neg_elbo_per_cell, lr_auc_ci
from vae import VAE, vae_loss

LOG2PI = float(np.log(2 * np.pi))


# ----------------------------------------------------------------------------- data
def generate_mog_data(K, d, n, seed=42, spacing_sd=6.0, rare_n=0, rare_distance_sd=4.0):
    """K common components (equal weights, unit variance) plus ``rare_n`` points of a rare component.

    Returns X [n + rare_n, d], labels (rare component = K), metadata.
    """
    rng = np.random.RandomState(seed)
    grid = [(i % 4, i // 4) for i in range(K)]
    means = np.zeros((K + (1 if rare_n else 0), d))
    for k, (gx, gy) in enumerate(grid):
        means[k, 0], means[k, 1] = gx * spacing_sd, gy * spacing_sd
    if rare_n:
        means[K] = means[0].copy()
        means[K, 2] = rare_distance_sd
    labels = np.r_[rng.randint(0, K, size=n), np.full(rare_n, K)]
    X = means[labels] + rng.standard_normal((len(labels), d))
    perm = rng.permutation(len(labels))
    meta = {'K': K, 'd': d, 'n_common': n, 'rare_n': rare_n, 'rare_component': K if rare_n else None,
            'rare_distance_sd': rare_distance_sd if rare_n else None, 'spacing_sd': spacing_sd, 'seed': seed,
            'means': means.tolist()}
    return X[perm].astype(np.float32), labels[perm], meta


def stratified_split(labels, holdout_fraction, seed=42, rare_component=None, rare_holdout_fraction=None):
    """Held-out indices per component (the rare component may use its own fraction)."""
    rng = np.random.RandomState(seed)
    unseen = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        frac = rare_holdout_fraction if (rare_component is not None and k == rare_component and rare_holdout_fraction) else holdout_fraction
        n_hold = int(round(len(idx) * frac))
        unseen.extend(rng.choice(idx, n_hold, replace=False).tolist())
    unseen = np.array(sorted(unseen))
    train = np.setdiff1d(np.arange(len(labels)), unseen)
    return train, unseen


def forget_indices(labels, train_idx, component, fraction=1.0, seed=42):
    """Training indices of ``component`` to forget: all of them or a seeded fraction."""
    rng = np.random.RandomState(seed)
    members = train_idx[labels[train_idx] == component]
    if fraction >= 1.0:
        return members
    return np.sort(rng.choice(members, int(round(len(members) * fraction)), replace=False))


def latent_mu(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.as_tensor(X).to(device))
    return mu.cpu().numpy()


def select_matched(baseline, X_forget, X_pool, k=10, min_n=200):
    """Indices into ``X_pool`` of the k nearest pool points of each forget point (baseline latent), deduplicated.

    An isolated forget set can share the same few nearest points, so k grows until at least ``min_n``
    distinct negatives are selected (or the pool is exhausted)."""
    nn = NearestNeighbors(n_neighbors=len(X_pool)).fit(latent_mu(baseline, X_pool))
    _, idx = nn.kneighbors(latent_mu(baseline, X_forget))
    kk = min(k, len(X_pool))
    while True:
        sel = np.unique(idx[:, :kk].flatten())
        if len(sel) >= min(min_n, len(X_pool)) or kk >= len(X_pool):
            return sel
        kk = min(len(X_pool), kk * 2)


# ----------------------------------------------------------------------------- models
def make_vae(d, hidden=(128, 64), latent=8):
    return VAE(input_dim=d, hidden_dims=list(hidden), latent_dim=latent, likelihood='gaussian', dropout=0.0, use_layer_norm=True)


def train_vae(X, seed, hidden=(128, 64), latent=8, epochs=200, lr=1e-3, batch_size=256, kl_warmup_epochs=20,
              free_bits=0.0, device='cpu', log=None):
    """Gaussian VAE trained from scratch on X with ``vae_loss`` (KL warm-up through beta)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_vae(X.shape[1], hidden, latent).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.as_tensor(X)), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / kl_warmup_epochs) if kl_warmup_epochs else 1.0
        model.train(); total = 0.0
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            loss, _, _ = vae_loss(x, model(x), likelihood='gaussian', beta=beta, free_bits=free_bits)
            loss.backward(); opt.step(); total += loss.item()
        if log and (epoch + 1) % 50 == 0:
            log(f"    epoch {epoch + 1}: loss {total / len(loader):.2f}")
    return model.eval()


def neg_elbo_with_grad(model, x):
    """Single-sample Gaussian negative ELBO per point, differentiable (for the methods' objectives)."""
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    rm, rl = model.decode(z)
    recon = 0.5 * (rl + (x - rm) ** 2 / rl.exp() + LOG2PI).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return recon + kl


def _retain_loader(X_retain, batch_size):
    return DataLoader(TensorDataset(torch.as_tensor(X_retain)), batch_size=batch_size, shuffle=True)


def retain_finetune(baseline, X_retain, epochs=10, lr=1e-4, batch_size=256, seed=42, device='cpu', callback=None):
    """Retain-only fine-tuning; ``callback(epoch, model)`` runs after each epoch when given."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = deepcopy(baseline).to(device); opt = optim.Adam(model.parameters(), lr=lr)
    loader = _retain_loader(X_retain, batch_size)
    for epoch in range(1, epochs + 1):
        model.train()
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            loss, _, _ = vae_loss(x, model(x), likelihood='gaussian')
            loss.backward(); opt.step()
        if callback: callback(epoch, model.eval())
    return model.eval()


def gradient_ascent_then_finetune(baseline, X_forget, X_retain, ascent_steps=20, ascent_lr=1e-4, finetune_epochs=10,
                                  lr=1e-4, batch_size=256, seed=42, device='cpu', clip=5.0):
    """Gradient ascent on the forget set's negative ELBO (clipped), then retain fine-tuning."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = deepcopy(baseline).to(device); opt = optim.Adam(model.parameters(), lr=ascent_lr)
    xf = torch.as_tensor(X_forget).to(device)
    model.train()
    for _ in range(ascent_steps):
        opt.zero_grad()
        (-neg_elbo_with_grad(model, xf).mean()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    return retain_finetune(model, X_retain, epochs=finetune_epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)


def forget_push(baseline, X_forget, X_retain, gamma=1.0, epochs=20, lr=1e-4, batch_size=256, seed=42, device='cpu'):
    """Continuous forget objective: retain ELBO plus gamma times the forget posteriors' KL to the prior
    (the paper's contrastive latent unlearning), with a fixed epoch budget and no stopping rule."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = deepcopy(baseline).to(device); opt = optim.Adam(model.parameters(), lr=lr)
    xf = torch.as_tensor(X_forget).to(device); loader = _retain_loader(X_retain, batch_size)
    for _ in range(epochs):
        model.train()
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            loss, _, _ = vae_loss(x, model(x), likelihood='gaussian')
            mu, logvar = model.encode(xf)
            kl_f = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()
            (loss + gamma * kl_f).backward(); opt.step()
    return model.eval()


def targeted_finetune(baseline, X_forget, X_retain, target, lambda_target=1.0, epochs=50, lr=1e-4, batch_size=256,
                      tol=0.2, patience=3, seed=42, device='cpu'):
    """Likelihood-targeted unlearning. ``target`` is a float (oracle level) or an array of calibration
    points whose mean negative ELBO under the current model (no gradient) is the level."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = deepcopy(baseline).to(device); opt = optim.Adam(model.parameters(), lr=lr)
    xf = torch.as_tensor(X_forget).to(device); loader = _retain_loader(X_retain, batch_size)
    xc = None if isinstance(target, (int, float)) else torch.as_tensor(np.asarray(target, dtype=np.float32)).to(device)
    best_state, best_gap, inside, history = deepcopy(model.state_dict()), float('inf'), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for (x,) in loader:
            x = x.to(device); opt.zero_grad()
            loss, _, _ = vae_loss(x, model(x), likelihood='gaussian')
            with torch.no_grad():
                t = float(target) if xc is None else neg_elbo_with_grad(model, xc).mean()
            penalty = (neg_elbo_with_grad(model, xf) - t).abs().mean()
            (loss + lambda_target * penalty).backward(); opt.step()
        model.eval()
        level = float(target) if xc is None else float(neg_elbo_per_cell(model, xc).mean())
        gap = float(neg_elbo_per_cell(model, xf).mean()) - level
        history.append({'epoch': epoch, 'gap': gap, 'level': level})
        if abs(gap) < best_gap:
            best_gap, best_state = abs(gap), deepcopy(model.state_dict())
        inside = inside + 1 if abs(gap) < tol else 0
        if inside >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


# ----------------------------------------------------------------------------- instruments
def feature_matrix(model, X, retain_z, device='cpu', seed=0):
    """Canonical attack features (v1 + k-NN distance to a retain sample), z draws seeded.

    The seed is applied inside a forked RNG so that measuring a model mid-training does not
    reset the training run's random stream."""
    x = torch.as_tensor(X).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        feats = extract_vae_features(model, x, torch.ones(len(x), 1), device, requires_grad=False)
    knn = compute_knn_distances(feats['z'].numpy(), retain_z, k=5)
    return build_attack_features(feats, knn_dist_retain=knn, variant='v1').float()


def train_attacker(F_pos, F_neg, seed=42, epochs=50, lr=1e-3, batch_size=128):
    """The canonical recipe: spectral-norm MLP [256, 256], dropout 0.3, 50 epochs, Adam 1e-3, BCE."""
    torch.manual_seed(seed)
    X = torch.cat([F_pos, F_neg]); y = torch.cat([torch.ones(len(F_pos)), torch.zeros(len(F_neg))])
    att = MLPAttacker(X.shape[1], hidden_dims=[256, 256], dropout=0.3, use_spectral_norm=True)
    opt = optim.Adam(att.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    att.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(att(xb).squeeze(1), yb).backward(); opt.step()
    return att.eval()


def attacker_auc(att, F_pos, F_neg):
    with torch.no_grad():
        s = torch.sigmoid(att(torch.cat([F_pos, F_neg])).squeeze(1)).numpy()
    return float(roc_auc_score(np.r_[np.ones(len(F_pos)), np.zeros(len(F_neg))], s))


def audit_auc(reference, target, X_pos, X_neg):
    """Reference audit: negELBO under the reference minus under the target, positives vs negatives."""
    s = np.r_[neg_elbo_per_cell(reference, torch.as_tensor(X_pos)) - neg_elbo_per_cell(target, torch.as_tensor(X_pos)),
              neg_elbo_per_cell(reference, torch.as_tensor(X_neg)) - neg_elbo_per_cell(target, torch.as_tensor(X_neg))]
    return float(roc_auc_score(np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))], s))


def audit_auc_ci(reference, target, X_pos, X_neg, n_boot=2000):
    """The audit with a percentile bootstrap over points (groups resampled separately); returns {'auc', 'ci95'}."""
    r = lr_auc_ci(neg_elbo_per_cell(reference, torch.as_tensor(X_pos)), neg_elbo_per_cell(reference, torch.as_tensor(X_neg)),
                  neg_elbo_per_cell(target, torch.as_tensor(X_pos)), neg_elbo_per_cell(target, torch.as_tensor(X_neg)), n_boot=n_boot, n_perm=0)
    return {'auc': r['auc'], 'ci95': r['ci95']}


class Instruments:
    """Everything needed to measure one model: matched negatives, attackers trained on the baseline, the reference."""

    def __init__(self, baseline, reference, X_forget, X_cross, X_within, X_retain, device='cpu', seed=42):
        self.X_forget, self.X_cross, self.X_within = X_forget, X_cross, X_within
        self.X_retain_sample = X_retain[np.random.RandomState(seed).choice(len(X_retain), min(5000, len(X_retain)), replace=False)]
        self.reference, self.device = reference, device
        self.retain_z = latent_mu(baseline, self.X_retain_sample, device)
        Ff, Fc, Fw = (feature_matrix(baseline, X, self.retain_z, device) for X in (X_forget, X_cross, X_within))
        # one fixed standardization from the baseline's own features; raw features (scale near 100 with a
        # 1:8 class imbalance) let the spectral-norm attacker collapse to the class prior
        allf = torch.cat([Ff, Fc, Fw]) if len(Fw) else torch.cat([Ff, Fc])
        self.f_mu, self.f_sd = allf.mean(0), allf.std(0) + 1e-6
        Ff, Fc, Fw = (self._std(F_) for F_ in (Ff, Fc, Fw))
        self.att_cross = train_attacker(Ff, Fc, seed=seed)
        self.att_within = train_attacker(Ff, Fw, seed=seed) if len(X_within) >= 5 else None

    def _std(self, F_):
        return (F_ - self.f_mu) / self.f_sd if len(F_) else F_

    def measure(self, model, ci=False):
        """With ``ci=True`` the two audits also carry a bootstrap interval (``audit_cross_ci``, ``audit_within_ci``)."""
        rz = latent_mu(model, self.X_retain_sample, self.device)
        Ff, Fc, Fw = (self._std(feature_matrix(model, X, rz, self.device)) for X in (self.X_forget, self.X_cross, self.X_within))
        out = {'forget_neg_elbo': float(neg_elbo_per_cell(model, torch.as_tensor(self.X_forget)).mean()),
               'cross_neg_elbo': float(neg_elbo_per_cell(model, torch.as_tensor(self.X_cross)).mean()),
               'within_neg_elbo': float(neg_elbo_per_cell(model, torch.as_tensor(self.X_within)).mean()) if len(self.X_within) else float('nan'),
               'retain_neg_elbo': float(neg_elbo_per_cell(model, torch.as_tensor(self.X_retain_sample)).mean()),
               'attacker_cross': attacker_auc(self.att_cross, Ff, Fc),
               'attacker_within': attacker_auc(self.att_within, Ff, Fw) if self.att_within is not None else float('nan'),
               'audit_cross': audit_auc(self.reference, model, self.X_forget, self.X_cross),
               'audit_within': audit_auc(self.reference, model, self.X_forget, self.X_within) if len(self.X_within) >= 5 else float('nan')}
        if ci:
            out['audit_cross_ci'] = audit_auc_ci(self.reference, model, self.X_forget, self.X_cross)['ci95']
            out['audit_within_ci'] = audit_auc_ci(self.reference, model, self.X_forget, self.X_within)['ci95'] if len(self.X_within) >= 5 else [float('nan'), float('nan')]
        return out


def advantage(auc):
    return 2 * abs(auc - 0.5)
