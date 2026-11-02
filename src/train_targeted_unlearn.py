"""Likelihood-targeted unlearning (LTU).

Every post-hoc method in the study either leaves the forget cells fitted almost as well
as the baseline fits them or pushes them past the point retraining reaches (Stage 1,
2026-09-08). None of their objectives encodes where the forget cells should end up. This
method gives the objective that target, and the target is observable without a retrain:
a forgotten cell should be fitted the way comparable never-seen cells are fitted, and the
matched negatives of the evaluation protocol are exactly those cells.

Objective, per step, for a retain batch R_b and the full forget set F:

    L = L_VAE(theta; R_b) + lambda * mean_{i in F} | negELBO_theta(x_i) - t_i(theta) |

    t_i(theta) = mean_{j in N(i)} negELBO_theta(x_j), the model's own negative ELBO on the
    k nearest matched negatives of x_i (in the baseline's latent space), treated as a
    constant at each step.

The penalty is two-sided, so it pulls the forget cells down toward the never-seen level
if they are fitted better than it (memorized) and up if they are fitted worse
(over-corrected). Training stops when the mean gap has stayed inside a tolerance for
`patience` epochs, and the returned model is the one with the smallest mean absolute gap.

Everything reported per epoch is measured with the training objective's own reduction
(reconstruction summed over genes, KL summed over latent dimensions), so no reweighting
of the two ELBO terms can occur (see LOSS_WEIGHTING_BUG.md), and with the deterministic
estimator of src/likelihood_audit.py (eval mode, 16 seeded z samples).
"""
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from vae import vae_loss
from likelihood_audit import neg_elbo_per_cell as audit_neg_elbo
from attacker import extract_vae_features, build_attack_features, compute_knn_distances


def neg_elbo_per_cell(model, x, lib, deterministic=True):
    """Per-cell negative ELBO with the training reduction. z = mu when deterministic."""
    mu, logvar = model.encode(x)
    z = mu if deterministic else model.reparameterize(mu, logvar)
    mean, _ = model.decode(z, lib)
    recon = ((x - mean) ** 2).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return recon + kl


def matched_neighbors(baseline, X_forget, X_matched, device, k=10):
    """Index of the k nearest matched negatives of each forget cell in baseline latent space."""
    baseline.eval()
    with torch.no_grad():
        mu_f, _ = baseline.encode(X_forget.to(device))
        mu_m, _ = baseline.encode(X_matched.to(device))
    nn = NearestNeighbors(n_neighbors=min(k, len(X_matched))).fit(mu_m.cpu().numpy())
    _, idx = nn.kneighbors(mu_f.cpu().numpy())
    return torch.as_tensor(idx)


def attack_features(model, X, retain_z, device):
    """70-dim canonical attack features (v1 + kNN distance to a retain sample)."""
    model.eval()
    x = X.to(device)
    feats = extract_vae_features(model, x, x.sum(1, keepdim=True), device, requires_grad=False)
    knn = compute_knn_distances(feats['z'].numpy(), retain_z, k=5)
    return build_attack_features(feats, knn_dist_retain=knn, variant='v1')


def retain_latents(model, X_retain_sample, device):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_retain_sample.to(device))
        return model.reparameterize(mu, logvar).cpu().numpy()


class Evaluator:
    """Per-epoch measurements: forget/target/matched negative ELBO, retain ELBO, MLP and LR AUC.

    Likelihoods use the deterministic audit estimator (eval mode, 16 seeded z samples), the same
    one as scripts/attack_lr_reference.py, so the notebook and the audit JSON agree.
    """

    def __init__(self, attacker, reference, X_forget, X_matched, X_retain_sample, device, extra_sets=None):
        self.attacker = attacker.to('cpu').eval() if attacker is not None else None
        self.X_forget, self.X_matched, self.X_retain_sample = X_forget, X_matched, X_retain_sample
        self.device = device
        self.extra_sets = dict(extra_sets or {})   # name -> tensor; each adds '<name>_neg_elbo' to the record
        self.labels = np.r_[np.ones(len(X_forget)), np.zeros(len(X_matched))]
        ref = reference.to('cpu').eval()
        self.ref_f = audit_neg_elbo(ref, X_forget)
        self.ref_m = audit_neg_elbo(ref, X_matched)

    def __call__(self, model, neighbors):
        """Measure ``model`` wherever it lives (CPU for saved checkpoints, MPS during training)."""
        model.eval()
        dev = next(model.parameters()).device
        nf = audit_neg_elbo(model, self.X_forget)
        nm = audit_neg_elbo(model, self.X_matched)
        nr = audit_neg_elbo(model, self.X_retain_sample)
        targets = nm[neighbors.numpy()].mean(1)
        if self.attacker is not None:
            rz = retain_latents(model, self.X_retain_sample, dev)
            feats = torch.cat([attack_features(model, self.X_forget, rz, dev),
                               attack_features(model, self.X_matched, rz, dev)], 0)
            with torch.no_grad():
                scores = torch.sigmoid(self.attacker(feats).squeeze()).numpy()
            mlp_auc = float(roc_auc_score(self.labels, scores))
        else:
            mlp_auc = float('nan')
        lr = np.r_[self.ref_f - nf, self.ref_m - nm]
        out = {'forget_neg_elbo': float(nf.mean()), 'target_neg_elbo': float(targets.mean()),
               'matched_neg_elbo': float(nm.mean()), 'retain_neg_elbo': float(nr.mean()),
               'gap': float((nf - targets).mean()), 'abs_gap': float(np.abs(nf - targets).mean()),
               'mlp_auc': mlp_auc, 'lr_auc': float(roc_auc_score(self.labels, lr))}
        for name, Xe in self.extra_sets.items():
            out[f'{name}_neg_elbo'] = float(audit_neg_elbo(model, Xe).mean())
        return out


def train_targeted(baseline, X_forget, X_matched, X_retain, neighbors, evaluator, device,
                   lambda_target=1.0, lr=1e-4, epochs=50, batch_size=256, tol=5.0, patience=3, seed=42,
                   log=print, eval_neighbors=None):
    """Likelihood-targeted unlearning from a baseline checkpoint. Returns (model, history).

    ``X_matched`` and ``neighbors`` define the target (the calibration cells and which of them each
    forget cell is compared to). ``eval_neighbors`` indexes the evaluator's own matched negatives and
    defaults to ``neighbors``; pass it when the calibration cells are not the evaluation negatives.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    en = neighbors if eval_neighbors is None else eval_neighbors
    model = deepcopy(baseline).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_retain, X_retain.sum(1, keepdim=True)), batch_size=batch_size, shuffle=True)
    xf, lf = X_forget.to(device), X_forget.sum(1, keepdim=True).to(device)
    xm, lm = X_matched.to(device), X_matched.sum(1, keepdim=True).to(device)
    nbr = neighbors.to(device)
    history = {k: [] for k in ['epoch', 'utility', 'penalty', 'forget_neg_elbo', 'target_neg_elbo', 'matched_neg_elbo',
                               'retain_neg_elbo', 'gap', 'abs_gap', 'mlp_auc', 'lr_auc']}
    m0 = evaluator(model, en)
    history['epoch'].append(0); history['utility'].append(float('nan')); history['penalty'].append(float('nan'))
    for k, v in m0.items():
        history[k].append(v)
    log(f"  epoch 0: forget={m0['forget_neg_elbo']:.1f} target={m0['target_neg_elbo']:.1f} gap={m0['gap']:+.1f} "
        f"MLP={m0['mlp_auc']:.3f} LR={m0['lr_auc']:.3f}")
    best_state, best_abs_gap, inside = deepcopy(model.state_dict()), abs(m0['abs_gap']), 0
    for epoch in range(1, epochs + 1):
        model.train()
        sums, n = {'utility': 0.0, 'penalty': 0.0}, 0
        for x_r, lib_r in loader:
            x_r, lib_r = x_r.to(device), lib_r.to(device)
            opt.zero_grad()
            out = model(x_r, library_size=lib_r)
            utility, _, _ = vae_loss(x_r, out, likelihood=model.likelihood, library_size=lib_r, beta=1.0)
            with torch.no_grad():
                t = neg_elbo_per_cell(model, xm, lm, deterministic=False)[nbr].mean(1)
            nf = neg_elbo_per_cell(model, xf, lf, deterministic=False)
            penalty = (nf - t).abs().mean()
            loss = utility + lambda_target * penalty
            loss.backward()
            opt.step()
            sums['utility'] += utility.item(); sums['penalty'] += penalty.item(); n += 1
        m = evaluator(model, en)
        history['epoch'].append(epoch); history['utility'].append(sums['utility'] / n); history['penalty'].append(sums['penalty'] / n)
        for k, v in m.items():
            history[k].append(v)
        log(f"  epoch {epoch}: forget={m['forget_neg_elbo']:.1f} target={m['target_neg_elbo']:.1f} gap={m['gap']:+.1f} "
            f"retain={m['retain_neg_elbo']:.1f} MLP={m['mlp_auc']:.3f} LR={m['lr_auc']:.3f}")
        if eval_neighbors is None:
            train_gap = m['gap']
        else:
            with torch.no_grad():
                t_now = audit_neg_elbo(model, xm)[nbr.cpu().numpy()].mean(1)
                train_gap = float((audit_neg_elbo(model, xf) - t_now).mean())
            log(f"    objective gap (calibration cells) {train_gap:+.1f}")
        inside = inside + 1 if abs(train_gap) < tol else 0
        cur_abs = m['abs_gap'] if eval_neighbors is None else abs(train_gap)
        if cur_abs < best_abs_gap:
            best_abs_gap, best_state = cur_abs, deepcopy(model.state_dict())
        if inside >= patience:
            log(f"  stop at epoch {epoch}: |gap| < {tol} for {patience} epochs")
            break
    model.load_state_dict(best_state)
    return model, history
