"""Long retain-only fine-tuning from a baseline checkpoint, measured every few epochs.

Question (notebook 42): the edit-then-fine-tune methods leave a memorized subpopulation fitted as
well as the baseline fits it after their ten retain epochs. Does the fit erode toward the retrained
model's level if the retain fine-tune simply runs on, or does the retain set keep the subpopulation
alive indefinitely? The same loop on TCGA-BRCA, whose retain set contains no Basal patient, is the
contrast. Training uses ``vae_loss`` with its own reduction; every measurement goes through the
``Evaluator`` of ``train_targeted_unlearn`` (deterministic audit estimator, canonical attacker).
"""
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from vae import vae_loss


def long_retain_finetune(baseline, X_retain, evaluator, eval_neighbors, device, epochs=200, lr=1e-4,
                         batch_size=256, eval_every=5, seed=42, log=print):
    """Fine-tune ``baseline`` on ``X_retain`` for ``epochs`` epochs. Returns (final model, history rows)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = deepcopy(baseline).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_retain, X_retain.sum(1, keepdim=True)), batch_size=batch_size, shuffle=True)
    history = []

    def measure(epoch, train_loss):
        rec = evaluator(model, eval_neighbors)
        rec.update({'epoch': epoch, 'train_loss': train_loss})
        history.append(rec)
        log(f"  epoch {epoch:3d}: forget={rec['forget_neg_elbo']:.1f} matched={rec['matched_neg_elbo']:.1f} "
            f"retain={rec['retain_neg_elbo']:.1f} MLP={rec['mlp_auc']:.3f} LR={rec['lr_auc']:.3f}")

    measure(0, float('nan'))
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x_r, lib_r in loader:
            x_r, lib_r = x_r.to(device), lib_r.to(device)
            opt.zero_grad()
            out = model(x_r, library_size=lib_r)
            loss, _, _ = vae_loss(x_r, out, likelihood=model.likelihood, library_size=lib_r, beta=1.0)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        if epoch % eval_every == 0 or epoch == epochs:
            measure(epoch, total / n)
    return model, history
