"""Synthetic sample generation and evaluation utilities.

Generates synthetic matched negatives from a reference model (typically the retrain
model) for within-subtype MIA evaluation. Also provides permutation testing.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

from vae import VAE
from attacker import MLPAttacker, extract_vae_features, build_attack_features


def generate_synthetic_negatives(
    reference_model,
    seed_samples,
    n_per_seed=40,
    noise_scale=1.0,
    device='cpu',
):
    """Generate synthetic samples by posterior sampling from a reference model.

    For each seed sample, encode through the reference model to get the posterior,
    then sample multiple z values and decode. The reference model should be one
    that has NOT memorized the seed samples (e.g., the retrain model).

    Args:
        reference_model: VAE model (typically retrain, which never saw the forget set)
        seed_samples: Real samples to use as seeds [n_seeds, n_genes]
        n_per_seed: Number of synthetic samples per seed
        noise_scale: Scale factor for posterior sampling (1.0 = standard, <1 = tighter)
        device: torch device

    Returns:
        Synthetic samples [n_seeds * n_per_seed, n_genes]
    """
    reference_model.eval()
    seed_samples = seed_samples.to(device)
    lib_sizes = seed_samples.sum(dim=1, keepdim=True)

    synthetic = []
    with torch.no_grad():
        mu, logvar = reference_model.encode(seed_samples)
        std = torch.exp(0.5 * logvar) * noise_scale

        for i in range(len(seed_samples)):
            for _ in range(n_per_seed):
                eps = torch.randn_like(std[i:i+1])
                z = mu[i:i+1] + eps * std[i:i+1]
                mean, _ = reference_model.decode(z, library_size=lib_sizes[i:i+1])
                synthetic.append(mean.cpu())

    return torch.cat(synthetic, dim=0)


def extract_mia_features(model, samples, device='cpu', variant='v1'):
    """Extract MIA features for a set of samples.

    Args:
        model: VAE model to extract features from
        samples: Input samples [n_samples, n_genes]
        device: torch device
        variant: Feature variant ('v1' = 69-dim)

    Returns:
        Feature tensor [n_samples, feature_dim]
    """
    model.eval()
    x = samples.to(device)
    lib = x.sum(dim=1, keepdim=True)
    vae_feats = extract_vae_features(model, x, lib, device, requires_grad=False)
    return build_attack_features(vae_feats, variant=variant)


def train_and_evaluate_attacker(
    pos_features, neg_features, device='cpu', epochs=100, lr=1e-3, seed=42
):
    """Train a fresh MLP attacker and compute AUC + advantage.

    Per mia-evaluation skill: spectral norm, [256,256], dropout 0.3.

    Returns:
        Dict with 'auc' and 'advantage'
    """
    torch.manual_seed(seed)

    input_dim = pos_features.shape[1]
    attacker = MLPAttacker(
        input_dim, [256, 256], dropout=0.3, use_spectral_norm=True
    ).to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.cat([pos_features, neg_features]).to(device)
    y = torch.cat([
        torch.ones(len(pos_features)),
        torch.zeros(len(neg_features))
    ]).unsqueeze(1).to(device)

    for _ in range(epochs):
        attacker.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(attacker(X), y)
        loss.backward()
        optimizer.step()

    attacker.eval()
    with torch.no_grad():
        preds = torch.sigmoid(attacker(X)).cpu().numpy().flatten()

    y_np = y.cpu().numpy().flatten()
    auc = float(roc_auc_score(y_np, preds))
    advantage = float(2 * abs(auc - 0.5))

    return {'auc': auc, 'advantage': advantage}


def permutation_test(
    pos_features, neg_features, n_permutations=1000, device='cpu', attacker_epochs=50
):
    """Permutation test for MIA significance.

    Shuffles member/non-member labels and retrains the attacker each time
    to build a null distribution of AUC values.

    Returns:
        Dict with 'observed_auc', 'p_value', 'null_mean', 'null_std', 'null_aucs'
    """
    observed = train_and_evaluate_attacker(
        pos_features, neg_features, device=device, epochs=100, seed=42
    )

    all_features = torch.cat([pos_features, neg_features])
    n_pos = len(pos_features)
    n_total = len(all_features)
    null_aucs = []

    for i in range(n_permutations):
        perm = np.random.permutation(n_total)
        perm_pos = all_features[perm[:n_pos]]
        perm_neg = all_features[perm[n_pos:]]
        result = train_and_evaluate_attacker(
            perm_pos, perm_neg, device=device, epochs=attacker_epochs, seed=i
        )
        null_aucs.append(result['auc'])

    null_aucs = np.array(null_aucs)
    p_value = float(np.mean(np.abs(null_aucs - 0.5) >= abs(observed['auc'] - 0.5)))

    return {
        'observed_auc': observed['auc'],
        'observed_advantage': observed['advantage'],
        'p_value': p_value,
        'null_mean': float(null_aucs.mean()),
        'null_std': float(null_aucs.std()),
        'null_aucs': null_aucs.tolist(),
    }
