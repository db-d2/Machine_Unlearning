"""Loss-based membership inference attack (low-assumption).

This is the simplest possible MIA: use reconstruction loss as the membership score.
Lower loss = more likely to be a training member.

Assumption level: LOW
- Only requires query access to the model
- No auxiliary models, no training data access, no learned parameters

Reference: Yeom et al. "Privacy Risk in Machine Learning"
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

from attacker_eval import compute_advantage, compute_attack_success, compute_confidence_interval


def compute_reconstruction_loss(model, x: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """Compute per-sample reconstruction loss.

    Args:
        model: VAE model with encode() and decode() methods
        x: Input samples [N, genes]
        device: Device to run on

    Returns:
        Per-sample reconstruction loss [N]
    """
    model.eval()
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        # Encode
        mu, logvar = model.encode(x)
        # Sample z (use mean for deterministic scoring)
        z = mu
        # Decode - handle both NB (returns tuple) and Gaussian decoders
        decode_output = model.decode(z)
        if isinstance(decode_output, tuple):
            recon = decode_output[0]  # NB decoder returns (mean, dispersion)
        else:
            recon = decode_output

        # Compute per-sample MSE loss
        loss = ((x - recon) ** 2).mean(dim=1)

    return loss.cpu().numpy()


def compute_elbo_components(model, x: torch.Tensor, device: str = 'cpu') -> Dict[str, np.ndarray]:
    """Compute per-sample ELBO components.

    Args:
        model: VAE model
        x: Input samples [N, genes]
        device: Device to run on

    Returns:
        Dictionary with 'recon_loss', 'kl_div', 'elbo' arrays
    """
    model.eval()
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = mu
        # Decode - handle both NB (returns tuple) and Gaussian decoders
        decode_output = model.decode(z)
        if isinstance(decode_output, tuple):
            recon = decode_output[0]  # NB decoder returns (mean, dispersion)
        else:
            recon = decode_output

        # Reconstruction loss (MSE)
        recon_loss = ((x - recon) ** 2).mean(dim=1)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

        # ELBO = -recon_loss - kl_div (higher is better, so negate for loss)
        elbo = recon_loss + kl_div

    return {
        'recon_loss': recon_loss.cpu().numpy(),
        'kl_div': kl_div.cpu().numpy(),
        'elbo': elbo.cpu().numpy()
    }


def loss_attack_scores(model, samples: torch.Tensor, device: str = 'cpu',
                       score_type: str = 'recon') -> np.ndarray:
    """Compute membership scores using loss-based attack.

    Lower score = more likely member (model reconstructs it better).

    Args:
        model: VAE model
        samples: Input samples [N, genes]
        device: Device to run on
        score_type: 'recon' (reconstruction loss), 'kl' (KL divergence), or 'elbo'

    Returns:
        Membership scores [N] (lower = more likely member)
    """
    components = compute_elbo_components(model, samples, device)

    if score_type == 'recon':
        return components['recon_loss']
    elif score_type == 'kl':
        return components['kl_div']
    elif score_type == 'elbo':
        return components['elbo']
    else:
        raise ValueError(f"Unknown score_type: {score_type}")


def loss_attack_auc(model, forget_samples: torch.Tensor, unseen_samples: torch.Tensor,
                    device: str = 'cpu', score_type: str = 'recon') -> Dict[str, float]:
    """Run loss-based MIA and compute AUC.

    Args:
        model: VAE model
        forget_samples: Forget set samples (members)
        unseen_samples: Unseen samples (non-members, ideally matched negatives)
        device: Device to run on
        score_type: 'recon', 'kl', or 'elbo'

    Returns:
        Dictionary with 'auc', 'advantage', 'attack_success'
    """
    # Compute scores
    forget_scores = loss_attack_scores(model, forget_samples, device, score_type)
    unseen_scores = loss_attack_scores(model, unseen_samples, device, score_type)

    # Create labels (1 = member, 0 = non-member)
    labels = np.concatenate([
        np.ones(len(forget_scores)),
        np.zeros(len(unseen_scores))
    ])

    # Combine scores (negate so higher = more likely member for ROC)
    scores = np.concatenate([forget_scores, unseen_scores])
    scores_for_roc = -scores  # Negate: lower loss = higher membership probability

    # Compute AUC
    auc = roc_auc_score(labels, scores_for_roc)

    return {
        'auc': float(auc),
        'advantage': compute_advantage(auc),
        'attack_success': compute_attack_success(auc),
        'score_type': score_type,
        'assumption_level': 'low'
    }


def run_loss_attack_suite(model, forget_samples: torch.Tensor, unseen_samples: torch.Tensor,
                          device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Run all loss-based attack variants.

    Args:
        model: VAE model
        forget_samples: Forget set samples (members)
        unseen_samples: Unseen samples (non-members)
        device: Device to run on

    Returns:
        Dictionary with results for each score type
    """
    results = {}

    for score_type in ['recon', 'kl', 'elbo']:
        results[f'loss_{score_type}'] = loss_attack_auc(
            model, forget_samples, unseen_samples, device, score_type
        )

    # Find worst case among loss attacks
    worst = max(results.values(), key=lambda x: x['advantage'])
    results['worst_case'] = {
        'advantage': worst['advantage'],
        'attack': f"loss_{worst['score_type']}"
    }

    return results
