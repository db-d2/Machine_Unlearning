"""Likelihood-ratio membership inference attack (high-assumption).

Compares likelihoods under target model vs reference model.
Score = log P(x | target) - log P(x | reference)
Higher score = sample more likely under target = more likely member.

Assumption level: HIGH
- Requires access to a reference model (e.g., retrained model or fresh init)
- Document this assumption explicitly in writeup

Reference: Carlini et al. "Membership Inference Attacks From First Principles"
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional

from attacker_eval import compute_advantage, compute_attack_success


def compute_log_likelihood(model, x: torch.Tensor, device: str = 'cpu',
                           n_samples: int = 1) -> np.ndarray:
    """Compute approximate log-likelihood for each sample.

    For VAEs, we use the ELBO as a lower bound on log-likelihood.
    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    Args:
        model: VAE model
        x: Input samples [N, genes]
        device: Device to run on
        n_samples: Number of z samples for importance-weighted estimate

    Returns:
        Per-sample log-likelihood estimate [N]
    """
    model.eval()
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        mu, logvar = model.encode(x)

        if n_samples == 1:
            # Simple ELBO estimate using mean
            z = mu
            decode_output = model.decode(z)
            # Handle both NB (returns tuple) and Gaussian decoders
            if isinstance(decode_output, tuple):
                recon = decode_output[0]
            else:
                recon = decode_output

            # Reconstruction term (negative MSE, treating as Gaussian likelihood)
            recon_term = -((x - recon) ** 2).sum(dim=1)

            # KL term
            kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

            # ELBO (lower bound on log p(x))
            elbo = recon_term - kl_term
        else:
            # Importance-weighted ELBO for tighter bound
            elbos = []
            for _ in range(n_samples):
                # Sample z
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                decode_output = model.decode(z)
                # Handle both NB (returns tuple) and Gaussian decoders
                if isinstance(decode_output, tuple):
                    recon = decode_output[0]
                else:
                    recon = decode_output
                recon_term = -((x - recon) ** 2).sum(dim=1)
                kl_term = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                elbos.append(recon_term - kl_term)

            # Average over samples
            elbo = torch.stack(elbos).mean(dim=0)

    return elbo.cpu().numpy()


def likelihood_ratio_scores(target_model, reference_model, samples: torch.Tensor,
                            device: str = 'cpu') -> np.ndarray:
    """Compute likelihood ratio scores.

    Score = log P(x | target) - log P(x | reference)
    Higher score = more likely member of target's training set.

    Args:
        target_model: The model we're attacking (trained with forget set)
        reference_model: Reference model (e.g., retrained without forget set)
        samples: Input samples [N, genes]
        device: Device to run on

    Returns:
        Likelihood ratio scores [N]
    """
    target_ll = compute_log_likelihood(target_model, samples, device)
    reference_ll = compute_log_likelihood(reference_model, samples, device)

    return target_ll - reference_ll


def likelihood_ratio_attack(target_model, reference_model,
                            forget_samples: torch.Tensor,
                            unseen_samples: torch.Tensor,
                            device: str = 'cpu') -> Dict[str, float]:
    """Run likelihood-ratio MIA.

    Args:
        target_model: The model we're attacking
        reference_model: Reference model (retrained or fresh init)
        forget_samples: Forget set samples (members)
        unseen_samples: Unseen samples (non-members)
        device: Device to run on

    Returns:
        Dictionary with 'auc', 'advantage', 'attack_success'
    """
    # Compute scores
    forget_scores = likelihood_ratio_scores(
        target_model, reference_model, forget_samples, device
    )
    unseen_scores = likelihood_ratio_scores(
        target_model, reference_model, unseen_samples, device
    )

    # Create labels
    labels = np.concatenate([
        np.ones(len(forget_scores)),
        np.zeros(len(unseen_scores))
    ])

    # Combine scores (higher = more likely member)
    scores = np.concatenate([forget_scores, unseen_scores])

    # Compute AUC
    auc = roc_auc_score(labels, scores)

    return {
        'auc': float(auc),
        'advantage': compute_advantage(auc),
        'attack_success': compute_attack_success(auc),
        'attack_type': 'likelihood_ratio',
        'assumption_level': 'high',
        'assumption': 'Attacker has access to reference model (retrained or similar)'
    }
