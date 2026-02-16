"""k-NN latent space membership inference attack (high-assumption).

Non-parametric attack using distance to training samples in latent space.
Score = average distance to k nearest training samples.
Lower distance = more likely member.

Assumption level: HIGH
- Requires access to latent embeddings of training samples (or proxy set)
- Document this assumption explicitly in writeup

Reference: Carlini et al. "Extracting Training Data from Diffusion Models"
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional

from attacker_eval import compute_advantage, compute_attack_success


def encode_samples(model, samples: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """Encode samples to latent space.

    Args:
        model: VAE model with encode() method
        samples: Input samples [N, genes]
        device: Device to run on

    Returns:
        Latent means [N, latent_dim]
    """
    model.eval()
    model = model.to(device)
    samples = samples.to(device)

    with torch.no_grad():
        mu, _ = model.encode(samples)

    return mu.cpu().numpy()


def build_knn_index(training_latents: np.ndarray, k: int = 10) -> NearestNeighbors:
    """Build k-NN index from training latent codes.

    Args:
        training_latents: Latent codes of training samples [N_train, latent_dim]
        k: Number of neighbors

    Returns:
        Fitted NearestNeighbors object
    """
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
    nn.fit(training_latents)
    return nn


def knn_attack_scores(nn_index: NearestNeighbors, query_latents: np.ndarray) -> np.ndarray:
    """Compute k-NN membership scores.

    Score = mean distance to k nearest neighbors.
    Lower score = closer to training data = more likely member.

    Args:
        nn_index: Fitted NearestNeighbors object
        query_latents: Latent codes of query samples [N_query, latent_dim]

    Returns:
        Mean distances [N_query]
    """
    distances, _ = nn_index.kneighbors(query_latents)
    return distances.mean(axis=1)


def knn_latent_attack(model, training_samples: torch.Tensor,
                      forget_samples: torch.Tensor,
                      unseen_samples: torch.Tensor,
                      device: str = 'cpu',
                      k: int = 10) -> Dict[str, float]:
    """Run k-NN latent space MIA.

    Args:
        model: VAE model
        training_samples: Training set samples (for building index)
        forget_samples: Forget set samples (members to detect)
        unseen_samples: Unseen samples (non-members)
        device: Device to run on
        k: Number of neighbors

    Returns:
        Dictionary with 'auc', 'advantage', 'attack_success'
    """
    # Encode all samples
    training_latents = encode_samples(model, training_samples, device)
    forget_latents = encode_samples(model, forget_samples, device)
    unseen_latents = encode_samples(model, unseen_samples, device)

    # Build k-NN index from training data
    nn_index = build_knn_index(training_latents, k=k)

    # Compute scores (distances)
    forget_scores = knn_attack_scores(nn_index, forget_latents)
    unseen_scores = knn_attack_scores(nn_index, unseen_latents)

    # Create labels
    labels = np.concatenate([
        np.ones(len(forget_scores)),
        np.zeros(len(unseen_scores))
    ])

    # Combine scores (negate: lower distance = higher membership probability)
    scores = np.concatenate([forget_scores, unseen_scores])
    scores_for_roc = -scores

    # Compute AUC
    auc = roc_auc_score(labels, scores_for_roc)

    return {
        'auc': float(auc),
        'advantage': compute_advantage(auc),
        'attack_success': compute_attack_success(auc),
        'attack_type': 'knn_latent',
        'k': k,
        'assumption_level': 'high',
        'assumption': 'Attacker has access to training set latent embeddings'
    }


def knn_latent_attack_precomputed(nn_index: NearestNeighbors,
                                   model,
                                   forget_samples: torch.Tensor,
                                   unseen_samples: torch.Tensor,
                                   device: str = 'cpu') -> Dict[str, float]:
    """Run k-NN attack with precomputed index (faster for multiple queries).

    Args:
        nn_index: Pre-built NearestNeighbors index
        model: VAE model (for encoding query samples)
        forget_samples: Forget set samples (members)
        unseen_samples: Unseen samples (non-members)
        device: Device to run on

    Returns:
        Dictionary with attack results
    """
    # Encode query samples
    forget_latents = encode_samples(model, forget_samples, device)
    unseen_latents = encode_samples(model, unseen_samples, device)

    # Compute scores
    forget_scores = knn_attack_scores(nn_index, forget_latents)
    unseen_scores = knn_attack_scores(nn_index, unseen_latents)

    # Create labels and scores
    labels = np.concatenate([
        np.ones(len(forget_scores)),
        np.zeros(len(unseen_scores))
    ])
    scores = np.concatenate([forget_scores, unseen_scores])
    scores_for_roc = -scores

    auc = roc_auc_score(labels, scores_for_roc)

    return {
        'auc': float(auc),
        'advantage': compute_advantage(auc),
        'attack_success': compute_attack_success(auc),
        'attack_type': 'knn_latent',
        'assumption_level': 'high'
    }
