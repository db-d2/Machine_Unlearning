"""Membership inference attacker for privacy auditing.

Implements a neural network-based membership inference attack to detect
whether a data point was used in training the VAE.

References:
    - Shokri et al. (2017). Membership Inference Attacks Against Machine Learning Models.
      IEEE S&P. https://doi.org/10.1109/SP.2017.41
    - Salem et al. (2019). ML-Leaks: Model and Data Independent Membership Inference Attacks
      and Defenses on Machine Learning Models. NDSS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict


class MLPAttacker(nn.Module):
    """MLP-based membership inference attacker.

    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        dropout: Dropout probability (default: 0.3)
        use_spectral_norm: Whether to use spectral normalization (default: False)
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 256], dropout: float = 0.3, use_spectral_norm: bool = False):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.extend([
                linear,
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.network = nn.Sequential(*layers)

        fc_out = nn.Linear(hidden_dims[-1], 1)
        if use_spectral_norm:
            fc_out = spectral_norm(fc_out)
        self.fc_out = fc_out

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits for membership prediction [batch_size, 1]
        """
        h = self.network(x)
        return self.fc_out(h)


def extract_vae_features(
    model,
    x: torch.Tensor,
    library_size: torch.Tensor,
    device: str = 'cpu',
    requires_grad: bool = False
) -> Dict[str, torch.Tensor]:
    """Extract features from VAE for membership inference.

    Args:
        model: Trained VAE model
        x: Input data [batch_size, input_dim]
        library_size: Library sizes [batch_size, 1]
        device: Device to run on
        requires_grad: If True, compute features with gradients (for unlearning training)

    Returns:
        Dictionary of features:
            - recon_nll: Reconstruction negative log-likelihood
            - kl: KL divergence
            - elbo: Total ELBO (recon + kl)
            - mu: Encoder mean
            - logvar: Encoder log variance
            - mu_norm: L2 norm of mu
            - logvar_norm: L2 norm of logvar
            - z: Latent code
    """
    if not requires_grad:
        model.eval()

    # Optionally enable gradients for unlearning training
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()

    with grad_context:
        x = x.to(device)
        library_size = library_size.to(device)

        # Forward pass
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)

        if model.likelihood == 'nb':
            mean, dispersion = model.decode(z, library_size)
            # Compute NB NLL (using MSE approximation for log-normalized data)
            recon_nll = F.mse_loss(mean, x, reduction='none').sum(dim=1)
        else:
            recon_mu, recon_logvar = model.decode(z)
            # Compute Gaussian NLL
            recon_nll = 0.5 * (
                recon_logvar
                + ((x - recon_mu) ** 2) / torch.exp(recon_logvar)
                + np.log(2 * np.pi)
            ).sum(dim=1)

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # ELBO
        elbo = recon_nll + kl

        # Norms
        mu_norm = torch.norm(mu, p=2, dim=1)
        logvar_norm = torch.norm(logvar, p=2, dim=1)

    # Keep tensors on device if gradients are needed
    if requires_grad:
        return {
            'recon_nll': recon_nll,
            'kl': kl,
            'elbo': elbo,
            'mu': mu,
            'logvar': logvar,
            'mu_norm': mu_norm,
            'logvar_norm': logvar_norm,
            'z': z
        }

    return {
        'recon_nll': recon_nll.cpu(),
        'kl': kl.cpu(),
        'elbo': elbo.cpu(),
        'mu': mu.cpu(),
        'logvar': logvar.cpu(),
        'mu_norm': mu_norm.cpu(),
        'logvar_norm': logvar_norm.cpu(),
        'z': z.cpu()
    }


def compute_knn_distances(
    query_z: np.ndarray,
    reference_z: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """Compute average k-NN distances from query to reference set.

    Args:
        query_z: Query latent codes [n_query, latent_dim]
        reference_z: Reference latent codes [n_ref, latent_dim]
        k: Number of nearest neighbors

    Returns:
        Average k-NN distances [n_query]
    """
    if len(reference_z) < k:
        k = len(reference_z)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(reference_z)
    distances, _ = nbrs.kneighbors(query_z)

    return distances.mean(axis=1)


def build_attack_features(
    vae_features: Dict[str, torch.Tensor],
    knn_dist_retain: np.ndarray = None,
    knn_dist_unseen: np.ndarray = None,
    variant: str = 'v1'
) -> torch.Tensor:
    """Build feature vector for membership inference attack.

    Supports multiple feature variants for multi-critic evaluation.

    Args:
        vae_features: Dictionary of VAE features from extract_vae_features
        knn_dist_retain: k-NN distances to retain set (optional)
        knn_dist_unseen: k-NN distances to unseen set (optional)
        variant: Feature variant ('v1' = baseline, 'v2' = extended, 'v3' = minimal)

    Returns:
        Feature tensor [batch_size, feature_dim]
    """
    if variant == 'v1':
        # A1: Baseline features (full)
        features = [
            vae_features['recon_nll'].unsqueeze(1),
            vae_features['kl'].unsqueeze(1),
            vae_features['elbo'].unsqueeze(1),
            vae_features['mu'],
            vae_features['logvar'],
            vae_features['mu_norm'].unsqueeze(1),
            vae_features['logvar_norm'].unsqueeze(1)
        ]
    elif variant == 'v2':
        # A2: Extended features (ELBO + mu/logvar, no norms)
        features = [
            vae_features['recon_nll'].unsqueeze(1),
            vae_features['kl'].unsqueeze(1),
            vae_features['elbo'].unsqueeze(1),
            vae_features['mu'],
            vae_features['logvar']
        ]
    elif variant == 'v3':
        # A3: Minimal features (ELBO parts only)
        features = [
            vae_features['recon_nll'].unsqueeze(1),
            vae_features['kl'].unsqueeze(1),
            vae_features['elbo'].unsqueeze(1)
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if knn_dist_retain is not None:
        features.append(torch.FloatTensor(knn_dist_retain).unsqueeze(1))

    if knn_dist_unseen is not None:
        features.append(torch.FloatTensor(knn_dist_unseen).unsqueeze(1))

    return torch.cat(features, dim=1)


def compute_attack_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute attack performance metrics.

    Args:
        predictions: Predicted probabilities [n_samples]
        labels: True labels (1 = member, 0 = non-member) [n_samples]

    Returns:
        Dictionary of metrics:
            - auc: ROC AUC score
            - accuracy: Classification accuracy at 0.5 threshold
            - tpr_at_fpr_01: TPR at 1% FPR
            - tpr_at_fpr_05: TPR at 5% FPR
    """
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

    # ROC AUC
    auc = roc_auc_score(labels, predictions)

    # Accuracy
    acc = accuracy_score(labels, (predictions >= 0.5).astype(int))

    # TPR at specific FPR
    fpr, tpr, thresholds = roc_curve(labels, predictions)

    # TPR at 1% FPR
    idx = np.where(fpr <= 0.01)[0]
    tpr_at_fpr_01 = tpr[idx[-1]] if len(idx) > 0 else 0.0

    # TPR at 5% FPR
    idx = np.where(fpr <= 0.05)[0]
    tpr_at_fpr_05 = tpr[idx[-1]] if len(idx) > 0 else 0.0

    return {
        'auc': float(auc),
        'accuracy': float(acc),
        'tpr_at_fpr_01': float(tpr_at_fpr_01),
        'tpr_at_fpr_05': float(tpr_at_fpr_05)
    }


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using bootstrap.

    Args:
        values: Array of values
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    from scipy import stats

    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)

    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)

    return (mean - ci, mean + ci)


class EMA:
    """Exponential Moving Average for model parameters.

    Used to stabilize adversarial training by maintaining an exponential
    moving average of model weights.

    Args:
        model: PyTorch model
        decay: EMA decay rate (default: 0.999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
