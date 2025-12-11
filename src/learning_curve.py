"""Learning curve utilities for AUC-vs-time tracking.

Implements wall-clock time logging during training to generate learning curves
comparing convergence rates of different unlearning methods.

Feedback item: "For efficiency/runtime you might want to log a metric like AUC vs
wall clock time, then plot a learning curve to compare the actual rates."
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score

from attacker import MLPAttacker, extract_vae_features, build_attack_features


class LearningCurveTracker:
    """Track AUC and other metrics vs wall-clock time during training.

    Usage:
        tracker = LearningCurveTracker()
        tracker.start()

        for step in training_loop:
            # ... training step ...
            if step % eval_interval == 0:
                auc = compute_auc(model, ...)
                tracker.log(step, auc=auc, loss=loss)

        tracker.save("learning_curve.json")
    """

    def __init__(self):
        self.start_time = None
        self.records = []

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.records = []

    def log(self, step: int, phase: str = "train", **metrics):
        """Log metrics at current time.

        Args:
            step: Training step or epoch number
            phase: Phase name (e.g., "scrub", "finetune", "train")
            **metrics: Any metrics to log (auc, loss, etc.)
        """
        if self.start_time is None:
            raise RuntimeError("Must call start() before logging")

        elapsed = time.time() - self.start_time

        record = {
            "wall_clock_seconds": elapsed,
            "step": step,
            "phase": phase,
            **metrics
        }
        self.records.append(record)

    def get_records(self) -> List[Dict]:
        """Get all logged records."""
        return self.records

    def save(self, path: str):
        """Save learning curve data to JSON."""
        with open(path, 'w') as f:
            json.dump({
                "total_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "records": self.records
            }, f, indent=2)


def compute_mia_auc(
    model,
    forget_x: torch.Tensor,
    forget_lib: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_lib: torch.Tensor,
    attacker: MLPAttacker,
    device: str = 'cpu',
    feature_variant: str = 'v1'
) -> float:
    """Compute MIA AUC for current model state.

    Args:
        model: VAE model (current state)
        forget_x: Forget set data
        forget_lib: Forget set library sizes
        unseen_x: Unseen (held-out) set data
        unseen_lib: Unseen set library sizes
        attacker: Trained MIA attacker
        device: Device to run on
        feature_variant: Feature variant for attacker

    Returns:
        AUC score
    """
    model.eval()
    attacker.eval()

    with torch.no_grad():
        # Extract features for forget set
        forget_features = extract_vae_features(
            model, forget_x, forget_lib, device=device
        )
        forget_attack_features = build_attack_features(
            forget_features, variant=feature_variant
        )

        # Extract features for unseen set
        unseen_features = extract_vae_features(
            model, unseen_x, unseen_lib, device=device
        )
        unseen_attack_features = build_attack_features(
            unseen_features, variant=feature_variant
        )

        # Create labels (forget=1, unseen=0)
        n_forget = len(forget_x)
        n_unseen = len(unseen_x)
        labels = np.concatenate([np.ones(n_forget), np.zeros(n_unseen)])

        # Get attacker predictions
        all_features = torch.cat([forget_attack_features, unseen_attack_features], dim=0)
        all_features = all_features.to(device)

        logits = attacker(all_features).squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()

    # Compute AUC
    auc = roc_auc_score(labels, predictions)

    return float(auc)


def load_attacker_for_eval(attacker_path: str, input_dim: int, device: str = 'cpu') -> MLPAttacker:
    """Load a trained attacker for evaluation.

    Args:
        attacker_path: Path to attacker checkpoint
        input_dim: Input dimension for attacker
        device: Device to load to

    Returns:
        Loaded attacker model
    """
    checkpoint = torch.load(attacker_path, map_location=device)

    # Get config from checkpoint if available
    config = checkpoint.get('config', {})
    hidden_dims = config.get('hidden_dims', [256, 256])
    dropout = config.get('dropout', 0.3)
    use_spectral_norm = config.get('use_spectral_norm', False)

    # Create attacker with correct settings
    attacker = MLPAttacker(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_spectral_norm=use_spectral_norm
    )

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        attacker.load_state_dict(checkpoint['model_state_dict'])
    else:
        attacker.load_state_dict(checkpoint)

    attacker.to(device)
    attacker.eval()

    return attacker


def prepare_eval_data(
    adata,
    split_data: Dict,
    matched_indices: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare forget and unseen data for AUC evaluation.

    Args:
        adata: AnnData object
        split_data: Split data with forget_indices, unseen_indices
        matched_indices: Optional matched negative indices (if None, uses unseen_indices)

    Returns:
        (forget_x, forget_lib, unseen_x, unseen_lib)
    """
    forget_indices = split_data['forget_indices']

    # Use matched negatives if available, otherwise use unseen
    if matched_indices is not None:
        unseen_indices = matched_indices
    elif 'unseen_indices' in split_data:
        unseen_indices = split_data['unseen_indices']
    else:
        # Fall back to retain indices (subsample to match forget size)
        retain_indices = np.array(split_data['retain_indices'])
        np.random.seed(42)
        unseen_indices = np.random.choice(
            retain_indices,
            size=min(len(forget_indices), len(retain_indices)),
            replace=False
        )

    # Extract data
    forget_x = torch.FloatTensor(
        adata.X[forget_indices].toarray()
        if hasattr(adata.X[forget_indices], 'toarray')
        else adata.X[forget_indices]
    )
    forget_lib = forget_x.sum(dim=1, keepdim=True)

    unseen_x = torch.FloatTensor(
        adata.X[unseen_indices].toarray()
        if hasattr(adata.X[unseen_indices], 'toarray')
        else adata.X[unseen_indices]
    )
    unseen_lib = unseen_x.sum(dim=1, keepdim=True)

    return forget_x, forget_lib, unseen_x, unseen_lib


def get_feature_dim(latent_dim: int, variant: str = 'v1') -> int:
    """Get feature dimension for attacker based on variant.

    Args:
        latent_dim: VAE latent dimension
        variant: Feature variant

    Returns:
        Feature dimension
    """
    if variant == 'v1':
        # recon_nll(1) + kl(1) + elbo(1) + mu(latent) + logvar(latent) + mu_norm(1) + logvar_norm(1)
        return 5 + 2 * latent_dim
    elif variant == 'v2':
        # recon_nll(1) + kl(1) + elbo(1) + mu(latent) + logvar(latent)
        return 3 + 2 * latent_dim
    elif variant == 'v3':
        # recon_nll(1) + kl(1) + elbo(1)
        return 3
    else:
        raise ValueError(f"Unknown variant: {variant}")
