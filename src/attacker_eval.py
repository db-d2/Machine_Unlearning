"""Cluster-conditioned evaluation for membership inference attacks.

This module implements evaluation methods that control for cluster identity
to avoid confounding between cluster membership and dataset membership.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, Tuple


def compute_advantage(auc: float) -> float:
    """Compute attack advantage from AUC.

    Advantage = 2 * |AUC - 0.5|

    This is direction-agnostic: AUC=0.38 and AUC=0.62 both give advantage=0.24.
    An advantage of 0 means the attacker is at chance (no distinguishing power).
    An advantage of 1 means perfect distinguishing (AUC=0 or AUC=1).
    """
    return 2 * abs(auc - 0.5)


def compute_attack_success(auc: float) -> float:
    """Compute attack success from AUC.

    Attack success = max(AUC, 1 - AUC)

    This is equivalent to advantage but on [0.5, 1] scale.
    A success of 0.5 means chance; 1.0 means perfect distinguishing.
    """
    return max(auc, 1 - auc)


def compute_attack_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute standard attack metrics including TPR@FPR and advantage."""
    auc = roc_auc_score(labels, predictions)
    accuracy = accuracy_score(labels, (predictions >= 0.5))

    # Advantage metric (direction-agnostic)
    advantage = compute_advantage(auc)
    attack_success = compute_attack_success(auc)

    # Compute TPR at specific FPR thresholds
    fpr, tpr, _ = roc_curve(labels, predictions)

    # TPR at 1% FPR
    idx_01 = np.where(fpr <= 0.01)[0]
    tpr_at_fpr_01 = tpr[idx_01[-1]] if len(idx_01) > 0 else 0.0

    # TPR at 5% FPR
    idx_05 = np.where(fpr <= 0.05)[0]
    tpr_at_fpr_05 = tpr[idx_05[-1]] if len(idx_05) > 0 else 0.0

    return {
        'auc': float(auc),
        'advantage': float(advantage),
        'attack_success': float(attack_success),
        'accuracy': float(accuracy),
        'tpr_at_fpr_01': float(tpr_at_fpr_01),
        'tpr_at_fpr_05': float(tpr_at_fpr_05)
    }


def compute_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """Compute confidence intervals for AUC and advantage via bootstrap.

    Returns:
        Dictionary with 'auc' and 'advantage' keys, each containing
        (mean, lower_bound, upper_bound) tuple.
    """
    rng = np.random.RandomState(seed)
    aucs = []
    advantages = []

    n_samples = len(labels)
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(n_samples, size=n_samples, replace=True)

        # Check if we have both classes
        if len(np.unique(labels[idx])) < 2:
            continue

        auc = roc_auc_score(labels[idx], predictions[idx])
        aucs.append(auc)
        advantages.append(compute_advantage(auc))

    aucs = np.array(aucs)
    advantages = np.array(advantages)

    alpha = 1 - confidence

    auc_result = (
        float(aucs.mean()),
        float(np.percentile(aucs, alpha/2 * 100)),
        float(np.percentile(aucs, (1 - alpha/2) * 100))
    )

    adv_result = (
        float(advantages.mean()),
        float(np.percentile(advantages, alpha/2 * 100)),
        float(np.percentile(advantages, (1 - alpha/2) * 100))
    )

    return {
        'auc': auc_result,
        'advantage': adv_result
    }


def cluster_conditioned_evaluation(
    predictions: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    method: str = 'residualize'
) -> Dict[str, float]:
    """Evaluate membership inference while controlling for cluster identity.

    Args:
        predictions: Attacker predictions (probabilities)
        labels: True membership labels (1=member, 0=non-member)
        clusters: Cluster assignments (string or int)
        method: 'residualize' or 'stratified'
            - residualize: Use logistic regression with cluster fixed effects
            - stratified: Compute within-cluster AUCs and average

    Returns:
        Dictionary with conditioned metrics
    """
    if method == 'residualize':
        return _residualized_evaluation(predictions, labels, clusters)
    elif method == 'stratified':
        return _stratified_evaluation(predictions, labels, clusters)
    else:
        raise ValueError(f"Unknown method: {method}")


def _residualized_evaluation(
    predictions: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray
) -> Dict[str, float]:
    """Residualize cluster effects via logistic regression.

    Fits: logit(P(member)) = beta_0 + beta_member * attacker_score + sum_c gamma_c * I(cluster=c)

    The conditioned AUC is computed from beta_member coefficient's predictions
    after removing cluster effects.
    """
    from sklearn.preprocessing import LabelEncoder

    # Encode clusters as integers
    le = LabelEncoder()
    cluster_encoded = le.fit_transform(clusters)
    n_clusters = len(le.classes_)

    # Create one-hot encoding for clusters (drop first for identifiability)
    cluster_onehot = np.zeros((len(clusters), n_clusters - 1))
    for i in range(1, n_clusters):
        cluster_onehot[:, i-1] = (cluster_encoded == i).astype(float)

    # Design matrix: [attacker_score, cluster_dummies]
    X = np.column_stack([predictions.reshape(-1, 1), cluster_onehot])

    # Fit logistic regression
    clf = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
    clf.fit(X, labels)

    # Get predictions using only the attacker score (without cluster effects)
    # This is equivalent to setting all cluster coefficients to their mean (0)
    X_no_cluster = np.column_stack([
        predictions.reshape(-1, 1),
        np.zeros((len(predictions), n_clusters - 1))
    ])

    conditioned_preds = clf.predict_proba(X_no_cluster)[:, 1]

    # Compute metrics on conditioned predictions
    metrics = compute_attack_metrics(conditioned_preds, labels)

    # Add coefficient info
    metrics['membership_coef'] = float(clf.coef_[0, 0])
    metrics['method'] = 'residualize'

    return metrics


def _stratified_evaluation(
    predictions: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray
) -> Dict[str, float]:
    """Compute within-cluster AUCs and average.

    For each cluster with both members and non-members, compute AUC.
    Average across clusters (weighted by size).
    """
    unique_clusters = np.unique(clusters)

    cluster_aucs = []
    cluster_weights = []

    for cluster in unique_clusters:
        mask = clusters == cluster
        cluster_preds = predictions[mask]
        cluster_labels = labels[mask]

        # Need both classes present
        if len(np.unique(cluster_labels)) < 2:
            continue

        auc = roc_auc_score(cluster_labels, cluster_preds)
        cluster_aucs.append(auc)
        cluster_weights.append(len(cluster_labels))

    if len(cluster_aucs) == 0:
        return compute_attack_metrics(predictions, labels)

    cluster_aucs = np.array(cluster_aucs)
    cluster_weights = np.array(cluster_weights)
    avg_auc = np.average(cluster_aucs, weights=cluster_weights)

    metrics = compute_attack_metrics(predictions, labels)
    metrics['auc'] = float(avg_auc)
    metrics['n_clusters_evaluated'] = len(cluster_aucs)
    metrics['method'] = 'stratified'

    return metrics


def matched_negative_evaluation(
    attacker,
    forget_features: torch.Tensor,
    matched_negative_features: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate F vs matched negatives (within-cluster comparison).

    This uses the latent k-NN matched negatives as a proxy for held-out
    cells from the same cluster.

    Args:
        attacker: Trained MLPAttacker model
        forget_features: Features for forget set (members)
        matched_negative_features: Features for matched negatives (non-members)
        device: Device to run on

    Returns:
        Metrics for F vs matched negatives
    """
    attacker.eval()

    # Create labels
    member_labels = np.ones(len(forget_features))
    nonmember_labels = np.zeros(len(matched_negative_features))

    # Combine
    all_features = torch.cat([forget_features, matched_negative_features], dim=0)
    all_labels = np.concatenate([member_labels, nonmember_labels])

    # Get predictions
    with torch.no_grad():
        all_features = all_features.to(device)
        logits = attacker(all_features).squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()

    # Compute metrics
    metrics = compute_attack_metrics(predictions, all_labels)

    # Add CI for AUC and advantage
    ci_results = compute_confidence_interval(predictions, all_labels)
    auc_mean, auc_lower, auc_upper = ci_results['auc']
    adv_mean, adv_lower, adv_upper = ci_results['advantage']

    metrics['auc_mean'] = auc_mean
    metrics['auc_ci_lower'] = auc_lower
    metrics['auc_ci_upper'] = auc_upper
    metrics['advantage_mean'] = adv_mean
    metrics['advantage_ci_lower'] = adv_lower
    metrics['advantage_ci_upper'] = adv_upper
    metrics['comparison'] = 'F_vs_matched_negatives'

    return metrics


def evaluate_with_conditioning(
    attacker,
    forget_features: torch.Tensor,
    negative_features: torch.Tensor,
    forget_clusters: np.ndarray,
    negative_clusters: np.ndarray,
    device: str = 'cpu',
    method: str = 'residualize'
) -> Dict[str, Dict[str, float]]:
    """Evaluate attacker with both global and cluster-conditioned metrics.

    Returns both global (confounded) and conditioned (true privacy) metrics.

    Returns:
        {
            'global': {auc, accuracy, ...},
            'conditioned': {auc, accuracy, ...}
        }
    """
    attacker.eval()

    # Create labels
    member_labels = np.ones(len(forget_features))
    nonmember_labels = np.zeros(len(negative_features))

    # Combine
    all_features = torch.cat([forget_features, negative_features], dim=0)
    all_labels = np.concatenate([member_labels, nonmember_labels])
    all_clusters = np.concatenate([forget_clusters, negative_clusters])

    # Get predictions
    with torch.no_grad():
        all_features = all_features.to(device)
        logits = attacker(all_features).squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()

    # Global metrics (confounded)
    global_metrics = compute_attack_metrics(predictions, all_labels)
    ci_results = compute_confidence_interval(predictions, all_labels)
    auc_mean, auc_lower, auc_upper = ci_results['auc']
    adv_mean, adv_lower, adv_upper = ci_results['advantage']

    global_metrics['auc_mean'] = auc_mean
    global_metrics['auc_ci_lower'] = auc_lower
    global_metrics['auc_ci_upper'] = auc_upper
    global_metrics['advantage_mean'] = adv_mean
    global_metrics['advantage_ci_lower'] = adv_lower
    global_metrics['advantage_ci_upper'] = adv_upper

    # Conditioned metrics (true privacy)
    conditioned_metrics = cluster_conditioned_evaluation(
        predictions, all_labels, all_clusters, method=method
    )

    return {
        'global': global_metrics,
        'conditioned': conditioned_metrics
    }


def load_matched_negatives(matched_path: str) -> np.ndarray:
    """Load matched negative indices from sanity check output."""
    import json

    with open(matched_path, 'r') as f:
        data = json.load(f)

    return np.array(data['matched_indices'])
