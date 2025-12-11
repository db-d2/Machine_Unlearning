"""Re-run structured audit with cluster conditioning.

This script implements the corrected baseline audit that controls for
cluster identity confounding.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc

from vae import VAE
from attacker import (
    MLPAttacker,
    extract_vae_features,
    compute_knn_distances,
    build_attack_features
)
from attacker_eval import (
    matched_negative_evaluation,
    evaluate_with_conditioning,
    load_matched_negatives
)
from utils import set_global_seed, create_run_metadata, save_metadata, DEVICE
from logging_utils import save_metrics_json


def load_forget_set(forget_set_path: str):
    """Load forget set indices and cluster from JSON file."""
    with open(forget_set_path, 'r') as f:
        forget_data = json.load(f)
    return np.array(forget_data['indices']), forget_data['cluster']


def extract_features_for_split(
    model,
    adata,
    indices: np.ndarray,
    batch_size: int,
    device: str,
    reference_z_retain: np.ndarray = None,
    reference_z_unseen: np.ndarray = None
):
    """Extract features for a data split."""
    features_list = []
    knn_retain_list = []
    knn_unseen_list = []

    # Process in batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]

        X = torch.FloatTensor(
            adata.X[batch_indices].toarray()
            if hasattr(adata.X[batch_indices], 'toarray')
            else adata.X[batch_indices]
        )
        lib_size = torch.FloatTensor(X.sum(dim=1, keepdim=True))

        # Extract VAE features
        vae_feats = extract_vae_features(model, X, lib_size, device)
        features_list.append(vae_feats)

        # Compute kNN distances if reference sets provided
        z_np = vae_feats['z'].numpy()

        if reference_z_retain is not None:
            knn_dist = compute_knn_distances(z_np, reference_z_retain, k=5)
            knn_retain_list.append(knn_dist)

        if reference_z_unseen is not None:
            knn_dist = compute_knn_distances(z_np, reference_z_unseen, k=5)
            knn_unseen_list.append(knn_dist)

    # Concatenate all batches
    combined_features = {}
    for key in features_list[0].keys():
        combined_features[key] = torch.cat([f[key] for f in features_list], dim=0)

    knn_retain = np.concatenate(knn_retain_list) if knn_retain_list else None
    knn_unseen = np.concatenate(knn_unseen_list) if knn_unseen_list else None

    return combined_features, knn_retain, knn_unseen


def train_attacker_epoch(model, dataloader, optimizer, device):
    """Train attacker for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(x).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_attacker(model, dataloader, device):
    """Evaluate attacker and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return predictions, labels


def main(args):
    set_global_seed(args.seed)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load forget set
    print("\nLoading forget set...")
    forget_indices, forget_cluster = load_forget_set(args.forget_set_path)
    print(f"Forget set: {len(forget_indices)} cells (Cluster {forget_cluster})")

    # Load matched negatives
    print("\nLoading matched negatives...")
    matched_indices = load_matched_negatives(args.matched_negatives_path)
    print(f"Matched negatives: {len(matched_indices)} cells")

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Total cells: {adata.shape}")

    # Get cluster labels
    if 'leiden' not in adata.obs:
        raise ValueError("Leiden clusters not found in adata.obs")

    clusters = adata.obs['leiden'].astype(str).values

    # Create splits
    n_cells = adata.n_obs
    all_indices = np.arange(n_cells)

    # Retain set (R) = all cells not in forget set
    retain_mask = np.ones(n_cells, dtype=bool)
    retain_mask[forget_indices] = False
    retain_indices = all_indices[retain_mask]

    # Split retain into train/test
    np.random.seed(args.seed)
    np.random.shuffle(retain_indices)
    n_train = int(0.85 * len(retain_indices))
    retain_train_indices = retain_indices[:n_train]
    retain_test_indices = retain_indices[n_train:]

    unseen_indices = retain_test_indices

    print(f"\nData splits:")
    print(f"  Forget (F): {len(forget_indices)}")
    print(f"  Matched negatives: {len(matched_indices)}")
    print(f"  Retain train: {len(retain_train_indices)}")
    print(f"  Retain test / Unseen: {len(unseen_indices)}")

    # Load VAE model
    print("\nLoading VAE model...")
    checkpoint = torch.load(args.model_path, map_location=device)

    model = VAE(
        input_dim=adata.n_vars,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        likelihood=args.likelihood
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"VAE loaded from {args.model_path}")

    # Extract features
    print("\nExtracting features...")

    # Reference latent codes for kNN
    print("  Computing reference latent codes...")
    ref_retain_feats, _, _ = extract_features_for_split(
        model, adata, retain_train_indices, args.batch_size, device
    )
    reference_z_retain = ref_retain_feats['z'].numpy()

    ref_unseen_feats, _, _ = extract_features_for_split(
        model, adata, unseen_indices, args.batch_size, device
    )
    reference_z_unseen = ref_unseen_feats['z'].numpy()

    # Extract features for all splits
    print("  Extracting features for forget set...")
    forget_feats, forget_knn_r, forget_knn_u = extract_features_for_split(
        model, adata, forget_indices, args.batch_size, device,
        reference_z_retain, reference_z_unseen
    )

    print("  Extracting features for matched negatives...")
    matched_feats, matched_knn_r, matched_knn_u = extract_features_for_split(
        model, adata, matched_indices, args.batch_size, device,
        reference_z_retain, reference_z_unseen
    )

    print("  Extracting features for retain set...")
    retain_feats, retain_knn_r, retain_knn_u = extract_features_for_split(
        model, adata, retain_test_indices, args.batch_size, device,
        reference_z_retain, reference_z_unseen
    )

    # Build attack features
    print("\nBuilding attack features...")
    forget_X = build_attack_features(forget_feats, forget_knn_r, forget_knn_u)
    matched_X = build_attack_features(matched_feats, matched_knn_r, matched_knn_u)
    retain_X = build_attack_features(retain_feats, retain_knn_r, retain_knn_u)

    feature_dim = forget_X.shape[1]
    print(f"Feature dimension: {feature_dim}")

    # Train two attackers with conditioning
    results = {}

    # Scenario 1: F vs Matched Negatives (within-cluster, using latent k-NN)
    print(f"\n{'='*60}")
    print(f"Training attacker: F vs Matched Negatives")
    print(f"{'='*60}")

    # Create labels
    member_labels = torch.ones(len(forget_X))
    matched_labels = torch.zeros(len(matched_X))

    # Combine and shuffle
    all_X_matched = torch.cat([forget_X, matched_X], dim=0)
    all_labels_matched = torch.cat([member_labels, matched_labels], dim=0)

    # Split into train/test
    indices = torch.randperm(len(all_X_matched))
    n_train_att = int(0.8 * len(all_X_matched))

    train_X = all_X_matched[indices[:n_train_att]]
    train_y = all_labels_matched[indices[:n_train_att]]
    test_X = all_X_matched[indices[n_train_att:]]
    test_y = all_labels_matched[indices[n_train_att:]]

    print(f"Attacker train: {len(train_X)}, test: {len(test_X)}")

    # Create dataloaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize attacker
    attacker_matched = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(attacker_matched.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining attacker for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_attacker_epoch(attacker_matched, train_loader, optimizer, device)

        if (epoch + 1) % 10 == 0:
            preds, labels = eval_attacker(attacker_matched, test_loader, device)
            from attacker_eval import compute_attack_metrics
            test_metrics = compute_attack_metrics(preds, labels)
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Test AUC: {test_metrics['auc']:.4f}")

    # Final evaluation with matched negatives
    matched_metrics = matched_negative_evaluation(
        attacker_matched, forget_X, matched_X, device
    )

    print(f"\nF vs Matched Negatives Results:")
    print(f"  AUC: {matched_metrics['auc']:.4f} "
          f"[{matched_metrics['auc_ci_lower']:.4f}, {matched_metrics['auc_ci_upper']:.4f}]")
    print(f"  Accuracy: {matched_metrics['accuracy']:.4f}")
    print(f"  TPR@1%FPR: {matched_metrics['tpr_at_fpr_01']:.4f}")
    print(f"  TPR@5%FPR: {matched_metrics['tpr_at_fpr_05']:.4f}")

    results['F_vs_Matched'] = matched_metrics

    # Save attacker model
    torch.save(attacker_matched.state_dict(), output_dir / "attacker_F_vs_Matched.pt")

    # Scenario 2: F vs Retain (cluster-conditioned)
    print(f"\n{'='*60}")
    print(f"Training attacker: F vs Retain (cluster-conditioned)")
    print(f"{'='*60}")

    # Create labels
    retain_labels = torch.zeros(len(retain_X))

    # Combine
    all_X_retain = torch.cat([forget_X, retain_X], dim=0)
    all_labels_retain = torch.cat([member_labels, retain_labels], dim=0)

    # Split into train/test
    indices = torch.randperm(len(all_X_retain))
    n_train_att = int(0.8 * len(all_X_retain))

    train_X = all_X_retain[indices[:n_train_att]]
    train_y = all_labels_retain[indices[:n_train_att]]
    test_X = all_X_retain[indices[n_train_att:]]
    test_y = all_labels_retain[indices[n_train_att:]]

    print(f"Attacker train: {len(train_X)}, test: {len(test_X)}")

    # Create dataloaders
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize attacker
    attacker_retain = MLPAttacker(
        input_dim=feature_dim,
        hidden_dims=[256, 256],
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(attacker_retain.parameters(), lr=args.lr)

    # Train
    print(f"\nTraining attacker for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_attacker_epoch(attacker_retain, train_loader, optimizer, device)

        if (epoch + 1) % 10 == 0:
            preds, labels = eval_attacker(attacker_retain, test_loader, device)
            from attacker_eval import compute_attack_metrics
            test_metrics = compute_attack_metrics(preds, labels)
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Test AUC: {test_metrics['auc']:.4f}")

    # Final evaluation with conditioning
    # Get cluster labels for forget and retain sets
    forget_clusters = clusters[forget_indices]
    retain_clusters = clusters[retain_test_indices]

    # Map test indices back to original features
    test_forget_mask = []
    test_retain_mask = []
    for idx in indices[n_train_att:].tolist():
        if idx < len(forget_X):
            test_forget_mask.append(idx)
        else:
            test_retain_mask.append(idx - len(forget_X))

    test_forget_mask = torch.tensor(test_forget_mask)
    test_retain_mask = torch.tensor(test_retain_mask)

    test_forget_X = forget_X[test_forget_mask]
    test_retain_X = retain_X[test_retain_mask]
    test_forget_clusters = forget_clusters[test_forget_mask.numpy()]
    test_retain_clusters = retain_clusters[test_retain_mask.numpy()]

    retain_metrics = evaluate_with_conditioning(
        attacker_retain,
        test_forget_X,
        test_retain_X,
        test_forget_clusters,
        test_retain_clusters,
        device,
        method=args.conditioning_method
    )

    print(f"\nF vs Retain Results:")
    print(f"  Global AUC (confounded): {retain_metrics['global']['auc']:.4f} "
          f"[{retain_metrics['global']['auc_ci_lower']:.4f}, "
          f"{retain_metrics['global']['auc_ci_upper']:.4f}]")
    print(f"  Conditioned AUC (true privacy): {retain_metrics['conditioned']['auc']:.4f}")
    print(f"  Global TPR@1%FPR: {retain_metrics['global']['tpr_at_fpr_01']:.4f}")
    print(f"  Global TPR@5%FPR: {retain_metrics['global']['tpr_at_fpr_05']:.4f}")

    results['F_vs_Retain'] = retain_metrics

    # Save attacker model
    torch.save(attacker_retain.state_dict(), output_dir / "attacker_F_vs_Retain.pt")

    # Save results
    save_metrics_json(results, output_dir, "attack_results_conditioned.json")

    # Save sanity metadata
    sanity_meta = {
        "conditioning_method": args.conditioning_method,
        "matched_negatives_method": "latent_knn",
        "matched_negatives_count": int(len(matched_indices)),
        "forget_cluster": forget_cluster,
        "n_clusters_in_retain": int(len(np.unique(retain_clusters)))
    }

    with open(output_dir / "sanity_meta.json", 'w') as f:
        json.dump(sanity_meta, f, indent=2)

    # Save metadata
    config = vars(args)
    metadata = create_run_metadata("P1.5.S6", config, args.seed)
    metadata['forget_set_size'] = int(len(forget_indices))
    metadata['matched_negatives_size'] = int(len(matched_indices))
    metadata['results'] = results
    save_metadata(metadata, output_dir)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"F vs Matched Negatives AUC: {results['F_vs_Matched']['auc']:.4f}")
    print(f"F vs Retain (global, confounded): {results['F_vs_Retain']['global']['auc']:.4f}")
    print(f"F vs Retain (conditioned, true): {results['F_vs_Retain']['conditioned']['auc']:.4f}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="data/adata_processed.h5ad")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained VAE checkpoint")
    parser.add_argument("--forget_set_path", type=str, required=True,
                       help="Path to forget set JSON")
    parser.add_argument("--matched_negatives_path", type=str, required=True,
                       help="Path to matched negatives JSON from S1")
    parser.add_argument("--output_dir", type=str, required=True)

    # VAE architecture (must match trained model)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 128])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--likelihood", type=str, default="nb")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    # Conditioning
    parser.add_argument("--conditioning_method", type=str, default="residualize",
                       choices=["residualize", "stratified"],
                       help="Method for cluster conditioning")

    args = parser.parse_args()
    main(args)
