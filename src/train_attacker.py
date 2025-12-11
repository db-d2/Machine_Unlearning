"""Train membership inference attacker for privacy auditing.

This script trains an MLP-based attacker to perform membership inference
on a trained VAE model. It evaluates two-negative privacy:
- F vs Unseen: Can attacker distinguish forget set from unseen test data?
- F vs Retain: Can attacker distinguish forget set from retained training data?
"""

import argparse
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split

from vae import VAE
from attacker import (
    MLPAttacker,
    extract_vae_features,
    compute_knn_distances,
    build_attack_features,
    compute_attack_metrics,
    compute_confidence_interval
)
from utils import set_global_seed, create_run_metadata, save_metadata, DEVICE
from logging_utils import save_metrics_json


def load_forget_set(forget_set_path: str) -> np.ndarray:
    """Load forget set indices from JSON file."""
    with open(forget_set_path, 'r') as f:
        forget_data = json.load(f)
    return np.array(forget_data['indices'])


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
    """Evaluate attacker."""
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

    return compute_attack_metrics(predictions, labels)


def main(args):
    set_global_seed(args.seed)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load forget set
    print("\nLoading forget set...")
    forget_indices = load_forget_set(args.forget_set_path)
    print(f"Forget set: {len(forget_indices)} cells")

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Total cells: {adata.shape}")

    # Create splits
    n_cells = adata.n_obs
    all_indices = np.arange(n_cells)

    # Retain set (R) = all cells not in forget set
    retain_mask = np.ones(n_cells, dtype=bool)
    retain_mask[forget_indices] = False
    retain_indices = all_indices[retain_mask]

    # Split retain into train/test for attacker
    # Use same split as VAE training (85/15)
    np.random.seed(args.seed)
    np.random.shuffle(retain_indices)
    n_train = int(0.85 * len(retain_indices))
    retain_train_indices = retain_indices[:n_train]
    retain_test_indices = retain_indices[n_train:]

    # Unseen set (U) = held-out test set (will overlap with retain_test, which is fine)
    unseen_indices = retain_test_indices

    print(f"\nData splits:")
    print(f"  Retain (R): {len(retain_indices)}")
    print(f"  Retain train: {len(retain_train_indices)}")
    print(f"  Retain test: {len(retain_test_indices)}")
    print(f"  Forget (F): {len(forget_indices)}")
    print(f"  Unseen (U): {len(unseen_indices)}")

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

    # Extract features and latent codes
    print("\nExtracting features...")

    # First, get reference latent codes for kNN
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

    print("  Extracting features for retain set...")
    retain_feats, retain_knn_r, retain_knn_u = extract_features_for_split(
        model, adata, retain_test_indices, args.batch_size, device,
        reference_z_retain, reference_z_unseen
    )

    print("  Extracting features for unseen set...")
    unseen_feats, unseen_knn_r, unseen_knn_u = extract_features_for_split(
        model, adata, unseen_indices, args.batch_size, device,
        reference_z_retain, reference_z_unseen
    )

    # Build attack features
    print("\nBuilding attack features...")
    forget_X = build_attack_features(forget_feats, forget_knn_r, forget_knn_u)
    retain_X = build_attack_features(retain_feats, retain_knn_r, retain_knn_u)
    unseen_X = build_attack_features(unseen_feats, unseen_knn_r, unseen_knn_u)

    feature_dim = forget_X.shape[1]
    print(f"Feature dimension: {feature_dim}")

    # Train two attackers: F vs Unseen, and F vs Retain
    results = {}

    for scenario, member_X, nonmember_X in [
        ("F_vs_Unseen", forget_X, unseen_X),
        ("F_vs_Retain", forget_X, retain_X)
    ]:
        print(f"\n{'='*60}")
        print(f"Training attacker: {scenario}")
        print(f"{'='*60}")

        # Create labels: 1 = member (forget), 0 = non-member
        member_labels = torch.ones(len(member_X))
        nonmember_labels = torch.zeros(len(nonmember_X))

        # Combine and shuffle
        all_X = torch.cat([member_X, nonmember_X], dim=0)
        all_labels = torch.cat([member_labels, nonmember_labels], dim=0)

        # Split into train/test for attacker
        indices = torch.randperm(len(all_X))
        n_train_att = int(0.8 * len(all_X))

        train_X = all_X[indices[:n_train_att]]
        train_y = all_labels[indices[:n_train_att]]
        test_X = all_X[indices[n_train_att:]]
        test_y = all_labels[indices[n_train_att:]]

        print(f"Attacker train: {len(train_X)}, test: {len(test_X)}")

        # Create dataloaders
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Initialize attacker
        attacker = MLPAttacker(
            input_dim=feature_dim,
            hidden_dims=[256, 256],
            dropout=0.3
        ).to(device)

        optimizer = optim.Adam(attacker.parameters(), lr=args.lr)

        # Train
        print(f"\nTraining attacker for {args.epochs} epochs...")
        best_test_auc = 0.0

        for epoch in range(args.epochs):
            train_loss = train_attacker_epoch(attacker, train_loader, optimizer, device)

            if (epoch + 1) % 10 == 0:
                test_metrics = eval_attacker(attacker, test_loader, device)
                print(f"Epoch {epoch+1}/{args.epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Test AUC: {test_metrics['auc']:.4f}")

                if test_metrics['auc'] > best_test_auc:
                    best_test_auc = test_metrics['auc']

        # Final evaluation
        final_metrics = eval_attacker(attacker, test_loader, device)

        print(f"\n{scenario} Results:")
        print(f"  AUC: {final_metrics['auc']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  TPR@1%FPR: {final_metrics['tpr_at_fpr_01']:.4f}")
        print(f"  TPR@5%FPR: {final_metrics['tpr_at_fpr_05']:.4f}")

        results[scenario] = final_metrics

        # Save attacker model
        torch.save(attacker.state_dict(), output_dir / f"attacker_{scenario}.pt")

    # Save results
    save_metrics_json(results, output_dir, "attack_results.json")

    # Save metadata
    config = vars(args)
    metadata = create_run_metadata("attacker_training", config, args.seed)
    metadata['forget_set_size'] = int(len(forget_indices))
    metadata['results'] = results
    save_metadata(metadata, output_dir)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"F vs Unseen AUC: {results['F_vs_Unseen']['auc']:.4f}")
    print(f"F vs Retain AUC: {results['F_vs_Retain']['auc']:.4f}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="data/adata_processed.h5ad")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--forget_set_path", type=str, required=True,
                        help="Path to forget set JSON")
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

    args = parser.parse_args()
    main(args)
