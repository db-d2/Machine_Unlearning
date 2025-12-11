"""Define forget sets for machine unlearning experiments.

Creates two types of forget sets:
- Structured: One complete cluster (smallest by default)
- Scattered: Random 10-20% of cells

References:
    - Traag et al. (2019). From Louvain to Leiden: guaranteeing well-connected
      communities. Scientific Reports. https://doi.org/10.1038/s41598-019-41695-z
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import scanpy as sc
from utils import set_global_seed, save_metadata, create_run_metadata


def compute_checksum(indices: np.ndarray) -> str:
    """Compute SHA256 checksum of cell indices."""
    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def define_structured_forget_set(
    adata,
    cluster_key: str = 'leiden',
    target_cluster: int = None,
    output_path: Path = None
):
    """
    Define structured forget set by removing one complete cluster.

    Args:
        adata: AnnData object with clustering results
        cluster_key: Key in adata.obs for cluster labels
        target_cluster: Specific cluster to remove (default: smallest)
        output_path: Path to save forget set indices

    Returns:
        Indices of cells in the forget set
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    # Get cluster sizes
    cluster_counts = adata.obs[cluster_key].value_counts()
    print(f"\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells")

    # Select target cluster (smallest by default)
    if target_cluster is None:
        target_cluster = cluster_counts.idxmin()

    print(f"\nSelected cluster {target_cluster} for structured forgetting")
    print(f"  Size: {cluster_counts[target_cluster]} cells ({100*cluster_counts[target_cluster]/len(adata):.1f}%)")

    # Get indices of cells in target cluster
    forget_mask = adata.obs[cluster_key] == target_cluster
    forget_indices = np.where(forget_mask)[0]

    # Save forget set
    if output_path:
        checksum = compute_checksum(forget_indices)
        output_data = {
            'indices': forget_indices.tolist(),
            'cluster': str(target_cluster),
            'size': int(len(forget_indices)),
            'fraction': float(len(forget_indices) / len(adata)),
            'checksum': checksum
        }

        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved structured forget set: {output_path}")
        print(f"  Checksum: {checksum}")

    return forget_indices


def define_scattered_forget_set(
    adata,
    fraction: float = 0.15,
    seed: int = 42,
    output_path: Path = None
):
    """
    Define scattered forget set by random sampling.

    Args:
        adata: AnnData object
        fraction: Fraction of cells to forget (0.1-0.2)
        seed: Random seed
        output_path: Path to save forget set indices

    Returns:
        Indices of cells in the forget set
    """
    if not (0.1 <= fraction <= 0.2):
        raise ValueError(f"Fraction must be in [0.1, 0.2], got {fraction}")

    np.random.seed(seed)

    n_cells = len(adata)
    n_forget = int(n_cells * fraction)

    # Random sampling without replacement
    forget_indices = np.random.choice(n_cells, size=n_forget, replace=False)
    forget_indices = np.sort(forget_indices)

    print(f"\nScattered forget set:")
    print(f"  Size: {len(forget_indices)} cells ({100*len(forget_indices)/n_cells:.1f}%)")

    # Save forget set
    if output_path:
        checksum = compute_checksum(forget_indices)
        output_data = {
            'indices': forget_indices.tolist(),
            'size': int(len(forget_indices)),
            'fraction': float(len(forget_indices) / n_cells),
            'seed': seed,
            'checksum': checksum
        }

        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved scattered forget set: {output_path}")
        print(f"  Checksum: {checksum}")

    return forget_indices


def main(args):
    set_global_seed(args.seed)

    # Configure scanpy for single-threaded execution to avoid OpenMP conflicts
    sc.settings.n_jobs = 1

    print("Loading preprocessed data...")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded: {adata.shape}")

    # Perform clustering if not already done
    if args.cluster_key not in adata.obs:
        print(f"\nClustering not found, running Leiden clustering...")

        # Compute PCA if not present
        if 'X_pca' not in adata.obsm:
            print("Computing PCA...")
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

        # Compute neighbors
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

        # Leiden clustering (single-threaded to avoid segfaults on Intel Mac)
        print("Running Leiden clustering...")
        sc.tl.leiden(adata, resolution=args.resolution, key_added=args.cluster_key)

        # Save updated adata
        adata.write(args.data_path)
        print(f"Saved clustering results to {args.data_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define structured forget set
    print("\n" + "="*60)
    print("STRUCTURED FORGET SET")
    print("="*60)

    structured_indices = define_structured_forget_set(
        adata,
        cluster_key=args.cluster_key,
        target_cluster=args.target_cluster,
        output_path=output_dir / "forget_structured.json"
    )

    # Define scattered forget set
    print("\n" + "="*60)
    print("SCATTERED FORGET SET")
    print("="*60)

    scattered_indices = define_scattered_forget_set(
        adata,
        fraction=args.scatter_fraction,
        seed=args.seed,
        output_path=output_dir / "forget_scattered.json"
    )

    # Save metadata
    config = vars(args)
    metadata = create_run_metadata("forget_set_definition", config, args.seed)
    metadata['structured_size'] = int(len(structured_indices))
    metadata['scattered_size'] = int(len(scattered_indices))
    metadata['total_cells'] = int(len(adata))

    save_metadata(metadata, output_dir)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells: {len(adata)}")
    print(f"Structured forget set: {len(structured_indices)} cells")
    print(f"Scattered forget set: {len(scattered_indices)} cells")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="data/adata_processed.h5ad")
    parser.add_argument("--output_dir", type=str, default="data/forget_sets")

    # Clustering
    parser.add_argument("--cluster_key", type=str, default="leiden")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--target_cluster", type=int, default=None,
                        help="Specific cluster to remove (default: smallest)")

    # Scattered
    parser.add_argument("--scatter_fraction", type=float, default=0.15,
                        help="Fraction of cells for scattered forget set (0.1-0.2)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
