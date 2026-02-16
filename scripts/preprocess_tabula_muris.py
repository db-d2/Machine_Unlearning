#!/usr/bin/env python
"""
Preprocess Tabula Muris 10x droplet data for cross-dataset validation.

This script applies the same preprocessing pipeline as PBMC-33k:
- QC filtering (min 200 genes, max 10% mito)
- Normalization (CP10K + log1p)
- HVG selection (2000 genes, Seurat v3)
- Clustering (PCA, k-NN, UMAP, Leiden)

Data source: https://figshare.com/articles/dataset/5968960
Paper: https://doi.org/10.1038/s41586-018-0590-4

Usage:
    python scripts/preprocess_tabula_muris.py

Output:
    data/tabula_muris_processed.h5ad
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import os
import argparse

from utils import set_global_seed, GLOBAL_SEED


def load_tissue_samples(droplet_dir: Path, exclude: list = None) -> sc.AnnData:
    """Load and concatenate all tissue samples from 10x format."""
    if exclude is None:
        exclude = []

    tissue_folders = sorted([
        d for d in os.listdir(droplet_dir)
        if os.path.isdir(droplet_dir / d) and d not in exclude
    ])

    print(f"Loading {len(tissue_folders)} tissue samples...")
    if exclude:
        print(f"  (excluding {len(exclude)} unfiltered samples)")

    adatas = []
    for folder in tissue_folders:
        path = droplet_dir / folder
        try:
            adata_tissue = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
            tissue_name = folder.split('-')[0]
            # Assign tissue and channel to all cells in this sample
            adata_tissue.obs['tissue'] = tissue_name
            adata_tissue.obs['channel'] = folder
            # Make cell barcodes unique by adding channel prefix
            adata_tissue.obs_names = [f'{folder}_{bc}' for bc in adata_tissue.obs_names]
            adatas.append(adata_tissue)
            print(f"  {folder}: {adata_tissue.n_obs} cells")
        except Exception as e:
            print(f"  {folder}: FAILED - {e}")

    print("\nConcatenating...")
    adata = sc.concat(adatas, join='outer')
    adata.obs_names_make_unique()

    return adata


def add_annotations(adata: sc.AnnData, annotations_path: Path) -> int:
    """Add cell type annotations from CSV file."""
    annotations = pd.read_csv(annotations_path, low_memory=False)
    print(f"Annotations file: {len(annotations)} cells")

    cell_id_col = 'cell' if 'cell' in annotations.columns else annotations.columns[0]

    # Create mapping from annotation cell IDs to our obs_names
    # Our format: "Tissue-Channel_Barcode-1" (e.g., "Lung-10X_P7_8_AAACGGGAGGATATAC-1")
    # Annotation format: "Channel_Barcode" (e.g., "10X_P7_8_AAACGGGAGGATATAC")

    # Build lookup from annotation key (channel_barcode) to full obs_name
    obs_lookup = {}
    for obs_name in adata.obs_names:
        # Extract channel_barcode from "Tissue-Channel_Barcode-1"
        # Split on first '-' to get tissue, rest is channel_barcode-1
        parts = obs_name.split('-', 1)
        if len(parts) == 2:
            channel_barcode_with_suffix = parts[1]  # "10X_P4_3_BARCODE-1"
            # Remove trailing -1
            if channel_barcode_with_suffix.endswith('-1'):
                key = channel_barcode_with_suffix[:-2]
            else:
                key = channel_barcode_with_suffix
            obs_lookup[key] = obs_name

    # Build annotation dict
    annotations_dict = annotations.set_index(cell_id_col).to_dict(orient='index')

    # Add annotation columns (initialize with empty string for h5ad compatibility)
    ann_cols = [c for c in annotations.columns if c != cell_id_col]
    for col in ann_cols:
        adata.obs[col] = ''

    matched = 0
    for ann_cell_id, ann_data in annotations_dict.items():
        if ann_cell_id in obs_lookup:
            obs_name = obs_lookup[ann_cell_id]
            matched += 1
            for col, val in ann_data.items():
                # Convert to string for h5ad compatibility
                adata.obs.loc[obs_name, col] = str(val) if pd.notna(val) else ''

    return matched


def apply_qc_filters(adata: sc.AnnData, min_genes: int = 200, max_mito_pct: float = 10.0) -> sc.AnnData:
    """Apply QC filters matching PBMC preprocessing."""
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    # Calculate QC metrics (mouse uses lowercase 'mt-')
    adata.var['mt'] = adata.var_names.str.lower().str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    n_before = adata.n_obs

    # Filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct, :].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    n_after = adata.n_obs
    print(f"QC filtering: {n_before} -> {n_after} cells ({n_before - n_after} removed, {100*(n_before-n_after)/n_before:.1f}%)")

    return adata


def normalize_and_hvg(adata: sc.AnnData, n_top_genes: int = 2000) -> sc.AnnData:
    """Normalize and select highly variable genes."""
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized data
    adata.raw = adata

    # HVG selection (Seurat v3 method, same as PBMC)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', layer='counts')
    n_hvg = adata.var['highly_variable'].sum()
    print(f"Selected {n_hvg} highly variable genes")

    # Subset to HVGs
    adata = adata[:, adata.var.highly_variable].copy()

    return adata


def cluster(adata: sc.AnnData, resolution: float = 0.5) -> sc.AnnData:
    """Run clustering pipeline."""
    # Scale for PCA
    sc.pp.scale(adata, max_value=10)

    # PCA
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    # UMAP
    sc.tl.umap(adata)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')

    return adata


def print_cluster_summary(adata: sc.AnnData, rare_threshold: int = 100):
    """Print cluster size summary."""
    cluster_counts = adata.obs['leiden'].value_counts().sort_values()

    print(f"\nClusters: {adata.obs['leiden'].nunique()}")
    print("\nCluster sizes:")
    for cluster, count in cluster_counts.items():
        marker = "<-- RARE" if count < rare_threshold else ""
        print(f"  Cluster {cluster}: {count:5d} {marker}")

    rare_clusters = cluster_counts[cluster_counts < rare_threshold]
    if len(rare_clusters) > 0:
        print(f"\nRare clusters (n < {rare_threshold}): {list(rare_clusters.index)}")
        print(f"Total rare cells: {rare_clusters.sum()}")

        # Find cluster closest to 30-50 cells (comparable to PBMC cluster 13)
        clusters_30_50 = cluster_counts[(cluster_counts >= 30) & (cluster_counts <= 50)]
        if len(clusters_30_50) > 0:
            print(f"\nClusters with 30-50 cells (comparable to PBMC cluster 13):")
            for cluster, count in clusters_30_50.items():
                print(f"  Cluster {cluster}: {count} cells")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Tabula Muris data')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--min-genes', type=int, default=200, help='Min genes per cell')
    parser.add_argument('--max-mito', type=float, default=10.0, help='Max mito percentage')
    parser.add_argument('--n-hvg', type=int, default=2000, help='Number of HVGs')
    parser.add_argument('--resolution', type=float, default=0.5, help='Leiden resolution')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED, help='Random seed')
    args = parser.parse_args()

    set_global_seed(args.seed)
    sc.settings.verbosity = 2

    DATA_DIR = Path(args.data_dir)
    DROPLET_DIR = DATA_DIR / 'tabula_muris_droplet' / 'droplet'
    ANNOTATIONS_PATH = DATA_DIR / 'annotations_droplet.csv'
    PROCESSED_PATH = DATA_DIR / 'tabula_muris_processed.h5ad'

    # Check data exists
    if not DROPLET_DIR.exists():
        print(f"ERROR: Droplet data not found at {DROPLET_DIR}")
        print("Please download from: https://figshare.com/articles/dataset/5968960")
        print("  curl -L -o data/droplet.zip 'https://ndownloader.figshare.com/files/10700167'")
        print("  unzip data/droplet.zip -d data/tabula_muris_droplet")
        sys.exit(1)

    # Samples to exclude (unfiltered 10x data with all barcodes)
    EXCLUDE = ['Lung-10X_P8_12', 'Lung-10X_P8_13', 'Trachea-10X_P8_14', 'Trachea-10X_P8_15']

    print("=" * 60)
    print("Tabula Muris Preprocessing")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"QC thresholds: min_genes={args.min_genes}, max_mito={args.max_mito}%")
    print(f"HVGs: {args.n_hvg}")
    print(f"Clustering resolution: {args.resolution}")
    print()

    # Load data
    print("=== Loading Data ===")
    adata = load_tissue_samples(DROPLET_DIR, exclude=EXCLUDE)
    print(f"Raw: {adata.n_obs} cells, {adata.n_vars} genes")

    # Add annotations
    print("\n=== Adding Annotations ===")
    if ANNOTATIONS_PATH.exists():
        matched = add_annotations(adata, ANNOTATIONS_PATH)
        print(f"Matched annotations: {matched} / {adata.n_obs}")
    else:
        print(f"Annotations file not found: {ANNOTATIONS_PATH}")

    # QC filtering
    print("\n=== QC Filtering ===")
    adata = apply_qc_filters(adata, min_genes=args.min_genes, max_mito_pct=args.max_mito)

    # Normalize and HVG
    print("\n=== Normalization & HVG Selection ===")
    adata = normalize_and_hvg(adata, n_top_genes=args.n_hvg)

    # Clustering
    print("\n=== Clustering ===")
    adata = cluster(adata, resolution=args.resolution)

    # Summary
    print_cluster_summary(adata)

    # Cells per tissue
    print("\nCells per tissue:")
    print(adata.obs['tissue'].value_counts())

    # Save
    print("\n=== Saving ===")
    # Convert categorical columns to strings for h5ad compatibility
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name == 'category':
            adata.obs[col] = adata.obs[col].astype(str)
        elif adata.obs[col].dtype == object:
            # Convert object columns to string, handling NaN
            adata.obs[col] = adata.obs[col].fillna('').astype(str)

    adata.write_h5ad(PROCESSED_PATH)
    print(f"Saved to {PROCESSED_PATH}")
    print(f"Final shape: {adata.shape}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
