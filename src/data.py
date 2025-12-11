"""Data loading and preprocessing for PBMC 33k dataset.

References:
    - Wolf et al. (2018). SCANPY: large-scale single-cell gene expression data analysis.
      Genome Biology. https://doi.org/10.1186/s13059-017-1382-0
    - Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
      Cell. https://doi.org/10.1016/j.cell.2019.05.031 (Seurat v3 HVG method)
"""

from pathlib import Path
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split


def load_10x_data(data_path: str, var_names: str = "gene_symbols") -> sc.AnnData:
    """
    Load 10x Genomics v2 data from matrix.mtx triplet format.

    Args:
        data_path: Path to directory containing matrix.mtx, genes.tsv, barcodes.tsv
        var_names: How to name variables ('gene_symbols' or 'gene_ids')

    Returns:
        AnnData object with raw counts
    """
    print(f"Loading 10x data from {data_path}")
    adata = sc.read_10x_mtx(
        data_path,
        var_names=var_names,
        cache=True
    )

    adata.var_names_make_unique()
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    return adata


def qc_filter(
    adata: sc.AnnData,
    min_genes: int = 200,
    max_mito_pct: float = 10.0
) -> sc.AnnData:
    """
    Apply quality control filtering to AnnData object.

    Standard scRNA-seq QC metrics following Luecken & Theis (2019).

    Args:
        adata: Input AnnData object
        min_genes: Minimum number of genes per cell
        max_mito_pct: Maximum percentage of mitochondrial genes

    Returns:
        Filtered AnnData object

    References:
        Luecken & Theis (2019). Current best practices in single-cell RNA-seq
        analysis: a tutorial. Mol Syst Biol. https://doi.org/10.15252/msb.20188746
    """
    print("Applying QC filters...")
    print(f"  Min genes per cell: {min_genes}")
    print(f"  Max mitochondrial %: {max_mito_pct}")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    n_cells_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct, :].copy()

    n_cells_after = adata.n_obs
    n_removed = n_cells_before - n_cells_after

    print(f"  Cells before QC: {n_cells_before}")
    print(f"  Cells after QC: {n_cells_after}")
    print(f"  Cells removed: {n_removed} ({100 * n_removed / n_cells_before:.1f}%)")

    return adata


def normalize_and_select_hvgs(
    adata: sc.AnnData,
    n_top_genes: int = 2000
) -> sc.AnnData:
    """
    Normalize counts and select highly variable genes.

    Normalization: Total count normalization followed by log1p transformation.
    HVG selection: Seurat v3 method (Stuart et al., 2019).

    Args:
        adata: Input AnnData object
        n_top_genes: Number of highly variable genes to select

    Returns:
        Processed AnnData object

    References:
        Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
        Cell. https://doi.org/10.1016/j.cell.2019.05.031
    """
    print("Normalizing and selecting HVGs...")

    # Total count normalization (CP10K)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Log transformation
    sc.pp.log1p(adata)

    # Seurat v3 HVG selection
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer=None,
        subset=False
    )

    n_hvg = adata.var["highly_variable"].sum()
    print(f"  Selected {n_hvg} highly variable genes")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print(f"  Final shape: {adata_hvg.n_obs} cells × {adata_hvg.n_vars} genes")

    return adata_hvg


def create_splits(
    adata: sc.AnnData,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Create train/val/test splits with a held-out set T ⊂ R.

    The holdout set T is a subset of the training data used specifically for
    measuring retraining equivalence (ELBO gap between unlearned and retrained models).

    Args:
        adata: Input AnnData object
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test', 'holdout' indices
    """
    print("Creating train/val/test splits...")

    n_cells = adata.n_obs
    indices = np.arange(n_cells)

    # Split into train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed
    )

    # Split train+val into train and val
    val_size = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed
    )

    # Reserve 10% of training data as holdout set T ⊂ R
    holdout_size = 0.1
    train_main_idx, holdout_idx = train_test_split(
        train_idx,
        test_size=holdout_size,
        random_state=seed
    )

    splits = {
        'train': train_main_idx,
        'val': val_idx,
        'test': test_idx,
        'holdout': holdout_idx
    }

    print(f"  Train: {len(train_main_idx)} cells")
    print(f"  Val: {len(val_idx)} cells")
    print(f"  Test: {len(test_idx)} cells")
    print(f"  Holdout (T ⊂ R for retraining equivalence): {len(holdout_idx)} cells")

    return splits


def load_and_preprocess_pbmc(
    data_path: str,
    output_path: str = "data/adata_processed.h5ad",
    min_genes: int = 200,
    max_mito_pct: float = 10.0,
    n_top_genes: int = 2000,
    seed: int = 42
) -> tuple:
    """
    Complete pipeline: load, QC, normalize, select HVGs, and split PBMC data.

    Args:
        data_path: Path to 10x data directory
        output_path: Path to save processed AnnData
        min_genes: Minimum genes per cell for QC
        max_mito_pct: Maximum mitochondrial percentage for QC
        n_top_genes: Number of HVGs to select
        seed: Random seed

    Returns:
        Tuple of (processed_adata, splits_dict)
    """
    adata = load_10x_data(data_path)
    adata = qc_filter(adata, min_genes=min_genes, max_mito_pct=max_mito_pct)
    adata_proc = normalize_and_select_hvgs(adata, n_top_genes=n_top_genes)
    splits = create_splits(adata_proc, seed=seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_proc.write_h5ad(output_path)
    print(f"\nProcessed data saved to {output_path}")

    return adata_proc, splits
