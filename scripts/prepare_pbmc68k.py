#!/usr/bin/env python3
"""PBMC-68k cohort (Zheng et al. 2017, 10x fresh_68k_pbmc_donor_a): the PBMC-33k recipe at twice the size.

Preprocessing, clustering, and the forget sets follow notebook 01 exactly (min 200 genes, mitochondrial
fraction below 10 percent, 2,000 highly variable genes, 50 PCs with 30 used, 15 neighbours, Leiden at
resolution 0.5). Two rare clusters are identified by marker genes rather than by cluster id, the
megakaryocytes (PPBP, PF4, GP9) and the plasmacytoid dendritic cells (LILRA4, IL3RA), and 40 percent
of each is held out so that within-type negatives of useful size exist; every other cell is held out
at 15 percent as on PBMC-33k. One training split serves both forget sets, as cluster 13 and cluster 12
share the PBMC-33k baseline.

Writes data/pbmc68k_processed.h5ad, outputs/pbmc68k/split_training.json, split_mk.json, split_pdc.json.
Run: PYTHONPATH=src python scripts/prepare_pbmc68k.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
from data import load_10x_data, qc_filter, normalize_and_select_hvgs

RAW = BASE / 'data/pbmc68k/filtered_matrices_mex/hg19'
ANNOT = BASE / 'data/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
PROCESSED = BASE / 'data/pbmc68k_processed.h5ad'
OUT = BASE / 'outputs/pbmc68k'
SEED = 42
MARKERS = {'megakaryocyte': ['PPBP', 'PF4', 'GP9'], 'pdc': ['LILRA4', 'IL3RA', 'CLEC4C']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROCESSED.exists():
        adata = sc.read_h5ad(PROCESSED); print(f"loaded {PROCESSED} {adata.shape}")
    else:
        adata = load_10x_data(str(RAW))
        adata = qc_filter(adata, min_genes=200, max_mito_pct=10.0)
        adata = normalize_and_select_hvgs(adata, n_top_genes=2000)
        ann = pd.read_csv(ANNOT, sep='\t').set_index('barcodes')['celltype']
        adata.obs['published_label'] = ann.reindex(adata.obs_names).astype(str).values
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=SEED)
        sc.tl.leiden(adata, resolution=0.5, key_added='leiden', random_state=SEED)
        adata.write_h5ad(PROCESSED); print(f"processed {adata.shape} -> {PROCESSED}")
    sizes = adata.obs['leiden'].value_counts()
    print("cluster sizes:", sizes.sort_values().to_dict())
    # identify the two rare clusters by marker means (the published labels call the megakaryocytes CD34+)
    rare = {}
    for name, genes in MARKERS.items():
        genes = [g for g in genes if g in adata.var_names]
        score = pd.Series(np.asarray(adata[:, genes].X.mean(axis=1)).ravel(), index=adata.obs_names)
        by_cluster = score.groupby(adata.obs['leiden'].values).mean().sort_values(ascending=False)
        rare[name] = str(by_cluster.index[0])
        print(f"{name}: cluster {rare[name]} (n={sizes[rare[name]]}), marker mean {by_cluster.iloc[0]:.2f} vs next {by_cluster.iloc[1]:.2f}; "
              f"published labels {adata.obs.loc[adata.obs.leiden == rare[name], 'published_label'].value_counts().head(3).to_dict()}")
    assert rare['megakaryocyte'] != rare['pdc']
    # one training / held-out split: rare clusters held out at 40 percent, everything else at 15 percent
    rng = np.random.RandomState(SEED)
    leiden = adata.obs['leiden'].astype(str).values
    unseen = []
    for c in sizes.index:
        idx = np.where(leiden == str(c))[0]
        frac = 0.4 if str(c) in rare.values() else 0.15
        unseen.extend(rng.choice(idx, int(round(len(idx) * frac)), replace=False).tolist())
    unseen = np.array(sorted(unseen)); training = np.setdiff1d(np.arange(adata.n_obs), unseen)
    json.dump({'training_indices': training.tolist(), 'unseen_indices': unseen.tolist(), 'n_training': int(len(training)),
               'n_unseen': int(len(unseen)), 'rare_clusters': rare, 'holdout_fraction': {'rare': 0.4, 'other': 0.15}, 'seed': SEED,
               'description': 'PBMC-68k: one training/held-out split; rare clusters held out at 40 percent, others at 15 percent'},
              open(OUT / 'split_training.json', 'w'))
    for key, name in [('mk', 'megakaryocyte'), ('pdc', 'pdc')]:
        c = rare[name]
        forget = training[leiden[training] == c]; retain = np.setdiff1d(training, forget)
        within = unseen[leiden[unseen] == c]
        split = {'forget_indices': forget.tolist(), 'retain_indices': retain.tolist(), 'unseen_indices': unseen.tolist(),
                 'within_type_unseen_indices': within.tolist(), 'n_forget': int(len(forget)), 'n_retain': int(len(retain)),
                 'n_unseen': int(len(unseen)), 'n_within_type_unseen': int(len(within)), 'forget_cluster': c, 'forget_type': name,
                 'description': f'PBMC-68k structured forget set: Leiden cluster {c} ({name}), all training cells of the cluster'}
        json.dump(split, open(OUT / f'split_{key}.json', 'w'))
        print(f"split_{key}: forget {len(forget)} retain {len(retain)} unseen {len(unseen)} within-type unseen {len(within)}")


if __name__ == '__main__':
    main()
