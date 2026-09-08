# Plan: Within-Subtype Evaluation on Larger Clusters (PBMC c7 + TM c28)

## Context

Within-subtype MIA on PBMC uses only 5 unseen megakaryocytes. TM has no within-subtype analysis. Testing larger clusters with more unseen cells gives stable AUC estimates where memorization may exist.

## Design

### PBMC Cluster 7 (CD14+ monocytes, 477 total, 397 train, 80 unseen)

Two forget set sizes as a within-subtype size ablation:
- **Forget-50:** Random 50 cells from cluster 7 training set; retrain on remaining 347 + all other clusters
- **Forget-100:** Random 100 cells from cluster 7 training set; retrain on remaining 297 + all other clusters

For each:
1. Train a retrain model that excludes the forget subset (but keeps the remaining cluster-7 cells)
2. Evaluate baseline and retrain with cross-cluster matching (k-NN from all unseen)
3. Evaluate baseline and retrain with within-cluster matching (80 unseen cluster-7 cells)
4. Compare AUC gap and confound fraction

### Tabula Muris Cluster 28 (394 total, 327 train, 67 unseen)

Same structure with forget-50 and forget-100:
- **Forget-50:** Random 50 from cluster 28 training; retrain on remaining 277 + all other clusters
- **Forget-100:** Random 100 from cluster 28 training; retrain on remaining 227 + all other clusters

## Implementation

### Source code
No new src/ changes. Reuse:
- `src/vae.py`, `src/train.py` (create_dataloader, train_epoch, get_kl_weight)
- `src/attacker.py`, `src/attacker_eval.py`
- `src/synthetic_utils.py` (extract_mia_features, train_and_evaluate_attacker)

### Notebook: `notebooks/39_larger_cluster_evaluation.ipynb`

Cells:
1. Setup (imports, seeds, paths)
2. **PBMC**: Load data, define cluster-7 forget subsets (50 and 100 random cells)
3. Train retrain-c7-50 model (exclude 50 specific cluster-7 cells)
4. Train retrain-c7-100 model (exclude 100 specific cluster-7 cells)
5. Cross-cluster and within-cluster MIA for forget-50: baseline vs retrain-c7-50
6. Cross-cluster and within-cluster MIA for forget-100: baseline vs retrain-c7-100
7. Figure: PBMC cluster 7 results (2x2: cross/within x 50/100)
8. **TM**: Load data, define cluster-28 forget subsets (50 and 100)
9. Train retrain-c28-50 and retrain-c28-100
10. MIA evaluation with both matching strategies
11. Figure: TM cluster 28 results
12. **Comparison**: All within-subtype results across datasets
13. Save results JSON

Estimated compute: ~60 min total (4 retrain models + MIA evaluations)

### Attacker features: 70-dim (v1 + kNN)

The canonical multi-seed evaluation (scripts/eval_multiseed.py) uses 70-dim features:
v1 (69 dims) + kNN distance to retain set (k=5) = 70 dims total.

This matches what NB03, NB19, NB25 use. The NB32-38 notebooks used 69-dim v1
WITHOUT the kNN feature, which is a methodological inconsistency.

For NB39, use the canonical 70-dim features:
```python
from attacker import extract_vae_features, build_attack_features, compute_knn_distances

vae_feats = extract_vae_features(model, x, lib, device)
retain_z = get_latent_codes(model, adata, retain_indices)
knn_dist = compute_knn_distances(z_forget, retain_z, k=5)
attack_feats = build_attack_features(vae_feats, knn_dist_retain=knn_dist, variant='v1')
# Result: 70 dims
```

### Skill validation

experiment-runner: markdown header, seeds, OUTPUT_DIR, JSON output, figures, summary
mia-evaluation: fresh attacker, spectral norm [256,256] d=0.3, **70-dim (v1 + kNN)**, advantage metric

### Output
- `outputs/larger_cluster_eval/results.json`
- `outputs/larger_cluster_eval/pbmc_c7_results.png`
- `outputs/larger_cluster_eval/tm_c28_results.png`
- `outputs/larger_cluster_eval/all_within_subtype_comparison.png`
- Retrain model checkpoints

### Writeup impact
DO NOT update writeup yet. Analyze results first, discuss with user, then decide what goes in.

### Statistical analysis
Run /statistical-analysis skill on the results: bootstrap CIs on within-cluster AUCs,
permutation test if sample sizes allow, effect size comparison across forget set sizes.

## Verification
- Cross-cluster AUC should be elevated (baseline sees cluster 7/28 biology)
- Within-cluster AUC should be near 0.5 if no sample-level memorization within the subtype
- OR within-cluster AUC gap (baseline - retrain) should be small relative to cross-cluster gap
- Confound fraction should be consistent with existing estimates (76-90%)

## Context

The within-subtype MIA evaluation on PBMC uses only 5 unseen megakaryocytes as negatives, producing noisy estimates (retrain AUC 0.609 > baseline 0.527). Tabula Muris has no within-subtype analysis at all. Testing larger clusters with more unseen cells gives stable within-subtype AUC estimates on datasets where memorization may exist.

## What we're adding

**NB39: Larger-cluster within-subtype evaluation**

Two new structured forget sets evaluated with both cross-cluster and within-cluster matching:

| Dataset | Cluster | Total | Train | Unseen | Current status |
|---------|---------|-------|-------|--------|---------------|
| PBMC | 7 | 477 | 397 | 80 | No within-cluster eval |
| Tabula Muris | 28 | 394 | 327 | 67 | No within-cluster eval |

For each:
1. Define forget set (all training cells from the cluster)
2. Compute matched negatives: cross-cluster (k-NN from all unseen) AND within-cluster (unseen cells from same cluster only)
3. Evaluate baseline and retrain models with fresh MIA attacker (per mia-evaluation skill)
4. Compare cross-cluster vs within-cluster AUC (quantify biology confound)

No new models need training — we use the existing PBMC baseline/retrain and Tabula Muris baseline/retrain checkpoints.

## Implementation

### Source code
No new `src/` changes needed. Uses existing:
- `src/vae.py` — VAE model loading
- `src/attacker.py` — `extract_vae_features`, `build_attack_features`
- `src/attacker_eval.py` — `matched_negative_evaluation`
- `src/synthetic_utils.py` — `extract_mia_features`, `train_and_evaluate_attacker`

### Notebook: `notebooks/39_larger_cluster_evaluation.ipynb`

Cells:
1. Setup (imports, seeds, paths) — per experiment-runner skill
2. **PBMC Part**: Load adata, define cluster 7 forget/retain split
3. Train retrain-c7 model (exclude cluster 7 from training, include cluster 13)
4. Compute cross-cluster matched negatives (k-NN k=10 from all unseen)
5. Compute within-cluster matched negatives (80 unseen cluster-7 cells)
6. MIA evaluation: baseline + retrain-c7 with both matching strategies
7. Figure: cross-cluster vs within-cluster AUC (PBMC cluster 7)
8. **TM Part**: Load TM adata, define cluster 28 forget/retain split
9. Train retrain-c28 model (exclude cluster 28 from training)
10. Compute both matching strategies
11. MIA evaluation: TM baseline + retrain-c28 with both matching strategies
12. Figure: cross-cluster vs within-cluster AUC (TM cluster 28)
13. **Comparison figure**: all within-subtype results across datasets/clusters
    - PBMC c13 (n=5 unseen), PBMC c7 (n=80 unseen)
    - TM c28 (n=67 unseen)
    - TCGA-BRCA Basal (n=30 unseen)
14. Save results JSON, summary table as DataFrame

Estimated compute: ~30 min total (two retrain models + MIA evaluations)

### Per skill validation checklist

**experiment-runner:**
- [ ] Markdown header + dependencies
- [ ] sys.path.insert, GLOBAL_SEED, OUTPUT_DIR
- [ ] Results saved to JSON
- [ ] Summary markdown at end
- [ ] Figures with dpi=150, no excessive prints
- [ ] At least 2 figures (one per dataset + comparison)

**mia-evaluation:**
- [ ] Fresh attacker per evaluation
- [ ] Spectral norm, [256,256], dropout 0.3
- [ ] 69-dim v1 features
- [ ] Advantage = 2*|AUC - 0.5|
- [ ] Both cross-cluster and within-cluster matching

### Key files
- `data/adata_processed.h5ad` — PBMC data
- `data/tabula_muris_processed.h5ad` — TM data
- `outputs/p1/baseline/best_model.pt` — PBMC baseline
- `outputs/p1/retrain_structured/best_model.pt` — PBMC retrain (trained without c13, but still contains c7)
- TM baseline/retrain checkpoints (need to verify paths)
- `outputs/p1/split_structured.json` — PBMC split (for train/unseen indices)
- `outputs/tabula_muris/split_structured.json` — TM split

### Training new retrain models

This experiment repeats the full protocol with a different cluster. We need NEW retrain models:

**PBMC:** Train retrain-c7 model on all training data EXCEPT cluster 7 (but INCLUDING cluster 13). This is the proper counterfactual for cluster 7.
- Training set: 28,124 - 397 = 27,727 cells
- Same config as original baseline (from checkpoint)
- ~15 min on CPU

**Tabula Muris:** Train retrain-c28 model on all training data EXCEPT cluster 28.
- Training set: 35,399 - 327 = 35,072 cells
- Same config as TM baseline
- ~15 min on CPU

The existing baseline models (trained on ALL data) serve as the "baseline" — they saw the forget cluster. The new retrain models never saw the forget cluster. The AUC gap between baseline and retrain measures genuine memorization, exactly as in the original experiment.

### Output
- `outputs/larger_cluster_eval/results.json`
- `outputs/larger_cluster_eval/pbmc_c7_comparison.png`
- `outputs/larger_cluster_eval/tm_c28_comparison.png`
- `outputs/larger_cluster_eval/all_datasets_within_subtype.png`

### Writeup impact
Add one paragraph to Section 5.5 (biology confound): "To validate with larger matched-negative pools, clusters with more unseen cells were tested..." with the results. Add data points to Figure 4 (confound comparison) or create supplementary table.

## Verification
- Run NB39
- Check that cross-cluster AUC is elevated (biology signal present)
- Check that within-cluster AUC drops (confound quantified)
- Compare confound fractions to existing PBMC c13 (90%) and TCGA-BRCA (76%)
- Verify results are consistent with the paper's narrative
