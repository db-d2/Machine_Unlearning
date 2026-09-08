# INSTRUCTIONS.md - Guide for Extending This Project

## Research Goal

**Address publication blockers** to bring the paper to conference/journal quality. The paper draft is complete with cross-dataset validation and size ablations. An external review (`study_suggestions.md`) identified 7 blockers. See `TASKS.md` for the full plan.

**Target framing:** A careful empirical evaluation of unlearning in single-cell VAEs showing that memorization concentrates in coherent biological subpopulations and that standard unlearning techniques fail, create artifacts, or work only in narrow regimes.

## Previous Feedback (Addressed)

The professor asked for improved generalization. All three items are now complete:
- **More datasets** → Tabula Muris added ✓
- **Larger forget sets** → Size ablation (n=10 to 500) ✓
- **More stress-testing** → Lambda sweep, cross-dataset, multi-cluster ✓

## Publication Blockers (Current Priority)

| ID | Issue | Fix |
|----|-------|-----|
| A | AUC symmetry (AUC < 0.5 ≠ success), arbitrary target band | Advantage metric, retrain bootstrap CI |
| B | Forget set = cluster, not patient/donor | Reframe as subpopulation unlearning |
| C | Single attacker family | Add threshold, likelihood-ratio, k-NN attacks |
| D | 3-4 seeds for key results | 10 seeds, bootstrap CIs |
| E | Utility metrics too coarse (identical MSE suspicious) | ELBO, silhouette, ARI, marker genes |
| F | Extra-gradient 4× slower than retraining | Reframe contribution as evaluation |

## Priority Tasks

### 1. Evaluation Framework Hardening (Blockers A, C)

Implement attack diversity and fix privacy metric reporting. This is the highest priority because reviewers who do not trust the evaluation will reject regardless of findings.

### 2. Additional Unlearning Baselines (Suggestion)

Add retain-only fine-tuning and gradient ascent to make method comparison less arbitrary.

### 3. Utility Expansion (Blocker E)

Add domain-relevant utility metrics expected by scRNA-seq reviewers.

### 4. Statistical Rigor (Blocker D)

Increase seeds to 10 for key results and add bootstrap CIs to all tables.

### 5. Writeup Revision (Blockers B, F)

Reframe as subpopulation unlearning. Add contribution bullets. Update all tables.

## Prerequisites

Before extending this project, ensure you have:
- Completed notebooks 01-10 successfully on PBMC-33k
- Environment set up (`pyenv activate stat4243`)
- ~10GB free disk space for new datasets
- GPU access recommended (CPU is 10x slower)

## Time and Resource Estimates

| Task | GPU Time | CPU Time | Disk |
|------|----------|----------|------|
| Data preprocessing | 2-5 min | 5-10 min | 500MB |
| Baseline VAE (100 epochs) | 10 min | 2 hours | 200MB |
| MIA evaluation (3 attackers) | 15 min | 1 hour | 50MB |
| Extra-gradient (50 epochs) | 42 min | 8 hours | 800MB |
| Fisher scrubbing | 2 min | 10 min | 100MB |
| Full pipeline (one dataset) | ~2 hours | ~12 hours | ~2GB |

Estimates based on NVIDIA RTX 3090 / Apple M1 Pro.

## 1. Adding a New Dataset

### Step 1: Obtain Data

Recommended scRNA-seq datasets for testing generalization:

| Dataset | Cells | Source | Why |
|---------|-------|--------|-----|
| Tabula Muris | ~100k | [link](https://tabula-muris.ds.czbiohub.org/) | Multi-tissue, diverse cell types |
| COVID PBMC | ~500k | [link](https://www.covid19cellatlas.org/) | Disease context, larger scale |
| Human Cell Atlas | ~1M+ | [link](https://data.humancellatlas.org/) | Reference dataset |
| Mouse Brain | ~1M | [link](https://portal.brain-map.org/) | Complex tissue |

### Step 2: Create Preprocessing Notebook

Create `notebooks/11_data_prep_[dataset].ipynb`:

```python
import scanpy as sc
import numpy as np

# Load raw data
adata = sc.read_10x_mtx('data/[dataset]/raw/')

# QC filtering (same as PBMC)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVG selection (match PBMC: 2000 genes)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

# Clustering
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Save
adata.write('data/[dataset]_processed.h5ad')
```

### Step 3: Identify Rare Clusters

```python
# Find rare clusters (analogous to cluster 13)
cluster_sizes = adata.obs['leiden'].value_counts()
rare_clusters = cluster_sizes[cluster_sizes < 100]
print(rare_clusters)

# Select one for structured forget set
# Prefer biologically distinct clusters (check marker genes)
```

### Step 4: Create Data Splits

```python
from sklearn.model_selection import train_test_split

# 85/15 train/unseen split
train_idx, unseen_idx = train_test_split(
    range(len(adata)), test_size=0.15, random_state=42
)

# Structured forget set (rare cluster)
structured_forget = adata.obs[adata.obs['leiden'] == '[RARE_CLUSTER]'].index

# Scattered forget set (random 30 from train)
scattered_forget = np.random.choice(train_idx, size=30, replace=False)

# Save splits
import json
splits = {
    'train_idx': train_idx.tolist(),
    'unseen_idx': unseen_idx.tolist(),
    'structured_forget': structured_forget.tolist(),
    'scattered_forget': scattered_forget.tolist()
}
with open(f'outputs/p1/[dataset]/splits.json', 'w') as f:
    json.dump(splits, f)
```

### Step 5: Run Full Pipeline

Create `notebooks/12_[dataset]_experiments.ipynb` that runs:
1. Baseline VAE training
2. MIA evaluation
3. Extra-gradient unlearning
4. Fisher unlearning
5. Post-hoc evaluation

### Verification Steps

After completing each stage, verify outputs:

```bash
# After data preprocessing (Step 2)
python -c "import scanpy as sc; adata = sc.read('data/[dataset]_processed.h5ad'); print(f'Shape: {adata.shape}, Clusters: {adata.obs.leiden.nunique()}')"
# Expected: Shape similar to (N, 2000), multiple clusters

# After creating splits (Step 4)
python -c "import json; d = json.load(open('outputs/p1/[dataset]/splits.json')); print(f'Train: {len(d[\"train_idx\"])}, Forget: {len(d[\"structured_forget\"])}')"
# Expected: Train >> Forget, Forget ~= 30

# After baseline training
ls outputs/p1/[dataset]/baseline/
# Expected: best_model.pt, training_history.json, tensorboard/

# After MIA evaluation
python -c "import json; print(json.load(open('outputs/p1/[dataset]/baseline/mia_results.json'))['auc'])"
# Expected: AUC 0.6-0.8 (indicates memorization)
```

## 2. Varying Forget Set Sizes (Extra-gradient ONLY)

**IMPORTANT:** Only test extra-gradient on STRUCTURED sets. Fisher is already characterized:
- Fisher works on scattered (AUC=0.50) but has utility issues (class acc 0.954→0.818)
- Fisher fails on structured (AUC=0.81)

### Experiment Design

Test sizes: n = [10, 30, 50, 100] for **structured** forget sets.

Cluster 13 has only 35 cells, so for n>35 combine multiple rare clusters:

| Size | Source |
|------|--------|
| n=10 | Subset of cluster 13 |
| n=30 | All of cluster 13 (baseline, already tested) |
| n=50 | Cluster 13 (35) + subset of cluster 12 (15 of 55) |
| n=100 | Clusters 13 (35) + 12 (55) + subset of 11 (10 of 85) |

### Implementation

```bash
# Create structured splits for each size
PYTHONPATH=src python scripts/create_structured_size_splits.py

# Run extra-gradient with multiple seeds
for size in 10 50 100; do
    for seed in 42 123 456 789 1011; do
        PYTHONPATH=src python src/train_unlearn_extragradient.py \
            --data_path data/adata_processed.h5ad \
            --split_path outputs/p4/extragradient_size/splits/split_structured_n${size}.json \
            --baseline_checkpoint outputs/p1/baseline/best_model.pt \
            --output_dir outputs/p4/extragradient_size/n${size}/seed${seed} \
            --lambda_privacy 10 \
            --seed $seed
    done
done
```

### Expected Outcomes

- Extra-gradient λ=10 should work at small n (10, 30) - **n=30 already confirmed**
- Key question: Does it still work when combining multiple clusters (n=50, 100)?
- If combining clusters fails, that reveals important limitations

### Verification

```bash
# Check post-hoc AUC for each size
for n in 10 50 100; do
    echo "n=$n:"
    cat outputs/p4/extragradient_size/n${n}/*/eval_results.json | jq -r '.auc'
done
# Success: AUC in [0.45, 0.52]
```

## 3. Cross-Dataset Results (Completed)

The following experiments are complete. See `TASKS.md` for the publication readiness plan.

### Results Table (Current)

| Dataset | Baseline Struct | EG-10 Struct | Fisher Struct | Baseline Scat | Fisher Scat |
|---------|----------------|--------------|---------------|---------------|-------------|
| PBMC-33k | 0.769 | 0.482 (target) | 0.814 (fail) | 0.525 | 0.499 (marginal) |
| Tabula Muris | 0.891 | 0.874 (fail) | 0.946 (fail) | 0.411 | 0.568 (worse) |

**Key findings:**
- Extra-gradient λ=10 works only on PBMC single-cluster n=30; fails on TM and multi-cluster
- Fisher shows consistent posterior collapse on structured sets across datasets
- Scattered baselines are near chance; Fisher provides marginal-to-negative improvement
- The structured-scattered distinction is about memorization strength, not method effectiveness

## 4. Publication Readiness Work (Phases 1-5: COMPLETE)

Phases 1-5 are done. Evaluation hardening, 8 methods, utility suite, multi-seed, and writeup revision are all complete. See `TASKS.md` for the completed checklist.

## 5. Fisher Overlap Analysis + Theory (Phase 6: CURRENT)

This adds a theoretical contribution explaining why parameter-space unlearning fails for generative models. Target venue: TMLR.

### Core Argument

Classifiers have block-diagonal Fisher structure: forget-set influence concentrates in output-layer weights for the forgotten class. VAE decoders share all output parameters across all input types. When the forget set differs on 337/2000 genes (17% of output dimensions), the forget-set Fisher has high cosine similarity with the retain-set Fisher, making selective parameter perturbation impossible without damaging retain performance.

### Step 1: Compute VAE Fisher Overlap

Create `scripts/fisher_overlap_analysis.py`:

```python
# Reuse compute_fisher_diagonal from src/train_fisher_unlearn.py
# Compute F_forget and F_retain separately
# Metrics:
#   cosine_sim = dot(F_forget, F_retain) / (norm(F_forget) * norm(F_retain))
#   effective_rank = exp(entropy(F / sum(F)))
#   top_k_overlap = |topK(F_forget) ∩ topK(F_retain)| / K
# Compute per-layer and globally
# Save to outputs/p6/fisher_overlap.json
```

Key implementation detail: the existing `compute_fisher_diagonal()` in `src/train_fisher_unlearn.py` computes F on retain set only. Run it twice (once with forget dataloader, once with retain dataloader) to get both.

### Step 2: Train Linear Probe Classifier

```python
# Freeze VAE encoder, train logistic regression on latent means
# 14 Leiden clusters as classes, 32-dim input
# Compute classifier Fisher on forget vs retain
# Expected: low overlap because forget class (cluster 13) has
#   dedicated output weights in the 14-class softmax
```

### Step 3: Create Figures

Three figures for the paper:
1. Per-layer cosine similarity heatmap (VAE vs classifier)
2. Effective rank bar chart showing Fisher concentration
3. Scatter plot of per-parameter Fisher magnitude (forget vs retain)

### Step 4: Write Proposition

Informal proposition with proof sketch:
- For shared-output generative models, Fisher overlap approaches 1 as output dimension grows
- For classifiers with class-specific output weights, Fisher overlap is bounded by class proportion
- Validate empirically with measured cosine similarity values

### Verification

```bash
# Run Fisher analysis
PYTHONPATH=src python scripts/fisher_overlap_analysis.py

# Check output
python -c "
import json
d = json.load(open('outputs/p6/fisher_overlap.json'))
print(f'Global cosine similarity: {d[\"global\"][\"cosine_similarity\"]:.3f}')
print(f'Effective rank (forget): {d[\"global\"][\"effective_rank_forget\"]:.1f}')
"
# Expected: cosine_sim > 0.9 for VAE, < 0.3 for classifier
```

## 5. Quick Reference Commands

```bash
# Activate environment
pyenv activate stat4243

# Run attack suite (after implementing additional attacks)
PYTHONPATH=src python scripts/run_attack_suite.py \
    --checkpoint outputs/p2/extragradient/lambda10/best_model.pt

# Run conditioned evaluation
PYTHONPATH=src python scripts/run_conditioned_eval.py \
    --checkpoint outputs/p2/extragradient/lambda10/best_model.pt

# Compute utility suite
PYTHONPATH=src python scripts/compute_utility_suite.py \
    --checkpoint outputs/p2/extragradient/lambda10/best_model.pt

# Train baseline on new dataset
PYTHONPATH=src python src/train.py \
    --data_path data/[dataset]_processed.h5ad \
    --output_dir outputs/p1/[dataset]/baseline

# Run extra-gradient
PYTHONPATH=src python src/train_unlearn_extragradient.py \
    --data_path data/[dataset]_processed.h5ad \
    --split_path outputs/p1/[dataset]/split_structured.json \
    --baseline_checkpoint outputs/p1/[dataset]/baseline/best_model.pt \
    --output_dir outputs/p2/[dataset]/extragradient \
    --lambda_privacy 10
```

## 6. Checklist Before Submission

See `TASKS.md` for the detailed checklist. Summary:

- [ ] Advantage metric in all tables
- [ ] Target band from retrain bootstrap CI
- [ ] 4 attacks (MLP, threshold, LR, k-NN)
- [ ] Worst-case attack reported
- [ ] 2 additional baselines (retain-finetune, gradient-ascent)
- [ ] Domain-relevant utility (ELBO, silhouette, ARI)
- [ ] 10 seeds for key results
- [ ] Bootstrap CIs on all values
- [ ] Subpopulation framing
- [ ] Explicit contribution bullets
- [ ] Prepare questions about further extensions
- [ ] Consider differential privacy integration (mentioned in Future Work)
