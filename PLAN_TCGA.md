# Implementation Plan: TCGA-BRCA + Unit Tests + Reproducibility Audit

## Overview

Three priorities:
1. **TCGA-BRCA dataset** — patient-level forget sets, third dataset, bulk RNA-seq generalization
2. **Unit tests** — prevent NB/MSE-type documentation drift, verify core functions
3. **Checkpoint/reproducibility audit** — trace every paper number to an artifact

---

## Phase 0: Prerequisites (30 min)

### Task 0.1: Verify/add Python dependencies
- `requests` (already in env) for GDC API
- `pytest>=7.0.0` for unit tests
- No R or TCGAbiolinks needed — download via GDC REST API or UCSC Xena

### Task 0.2: Obtain L1000 landmark gene list
- 978 genes from Broad Institute clue.io
- Save to `data/tcga_brca/lincs_l1000_genes.txt`

---

## Phase 1: TCGA-BRCA Data Preparation (2-3 hours)

### Task 1.1: Create `notebooks/31_tcga_brca_data_prep.ipynb`
**Pattern:** Follow `notebooks/11_tabula_muris_data_prep.ipynb`

Steps:
1. Download TCGA-BRCA HTSeq counts from GDC or UCSC Xena
2. Map clinical metadata (molecular subtypes: LumA, LumB, Basal, Her2, Normal)
3. Filter to L1000 genes (978 genes)
4. **Normalize with `sc.pp.normalize_total(target_sum=1e4)` + `sc.pp.log1p()`** — same pipeline as scRNA-seq, keeps softmax+library_size mechanism valid
5. Run Leiden clustering for pipeline compatibility
6. Visualize subtype distributions
7. Save to `data/tcga_brca_processed.h5ad`

**Key decision: use same normalization as scRNA-seq (Option A).** This makes the softmax output layer and MSE loss work identically. The alternative (VST) would require architectural changes.

**Subtype distribution (approximate):**
| Subtype | Count | % |
|---------|-------|---|
| LumA | ~500 | 46% |
| LumB | ~200 | 18% |
| Basal | ~190 | 17% |
| Her2 | ~82 | 7% |
| Normal | ~140 | 13% |

### Task 1.2: Define forget sets
**Output:** `outputs/tcga_brca/split_structured.json`, `split_patient_N.json`, `split_scattered.json`

Three types:
1. **Structured** — all Normal-like subtype patients (~120 in train split). Analogous to cluster 13 in PBMC.
2. **Scattered** — 30 random patients from training set
3. **Patient-level** (NEW) — groups of 1, 5, 10, 20 patients from same subtype

Split: 85/15 train/unseen, seed=42 (same convention)

---

## Phase 2: TCGA-BRCA Baseline VAE (1-3 hours)

### Task 2.1: Create `notebooks/32_tcga_brca_baseline.ipynb`
**Pattern:** Follow `notebooks/12_tabula_muris_baseline.ipynb`

Config:
```
input_dim: 978
hidden_dims: [1024, 512, 128]
latent_dim: 32
likelihood: nb  (MSE in practice)
dropout: 0.1
batch_size: 128  (smaller dataset, reduce from 256)
lr: 1e-4
epochs: 100
seed: 42
```

**Parameter count at 978 genes: ~4.2M** (vs 7.35M at 2000 genes)

**Expected behavior:** High memorization due to low sample-to-parameter ratio (~930 samples / 4.2M params). This HELPS the paper's argument.

Steps:
1. Load processed data
2. Create 85/15 split, save JSON
3. Train baseline VAE
4. Visualize latent space by subtype
5. Train retrain model (exclude structured forget set)
6. Create matched negatives (k=10 in unseen set)
7. Train fresh MIA attacker (F vs matched neg)
8. Report baseline and retrain AUC

### Task 2.2: Baseline MIA evaluation
**Output:** `outputs/tcga_brca/baseline_results.json`

MIA features: 69-dim v1 (32 mu + 32 logvar + recon + kl + elbo + 2 norms)
Fresh attacker: [256, 256], spectral norm, dropout 0.3

---

## Phase 3: TCGA-BRCA Experiments (4-8 hours compute)

### Task 3.1: Unlearning on structured forget set
**Create:** `notebooks/33_tcga_brca_unlearning.ipynb`

Run 8 methods on structured forget set (Normal-like subtype):
- Extra-gradient (lambda sweep 5, 10, 15, 20)
- Fisher scrubbing
- Retain-only fine-tuning
- Gradient ascent + fine-tune
- SSD
- SCRUB
- Contrastive latent
- DP-SGD baseline

All `src/train_*.py` scripts accept `--data_path` and `--split_path` — no code changes needed.

### Task 3.2: Patient-level unlearning (NEW contribution)
**Create:** `notebooks/34_tcga_brca_patient_unlearning.ipynb`

For each patient-level size (1, 5, 10, 20 patients):
1. Define forget set from specific patient IDs
2. Run extra-gradient and Fisher scrubbing
3. Train fresh attacker
4. Evaluate MIA AUC

Matched negatives: restrict to same-subtype neighbors in unseen set.

**Key question this answers:** "Can we unlearn a specific patient's data from a gene-expression VAE?"

### Task 3.3: Fisher overlap analysis
**Create:** `notebooks/35_tcga_brca_fisher_overlap.ipynb`
**Pattern:** Follow `notebooks/26_fisher_overlap_analysis.ipynb`

Compute:
- Diagonal Fisher for forget and retain sets
- Per-layer cosine similarity
- Comparison with PBMC (0.306) and classifier (0.018)
- Gene-level overlap statistics

### Task 3.4: Cross-domain comparison
**Create:** `notebooks/36_cross_domain_comparison.ipynb`

Consolidate results across all 3 datasets:

| Dataset | Type | n_train | n_genes | Baseline AUC | Fisher cosine |
|---------|------|---------|---------|-------------|---------------|
| PBMC-33k | scRNA | 28,124 | 2,000 | 0.769 | 0.306 |
| Tabula Muris | scRNA | 35,399 | 2,000 | 0.891 | N/A |
| TCGA-BRCA | bulk RNA | ~930 | 978 | TBD | TBD |

---

## Phase 4: Writeup Updates (2-4 hours)

### Task 4.1: Update Writeup.tex
- Section 2: Add TCGA-BRCA to dataset table
- Section 4: Add results rows for TCGA-BRCA
- Section 5: Patient-level unlearning subsection
- Section 6: TCGA-BRCA Fisher overlap numbers
- Discussion: Cross-domain generalization evidence
- Limitations: Update "clinical relevance" paragraph (now addressed)

### Task 4.2: Add Ozturk et al. citation to references.bib

---

## Phase 5: Unit Tests (2-3 hours, independent of Phases 0-4)

### Task 5.1: Test infrastructure
**Create:** `tests/__init__.py`, `tests/conftest.py`

Fixtures:
- Small synthetic AnnData (100 cells, 50 genes, 5 clusters)
- Small VAE (input_dim=50, hidden_dims=[32, 16], latent_dim=8)
- Small attacker
- Deterministic seed

### Task 5.2: `tests/test_vae.py`
1. `test_nb_loss_is_mse` — verify nb_loss computes MSE, ignores dispersion
2. `test_kl_divergence_zero` — mu=0, logvar=0 gives KL=0
3. `test_kl_divergence_known` — mu=1, logvar=0 gives KL=0.5
4. `test_vae_loss_composition` — total = recon + beta*kl
5. `test_decoder_softmax_sums_to_one` — without library_size, output sums to 1
6. `test_decoder_with_library_size` — output sums to library_size
7. `test_vae_forward_shapes` — correct output shapes

### Task 5.3: `tests/test_attacker.py`
1. `test_feature_dim_v1` — 2*latent_dim + 5 = 69 for z=32
2. `test_feature_dim_v2` — 2*latent_dim + 3
3. `test_feature_dim_v3` — 3
4. `test_extract_features_keys` — all expected keys present
5. `test_attacker_output_shape` — [batch, 1] output

### Task 5.4: `tests/test_fisher.py`
1. `test_fisher_diagonal_positive` — all entries >= 0
2. `test_fisher_linear_analytical` — matches known form for linear model
3. `test_cosine_sim_orthogonal` — returns 0
4. `test_cosine_sim_parallel` — returns 1
5. `test_effective_rank_uniform` — rank = n

### Task 5.5: `tests/test_data.py`
1. `test_splits_deterministic` — same seed = same split
2. `test_splits_no_overlap` — forget, retain, unseen are disjoint
3. `test_splits_cover_all` — union = all indices
4. `test_matched_negatives_from_unseen` — matched are subset of unseen
5. `test_splits_proportions` — ~85/15

### Task 5.6: Run all tests
```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Phase 6: Checkpoint/Reproducibility Audit (2-3 hours)

### Task 6.1: Create `outputs/manifest.json`
Map every number in the paper to source:
```json
{
  "table1_baseline_structured": {
    "value": 0.769,
    "source": "outputs/attack_suite_results.json",
    "key": "baseline.structured.auc",
    "notebook": "notebooks/15_attack_suite.ipynb"
  },
  ...
}
```

### Task 6.2: Create `scripts/verify_manifest.py`
- Load manifest
- Check source files exist
- Extract values, compare to claims
- Report mismatches

---

## Source Code Changes Required

### No changes needed:
- `src/vae.py` — input_dim already configurable
- `src/attacker.py` — features computed from latent dim, not gene count
- `src/attacker_eval.py` — data-agnostic
- All `src/train_*.py` — accept `--data_path` argument

### Minor additions:
- **`src/fisher_utils.py` line 47**: `GeneExprDataset` hardcodes `adata.obs['leiden']`. Add `label_key` parameter:
  ```python
  def __init__(self, adata, indices, label_key='leiden'):
      ...
      self.y = torch.LongTensor(adata.obs[label_key].values[indices].astype(int))
  ```
- **`src/define_forget_sets.py`**: Add `define_patient_forget_set(adata, patient_ids, train_indices)` function

### New files:
- 6 notebooks (31-36)
- 1 config file
- 5 test files + conftest
- 1 manifest JSON + 1 verification script

---

## Dependency Graph

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                                      │
                                      ├── Task 3.1 (structured unlearning)
                                      ├── Task 3.2 (patient-level)  ← NEW
                                      ├── Task 3.3 (Fisher overlap)
                                      └── Task 3.4 (cross-domain)

Phase 5 (unit tests) ── runs independently, start anytime
Phase 6 (audit) ── runs after all notebooks complete
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Softmax incompatible with bulk RNA | Use same normalize_total+log1p as scRNA |
| Small dataset = noisy MIA | Multi-seed (5-10), bootstrap CIs |
| TCGA download issues | Fallback: UCSC Xena pre-processed data |
| L1000 gene overlap incomplete | Check intersection, use available subset |
| Patient-level Fisher too noisy (n=1) | Focus Fisher on n>=20 forget sets |
