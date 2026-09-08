# Plan: Synthetic Augmentation Experiment

## Concept

The Fisher analysis identifies WHERE memorization concentrates (rare subpopulations
with high Fisher overlap). The constructive response is to augment those subpopulations
with synthetic data DURING TRAINING so the model generalizes from the subtype distribution
rather than memorizing specific training samples.

This is prevention, not post-hoc unlearning. The question:

> If we train the VAE on retain + synthetic megakaryocytes (instead of retain + real
> megakaryocytes), does the model learn megakaryocyte biology without memorizing
> specific cells?

## Experimental Design

### What we compare (4 models)

| Model | Training data | Expected memorization |
|-------|--------------|----------------------|
| Baseline | retain + 30 real c13 | High (current: AUC 0.783) |
| Retrain | retain only (no c13) | None (current: AUC 0.523) |
| Augmented-match | retain + 30 synthetic c13 | Low (replaces real with synthetic) |
| Augmented-over | retain + 100 synthetic c13 | Lower (more synthetic diversity) |

The augmented models see the same number (or more) megakaryocyte-like cells as
the baseline, but none of them are the ACTUAL training samples. If the MIA can
still detect the original forget-set cells as members, the memorization was in
the retain set's interaction with cluster 13, not in the cluster 13 cells themselves.

### How to generate synthetic megakaryocytes

Three strategies, ordered by complexity:

**Strategy 1: Bootstrap + noise (simplest, no model needed)**
- Take the 5 unseen cluster-13 cells (NOT from training set)
- Bootstrap resample to N cells
- Add small Gaussian noise (gene-wise, scaled to gene variance)
- Preserves sparsity, realistic distribution
- Limitation: only 5 seed cells, limited diversity

**Strategy 2: Fit per-gene distributions to cluster 13 (statistical)**
- For each gene, fit the expression distribution across all 35 cluster-13 cells
  (30 train + 5 unseen, or just the 5 unseen to be strict)
- Sample from per-gene distributions independently
- Apply gene-gene correlation structure via copula or PCA
- More diverse than bootstrap, still realistic

**Strategy 3: scDesign2-style generation (most realistic)**
- Fit marginal distributions per gene + correlation via Gaussian copula
- This is what Golob et al. attacked — we'd be using the same generative approach
- Most biologically plausible but most complex

**Recommendation: Strategy 1 for the paper.** It's simplest, clearly doesn't use
the baseline model (avoiding circularity), and the question is whether ANY
synthetic replacement reduces memorization. If bootstrap+noise works, we don't need
fancier generation. If it fails, fancier generation won't help either because
the failure would be about the retain set's influence, not the synthetic quality.

**IMPORTANT: Use only the 5 UNSEEN cluster-13 cells as seeds.**
Using the 30 training cells would be circular — we'd be generating synthetic data
from the very samples we're trying to prevent memorization of. The 5 unseen cells
provide the subtype's biology without the specific training samples.

### Evaluation (per mia-evaluation skill)

For each augmented model:
1. Extract 69-dim v1 features from the AUGMENTED model (not baseline)
   for the 30 original forget cells and 194 matched negatives
2. Train fresh attacker (spectral norm, [256,256], dropout 0.3)
3. Compute AUC and advantage
4. Compare to baseline (0.783) and retrain (0.523)

Also evaluate with within-cluster matching (5 real unseen c13) for clean comparison.

### Fisher overlap analysis (per fisher-analysis skill)

For the augmented model:
1. Compute Fisher on forget set (the 30 original real c13 cells) and retain set
2. Compare global cosine to baseline (0.306) and retrain
3. If augmentation reduces Fisher overlap, it changed the parameter-space geometry

### Utility evaluation

The augmented model should still reconstruct megakaryocytes well (it was trained
on synthetic ones). Check:
1. Reconstruction loss on forget set (should be similar to baseline)
2. ELBO on held-out data
3. Latent space visualization (c13 should still form a cluster)

## Implementation

### Source code: `src/augmentation_utils.py`

```python
def generate_bootstrap_augmented(seed_samples, n_synthetic, noise_scale=0.1, seed=42):
    """Bootstrap resample + gaussian noise from seed samples."""

def train_augmented_vae(adata, retain_indices, synthetic_data, config, device, seed=42):
    """Train VAE on retain + synthetic data."""
```

### Notebook: `notebooks/37_synthetic_augmentation.ipynb`

Cells:
1. Setup (imports, paths, seeds)
2. Load data, splits, baseline/retrain models
3. Generate synthetic megakaryocytes (bootstrap + noise from 5 unseen c13)
4. Validate synthetic data (UMAP, expression correlation, sparsity check)
5. Train augmented model (retain + 30 synthetic c13)
6. Train augmented-over model (retain + 100 synthetic c13)
7. MIA evaluation: all 4 models with cross-cluster matching
8. MIA evaluation: all 4 models with within-cluster matching (5 real c13)
9. Fisher overlap comparison (augmented vs baseline vs retrain)
10. Utility comparison (reconstruction loss, ELBO, latent space)
11. Summary figure: AUC by model, Fisher cosine by model
12. Save results JSON

### Expected outcomes

**If augmentation reduces memorization (AUC closer to retrain):**
- The model learned megakaryocyte biology from synthetic data without memorizing
  the specific training cells
- This validates the Fisher diagnostic + synthetic augmentation as a prevention strategy
- Publishable constructive contribution

**If augmentation doesn't help (AUC stays near baseline):**
- The memorization comes from the retain set's interaction with the rare subtype,
  not from the forget set itself
- Even seeing NO real megakaryocytes (retrain) doesn't fully prevent the model from
  having a distinctive representation of the megakaryocyte region
- This would explain the retrain AUC of 0.523 (not exactly 0.5)
- Still informative: augmentation doesn't work because memorization is deeper than
  sample-level

**If augmentation helps partially (AUC between baseline and retrain):**
- The synthetic data dilutes but doesn't eliminate memorization
- Suggests a combination of sample-level and subtype-level effects
- Still publishable: augmentation as a mitigation (not a solution)

## Risk assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Bootstrap from 5 cells lacks diversity | Synthetic cells too similar, model overfits to them | Add noise, check diversity in latent space |
| Training on synthetic changes model behavior unpredictably | Utility degrades | Monitor ELBO on retain set during training |
| Evaluation is confounded (same as NB33/NB34b) | Results hard to interpret | Use BOTH cross-cluster and within-cluster matching |
| Training time | Delays | ~10 min per model on CPU, 2 models = 20 min |

## Execution

1. Create `src/augmentation_utils.py`
2. Create + execute `notebooks/37_synthetic_augmentation.ipynb`
3. Update memory with findings
4. If successful, this becomes the constructive contribution in the paper
