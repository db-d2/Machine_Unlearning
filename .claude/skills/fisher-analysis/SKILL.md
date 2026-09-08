---
name: fisher-analysis
description: Compute and analyze Fisher information overlap between forget and retain sets. Use when running Fisher overlap analysis, computing the per-sample diagonal Fisher, comparing generative vs classifier overlap, or running the phase audit of an edit-then-fine-tune method.
---

# Fisher Overlap Analysis

## Purpose

Fisher overlap measures how much the forget-set and retain-set parameter importance vectors align. High overlap means an edit weighted by forget-set importance disturbs retain-set curvature in proportion (the remark under Proposition 1). What it does NOT mean, per the phase audit of 2026-09-08: it does not block the edit. The scrub moves forget and retain loss together, and the retain fine-tune that repairs the retain set undoes the edit through the same shared parameters. Overlap is the mechanism of the signal-intact failure mode, not of the over-corrected one.

## Core Functions (src/fisher_utils.py)

```python
from fisher_utils import (
    AnnDataDataset,               # (x, library_size) dataset
    GeneExprDataset,              # (x, label) dataset, label_key param
    compute_vae_fisher,           # per-sample diagonal Fisher for a VAE (negative ELBO, eval mode)
    compute_classifier_fisher,    # per-sample diagonal Fisher for a classifier
    compute_conditional_vae_fisher,
    cosine_sim, effective_rank, top_k_overlap,
    vae_layer_category, classifier_layer_category
)
```
Estimator: F_ii = (1/N) sum_n (dL(x_n)/dtheta_i)^2, one backward pass per sample. The older batch-summed estimator in `scripts/fisher_overlap_analysis.py` (default damping 1e-5) is superseded; its numbers (global 0.306, 17x, median 0.877) must not be cited.

## Standard Workflow (NB26 pattern)

1. Load the baseline model and the structured split.
2. Build forget (30) and retain (28,094) `AnnDataDataset` loaders.
3. `fisher_forget = compute_vae_fisher(model, forget_loader, device, damping=0.0)`; same for retain. The canonical file's metadata records damping 0; damping large enough to matter inflates the cosine.
4. Global cosine, then per-category cosines grouped by `vae_layer_category(name)` (Encoder, Bottleneck, Decoder hidden, Decoder output).
5. Classifier comparison: train a linear probe (or the deep MLP of NB27) on cluster labels and repeat.
6. Save to `outputs/p6/fisher_overlap_results.json` (schema below) and load through `canonical.pbmc_fisher()`.

## Key Numbers (PBMC baseline, `outputs/p6/fisher_overlap_results.json`)

- VAE global 0.209; encoder 0.345; bottleneck 0.428; decoder hidden 0.351; decoder output 0.493 (fc_mean 0.485; fc_dispersion 0.996 is a vestigial-head artifact).
- Linear classifier (462 params, 14 classes) 0.018. Ratios 19x to 27x.
- Gene level: 1,453 of 2,000 genes (73%) above 0.5, median 0.857; log-Fisher r 0.71.
- Deep MLP classifier (NB27; 1,093,262 params, 95.8% accuracy): shared hidden 0.527, class-specific output 0.006.
- z=8 VAE (NB28): global 0.349, decoder output 0.543.
- Conditional VAE (NB29): cluster-specific fc_mean columns 0.0, shared columns 0.379; encoder 0.617, bottleneck 0.653, decoder hidden 0.508. Fisher scrub on it (NB30): advantage 0.759 to 0.709.
- TCGA-BRCA (NB35): global 0.753; every layer 0.73 to 0.90; all 978 genes above 0.5.

## Proposition 1 Verification (NB30, `outputs/p6/nb30_results.json`)

For a linear decoder f(z) = Wz + b with squared error, the diagonal Fisher factorizes: cos(F^F, F^R) = cos(sigma^F, sigma^R) x cos(nu^F, nu^R), where sigma is residual variance per gene and nu the latent second moment.
- fc_mean Fisher is near rank-1 (top singular value explains 97.5% forget, 94.3% retain).
- Fisher-marginal factors cos(sigma) 0.830 x cos(nu) 0.600 = 0.498 vs measured 0.485 (2.75% over). Data-direct factors 0.848 x 0.629 = 0.533.

## Phase audit (`scripts/fisher_phase_audit.py` -> `outputs/p6/phase_audit.json`)

Records forget, matched, retain, and held-out negELBO plus MLP and reference-LR AUC after each phase of Fisher scrubbing (max step 1e-3, 3e-3, 1e-2) and gradient ascent, on CPU (`PHASE_AUDIT_DEVICE`, default cpu; on MPS the large-step endpoints are not reproducible) with the deterministic estimator. Baseline forget 270; a clean retrain fits the forget cells at 334 to 337. Step 1e-3: 5,569 after the scrub (retain 1,250), 308 after the fine-tune (LR 0.987). Step 3e-3: 377 after the fine-tune (LR 0.173). Step 1e-2: 723 (LR 0.155). Gradient ascent: 270 after the ascent (unchanged), 296 after the fine-tune (LR 0.996). Run it for any new two-phase method.

## Output Format

```python
{'vae': {'global_cosine': float, 'per_category': {cat: {'cosine': float, 'n_params': int, ...}},
         'log_fisher_correlation': float, 'total_params': int, ...},
 'classifier': {'global_cosine': float, 'total_params': int, 'n_classes': int, ...},
 'gene_analysis': {'median_cosine': float, 'n_genes_cos_gt_0.5': int, 'n_genes_total': int, ...},
 'metadata': {'n_forget': 30, 'n_retain': 28094, 'damping': 0.0}}
```
