# TASKS.md - Publication Readiness Plan

## Overview

Paper draft is complete with 8 unlearning methods, 2 datasets, attack suite, utility suite, and multi-seed evaluation. Phases 1-5 (evaluation hardening, method breadth, utility, seeds, writeup) are done. Phase 6a (Fisher overlap analysis, figures, NB26) is done. The remaining work is Phase 6b: strengthening the theoretical contribution based on a peer-review simulation that identified 6 weaknesses (W1-W4, W6-W7).

**Target venue:** NeurIPS 2025 (professor to advise on venue).

**Current acceptance estimate:** 50-60% at TMLR. Phase 6b improvements should push to 65-75%.

### Success Criterion

> **Robust unlearning is achieved when the worst-case attack advantage across all attacks is at or below the upper bound of the retrain advantage 95% CI.**

Where:
- `advantage = 2 × |AUC - 0.5|` (symmetric, direction-agnostic)
- Worst-case = maximum advantage across all attack types
- Retrain CI = bootstrap 95% confidence interval

---

## Completed Work

### Phases 1-5 (2026-01 to 2026-02-08)
- [x] PBMC-33k + Tabula Muris baselines, structured/scattered splits, matched negatives
- [x] 8 unlearning methods: frozen critics, extra-gradient, Fisher, retain-FT, gradient ascent, SSD, SCRUB, contrastive latent
- [x] DP-SGD baseline via Opacus 1.4.0
- [x] Attack suite: threshold (recon, KL, ELBO), likelihood-ratio, kNN latent, MLP
- [x] Utility metrics: ELBO, silhouette, ARI, marker gene correlation
- [x] Multi-seed evaluation with fresh attackers (bug fix 2026-02-08)
- [x] Size ablation, lambda sweep, cross-dataset validation
- [x] Writeup with all results, discussion, NeurIPS checklist

### Phase 6a: Fisher Overlap Analysis (2026-02-09 to 2026-02-15)
- [x] `scripts/fisher_overlap_analysis.py` (damping=1e-5, outputs/p6/fisher_overlap.json)
- [x] NB26: Fisher overlap analysis notebook (damping=1e-8, canonical results)
  - Diagonal Fisher computed on forget (30 cells) and retain (28,094 cells)
  - Linear classifier probe (14 classes, 462 params, 93.7% accuracy)
  - 4 figures: per-layer overlap, effective rank, scatter, gene analysis
  - Results saved to `outputs/p6/fisher_overlap_results.json`
- [x] Section 6 in Writeup.tex with Table 6 (Fisher overlap by layer category)
- [x] All paper numbers updated to match NB26 (damping=1e-8)
  - Global cosine: 0.306, classifier: 0.018, ratio: 17x
  - Per-category cosines, effective rank ratios, gene-level stats
- [x] Writeup humanizer pass (2 rounds, all AI patterns removed)
- [x] Validation audit (5 parallel agents): fixed duplicate label, tautology, overclaiming, Yeom attribution, SSD description, scattered AUC range, Streisand citation, checklist items
- [x] NB26 analysis section added and humanized
- [x] Related work updated with Golatkar 2020 for Streisand effect

---

## Phase 6b: Theory Strengthening (CURRENT)

Peer-review simulation identified 6 weaknesses. The tasks below address them.

### Weakness Summary

| ID | Issue | Severity | Task |
|----|-------|----------|------|
| W1 | Single VAE architecture (z=32 only) | Major | Task 4 |
| W2 | Diagonal Fisher approximation not discussed | Major | Task 1 |
| W3 | Classifier comparison is asymmetric (462 vs 7.35M params) | Major | Task 3 |
| W4 | No proposed solution (diagnosis only) | Moderate | Task 6 |
| W6 | Feldman 2020 missing (structured memorization is expected) | Moderate | Task 2 |
| W7 | No formal proposition (empirical observation, not theory) | Moderate | Task 5 |

### Task 1: Add diagonal Fisher defense paragraph [W2]

**Status:** DONE
**Estimated time:** 30 min
**Dependencies:** None

Add 1 paragraph to Section 6 of Writeup.tex acknowledging the diagonal Fisher approximation and explaining why it's the right metric here:
- Cite Kunstner et al. (ICML 2019) on limitations of diagonal Fisher
- Key defense: Fisher scrubbing, SSD, and SCRUB all USE the diagonal Fisher
- The overlap characterizes these methods as implemented, not the true Fisher
- Add Kunstner 2019 to references.bib

### Task 2: Add Feldman 2020 and missing citations [W6]

**Status:** DONE
**Estimated time:** 20 min
**Dependencies:** None

- Add Feldman 2020 "Does Learning Require Memorization?" (NeurIPS 2020) to references.bib
- Cite in Introduction where structured memorization is discussed
- Connect the structured/scattered dichotomy to Feldman's long-tail theory
- Optionally add Jia et al. 2023 "Model Sparsity Can Simplify Machine Unlearning"

### Task 3: Train deep MLP classifier and compute Fisher overlap [W3]

**Status:** DONE (NB27, 2026-02-16)
**Dependencies:** None

Architecture: 2000 → [512, 128] → 14, LayerNorm + Dropout(0.3), weight_decay=1e-2.
1.09M params. Early-stopped at 95.2% accuracy (8 epochs) to keep Fisher meaningful.
- Shared hidden cosine: 0.262 (matches VAE encoder 0.273, decoder hidden 0.232)
- Class-specific output cosine: 0.010 (matches linear clf 0.018)
- Log-Fisher correlation: 0.752
- Results: `outputs/p6/fisher_overlap_deep_clf.json`

### Task 4: Test VAE architecture variant (z=8) [W1]

**Status:** DONE (NB28, 2026-02-16)
**Dependencies:** None

Same hidden dims, latent_dim=8 instead of 32. Best epoch 77, val_loss 366.3.
- Global cosine: 0.846 (higher than z=32's 0.306, driven by bottleneck)
- Bottleneck cosine: 0.858 (vs z=32's 0.291)
- Encoder: 0.314, Decoder hidden: 0.216, Decoder output: 0.299
- Overlap persists and increases with smaller latent dim
- Results: `outputs/p6/fisher_overlap_z8.json`, checkpoint: `outputs/p6/vae_z8_best_model.pt`

### Task 5: Write Proposition 1 [W7]

**Status:** DONE (2026-02-16)
**Dependencies:** Tasks 3 and 4 (experimental numbers inform the statement)

Proposition 1 added to Section 6 of Writeup.tex. For linear decoder f(z) = Wz + b with MSE loss:
- Diagonal Fisher factorizes as F_{dh} = 4 E[e_d²] E[z_h²] (outer product structure)
- Fisher cosine = cos(σ^F, σ^R) · cos(ν^F, ν^R) (exact identity)
- For VAEs with KL regularization, ν^F ≈ ν^R ≈ 1, so cosine ≈ cos(σ^F, σ^R)
- With D=2000 shared output dims, residual profiles overlap → high cosine
- For classifiers, residual variance concentrates on class-specific logits → low cosine
- Measured fc_mean cosine (0.360) below linear prediction due to nonlinearity
- NeurIPS checklist updated from "Partial" to "Yes"
- Contributions list updated to mention the proposition

### Task 6: Test cluster-conditional decoder [W4]

**Status:** DONE (NB29, 2026-02-16)
**Dependencies:** None

Conditional VAE with cluster one-hot concatenated to fc_mean input ([1024+14] = 1038).
Best epoch 95, val_loss 365.8.
- Global cosine: 0.432
- Decoder output cosine: 0.246 (vs standard 0.362)
- fc_mean decomposition: shared cols 0.222, cluster cols ~0.000 (1.2e-8)
- Cluster columns achieve full disentanglement (like classifier)
- Shared columns retain irreducible overlap from shared hidden features
- Results: `outputs/p6/fisher_overlap_conditional.json`, checkpoint: `outputs/p6/cvae_best_model.pt`

---

## Execution Plan

**Wave 1 (parallel, ~30 min):** Tasks 1 + 2
Quick Writeup.tex and references.bib edits. No code.

**Wave 2 (parallel, 4-12 hours each):** Tasks 3 + 4 + 6
Three independent experiments. Can run concurrently.
- Task 3: Deep MLP classifier Fisher overlap
- Task 4: VAE z=8 Fisher overlap
- Task 6: Cluster-conditional decoder + Fisher scrubbing test

**Wave 3 (after experiments, 2-4 hours):** Task 5
Write Proposition 1, informed by experimental numbers from Tasks 3, 4, 6.

**Wave 4 (final, ~1 hour):** Integrate all results into Writeup.tex
Update Section 6 table, prose, abstract, conclusion. Final humanizer pass.

---

## New Files to Create (Phase 6b)

| File | Purpose | Task |
|------|---------|------|
| `src/vae_conditional.py` | Cluster-conditional VAE decoder | Task 6 |
| `outputs/p6/fisher_overlap_z8.json` | z=8 variant Fisher results | Task 4 |
| `outputs/p6/fisher_overlap_deep_clf.json` | Deep classifier Fisher results | Task 3 |
| `outputs/p6/fisher_overlap_conditional.json` | Conditional decoder Fisher results | Task 6 |

---

## Success Criteria

**Paper is ready for submission when all checked:**

Evaluation (phases 1-5, COMPLETE):
- [x] Advantage metric, success = worst-case ≤ retrain CI upper
- [x] 5 attack types, 8 unlearning methods, DP-SGD baseline
- [x] Domain-relevant utility (ELBO, ARI, marker correlation)
- [x] Multi-seed with bootstrap CIs, 2 datasets
- [x] Subpopulation framing, humanized writing, references validated

Theory (phase 6a, COMPLETE):
- [x] Fisher overlap computed for VAE (global cosine = 0.306)
- [x] Linear classifier comparison (cosine = 0.018, 17x lower)
- [x] 4 figures showing structural difference
- [x] Section 6 in Writeup.tex with Table 6

Theory strengthening (phase 6b, COMPLETE):
- [x] Diagonal Fisher defense paragraph (Task 1)
- [x] Feldman 2020 citation for structured memorization (Task 2)
- [x] Deep MLP classifier with shared hidden layers (Task 3)
- [x] VAE z=8 architecture variant (Task 4)
- [x] Proposition 1 with proof sketch (Task 5)
- [x] Cluster-conditional decoder experiment (Task 6)
- [x] Proposition 1 empirical verification (NB30 Part A)
- [x] Conditional VAE Fisher scrubbing test (NB30 Part B)
