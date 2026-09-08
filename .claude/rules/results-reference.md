---
paths:
  - "Writeup.tex"
  - "Writeup.md"
  - "index.md"
  - "README.md"
  - "notebooks/20*"
  - "notebooks/25*"
  - "notebooks/36*"
  - "notebooks/41*"
---

# Established Results Reference (Stage 1 re-derivation, 2026-09-08)

Every number below traces to a JSON under `outputs/` and is loaded in code through
`src/canonical.py`. Never type a result into a notebook, script, or document by hand.
Pre-Stage-1 numbers (July 2026 and earlier) are listed at the end as superseded; do not
cite them. Claim-by-claim verdicts: `CLAIMS_AUDIT.md`. Picture and decision:
`STAGE1_ANALYSIS.md`. Regenerated comparison tables: `CLAIMS_AUDIT_TABLES.md`
(`scripts/claims_audit.py`).

## Protocol facts that determine the numbers

- Every method checkpoint descends from the July-4 baseline (`outputs/p1/baseline/best_model.pt`; sha256 and parent in `outputs/checkpoint_manifest.json`) and was re-derived with the fixed code (`LOSS_WEIGHTING_BUG.md`).
- Retrain reference = five clean retain-split retrains (`outputs/p1/retrain_clean/seed_{42..46}`, `scripts/retrain_clean.py`). The April retrains under `outputs/p1/multiseed_retrain/` had trained on 159 to 170 of the 194 matched negatives; they survive only as the `retrain_leaky` row.
- One CPU attacker (70-dim features, trained on the baseline's forget cells vs the 194 matched negatives) scores every checkpoint in one pass (`scripts/save_persample_scores.py`, then `scripts/nested_bootstrap_all.py`). A fresh attacker instance drifts by 0.001 to 0.02 AUC (the baseline scored 0.792 in one pass and 0.801 in the next), so quote to two decimals and never mix passes.
- Success rule: method advantage at or below the clean-retrain nested-CI upper bound, 0.182.
- Reference-model LR audit (`scripts/attack_lr_reference.py` -> `outputs/attack_lr_reference.json`): score = negELBO under clean retrain seed 42 minus negELBO under the target, forget vs matched, CPU, z = mu. AUC near 1: the forget cells are still fitted better than a never-trained model fits them (signal intact). Near 0: fitted worse (over-corrected). Floor: retrain seeds 43 to 46 scored against seed 42 give 0.24 to 0.52 (mean 0.37). The estimator is `src/likelihood_audit.py` (eval mode, 16 seeded z samples, CPU); two runs agree exactly. The VAE's unseeded single-sample forward pass, used before 2026-09-08, moved a 30-cell AUC by up to 0.14 between runs.

## PBMC-33k cluster 13 (30 megakaryocytes vs 194 matched negatives)

Sources: `outputs/p4/multiseed/nested_ci_all.json`, `persample/*.json`, `utility_all.json`, `outputs/attack_lr_reference.json`.

| Model | seeds | MLP AUC | advantage [nested 95% CI] | diff-from-retrain CI | LR audit | ELBO | marker r | verdict |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1 | 0.801 | 0.601 | | 0.996 | 363.9 | 0.831 | anchor |
| Retrain, clean | 5 | 0.529 | 0.057 [0.044, 0.182] | | 0.37 (0.24 to 0.52; seeds 43-46 vs 42) | 364.2 | 0.831 | reference |
| Retain-only fine-tune | 5 | 0.669 | 0.339 [0.236, 0.439] | [0.109, 0.352] | 0.991 | 363.0 | 0.832 | FAIL, signal intact |
| Gradient ascent | 5 | 0.706 | 0.412 [0.311, 0.510] | [0.180, 0.424] | 0.995 | 363.2 | 0.831 | FAIL, signal intact |
| SSD | 3 | 0.701 | 0.402 [0.276, 0.527] | [0.153, 0.437] | 0.995 | 363.6 | 0.831 | FAIL, signal intact |
| SCRUB | 3 | 0.716 | 0.431 [0.300, 0.555] | [0.178, 0.466] | 0.991 | 363.5 | 0.831 | FAIL, signal intact |
| Moon feature unlearning | 3 | 0.740 | 0.480 [0.355, 0.601] | [0.227, 0.505] | 0.996 | 364.5 | 0.831 | FAIL, signal intact |
| Fisher scrubbing (fixed, max step 1e-3) | 3 | 0.673 | 0.345 [0.216, 0.470] | [0.087, 0.386] | 0.991 | 363.6 | 0.832 | FAIL, signal intact |
| Fisher, max step 3e-3 / 1e-2 | 1 / 1 | 0.312 / 0.445 | 0.243 [0.052, 0.475] (two runs pooled) | [-0.066, 0.381] | 0.126 / 0.256 | 363.8 / 370.7 | 0.831 / 0.830 | over-corrected |
| Extra-gradient lambda=10 | 10 | 0.411 | 0.225 [0.149, 0.329] | [0.015, 0.243], Holm reject | 0.218 (7 seeds 0.007 to 0.15; 3 seeds 0.58 to 0.74) | 363.8 | 0.832 | FAIL, over-corrected, bimodal |
| Extra-gradient lambda=1 | 10 | 0.336 | 0.327 [0.204, 0.482] | [0.074, 0.385] | 0.049 | 363.3 | 0.831 | FAIL, over-corrected |
| Contrastive | 3 | 0.170 | 0.661 [0.561, 0.750] | [0.433, 0.669] | 0.004 | 363.7 | 0.831 | FAIL, over-corrected |
| Frozen critics, 4 configs | 3 each | 0.003 to 0.008 | 0.985 to 0.995 | [0.80, 0.95] | 0.002 to 0.007 | 363.8 (one critic) / 403 to 406 (three) | 0.831 / 0.832 | FAIL, over-corrected |
| DP-SGD (from scratch) | 3 | 0.466 | 0.068 [0.028, 0.195] | [-0.109, 0.109] | 0.502 | 402.9 | 0.789 | at retrain level |
| Retrain, leaky (April; superseded) | 5 | 0.579 | 0.158 [0.083, 0.261] | | 0.799 (the audit reads its training on the negatives) | 361.9 | 0.834 | not a reference |

Per-seed extra-gradient lambda=10 MLP AUC: 0.29, 0.30, 0.32, 0.38, 0.39, 0.40, 0.41 (over-corrected) and 0.53, 0.53, 0.57 (signal partly intact); none in the retrain band.
Null floor of the folded advantage estimator (30 vs 194, shuffled labels): 0.090 [0.004, 0.253].
Refit control (`retrain_refit.json`; fresh attacker on each model's own features, 20 x 5-fold out-of-fold): baseline CV AUC 0.797 (advantage 0.595); clean retrain advantage 0.035 [-0.029, 0.100] (AUC 0.507).
Two failure modes, no utility cost. Edit-then-fine-tune methods leave the forget cells fitted as well as the baseline does (LR 0.991 to 0.996 against the baseline's 0.996). Continuous forget-objective methods push them past the retrain level (LR 0.004 to 0.26). Every post-hoc method sits at baseline ELBO (363.0 to 364.5) and marker r (0.831 to 0.832) except the three-critic frozen runs (403 to 406).

## Phase audit (`outputs/p6/phase_audit.json`, negative ELBO per forget cell)

CPU run (`PHASE_AUDIT_DEVICE=cpu`, the default; the cached per-sample retain Fisher). Baseline 269.5; a clean retrain fits the forget cells at 334 (seed 42) to 337 (five-seed mean), never-seen matched cells sit at 333, retain and held-out cells at 363.
Fisher, max step 1e-3: 5,569 after the scrub (retain 1,250, held-out 1,284; MLP 0.016, LR 0.015), 308 after the retain fine-tune (MLP 0.666, LR 0.987). Step 3e-3: 59,490 after the scrub, 377 after fine-tune (MLP 0.386, LR 0.173). Step 1e-2: 216,484 after the scrub, 723 after fine-tune (retain 370; MLP 0.376, LR 0.155); this endpoint is not reproducible on MPS (518 and 1,076 in two MPS runs), which is why the audit runs on CPU.
Gradient ascent: 269.5 after the ascent phase (unchanged; MLP 0.802, LR 0.996), 296 after the retain fine-tune (retain 361.6; MLP 0.716, LR 0.996).
Reading: the scrub damages retain and forget together (the shared directions of Proposition 1); the retain fine-tune repairs the retain set and restores the forget set with it. That is the mechanism of the signal-intact mode. The over-corrected mode is a missing target, not entanglement.

## Attack suite, PBMC (`outputs/attack_suite_results.json`; library size restored; retrain = clean seed 42)

| Attack | Baseline AUC (adv) | Retrain | Extra-gradient | Fisher |
|---|---|---|---|---|
| Reconstruction threshold | 0.780 (0.560) | 0.537 (0.075) | 0.301 (0.399) | 0.688 (0.375) |
| KL threshold | 0.073 (0.854) | 0.130 (0.740) | 0.441 (0.117) | 0.079 (0.843) |
| ELBO threshold | 0.073 (0.854) | 0.129 (0.742) | 0.439 (0.123) | 0.079 (0.842) |
| k-NN latent (k=10) | 0.009 (0.982) | 0.019 (0.962) | 0.159 (0.683) | 0.010 (0.980) |

Reconstruction is a genuine membership attack (the clean retrain is near chance). KL, ELBO, and k-NN score the clean retrain far from chance, so they read biology. The suite's old likelihood-ratio row was one statistic reported with opposite signs; the reference-model LR audit replaces it.

## Cross-cohort and confound results

Tabula Muris cluster 33 (82 cardiac muscle cells vs 135 matched; `outputs/tabula_muris/unlearning_results.json`, one attacker): baseline 0.903, retrain 0.943, extra-gradient 0.723, Fisher 0.944. Scattered (30 vs 285): baseline 0.410, Fisher 0.408. Reference-LR against the single TM retrain (`outputs/attack_lr_reference_multi.json`): baseline 0.877, extra-gradient 0.947, Fisher 0.346 (no second retrain, so no floor).

TCGA-BRCA Basal (158 patients; `outputs/tcga_brca/unlearning/unlearning_results.json`; `within_subtype_eval/within_subtype_results.json`): cross-subtype (89 matched) baseline 0.803, retrain 0.830, Fisher 0.784, the other five 0.829 to 0.831. Within-subtype (30 unseen Basal) baseline 0.580, retrain 0.572, Fisher 0.543, the other five 0.571 to 0.572. Patient level, baseline / retrain: n=5 0.360 / 0.327; n=10 0.443 / 0.417; n=20 0.422 / 0.415. Reference-LR against the TCGA retrain: baseline 0.833, Fisher 0.191, retain-FT 0.457, gradient ascent 0.397, SSD 0.454, SCRUB 0.533, contrastive 0.519. Fisher global cosine 0.753 (`outputs/tcga_brca/fisher_overlap_results.json`).

PBMC cluster 12 (49 cells vs 57 matched; `outputs/p1_c12/scores_c12_v2.json`): baseline AUC 0.869 (advantage 0.737); clean retrain 0.814 (0.628 [0.527, 0.723]); retain-FT 0.867 (0.734); Fisher 0.858 (0.717, marker r 0.831, no collapse); extra-gradient 0.410 (0.330 [0.178, 0.482]; per-seed AUC 0.34, 0.28, 0.61).

PBMC cluster 7 (`outputs/larger_cluster_eval/results.json`): n=50 cross-cluster 0.665 to within-cluster 0.527 (retrain 0.655 to 0.518); n=100 0.713 to 0.481 (retrain 0.701 to 0.477). Tabula Muris cluster 28: n=50 cross 0.400, within 0.565; n=100 cross 0.617, within 0.402. PBMC cluster-13 within-cluster control (5 unseen megakaryocytes): 0.596, confound-fraction CI [-0.47, 0.98], unresolved.

## Size ablations (`outputs/p4/size_ablations_v2.json`; per-size clean retrains)

Rescored 2026-09-08 with the deterministic audit estimator (one fresh CPU attacker per size).
Structured, extra-gradient lambda=10 (3 seeds), baseline / retrain / extra-gradient AUC: n=10 0.567 / 0.454 / 0.326; n=50 0.713 / 0.565 / 0.374; n=100 0.764 / 0.668 / 0.411. Extra-gradient advantage 0.35 [0.14, 0.63], 0.25 [0.11, 0.41], 0.18 [0.07, 0.32]; audit 0.19, 0.12, 0.13 (over-corrected at every size; baseline 0.87 to 0.91, per-size retrain 0.32, 0.60, 0.57). The per-size retrains (advantage 0.11, 0.13, 0.34) show the multi-cluster sets are confound-affected.
Scattered, Fisher (10 seeds), baseline / retrain / Fisher AUC: n=10 0.554 / 0.555 / 0.543; n=30 0.588 / 0.581 / 0.585; n=50 0.574 / 0.568 / 0.570; n=100 0.550 / 0.543 / 0.547; n=500 0.517 / 0.510 / 0.515. The attacker finds nothing at any size. The audit does: baseline 0.69 to 0.79 against a retrain floor of 0.50 to 0.53 (n at least 30; 0.68 at n=10, where 10 cells against 99 negatives is noisy), and Fisher reduces it only partly (0.44 to 0.65).

## Constructive approaches (against the clean retrain)

RAU (`outputs/rau/rau_results.json`): KL to the retrain posterior falls from 180 to 0.03 to 0.23 across lambda in {0.1, 1, 10, 100}; cross-cluster AUC 0.28 to 0.35 (baseline 0.817, retrain 0.549); LR 0.007 to 0.018. Over-corrected despite the posterior match. TCGA: no-op (0.827 cross, 0.571 within).
Augmentation (`outputs/augmentation/augmentation_results.json`; 5 seed cells): cross-cluster aug-30 0.744, aug-100 0.795 (baseline 0.803, retrain 0.551); within-cluster 0.827 and 1.000 (baseline 0.607, retrain 0.713); LR 0.985 and 0.984. A model that never saw the forget cells is scored as if it had memorized them because it learned megakaryocytes. For a whole-subpopulation forget set, membership of the individuals and knowledge of the subpopulation are one quantity. TCGA within-subtype 0.581 (baseline 0.582, retrain 0.573).

## Likelihood-targeted unlearning (NB41; `outputs/p7/targeted_unlearning.json`, `src/train_targeted_unlearn.py`)

Objective: retain `vae_loss` + lambda x mean over forget cells of |negELBO(cell) - target|, lambda 1, Adam 1e-4, stop once the mean gap stays within 5 nats for 3 epochs; ten seeds per variant; the deterministic estimator throughout; the notebook's own fresh CPU attacker (baseline 0.802 under it). Calibration axis (Part A; forget-cell negELBO): baseline 270, clean retrain 337 (five seeds; matched negatives 333), edit-then-fine-tune methods 292 to 306, extra-gradient 362 and 378, RAU 406, contrastive 421, frozen critics above 1,000.

| Variant | Target | Forget negELBO | MLP AUC | Advantage [nested CI] | Audit | ELBO / marker r | Reading |
|---|---|---|---|---|---|---|---|
| B, look-alike | each forget cell's 10 nearest evaluation negatives (the model fits them at about 300) | 309 | 0.75 | 0.503 (0 of 10 seeds at or below 0.182) | 0.62 | 364.7 / 0.830 | short of the retrain level |
| C, cohort average | mean over the 194 evaluation negatives (335; the retrain's level on this cohort) | 337 | 0.58 | 0.166 [0.111, 0.212] (6 of 10 seeds at or below 0.182) | 0.41 (band 0.24 to 0.52) | 365.0 / 0.830 | the level was the missing ingredient; optimistic, since the target was read off the evaluation's own negatives |
| D, disjoint look-alikes | 88 unseen cells ranked 11 to 30 nearest the forget cells, none an evaluation negative (fitted at 292 by baseline and retrain alike) | 296 | 0.83 | 0.668 [0.630, 0.700] | 0.70 | 364.7 / 0.830 | the observable substitute is the wrong level |

Lambda ablation, seed 42, look-alike target: lambda 0.3 / 1 / 3 give MLP 0.777 / 0.746 / 0.725 and audit 0.59 / 0.53 / 0.50. Why the look-alike level is wrong (`outputs/p7/calibration_levels.json`, `scripts/calibration_levels.py`): the five clean retrains fit the forget megakaryocytes at 334 to 342 but their ten nearest neighbours at 298 to 301 and the 88 disjoint look-alikes at 292 to 293, the same as the baseline does (287 and 292). A model without the subpopulation fits an unseen member of it worse than that member's nearest known-type neighbours. The target is the fit of a type the model does not know, and no cell it does know supplies it; without a retrain it is not observable.

## Fisher overlap (`outputs/p6/fisher_overlap_results.json`; per-sample estimator, damping 0)

| Category | params | cosine |
|---|---|---|
| Encoder | 2,642,816 | 0.345 |
| Bottleneck | 8,256 | 0.428 |
| Decoder hidden | 598,912 | 0.351 |
| Decoder output (fc_mean 0.485; fc_dispersion 0.996 is a vestigial-head artifact) | 4,100,000 | 0.493 |
| VAE global | 7,349,984 | 0.209 |
| Linear classifier (14 classes) | 462 | 0.018 |

Ratios 19x (encoder) to 27x (decoder output) over the classifier. Gene level: 1,453 of 2,000 genes (73%) with cosine above 0.5, median 0.857; log-Fisher correlation 0.71; top-1% overlap 0.28, top-10% 0.45.
Controls. Deep MLP classifier (`fisher_overlap_deep_clf.json`; 1,093,262 params, 95.8% accuracy): shared hidden 0.527, class-specific output 0.006, global 0.301. z=8 VAE (`fisher_overlap_z8.json`): global 0.349, encoder 0.466, decoder output 0.543. Conditional VAE (`fisher_overlap_conditional.json`): cluster-specific fc_mean columns 0.0, shared columns 0.379, full fc_mean 0.016; encoder 0.617, bottleneck 0.653, decoder hidden 0.508, global 0.625. Fisher scrub on the conditional VAE moves advantage 0.759 to 0.709 (`nb30_results.json`).
Proposition 1 (`nb30_results.json`): measured fc_mean cosine 0.485; Fisher-marginal factors cos(sigma) 0.830 x cos(nu) 0.600 = 0.498, 2.75% above measured; data-direct factors 0.848 x 0.629 = 0.533; SVD top-1 share 0.975 (forget) and 0.943 (retain).

## Superseded (do not cite)

Pre-Stage-1 (July 2026; leaky retrain, buggy Fisher, extra-gradient, and attack code): retrain 0.578 / 0.156 [0.082, 0.258] and threshold 0.258; Fisher 0.808 / 0.615 with "posterior collapse" (ELBO 490, KL 0.007, marker r 0.628); extra-gradient 0.433 / 0.281 with ELBO 475 and marker r 0.691; retain-FT 0.333; gradient ascent 0.396; SSD 0.435; SCRUB 0.411; contrastive 0.673; DP-SGD 0.045; refit retrain 0.134 [0.05, 0.22]; attack-suite reconstruction "identical 0.438 across models"; frozen critics AUC 0.98 to 0.997 (December, single seed); Tabula Muris baseline 0.891 / retrain 0.944 / extra-gradient 0.874 / Fisher scattered 0.568; cluster 12 Fisher marker r 0.21 and extra-gradient advantage 0.027; the December single-seed tables (baseline 0.769, extra-gradient 0.482 and 0.382, Fisher 0.814 and 0.499, structured size ablation 0.37 / 0.48 / 0.61 / 0.59, scattered 0.588 / 0.601 / 0.574 / 0.547 / 0.516); MPS-era values (baseline 0.783 / 0.565, retrain 0.574 / 0.148 / 0.229); Fisher batch-summed cosines (global 0.306, 17x, median 0.877) and the old per-category table (0.273 / 0.291 / 0.232); conditional-VAE encoder 0.433 and bottleneck 0.508; deep-classifier accuracy 95.2%; the Proposition 1 "0.41 vs 0.37, 11%" pair. Superseded files: `outputs/_superseded_20260908/`.
