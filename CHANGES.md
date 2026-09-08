# Changes from initial paper to current version

## Overview

The initial paper (Writeup_old.pdf) was a 4-page IEEE-format report comparing two unlearning methods (extra-gradient, Fisher) on PBMC-33k. The current version (Writeup.pdf) is a 17-page NeurIPS submission testing eight methods on two datasets. The main conclusion flipped: the old paper said extra-gradient succeeds, the new paper says everything fails, and Section 6 explains why.

---

## Framing and conclusion

The original paper concluded that extra-gradient with lambda=10 achieves retrain-equivalent privacy on PBMC-33k (AUC=0.482, within 0.001 of the retrain floor). That was one seed. With 10 seeds, mean advantage=0.300, 95% CI [0.226, 0.374], which exceeds the retrain CI upper bound (0.266). So it fails. All eight approximate methods fail on structured forget sets now.

## Metric change

The original paper used raw AUC with a +/-0.03 target band around the retrain floor (0.481). The current paper switched to membership inference advantage = 2|AUC - 0.5|, adapted from Yeom et al. [2018]. Advantage is direction-agnostic: AUC=0.38 (over-unlearning) and AUC=0.62 (under-unlearning) both give advantage=0.24, since an adversary can flip predictions. The old metric would have counted AUC=0.38 as "below target" without flagging it as a failure.

---

## New experimental content

### Second dataset: Tabula Muris
41,647 cells from 12 mouse tissues, 35 Leiden clusters. Structured forget set is cluster 33 (82 cardiac muscle cells). Extra-gradient does not transfer (AUC=0.874 regardless of lambda). Fisher fails on structured sets in both datasets (PBMC 0.814, TM 0.946).

### Six additional unlearning methods

| Method | Source file | Notebook | Result |
|--------|-----------|----------|--------|
| Retain-only fine-tuning | `src/train_retain_finetune.py` | NB16 | Advantage=0.331, no privacy gain |
| Gradient ascent | `src/train_gradient_ascent.py` | NB17 | Advantage=0.404, no privacy gain |
| SSD | `src/train_ssd.py` | NB21 | Advantage=0.450, dampens but does not unlearn |
| Contrastive latent | `src/train_contrastive_unlearn.py` | NB22 | AUC=0.153, Streisand effect (advantage=0.695) |
| DP-SGD | `src/train_dp.py` | NB23 | Advantage=0.072, closest to retrain but trains from scratch |
| SCRUB | `src/train_scrub.py` | NB24 | Advantage=0.474, retain objective dominates |

### Attack diversity (Section 5.3)
Four attack families beyond the trained MLP, implemented in NB15 and reported in Table 4:
- Threshold attacks on reconstruction loss, KL divergence, and ELBO individually (`src/threshold_attack.py`)
- Likelihood-ratio attack comparing per-sample loss against a reference model (`src/likelihood_ratio_attack.py`)
- k-NN latent attack classifying membership by distance in latent space (`src/knn_attack.py`)

High-assumption attacks (k-NN, likelihood ratio) show advantage >= 0.73 even on the retrain model, meaning they pick up cell-type identity rather than membership.

### Utility evaluation expansion
`src/utility_metrics.py` computes ELBO, KL divergence, ARI, and marker gene correlation. NB18 runs these across all methods. Table 6 and Figure 2 (privacy-utility tradeoff) are new. The original paper only had a 4-row table with reconstruction MSE, marker MSE, classification accuracy, and runtime.

### Forget set size ablation (Appendix B)
NB09 runs extra-gradient at forget set sizes 10, 30, 50, and 100 on PBMC-33k. Larger sets span multiple clusters and the method cannot unlearn them. A scattered-set ablation (n=10 to 500) confirmed scattered cells have low baseline memorization and Fisher provides marginal improvement at best.

### Matched negative validation (Appendix C)
NB19 tests a within-cluster holdout using only the 5 unseen cluster-13 cells. Baseline AUC drops from 0.769 to 0.527, confirming the biology confound in the default matched negatives. Relative method rankings hold.

---

## Fisher overlap theory (Section 6)

The biggest addition to the paper. The old version mentioned informally that VAE decoders share parameters and speculated about why Fisher fails on structured sets. The new version quantifies this with diagonal Fisher overlap measurements and a formal proposition.

### Core analysis (NB26)
`notebooks/26_fisher_overlap_analysis.ipynb` computes diagonal Fisher on the forget set and retain set, measures cosine similarity per layer category. VAE global cosine = 0.306; linear classifier on the same data = 0.018. The 17x ratio is the central number in the paper. Results in `outputs/p6/fisher_overlap_results.json`.

### Proposition 1 and Corollary 2
For a linear decoder f(z)=Wz+b with squared-error loss, diagonal Fisher factorizes as cos(F_forget, F_retain) = cos(sigma_forget, sigma_retain) * cos(nu_forget, nu_retain), where sigma is residual variance per output dimension and nu is the latent second moment. Corollary 2 gives scaling bounds: cos(sigma) >= (D-M)/(D-M+MV^2) for generative models, cos(sigma) = 1/sqrt(C) for single-class classifier forget sets. NB30 checks the factorization empirically (predicted=0.41, measured=0.37, 11% error).

### Deep MLP classifier (NB27)
`notebooks/27_deep_clf_fisher.ipynb` trains a 1.09M-parameter classifier (2000 -> [512, 128] -> 14) to control for model capacity. Shared hidden cosine=0.262 (comparable to VAE encoder 0.273); class-specific output cosine=0.010 (comparable to linear classifier 0.018). Model size is not the explanation. Results in `outputs/p6/fisher_overlap_deep_clf.json`.

### VAE z=8 (NB28)
`notebooks/28_vae_z8_fisher.ipynb` repeats the analysis with latent dim 8 instead of 32. Global cosine goes up to 0.846, driven by the bottleneck (0.858 vs. 0.291 for z=32). Smaller latent space concentrates the representation and makes overlap worse. Checkpoint at `outputs/p6/vae_z8_best_model.pt`.

### Conditional VAE (NB29)
`notebooks/29_conditional_vae_fisher.ipynb` and `src/vae_conditional.py` condition the decoder on a 14-dimensional cluster one-hot. Cluster-specific output columns achieve cosine near 0 (1.2e-8), but the shared hidden representation still has cosine=0.222. Fisher scrubbing on this model barely moves advantage (0.73 to 0.72) because overlap remains in the encoder (0.433), bottleneck (0.508), and decoder hidden layers (0.346). Results in `outputs/p6/fisher_overlap_conditional.json`.

### Proposition verification and CVAE scrubbing (NB30)
`notebooks/30_proposition_test_and_cvae_scrub.ipynb` runs SVD on fc_mean Fisher (showing ~95% rank-1), checks the factorized prediction against measurements, and tests Fisher scrubbing on the conditional VAE. Results in `outputs/p6/nb30_results.json`.

### Supporting module
`src/fisher_utils.py` consolidates shared functions for NB26-30: dataset classes, Fisher computation, cosine similarity, effective rank, top-k overlap, and layer categorization.

---

## New citations and why they are cited

### Carried over from the original paper (9 total)
Bourtoule et al. 2021, Cao and Yang 2015, Chavdarova et al. 2019, Golatkar et al. 2020, Hayes et al. 2024, Lopez et al. 2018, Nasr et al. 2018, Shokri et al. 2017, Wolf et al. 2018.

### Added in the current paper (16 new)

| Citation | Why it is cited |
|----------|----------------|
| Feldman 2020 | Theoretical argument that memorization of rare subpopulations may be necessary for generalization. Motivates why structured memorization is expected (Introduction). |
| Yeom et al. 2018 | Defined membership inference advantage as a privacy metric. The paper adapts their definition to advantage = 2\|AUC - 0.5\|. |
| Abadi et al. 2016 | DP-SGD: per-sample gradient clipping + Gaussian noise for formal (epsilon, delta) privacy. Used as the formal privacy baseline experiment. |
| Carlini et al. 2022 | Likelihood ratio attack (LiRA): trains shadow models, compares per-example difficulty scores against calibrated null distributions. One of the four attack families in Section 5.3. |
| Thudi et al. 2022 | Argued that without auditable definitions, unlearning claims are unverifiable in practice. Cited in the discussion (Section 7) on verification. |
| Moon et al. 2024 | Latent-space interventions for removing class-level features from GANs/VAEs. Distinguished from this work because their goal is preventing generation of a class, not preventing membership detection. |
| Foster et al. 2024 | SSD: dampens parameters proportional to forget-set Fisher importance. One of the eight tested methods. |
| Kurmanji et al. 2023 | SCRUB: teacher-student distillation, student matches teacher on retain data while diverging on forget data. One of the eight tested methods. |
| Kunstner et al. 2019 | Showed diagonal Fisher can misrepresent true Fisher in deep networks. Cited in Section 6 to acknowledge the limitation of diagonal Fisher, while arguing the analysis is still valid since the methods themselves use diagonal Fisher internally. |
| Basu et al. 2021 | Influence functions are unreliable in non-convex models. Cited in related work to question influence-based parameter selection for unlearning. |
| Ginart et al. 2019 | Deletion-efficient algorithms for k-means. Cited in related work as an example of exact unlearning that requires convexity. |
| Guo et al. 2020 | Certified removal via Newton-step updates for convex models. Same context as Ginart. |
| Sekhari et al. 2021 | Sample complexity bounds for unlearning under strong convexity. Same context. |
| Neel et al. 2021 | Noisy gradient descent admits deletion guarantees in smooth settings. Same context. |
| Izzo et al. 2021 | Projective residual updates for approximate deletion in linear models. Same context. |
| Tabula Muris Consortium 2018 | Source of the Tabula Muris dataset (41,647 cells from 20 mouse organs). |

---

## New code

### Source modules (src/)

| File | What it does |
|------|-------------|
| `train_retain_finetune.py` | Fine-tune VAE on retain set only |
| `train_gradient_ascent.py` | Gradient ascent on forget set + retain fine-tuning |
| `train_ssd.py` | Selective Synaptic Dampening |
| `train_contrastive_unlearn.py` | Contrastive latent unlearning (push forget representations to prior) |
| `train_scrub.py` | SCRUB teacher-student distillation adapted for VAEs |
| `train_dp.py` | DP-SGD training via Opacus |
| `vae_conditional.py` | Conditional VAE with cluster one-hot decoder input |
| `attacker_eval.py` | Post-hoc fresh attacker evaluation |
| `utility_metrics.py` | ELBO, KL, ARI, marker gene correlation |
| `fisher_utils.py` | Shared Fisher analysis functions for NB26-30 |
| `threshold_attack.py` | Threshold-based MIA on individual loss components |
| `likelihood_ratio_attack.py` | Likelihood ratio attack |
| `knn_attack.py` | k-NN latent space attack |

### Notebooks added (15 new, NB11-30 minus NB14 and NB20 which are consolidation)

| Notebook | Purpose |
|----------|---------|
| NB11-13 | Tabula Muris data prep, baseline, and unlearning |
| NB15 | Attack suite (threshold, LR, k-NN, MLP) |
| NB16 | Retain-only fine-tuning experiments |
| NB17 | Gradient ascent experiments |
| NB18 | Utility suite across all methods |
| NB19 | Matched negative validation (within-cluster holdout) |
| NB21 | SSD experiments |
| NB22 | Contrastive latent experiments |
| NB23 | DP-SGD experiments |
| NB24 | SCRUB experiments |
| NB25 | Consolidated evaluation |
| NB26 | Fisher overlap analysis (canonical, Section 6) |
| NB27 | Deep MLP classifier Fisher overlap |
| NB28 | VAE z=8 Fisher overlap |
| NB29 | Conditional VAE Fisher overlap |
| NB30 | Proposition verification + CVAE scrubbing |

### Scripts

| File | What it does |
|------|-------------|
| `scripts/fisher_overlap_analysis.py` | Fisher overlap computation pipeline |
| `scripts/run_deep_clf_fisher.py` | Deep MLP classifier Fisher pipeline |
| `scripts/run_vae_z8_fisher.py` | VAE z=8 Fisher pipeline |
