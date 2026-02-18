# Fishing for Privacy: Machine Unlearning for Single-Cell VAEs

**David Benson**
Columbia University
dmb2262@columbia.edu

## Abstract

Single-cell RNA sequencing models can memorize individual training samples, creating privacy risks when the data contains sensitive biological information. Eight unlearning methods were evaluated against four attack families on two datasets (PBMC-33k and Tabula Muris). All eight methods fail on structured (biologically coherent) forget sets. Methods that treat unlearning as a small parameter perturbation (retain-only fine-tuning, gradient ascent, SSD, SCRUB) preserve utility perfectly but produce no measurable privacy improvement. Fisher scrubbing and contrastive latent unlearning make the model detectably worse rather than detectably better. Extra-gradient co-training shows high variance across seeds (mean advantage = 0.300, 95% CI [0.226, 0.374]). DP-SGD trained from scratch on the retain set comes closest to the retrain baseline (advantage = 0.072 vs. 0.046), but at a real utility cost and by construction, not by unlearning. A Fisher information analysis reveals the structural cause: the VAE's shared decoder produces 17x higher Fisher overlap between forget and retain sets than a classifier on the same data, so selective parameter perturbation cannot cleanly separate the two. Full retraining remains the only dependable option for structured forget sets.

## 1. Introduction

Deep learning models memorize training data, and for rare subpopulations this memorization may be necessary for generalization (Feldman 2020). When those models are trained on sensitive biological information, an attacker can determine whether a specific sample was part of the training set through a membership inference attack (MIA).

This paper tests machine unlearning for VAEs trained on single-cell RNA sequencing (scRNA-seq) data, where gene expression profiles can reveal disease status, genetic predispositions, and other personal health information. The work addresses subpopulation unlearning, meaning the removal of entire biological groups such as rare cell types, not individual samples.

Privacy leakage is measured by membership inference advantage (adapted from Yeom et al. 2018): advantage = 2|AUC - 0.5|. This is direction-agnostic, so AUC = 0.38 (over-unlearning) and AUC = 0.62 (under-unlearning) both give advantage = 0.24. Unlearning succeeds when the post-hoc advantage falls within the 95% CI of the retrain model's advantage.

**Contributions.**

1. An evaluation protocol for MIAs in single-cell VAEs using biologically matched negatives.
2. Evidence that memorization concentrates in structured subpopulations (baseline MIA AUC of 0.78-0.89 for structured clusters; 0.41-0.53 for scattered cells).
3. A systematic comparison of eight unlearning methods plus DP-SGD, against four attack families, with utility evaluation across four metrics.
4. A catalog of failure modes: posterior collapse, critic exploitation, Streisand effect, parameter-space ineffectiveness, and dataset dependence.
5. A Fisher information analysis showing the VAE's shared decoder creates cosine similarity of 0.306 between forget-set and retain-set Fisher diagonals, 17x higher than a classifier (0.018). A proposition formalizes this gap for linear decoders, showing the Fisher cosine factorizes into residual-profile and latent-moment components, with the residual-profile factor scaling as 1 - O(M/D) for generative models and O(1/sqrt(C)) for classifiers.

## 2. Datasets

| | PBMC-33k | Tabula Muris |
|---|---|---|
| Cells | 33,088 | 41,647 |
| HVGs | 2,000 | 2,000 |
| Clusters | 14 | 35 |
| Tissues | 1 (blood) | 12 |
| Train / Unseen | 28,124 / 4,964 | 35,399 / 6,248 |
| Structured forget set | Cluster 13 (30 megakaryocytes) | Cluster 33 (82 cardiac muscle) |
| Scattered forget set | 35 random | 30 random |
| Matched negatives | 194 | 137 |

The PBMC-33k dataset consists of 33,088 peripheral blood mononuclear cells from 10x Genomics, preprocessed with Scanpy (Wolf et al. 2018). The Tabula Muris dataset has 41,647 cells from 12 mouse tissues. Matched negatives were selected as the unseen cells closest to the forget set in the baseline model's latent space (k-NN with k=10).

## 3. Methods

**VAE architecture.** Encoder: 2000 input genes through [1024, 512, 128] to latent mean and log-variance (z=32). Decoder reverses this to negative binomial parameters. Layer normalization and dropout (0.1) after each hidden layer. 7.35M total parameters.

**Eight unlearning methods tested:**

1. **Retain-only fine-tuning.** Fine-tune on the 28,094 retain cells.
2. **Gradient ascent.** Maximize loss on forget set, then fine-tune on retain set.
3. **Frozen critics.** Freeze pre-trained attackers, update VAE to minimize their success.
4. **Extra-gradient co-training.** Min-max game with extragradient updates, TTUR (attacker LR 5x higher), 3 co-trained critics, lambda=10, 50 epochs.
5. **Fisher scrubbing** (Golatkar et al. 2020). Perturb parameters inversely proportional to Fisher curvature. alpha=1e-4, lambda=0.1, 100 steps + 10 finetune epochs.
6. **SSD** (Foster et al. 2024). Dampen parameters proportional to forget-set Fisher importance.
7. **Contrastive latent.** Push forget-set latent representations toward prior N(0,I), preserve retain-set representations.
8. **SCRUB** (Kurmanji et al. 2023). Teacher-student distillation: match teacher on retain data, diverge on forget data.

**DP-SGD baseline.** Trains from scratch on retain set with per-sample gradient clipping and Gaussian noise (Abadi et al. 2016). Privacy by exclusion, not by unlearning.

**Attack suite.** Trained MLP (69-dim features, spectral norm), threshold attacks (reconstruction, KL, ELBO), likelihood ratio, k-NN latent.

## 4. Results

### Main results (PBMC-33k structured forget set)

| Method | Seeds | AUC | Advantage | Marker r | Status |
|--------|-------|-----|-----------|----------|--------|
| Baseline | — | 0.783 | 0.565 | 0.831 | — |
| Retain-only fine-tune | 5 | 0.665 +/- 0.007 | 0.331 | 0.832 | FAIL |
| Gradient ascent | 5 | 0.702 +/- 0.004 | 0.404 | 0.832 | FAIL |
| SSD (alpha=1.0) | 3 | 0.725 +/- 0.001 | 0.450 | 0.831 | FAIL |
| SCRUB (alpha_f=1.0) | 3 | 0.737 +/- 0.002 | 0.474 | 0.832 | FAIL |
| Contrastive latent | 3 | 0.153 +/- 0.032 | 0.695 | 0.832 | FAIL (Streisand) |
| Fisher scrubbing | 3 | 0.814 +/- 0.003 | 0.628 | — | FAIL (worse) |
| Extra-gradient (lambda=10) | 10 | 0.429 +/- 0.142 | 0.300 | 0.789 | FAIL |
| DP-SGD (eps=10) | 3 | 0.464 +/- 0.024 | 0.072 | 0.787 | Near target |
| Retrain | — | 0.523 | 0.046 | 0.829 | TARGET |

Retrain advantage = 0.046, 95% CI upper bound = 0.266. No approximate method achieves mean advantage within this interval.

### Fisher by forget set type

Fisher achieves AUC = 0.499 on scattered sets (near chance, but baseline is already 0.525) and AUC = 0.814 on structured sets (worse than baseline). The memorization problem concentrates in structured subpopulations, and Fisher fails where it is most needed.

### Cross-dataset (Tabula Muris)

Extra-gradient does not transfer (AUC = 0.874 on TM vs. 0.482 on PBMC). Fisher fails on structured sets in both datasets (PBMC 0.814, TM 0.946). The Tabula Muris retrain model itself has AUC = 0.944 (exceeding baseline 0.891), confirming that the attacker detects biology rather than membership.

### Utility

Five methods (retain-FT, gradient ascent, SSD, SCRUB, contrastive) preserve utility identically to baseline (ELBO ~364, marker r >= 0.831). These methods barely change the model, which is why they fail on privacy. Extra-gradient and DP-SGD trade utility for privacy (ELBO ~403, marker r ~0.789). Fisher is worst (ELBO = 490, marker r = 0.628, KL = 0.007 due to posterior collapse).

## 5. Why parameter-space methods fail

### Fisher information overlap

The diagonal Fisher was computed on the forget set (30 cells) and retain set (28,094 cells) from the baseline PBMC model.

| Layer category | Parameters | Cosine similarity |
|---|---|---|
| Encoder | 2,642,816 | 0.273 |
| Bottleneck | 8,256 | 0.291 |
| Decoder hidden | 598,912 | 0.232 |
| Decoder output | 4,100,000 | 0.362 |
| **VAE global** | **7,349,984** | **0.306** |
| **Classifier** | **462** | **0.018** |

The 17x gap between VAE (0.306) and classifier (0.018) arises because the VAE's output layer is shared across all 2,000 genes, while the classifier's output weights are specific to each of the 14 classes.

### Proposition 1 (Fisher factorization)

For a linear decoder f(z) = Wz + b with squared-error loss, the diagonal Fisher factorizes as F_dh = 4 E[e_d^2] E[z_h^2], giving

cos(F^F, F^R) = cos(sigma^F, sigma^R) * cos(nu^F, nu^R)

where sigma is the residual variance profile (R^D) and nu is the latent second moment (R^H).

### Corollary 2 (Dimensional scaling)

Part (i): For a generative model with D output dimensions, of which only M differ between forget and retain sets, cos(sigma) >= (D - M) / (D - M + MV^2). With D=2000, M=100, V=3, the bound is 0.68. The data-direct cos(sigma) = 0.83 satisfies this. The Fisher-marginal cos(sigma) = 0.51 is lower because the NB likelihood departs from linear-MSE assumptions.

Part (ii): For a single-class forget set in a C-class classifier, cos(sigma) = 1/sqrt(C). With C=14, this gives 0.27. The measured classifier cosine of 0.018 is lower still.

### Empirical verification

The fc_mean Fisher matrix is approximately rank-1 (top singular value explains 94-96% of the Frobenius norm). The factorized prediction cos(sigma)*cos(nu) = 0.41 overestimates the measured 0.37 by 11%, with the softmax nonlinearity as the dominant error source.

### Controls

**Model capacity.** A deep MLP classifier (2000 -> [512, 128] -> 14, 1.09M params, 95.2% accuracy) has shared-hidden cosine = 0.262 and class-specific output cosine = 0.010. Overlap depends on shared-vs-class-specific, not model size.

**Architecture generalization.** A VAE with z=8 gives global cosine = 0.846 (higher than z=32). Smaller latent dimension concentrates overlap in the bottleneck (0.858 vs. 0.291).

**Cluster-conditional decoder.** Conditioning the output layer on a 14-dim cluster one-hot achieves near-zero overlap in the cluster-specific columns (1.2e-8) but irreducible overlap persists in the shared hidden layers (encoder 0.433, bottleneck 0.508, decoder hidden 0.346). Fisher scrubbing on the conditional VAE gives advantage = 0.72, no improvement over the standard VAE's 0.63. The shared encoder dominates because 64 of 69 MIA features come from encoder outputs.

## 6. Discussion

Three failure modes emerge: (1) methods that preserve utility perfectly but produce no privacy improvement, (2) methods that create detectable artifacts (Streisand effect), and (3) methods that trade utility for privacy but cannot reach the retrain target.

The Fisher overlap analysis identifies the structural cause. Any generative model with shared output parameters will have high Fisher overlap between forget and retain sets, making parameter-space unlearning methods fundamentally harder than in classifiers where class-specific weights create a low-overlap regime.

Full retraining remains the only reliable option for structured forget sets.

## References

- Abadi et al. (2016). Deep Learning with Differential Privacy. CCS.
- Bourtoule et al. (2021). Machine Unlearning. IEEE S&P.
- Cao & Yang (2015). Towards Making Systems Forget with Machine Unlearning. IEEE S&P.
- Carlini et al. (2022). Membership Inference Attacks From First Principles. IEEE S&P.
- Chavdarova et al. (2019). Reducing Noise in GAN Training with Variance Reduced Extragradient. NeurIPS.
- Feldman (2020). Does Learning Require Memorization? STOC.
- Foster et al. (2024). Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening. AAAI.
- Ginart et al. (2019). Making AI Forget You: Data Deletion in Machine Learning. NeurIPS.
- Golatkar et al. (2020). Eternal Sunshine of the Spotless Net. CVPR.
- Guo et al. (2020). Certified Data Removal from Machine Learning Models. ICML.
- Hayes et al. (2024). Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy. SaTML.
- Izzo et al. (2021). Approximate Data Deletion from Machine Learning Models. AISTATS.
- Kunstner et al. (2019). Limitations of the Empirical Fisher Approximation for Natural Gradient Descent. NeurIPS.
- Kurmanji et al. (2023). Towards Unbounded Machine Unlearning. NeurIPS.
- Lopez et al. (2018). Deep Generative Modeling for Single-cell Transcriptomics. Nature Methods.
- Moon et al. (2024). Feature Unlearning for Pre-trained GANs and VAEs. AAAI.
- Nasr et al. (2018). Machine Learning with Membership Privacy Using Adversarial Regularization. CCS.
- Neel et al. (2021). Descent-to-Delete: Gradient-Based Methods for Machine Unlearning. ALT.
- Sekhari et al. (2021). Remember What You Want to Forget: Algorithms for Machine Unlearning. NeurIPS.
- Shokri et al. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P.
- Tabula Muris Consortium (2018). Single-cell Transcriptomics of 20 Mouse Organs Creates a Tabula Muris. Nature.
- Thudi et al. (2022). On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning. USENIX Security.
- Wolf et al. (2018). SCANPY: Large-scale Single-cell Gene Expression Data Analysis. Genome Biology.
- Yeom et al. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. CSF.
- Basu et al. (2021). Influence Functions in Deep Learning Are Fragile. ICLR.

Code: https://github.com/db-d2/Machine_Unlearning
