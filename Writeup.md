# Fishing for Privacy: Machine Unlearning for Gene-Expression VAEs

**David Benson**
Columbia University
dmb2262@columbia.edu

## Abstract

Variational autoencoders trained on gene-expression data memorize rare biological subpopulations, and those are the samples where a membership inference attack (MIA) is most dangerous. A few patients with an unusual disease subtype, or a small cluster of rare cells, can be individually identifiable from model behavior. Can such memorization be selectively removed without retraining the model, and without destroying its utility? That requires fixing the measurement first. MIA protocols that draw matched negatives from different biological classes than the forget set can conflate subtype identity with training membership, and this confound affects general-purpose attacks (threshold, likelihood ratio, k-NN) along with the distance-based methods used in recent transcriptomics benchmarks. Within-subtype matching, a trained MLP attacker on model-internal features, and a multi-seed retrain baseline recover a smaller memorization signal that sits almost entirely in biologically coherent subpopulations. Under this corrected evaluation, eleven unlearning approaches all fail on structured forget sets across three datasets (PBMC-33k, Tabula Muris, TCGA-BRCA) and two modalities. Methods that preserve utility produce no privacy improvement. The methods that do reduce the membership signal destroy the model through posterior collapse or Streisand effects. The underlying cause is structural. Fisher information overlap in the VAE's shared decoder output layer is roughly 27x higher between forget and retain sets than in a classifier's class-specific output (0.485 vs 0.018 under a per-sample estimator), and Proposition 1 formalizes why. With D shared output dimensions the overlap grows as 1 - O(M/D), while a classifier's class-specific heads give overlap O(1/sqrt(C)). If privacy actually matters, retrain.

## 1. Introduction

Deep learning models memorize training data, and for rare subpopulations this memorization may be necessary for generalization (Feldman 2020). When those models are trained on sensitive biological information, an attacker can determine whether a specific sample was part of the training set through a membership inference attack (MIA). Gene expression profiles can reveal disease status, genetic predispositions, and other personal health information. Rare cell types and disease subtypes may be individually identifiable even within large, aggregated datasets.

Can a gene-expression VAE selectively forget a rare biological subpopulation without retraining and without destroying model utility? This question arises when all cells from a specific cluster or all patients of a molecular subtype must be removed from a trained model. Experiments here use cell-type clusters (PBMC-33k, Tabula Muris) and disease subtypes (TCGA-BRCA) as forget sets across two modalities (single-cell and bulk RNA-seq).

Before testing unlearning methods, the evaluation itself had to be fixed. MIA protocols that draw matched negatives from different biological classes than the forget set can conflate subtype identity with training membership. This affects general-purpose attacks (threshold, likelihood ratio, k-NN) as well as the distance-based methods used in recent transcriptomics benchmarks. Within-subtype matching, a trained MLP attacker on model-internal features, and a multi-seed retrain baseline together isolate the genuine memorization signal, which concentrates in biologically coherent subpopulations.

Under this corrected evaluation, no tested method succeeds. Eleven unlearning approaches all fail on structured forget sets. Methods that preserve utility produce no privacy improvement. Methods that reduce the membership signal destroy the model through posterior collapse or Streisand effects. The cause is Fisher information overlap in the VAE's shared decoder, which prevents selective parameter perturbation.

**Contributions.**

1. A corrected evaluation protocol for MIA on biological generative models. Cross-class matched-negative protocols can conflate cell-type identity with training membership. Within-subtype matching, a trained attacker on model-internal features, multi-seed retrain baseline comparison, and attack diversity analysis isolate the genuine memorization signal, which concentrates in biologically coherent subpopulations.
2. A systematic evaluation of eleven unlearning approaches (nine approximate methods plus training-time synthetic augmentation and representation alignment) across three datasets and two modalities. On structured forget sets, methods either fail to reduce the memorization signal or reduce it but destroy model utility. On scattered forget sets, memorization is negligible and unlearning is unnecessary. On TCGA-BRCA, no genuine memorization is detectable under the attack suite after within-subtype correction.
3. A structural explanation via Fisher information overlap. The VAE's shared decoder output layer creates roughly 27x higher Fisher alignment between forget and retain sets than a classifier's class-specific output (0.485 vs. 0.018 on PBMC; 0.75 on TCGA-BRCA), while the global-parameter cosine is 0.21. Proposition 1 formalizes this for linear decoders, with dimensional scaling bounds showing generative-model overlap grows as 1 - O(M/D) while classifier overlap scales as 1/sqrt(C).

## 2. Background and threat model

**Unlearning and MIA.** Machine unlearning produces parameters from a trained model, forget set, and retain set such that a post-hoc MIA cannot distinguish forget samples from never-seen samples (Cao and Yang 2015; Bourtoule et al. 2021). Theoretical guarantees exist for restricted model classes (Ginart et al. 2019; Guo et al. 2020; Sekhari et al. 2021; Neel et al. 2021; Izzo et al. 2021) but depend on convexity or smoothness that deep generative models violate. MIA determines training-set membership via shadow models (Shokri et al. 2017) or likelihood ratios (Carlini et al. 2022). Privacy leakage is reported as membership inference advantage (adapted from Yeom et al. 2018): `advantage = 2|AUC - 0.5|`, direction-agnostic because an adversary can always flip predictions. Hayes et al. (2024) showed that inexact unlearning can leave forgotten samples more detectable (Streisand effect); Thudi et al. (2022) argued that without auditable definitions, unlearning claims are unverifiable.

**Threat model.** The scope is subpopulation unlearning, the removal of entire biological groups (rare cell types, disease subtypes) rather than individual samples. Data poisoning, adversarial extraction, and reconstruction attacks are out of scope. The adversary has query access and trains a dedicated attacker on data they control, which is a stronger setting than off-the-shelf distance-based MIA such as GAN-leaks, Monte Carlo, or Mahalanobis distance. A model that faithfully captures subtype biology will by construction produce outputs close to real training members of the same subtype, yielding high MIA scores that reflect biological fidelity rather than memorization. The trained MLP attacker used here (70-dim features from the VAE latent space) learns which feature combinations are informative for membership specifically, as evidenced by its near-chance retrain advantage (0.156) compared to distance-based methods (advantage >= 0.43 on the retrain model).

**Related work.** Fisher scrubbing (Golatkar et al. 2020), SSD (Foster et al. 2024), and SCRUB (Kurmanji et al. 2023) all assume forget-set influence concentrates in identifiable parameter subsets, an assumption Section 7 shows breaks down in shared-decoder architectures. Nasr et al. (2018) and Chavdarova et al. (2019) provide the adversarial min-max and extragradient machinery used in extra-gradient co-training; Abadi et al. (2016) introduced DP-SGD, used here as a formal privacy baseline. Moon et al. (2024) proposes feature unlearning for pre-trained generators by decoder fine-tuning, evaluated here as a post-hoc method. Cheng et al. (2026) studies retain-forget entanglement in classifiers via Wasserstein-2 gradient projection; Section 7 argues this projection cannot separate forget from retain in the shared-decoder regime where the two Fisher directions are nearly aligned. On biological data, Walker et al. (2024) showed single-cell count matrices leak private information via eQTL linkage. Golob et al. (2026) introduced scMAMA-MIA for scRNA synthetic data and observed per-cell-type vulnerability but set it aside as out of scope; Ozturk et al. (2026) reported rho = 0.92 between differential expression recovery and MIA vulnerability across 11 generative models on TCGA. Both use distance-based MIA on synthetic outputs and do not stratify by subtype. Lopez et al. (2018) developed scVI, the architecture tested here.

## 3. Datasets and setup

| Property | PBMC-33k | Tabula Muris | TCGA-BRCA |
|:---|:---|:---|:---|
| Type | scRNA-seq | scRNA-seq | bulk RNA-seq |
| Samples | 33,088 cells | 41,647 cells | 1,089 patients |
| Genes | 2,000 HVGs | 2,000 HVGs | 978 (L1000) |
| Clusters | 14 Leiden | 35 Leiden | 5 PAM50 subtypes |
| Train / Unseen | 28,124 / 4,964 | 35,399 / 6,248 | ~925 / ~164 |
| Forget set | Cluster 13, 30 megakaryocytes | Cluster 33, 82 cardiac muscle | 158 Basal patients |
| Matched neg. | 194 | 137 | 121 cross / 30 within |

The PBMC-33k dataset consists of 33,088 peripheral blood mononuclear cells from 10x Genomics, preprocessed with Scanpy (Wolf et al. 2018). The Tabula Muris dataset has 41,647 cells from 12 mouse tissues. TCGA-BRCA was obtained from the ELSA Benchmarks (Ozturk et al. 2026), VST-normalized, restricted to the 978 LINCS L1000 landmark genes (Subramanian et al. 2017). Matched negatives are the unseen samples closest to the forget set in baseline latent space (k-NN with k=10).

To control for biological confounds, a within-subtype holdout variant restricts matched negatives to the same biological group as the forget set. On PBMC, this drops baseline AUC from 0.769 to 0.527 (5 unseen megakaryocytes); on TCGA-BRCA, baseline and retrain AUCs converge to 0.576 (30 unseen Basal patients).

## 4. Methods

**VAE architecture.** The VAE follows scVI design principles (Lopez et al. 2018). Encoder: input genes through [1024, 512, 128] to latent mean and log-variance (z = 32). Decoder reverses this through [128, 512, 1024] to a softmax output across genes. Layer normalization and dropout (0.1) follow each hidden layer. Because the input data are log-normalized, the reconstruction term is mean squared error (Gaussian observation model on the transformed counts); the training objective is the negative ELBO.

**Nine post-hoc unlearning methods plus DP-SGD baseline.** **Retain-only fine-tuning** continues VAE training on the retain set. **Gradient ascent** maximizes forget-set loss then fine-tunes on the retain set. **Frozen critics** freeze pre-trained MIA attackers and update the VAE to minimize critic success. **Fisher information scrubbing** (Golatkar et al. 2020) updates parameters inversely proportional to the diagonal Fisher. **SSD** (Foster et al. 2024) dampens parameters multiplicatively in proportion to forget-set Fisher importance. **SCRUB** (Kurmanji et al. 2023) uses teacher-student distillation that matches on retain data and diverges on forget data. **Moon feature unlearning** (Moon et al. 2024) builds a forget-versus-retain latent direction and fine-tunes the decoder so forget-set latents reconstruct with that direction removed while retain reconstructions are preserved, leaving the encoder fixed. **Contrastive latent unlearning** pushes forget-set posteriors toward the prior while preserving retain-set representations. **Extra-gradient co-training** frames unlearning as a min-max game and uses the two-step extragradient update of Chavdarova et al. (2019) with TTUR, three co-trained critics, and 50 epochs. **DP-SGD** (Abadi et al. 2016) trains from scratch on only the retain set, included as a formal privacy baseline.

**Two constructive approaches.** **Training-time synthetic augmentation** trains VAEs on retain + bootstrap-resampled synthetic forget-class samples generated from k unseen seed cells with Gaussian noise. **Representation alignment unlearning (RAU)** fine-tunes the baseline so forget-sample posteriors match the retrain model's.

**Attack suite.** Canonical attacker: trained MLP on 70-dim features (69 from VAE latent space + k-NN distance to retain; spectral normalization, two 256-unit hidden layers, dropout 0.3). Additional families: threshold attacks on reconstruction / KL / ELBO (Yeom et al. 2018); likelihood ratio (Carlini et al. 2022); k-NN latent (Shokri et al. 2017).

**Multi-seed retrain reference.** Five retrain seeds (42-46) trained with the canonical protocol yield fresh-attacker advantage 0.156 (AUC = 0.578) with nested 95% CI [0.082, 0.258]. Scoring is CPU-deterministic with a single attacker applied to every model, since MPS is not bit-reproducible run to run. The nested CI combines cross-seed variance (outer) with within-seed bootstrap uncertainty on the matched-negative pool (inner). Each method is compared to retrain by a nested bootstrap of the advantage difference.

## 5. The biology confound

On TCGA-BRCA (1,089 breast cancer patients, bulk RNA-seq), the structured forget set is 158 Basal-subtype patients (17% of training). With standard cross-subtype matching, baseline AUC = 0.821 and retrain AUC = 0.860. The retrain model, which never saw the Basal patients, is *better* at detecting them than the baseline. All six tested unlearning methods converge to the retrain floor (AUC ~ 0.862).

Within-subtype matching (restricting matched negatives to 30 unseen Basal patients) resolves the cross-subtype confound. Baseline and retrain advantages both reach ~0.15 and become indistinguishable from each other. The entire cross-subtype gap was driven by Basal identity. The residual within-subtype advantage of 0.15 is not zero, so the attacker still detects something, but whatever it detects is equally present in the retrain model that never saw the Basal patients, and is therefore not a membership signal. Patient-level experiments at n_f in {5, 10, 20} show the same pattern: baseline and retrain advantages overlap within their CIs at every size.

The confound replicates on PBMC. Within-cluster matching (5 unseen megakaryocytes) drops baseline AUC from 0.769 to 0.527 while retrain rises from 0.495 to 0.609, indicating that most of the cross-cluster above-chance signal is attributable to subtype identity rather than memorization. Replication on PBMC cluster 7 (CD14+ monocytes, 80 unseen cells) shows a similar pattern (cross-cluster AUC = 0.665, within-cluster AUC = 0.527). A second rare cluster tells the same story from the retrain side: PBMC cluster 12 (49 cells) has baseline AUC 0.87 (advantage 0.73), yet a model retrained without those cells still reaches AUC 0.82 (advantage 0.64), leaving a genuine membership gap of only ~0.09 versus cluster 13's 0.21. Against this second forget set no post-hoc method matches retrain while preserving utility; retain-only fine-tuning stays at baseline (0.72), Fisher again collapses the posterior (marker r 0.34), and the extra-gradient lambda=10 that reaches the floor on cluster 13 over-unlearns here (advantage 0.00, marker r 0.76), so the one competitive method does not transfer across forget sets. Tabula Muris cluster 28 (67 unseen) shows no detectable per-sample memorization under either matching strategy. High-assumption attacks (k-NN, likelihood ratio) achieve advantage >= 0.73 on the retrain model because they detect biological structure regardless of training history.

The confound is consistent with two recent benchmarks that report high MIA scores on scRNA and bulk RNA using distance-based MIA on synthetic outputs. Ozturk et al. (2026) report rho = 0.92 between differential expression recovery and MIA vulnerability across 11 generative models (TCGA-COMBINED, n = 11 so the correlation is suggestive), and Golob et al. (2026) observe cell-type-dependent vulnerability in scMAMA-MIA but set the dependence aside as out of scope. Distance-based methods on synthetic outputs cannot distinguish biological signal preservation from individual-level memorization; a trained MLP attacker on model-internal features can, as its lower retrain advantage (0.156) compared to baseline (0.582) on PBMC shows.

## 6. Unlearning results

### 6.1 Main results (PBMC-33k structured forget set)

| Method | Seeds | AUC | Advantage [95% CI] | Marker r | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| Baseline (no unlearning) | 1 | 0.791 | 0.582 | 0.831 | anchor |
| Retain-only fine-tune | 5 | 0.666 | 0.333 [0.23, 0.43] | 0.832 | FAIL |
| Gradient ascent | 5 | 0.698 | 0.396 [0.30, 0.49] | 0.832 | FAIL |
| SSD (alpha=1.0) | 3 | 0.718 | 0.435 [0.31, 0.56] | 0.831 | FAIL |
| SCRUB (alpha_f=1.0) | 3 | 0.706 | 0.411 [0.28, 0.54] | 0.832 | FAIL |
| Moon feature-unlearn | 3 | 0.740 | 0.480 [0.36, 0.60] | 0.831 | FAIL |
| Contrastive (gamma=1.0) | 3 | 0.164 | 0.673 [0.57, 0.75] | 0.832 | FAIL (Streisand) |
| Fisher scrubbing | 1 | 0.808 | 0.615 [0.45, 0.77] | - | FAIL (worse) |
| Extra-gradient (lambda=10) | 10 | 0.433 | 0.281 [0.20, 0.37] | 0.789 | FAIL (marginal) |
| DP-SGD (epsilon=10) | 3 | 0.478 | 0.045 [0.03, 0.18] | 0.787 | ~ retrain\* |
| **Full retrain** | **5** | **0.578** | **0.156 [0.08, 0.26]** | **0.829** | **TARGET** |

The multi-seed retrain reference has advantage 0.156 with nested 95% CI [0.082, 0.258]. Every post-hoc method's point advantage exceeds this bound. All except extra-gradient also have an advantage-difference CI that excludes zero; extra-gradient's point advantage (0.281) is above the bound while its difference from retrain is not statistically resolved.

\*DP-SGD's advantage 0.045 is indistinguishable from retrain (its difference CI includes zero). DP-SGD trains from scratch with formal differential privacy; the forget set was never in training. Privacy by exclusion, not unlearning.

**Failure modes.** Five methods (retain-FT, gradient ascent, SSD, SCRUB, Moon) preserve utility (marker r >= 0.831, matching baseline) but produce no useful privacy reduction; thirty forget-set cells leave too small a gradient signal against 28,094 retain cells. Moon is the sharpest case, fine-tuning only the decoder so the encoder-borne signal that supplies most of the attacker's 70 features survives untouched (advantage 0.480 at baseline utility). Three methods (contrastive, Fisher, frozen critics) make the model worse rather than better. Contrastive drops AUC to 0.164 (advantage 0.673), a Streisand effect (Hayes et al. 2024). Fisher scrubbing collapses the posterior (KL falls from 10.55 to 0.007) and advantage rises to 0.615. Extra-gradient has high per-seed variance (sigma_AUC = 0.142) and is the one method whose separation from retrain is not statistically resolved (its advantage-difference CI includes zero), though its point advantage 0.281 still exceeds the retrain bound.

### 6.2 Attack diversity

Reconstruction-loss thresholding gives identical advantage (0.438) across all models because reconstruction quality is preserved, so the membership signal lives in the KL and latent geometry. ELBO and KL thresholds drop under extra-gradient (from 0.838 to 0.023 for ELBO), but Fisher's apparent KL drop (0.841 -> 0.272) is posterior collapse, not unlearning. High-assumption attacks (k-NN latent, likelihood ratio) keep advantage >= 0.73 even on retrain, so they detect cell-type identity, not membership. Only the trained MLP on internal features is near-chance on retrain.

### 6.3 Cross-dataset validation (Tabula Muris)

The Tabula Muris retrain model has AUC = 0.944, exceeding the baseline (0.891). Since retrain never saw cluster 33, the attacker is detecting cardiac muscle biology rather than membership; TM does not measure membership leakage at the per-cell level. Extra-gradient (0.874) and Fisher (0.946) therefore confirm rather than refute the biology-confound thesis; no method moves the AUC below the retrain ceiling.

### 6.4 Constructive approaches

**Synthetic augmentation at training time.** VAEs trained on retain + bootstrap-resampled synthetic megakaryocytes (from 5 unseen seed cells with Gaussian noise). On PBMC with within-cluster matching, the augmented model achieves AUC = 0.82-1.00, *worse* than baseline (0.63); the model's representation of the megakaryocyte region is now shaped by the 5 seed cells, and the MIA detects the shift. Augmentation replaces one memorization pattern with another. On TCGA-BRCA the augmented model (0.581) matches baseline (0.578) and retrain (0.573), consistent with no patient-level memorization.

**Representation alignment unlearning (RAU).** Fine-tunes the baseline so forget-sample posteriors match the retrain model's. A sweep over lambda in {0.1, 1, 10, 100} aligns posteriors (KL to retrain drops from 139 to 0.02) but pushes cross-cluster AUC to 0.10-0.13, a Streisand effect. The structural limit extends to representation space; any modification to forget-sample treatment, in parameter space (Fisher, SSD) or representation space (RAU, contrastive), creates a detectable trace.

### 6.5 Utility evaluation

Utility on the held-out set (4,964 cells). Six methods (retain-FT, gradient ascent, SSD, SCRUB, Moon, contrastive) preserve utility (ELBO ~ 364, marker r >= 0.831) but barely change the model, which is why they fail on privacy. Extra-gradient and DP-SGD degrade ELBO by ~40 and marker r by ~5%, the cost of any actual signal reduction. Fisher collapses the posterior entirely (ELBO = 490, marker r = 0.628, KL = 0.007). ARI is preserved throughout, so global cluster structure survives even when gene-level reconstruction degrades.

## 7. Why parameter-space methods fail (Fisher information analysis)

### 7.1 Fisher overlap across architectures

Parameter-space unlearning (Fisher scrubbing, SSD, SCRUB) modifies parameters identified as important for the forget set via the diagonal Fisher. For this to work, forget-set Fisher and retain-set Fisher must be separable; cosine similarity measures that separability.

| Layer category | Parameters | PBMC cosine |
|---|---|---|
| Encoder | 2,642,816 | 0.273 |
| Bottleneck | 8,256 | 0.291 |
| Decoder hidden | 598,912 | 0.232 |
| Decoder output | 4,100,000 | 0.493 |
| **VAE global (PBMC)** | **7,349,984** | **0.209** |
| **VAE global (TCGA-BRCA)** | **~4.2M** | **0.753** |
| **Linear classifier** | **462** | **0.018** |
| **Deep MLP classifier** (shared hidden / class-specific output) | **1.09M** | **0.53 / 0.006** |

The decoder's mean output layer, whose 1,024-dimensional hidden representation is shared across all 2,000 output genes, has forget-retain Fisher cosine 0.485 versus 0.018 for the classifier's class-specific output, a 27x gap; perturbing any column affects reconstruction of every cell type. The cosine across all parameters is lower (0.209) because forget and retain influence concentrate in different high-magnitude layers. At the gene level, 73% of genes (1,453/2,000) have Fisher cosine > 0.5 with median 0.857. The classifier's low overlap (0.018) arises because forget-set Fisher concentrates on the 32 weights feeding the cluster-13 logit while retain-set Fisher spreads across 14 class logits.

### 7.2 Proposition 1 (Fisher factorization for linear decoders)

Let f(z) = Wz + b with W in R^{D x H} and squared-error loss. If residuals e_d = x_d - f(z)_d are independent of latent activations z_h, the diagonal Fisher factorizes as F_{dh} = 4 E[e_d^2] E[z_h^2], and

cos(F^F, F^R) = cos(sigma^F, sigma^R) * cos(nu^F, nu^R)

where sigma^S in R^D has entries sigma^S_d = E_{x ~ S}[e_d^2] (residual variance per output dimension) and nu^S in R^H has entries nu^S_h = E_{x ~ S}[z_h^2] (latent second moment).

*Proof sketch.* d L / d W_{dh} = -2 e_d z_h, so F_{dh} = 4 E[e_d^2 z_h^2] = 4 sigma_d nu_h under independence. Then the inner product factors accordingly and dividing by norms gives the factorization.

*Remark.* A perturbation weighted by forget-set Fisher importance disturbs retain-set curvature by cos(F^F, F^R) * ||F^R|| / ||F^F||. Near-zero cosine (classifier) admits perturbation directions that primarily affect the forget set; cosine = 0.485 in the shared decoder output (VAE) does not.

### 7.3 Corollary 2 (Dimensional scaling)

Part (i). Suppose only M of the D output dimensions differ between forget and retain, with bounded relative residual variance V. Then cos(sigma^F, sigma^R) >= (D - M) / (D - M + M V^2).

Part (ii). For a single-class forget set where sigma^F is supported on a single coordinate, cos(sigma^F, sigma^R) = sigma^R_k / ||sigma^R||, which equals 1/sqrt(C) for balanced C-class residuals.

For PBMC (D = 2000, M = 100, V = 3) the bound is 0.68, which the data-direct cos(sigma) = 0.83 satisfies. For a 14-class classifier, part (ii) predicts 1/sqrt(14) ~ 0.27 as an upper bound; the measured 0.018 is lower still because the forget-class gradient is more concentrated than a point mass at 95% accuracy.

### 7.4 Empirical verification

The fc_mean Fisher matrix is approximately rank-1 (leading singular value explains 94-96% of Frobenius norm), consistent with F_{dh} ~ 4 sigma_d nu_h. The factorized prediction cos(sigma) * cos(nu) = 0.41 overshoots the measured 0.37 by 11%, with the softmax output Jacobian as the dominant source of error.

### 7.5 Controls

**Model capacity.** A deep MLP classifier (2000 -> [512, 128] -> 14, 1.09M parameters, 95.2% accuracy) has shared-hidden cosine = 0.53 (matching VAE encoder 0.35) and class-specific output cosine = 0.006 (matching the linear probe 0.018), so overlap depends on parameter sharing rather than capacity.

**Architecture generalization.** Reducing VAE latent dimension to z = 8 raises global cosine to 0.35 from z=32's 0.21; smaller latent spaces still concentrate the overlap.

**Cluster-conditional decoder.** A conditional VAE with a 14-dim cluster one-hot concatenated to the fc_mean input gives near-orthogonal cluster-specific output columns (~1e-8) but still fails to unlearn, because encoder (0.433) and bottleneck (0.508) overlaps dominate the MIA signal. Fisher scrubbing on the conditional VAE gives advantage = 0.72, no improvement over the standard VAE's 0.63.

### 7.6 Implication

With the shared decoder output cosine 0.49 (global 0.21), Fisher-weighted perturbations either damage retain performance or leave forget-set influence intact. Classifier unlearning works because class-specific output weights create a low-overlap regime the VAE does not. The same problem applies to any generative model with shared output parameters. The analysis uses the diagonal Fisher, which Kunstner et al. (2019) showed can misrepresent the true Fisher in deep networks; the concern is limited because Fisher scrubbing, SSD, and SCRUB themselves use the diagonal, and the log-Fisher correlation (r = 0.71) and per-gene pattern (73% > 0.5) corroborate at different granularities. Two recent methods target this entanglement from opposite ends and fail for the same reason. Cheng et al. (2026) project the gradient off retain-degrading directions, but in the shared decoder the forget and retain Fisher directions are nearly aligned (cosine 0.485 versus a classifier's 0.018), so any projection that spares retain also spares forget. Moon feature unlearning edits the decoder along a forget-versus-retain latent direction yet leaves advantage at 0.480 at baseline utility, because the attacker reads the untouched encoder. Gradient projection in parameter space and feature editing in decoder space both assume a forget-specific direction the shared-decoder VAE does not provide.

## 8. Discussion

**Unlearning as a detectable operation.** Contrastive, Fisher scrubbing, frozen critics, and RAU all produce behavioral changes the attacker picks up. Contrastive creates an unnatural latent location, Fisher collapses the posterior, frozen critics drive the VAE toward critic-specific blind spots, and RAU leaves a representational gap between student and retrain model. Synthetic augmentation substitutes one detectable signature for another. This is the Streisand effect (Golatkar et al. 2020; Hayes et al. 2024). Verification is a related problem. The only ground truth is the retrain model, which the data controller is trying to avoid running; DP-SGD offers a formal guarantee but requires training from scratch.

**Limitations.** Privacy guarantees are empirical (except DP-SGD). Tabula Muris is confounded by tissue-of-origin signals; within-subtype PBMC uses only 5 unseen megakaryocytes; TCGA-BRCA shows no within-subtype gap, so it cannot test methods that need one. Only VAE architectures were tested. The size ablation conflates size with heterogeneity.

## 9. Conclusion

Structured subpopulations are memorized. Scattered samples are not. All eleven approaches fail on the structured case across three datasets and two modalities. Fisher overlap in the shared decoder output is 27x the classifier's, a structural limit of the architecture; the corrected evaluation protocol is what makes the limit visible. Retraining is the only dependable option.

## References

- Abadi et al. (2016). Deep Learning with Differential Privacy. CCS.
- Bourtoule et al. (2021). Machine Unlearning. IEEE S&P.
- Cao and Yang (2015). Towards Making Systems Forget with Machine Unlearning. IEEE S&P.
- Carlini et al. (2022). Membership Inference Attacks From First Principles. IEEE S&P.
- Chavdarova et al. (2019). Reducing Noise in GAN Training with Variance Reduced Extragradient. NeurIPS.
- Cheng et al. (2026). Retain-Forget Entanglement in Machine Unlearning via Wasserstein Regularization. ICLR.
- Feldman (2020). Does Learning Require Memorization? STOC.
- Foster et al. (2024). Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening. AAAI.
- Ginart et al. (2019). Making AI Forget You: Data Deletion in Machine Learning. NeurIPS.
- Golatkar et al. (2020). Eternal Sunshine of the Spotless Net. CVPR.
- Golob et al. (2026). scMAMA-MIA: A Membership Inference Benchmark for Synthetic Single-Cell RNA-seq Data.
- Guo et al. (2020). Certified Data Removal from Machine Learning Models. ICML.
- Hayes et al. (2024). Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy. SaTML.
- Izzo et al. (2021). Approximate Data Deletion from Machine Learning Models. AISTATS.
- Kunstner et al. (2019). Limitations of the Empirical Fisher Approximation for Natural Gradient Descent. NeurIPS.
- Kurmanji et al. (2023). Towards Unbounded Machine Unlearning. NeurIPS.
- Lopez et al. (2018). Deep Generative Modeling for Single-cell Transcriptomics. Nature Methods.
- Moon et al. (2024). Feature Unlearning for Pre-trained GANs and VAEs. AAAI.
- Nasr et al. (2018). Machine Learning with Membership Privacy Using Adversarial Regularization. CCS.
- Neel et al. (2021). Descent-to-Delete: Gradient-Based Methods for Machine Unlearning. ALT.
- Ozturk et al. (2026). ELSA: A Benchmark for Privacy Evaluation in Generative Models for Bulk RNA-seq.
- Sekhari et al. (2021). Remember What You Want to Forget: Algorithms for Machine Unlearning. NeurIPS.
- Shokri et al. (2017). Membership Inference Attacks Against Machine Learning Models. IEEE S&P.
- Subramanian et al. (2017). A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. Cell.
- Tabula Muris Consortium (2018). Single-cell Transcriptomics of 20 Mouse Organs Creates a Tabula Muris. Nature.
- Thudi et al. (2022). On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning. USENIX Security.
- Walker et al. (2024). Privacy Risks in Single-Cell Data: Re-identification via eQTL-Based Linking. Cell.
- Wolf et al. (2018). SCANPY: Large-scale Single-cell Gene Expression Data Analysis. Genome Biology.
- Yeom et al. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. CSF.

Code: <https://github.com/db-d2/Machine_Unlearning>
