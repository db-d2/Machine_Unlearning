# Narrative Summary

This paper studies whether training samples can be selectively removed ("unlearned") from variational autoencoders trained on gene-expression data. Across three datasets (PBMC-33k single-cell, Tabula Muris single-cell, TCGA-BRCA bulk RNA-seq) and eight unlearning methods, I found that standard membership inference attacks overestimate memorization by 30-90% because they conflate biological subtype identity with training membership. After controlling for this confound via within-subtype matching, genuine memorization is detectable only on PBMC (AUC gap of 0.260 between baseline and retrain models) and concentrates in rare cell types rather than random samples. All eight approximate unlearning methods fail to close even this reduced gap. A Fisher information overlap analysis explains why: in VAE decoders, forget-set and retain-set parameter importance vectors are 17x more aligned than in classifiers (cosine 0.306 vs 0.018), making selective parameter perturbation structurally impossible.

---

## The Full Narrative

### Setting

Single-cell and bulk RNA-seq data are increasingly analyzed with deep generative models, particularly variational autoencoders (scVI and similar). These models learn latent representations that capture biological variation across cell types, disease states, and individual patients. Because they are trained on human-derived data, there is practical interest in removing specific training samples from a trained model without retraining from scratch. This is the machine unlearning problem, motivated by privacy regulations (GDPR "right to erasure") and the cost of full retraining.

Eight approximate unlearning methods exist in the ML literature, from simple fine-tuning to adversarial co-training and Fisher information scrubbing. All were designed and evaluated on image classifiers. I haven't found any that tested them on generative models for high-dimensional biological data.

### Memorization is real but overestimated

Using membership inference attacks (MIAs), I found that the VAE memorizes structured biological subpopulations. On PBMC-33k, an MLP attacker trained on latent-space features achieves AUC 0.783 on 30 rare megakaryocytes (cluster 13) versus biologically matched negatives, compared to a retrain floor of 0.523. The gap of 0.260 represents genuine memorization. In contrast, 30 randomly selected cells ("scattered" forget set) show AUC 0.525 (indistinguishable from chance). Memorization concentrates where the training data has distinctive, low-frequency structure, consistent with Feldman's (2020) long-tail memorization theory.

The standard MIA evaluation inflates this signal by 30-90% depending on the matching strategy. The matched negatives (k-NN from the unseen set) are mostly from different cell types or subtypes than the forget set. An attacker can exploit this distributional difference to achieve high AUC even on a retrain model that never saw the forget set.

I tried to quantify this confound with two experiments:

1. *Within-cluster holdout on PBMC.* Restricting matched negatives to the 5 unseen megakaryocytes drops baseline AUC from 0.769 to 0.527 and retrain AUC rises from 0.495 to 0.609. Approximately 90% of the MIA signal with standard matching is attributable to cell-type identity, not memorization.

2. *Within-subtype matching on TCGA-BRCA.* On 1,089 breast cancer patients (bulk RNA-seq, Ozturk et al. 2026 benchmark), restricting matched negatives to 30 unseen Basal patients drops baseline AUC from 0.821 to 0.576. The retrain model gives an identical 0.576. There is zero detectable patient-level memorization on this dataset.

This confound is not unique to our evaluation. Ozturk et al. (2026) report a Spearman correlation of 0.92 between differential expression recovery and MIA vulnerability across 11 generative models on TCGA-COMBINED, interpreting it as a privacy-utility tradeoff. I propose an alternative explanation: models that better capture biological structure appear to "memorize" more because the MIA detects biological signal preservation, not individual training membership. Golob et al. (2026) observe cell-type-dependent MIA vulnerability in their scMAMA-MIA evaluation of synthetic single-cell data (Figure 5) but explicitly set it aside as "not relevant." Both papers call for subtype-stratified evaluation as future work. I believe this provides it.

Five attack types tested on PBMC show different behavior on the retrain model (which never saw the forget set):

| Attack | Retrain AUC | Retrain Advantage | What it detects |
|--------|------------|-------------------|----------------|
| Reconstruction threshold | 0.719 | 0.438 | Gene expression magnitude (biology) |
| KL divergence threshold | 0.134 | 0.733 | Latent distribution shift (biology) |
| ELBO threshold | 0.137 | 0.725 | Combined (biology) |
| Likelihood ratio | 0.920 | 0.839 | Relative model fit (biology) |
| k-NN latent | 0.014 | 0.971 | Latent proximity (biology) |
| **MLP attacker** | **0.523** | **0.046** | **Membership (memorization)** |

Only the trained MLP attacker with matched negatives achieves retrain advantage near zero, confirming it isolates memorization. All other attacks detect cell-type identity regardless of training history. The high-assumption attacks (k-NN, likelihood ratio) have advantage exceeding 0.83 on the retrain model because megakaryocytes occupy a distinct latent region as a biological fact, not as a training artifact.

MIA evaluation for unlearning in biological generative models should use (a) a trained attacker rather than fixed-threshold statistics, (b) biologically matched negatives or within-subtype holdout controls, and (c) comparison against a retrain baseline to separate memorization from biology. Without these controls, MIA results are uninterpretable for structured forget sets.

### Threat model and attacker design

The choice of MIA methodology is a choice of threat model. In security evaluation, Kerckhoffs's principle holds that a system should be secure even if everything about it, except the key, is public knowledge. Applied to privacy evaluation, this means: evaluate against the strongest realistic adversary, not a convenient one. A weak attack that fails to detect membership provides a false sense of privacy.

Two threat models are relevant for biological generative models:

**Synthetic data release (Ozturk, Golob).** The model owner generates synthetic data and releases only that. The attacker sees the synthetic dataset and their own real samples but cannot query the model. Distance-based methods (GAN-leaks, Monte Carlo) and distributional attacks (scMAMA-MIA) are appropriate here. However, these methods have no mechanism to distinguish "the model learned this subtype's biology" from "the model memorized this specific patient." A model that faithfully generates Basal-like synthetic expression profiles will produce samples close to real Basal training samples regardless of individual-level memorization. Under this threat model, the biology confound is invisible and MIA risk is systematically overestimated for well-performing models — which is exactly what Ozturk et al.'s rho=0.92 correlation between utility and MIA risk reflects.

**Model deployment (my scenario).** The trained model is deployed via an API, shared as weights, or used in a federated setting. The attacker can query the model with any input and observe latent codes, reconstruction quality, and other internal statistics. This applies when a someone deploys a VAE for single-cell analysis or a consortium shares a trained model across institutions. The strongest attack under this model trains a custom classifier on features extracted from the model's internals. This paper's MLP attacker, trained on 69-dimensional feature vectors (latent mean, log-variance, reconstruction loss, KL divergence, ELBO, L2 norms) with member/non-member labels, is this worst-case adversary.

The advantage of the trained attacker is that it can learn which feature *combinations* are informative for membership specifically, rather than for subtype identity. The evidence: on the retrain model (which never saw the forget set), the MLP attacker achieves AUC 0.523 (advantage 0.046, near chance). It learned that without memorization, the features are not distinguishable. The off-the-shelf distance methods cannot make this distinction — they achieve advantage exceeding 0.73 on the retrain model because they detect biology, not membership.

The data poisoning scenario further motivates aggressive attacker design. If an adversary can inject samples into a training pipeline (e.g., a multi-site single-cell atlas), they need to verify inclusion to exploit the poisoned model. An MIA that merely measures distance to synthetic outputs cannot serve this purpose; the adversary needs model-level access and a trained attacker to reliably detect their injected samples among thousands of biologically similar cells. Pawelczyk et al. (2025) showed that unlearning methods fail to remove data poisoning effects in classifiers. Whether this extends to biological generative models is an open question that requires the strong-attacker evaluation framework this model provides.

### All unlearning methods fail?

Eight methods tested on the genuine memorization signal (PBMC MLP attacker with matched negatives):

| Method | Seeds | AUC (mean +/- SD) | Advantage | Status |
|--------|-------|-------------------|-----------|--------|
| Baseline | — | 0.783 | 0.565 | — |
| Retrain | — | 0.523 | 0.046 | Target |
| Extra-gradient (lambda=10) | 10 | 0.429 +/- 0.142 | 0.300 | FAIL |
| Retain fine-tune | 5 | 0.665 +/- 0.007 | 0.331 | FAIL |
| Gradient ascent | 5 | 0.702 +/- 0.004 | 0.404 | FAIL |
| Fisher scrubbing | 3 | 0.814 +/- 0.003 | 0.628 | FAIL (worse) |
| SSD | 3 | 0.725 +/- 0.001 | 0.450 | FAIL |
| SCRUB | 3 | 0.737 +/- 0.002 | 0.474 | FAIL |
| Contrastive latent | 3 | 0.153 +/- 0.032 | 0.695 | FAIL (Streisand) |
| DP-SGD (epsilon=10) | 3 | 0.464 +/- 0.024 | 0.072 | Near target |

No approximate method achieves mean advantage within the retrain 95% CI upper bound (0.266). The failure modes are diverse: Fisher scrubbing causes posterior collapse (AUC increases to 0.814), contrastive unlearning creates a Streisand effect (AUC = 0.153, more distinguishable than baseline), extra-gradient shows high variance (3 of 10 seeds below retrain), and retain fine-tuning barely changes the model. Only DP-SGD (which trains from scratch with formal privacy guarantees) approaches the target, at the cost of utility degradation.

Cross-dataset evidence: on Tabula Muris, extra-gradient fails entirely (AUC 0.874, all lambda values). On TCGA-BRCA, all six tested methods converge to the retrain floor (~0.576 with within-subtype matching), but this is because there is nothing to unlearn. The methods "succeed" trivially when there is no memorization to remove.

### Fisher overlap explains why

The failure of parameter-space methods (Fisher scrubbing, SSD, gradient ascent) has a structural explanation. In a VAE with a shared decoder, the diagonal Fisher information for the forget set and retain set are highly aligned because both sets contribute gradients through the same output weights.

The cosine similarity between forget-set and retain-set Fisher diagonals is 0.306 for the PBMC VAE (z=32, 7.35M parameters) and 0.018 for a linear classifier on the same data. The 17x ratio means that for every unit of forget-set-relevant curvature the VAE must modify, it disturbs 17x more retain-set curvature than a classifier would. The overlap concentrates in the decoder output layers (cosine 0.362), where 56% of parameters reside. At the gene level, 73% of output dimensions have cosine exceeding 0.5.

On TCGA-BRCA (978 genes, 4.2M parameters, 17% forget fraction), the global cosine rises to 0.905, with 100% of genes above 0.5. The overlap is worse with a larger forget fraction, consistent with the dimensional scaling prediction.

A deep MLP classifier with comparable capacity (1.09M parameters, shared hidden layers) shows the same pattern in its shared layers (cosine 0.262) but near-zero overlap in its class-specific output layer (cosine 0.010). This confirms that overlap depends on whether output parameters are shared across classes (generative model) or class-specific (classifier), not on model capacity.

**Proposition 1** For a linear decoder with squared-error loss, the diagonal Fisher factorizes as the product of a residual-variance cosine (across output dimensions) and a latent-moment cosine (across hidden dimensions). This factorization predicts the measured Fisher cosine within 11% (0.41 predicted vs 0.37 measured), with the residual error attributable to the softmax nonlinearity in the actual decoder. Corollary 2 provides dimensional scaling bounds showing that generative model overlap grows as 1 - O(M/D) while classifier overlap scales as 1/sqrt(C), formalizing the architectural gap.

Parameter-space unlearning methods designed for classifiers (where forget-class influence concentrates in class-specific output weights) cannot transfer to generative models (where influence is diffused across shared decoder weights). This is not a tuning problem but a structural limit.

### Cross-Dataset Summary

| | PBMC-33k | Tabula Muris | TCGA-BRCA |
|---|---|---|---|
| Type | scRNA-seq | scRNA-seq | Bulk RNA-seq |
| Training samples | 28,124 | 35,399 | 925 |
| Genes | 2,000 | 2,000 | 978 |
| Forget set | 30 cells (0.1%) | 82 cells (0.2%) | 158 patients (17%) |
| Forget type | Rare cell type | Rare cell type | Disease subtype |
| Baseline AUC (cross-sub) | 0.783 | 0.891 | 0.821 |
| Retrain AUC (cross-sub) | 0.523 | 0.944 | 0.860 |
| Baseline AUC (within-sub) | 0.527 | N/A | 0.576 |
| Retrain AUC (within-sub) | 0.609 | N/A | 0.576 |
| Biology confound | ~90% | Severe (retrain > baseline) | ~76% |
| Genuine memorization | Yes (gap 0.260) | Unclear (confounded) | None detected |
| Fisher global cosine | 0.306 | N/A | 0.905 |
| Methods fail? | Yes (8/8) | Yes (2/2 tested) | Moot (nothing to unlearn) |

### In Practice

1. **For privacy evaluation.** Standard MIA evaluation is unreliable for structured biological forget sets. Within-subtype matching or calibrated evaluation (using the retrain model as reference) is necessary. The five-attack-type diversity analysis should be standard practice: if high-assumption attacks show high advantage on the retrain model, the evaluation is confounded.

2. **For machine unlearning.** Approximate unlearning methods designed for classifiers do not transfer to VAEs. The shared-decoder architecture creates a fundamental obstacle. Retraining from scratch remains the only reliable approach. DP-SGD provides formal guarantees but is not an unlearning method (it requires training from scratch).

3. **For biological generative models.** VAEs memorize rare, biologically coherent subpopulations, not random samples. This has implications for data release: releasing a trained VAE model may leak information about rare cell types or disease subtypes even if no individual is identifiable. The magnitude of this leakage is smaller than standard MIA evaluation suggests (genuine memorization gap ~0.260 on PBMC, vs zero on TCGA-BRCA).

---

## Positioning vs Concurrent Work

| Paper | What they do | Differences |
|-------|-------------|--------------------------------|
| Ozturk et al. (2026) | Benchmark 11 generative models on TCGA; observe rho=0.92 between DE recovery and MIA risk (Figure 8, TCGA-COMBINED) | Interpret rho=0.92 as privacy-utility tradeoff, not as evidence that MIA detects biology. No within-subtype MIA stratification. Call for "sensitive attribute stratification (e.g. race, gender)" as future work (their quote). Use distance-based MIA (GAN-leaks, Monte Carlo, DOMIAS-KDE) on synthetic data outputs, not model internals. |
| Golob et al. (2026) | scMAMA-MIA on synthetic scRNA-seq; compute per-cell-type Mahalanobis distances to scDesign2 copula parameters | Show per-cell-type MIA signal variation (Figure 5) but state that "verifying why distances to some cell types are much greater than others will require further investigation, and is not relevant to this work." No within-cell-type matching for member/non-member sets (donor-level random split). Attack exploits cell-type copula structure without controlling for it. |
| Hayes et al. (2024) | Critique MIA for unlearning evaluation; per-example vs population attacks; tested on ResNet-18 (CIFAR-10) and PaLM 2 | Classifier and LLM focused. No biological or domain-specific confounds discussed. Their methodological concerns (per-example vulnerability variance, retain-set privacy) are complementary to ours. |
| Cheng et al. (ICLR 2026) | "Retain-forget entanglement" via Wasserstein-2 regularization and augmented Lagrangian | Classifier only (ResNet-18, RoBERTa, ViT). Entanglement defined as semantic proximity (same superclass), not statistical overlap. No Fisher analysis, no generative models. |
| Moon et al. (2024) | Latent-space feature removal for pre-trained GANs and VAEs via vector arithmetic | Does not use Fisher information. Operates in latent space, not parameter space. Goal is preventing generation of a class, not preventing detection of training membership. No forget/retain overlap analysis. |

---

## Note on MIA Methodology Differences

An important distinction between my approach and concurrent work:

**Ozturk et al. use off-the-shelf, distance-based MIA methods** (GAN-leaks, Monte Carlo, confidence-based RF). These operate on the released synthetic data, not on model internals. GAN-leaks scores each real sample by its minimum distance to the nearest synthetic sample. No attacker is trained on member/non-member labels. These methods are inherently more susceptible to the biology confound because a model that generates biologically faithful synthetic data will by construction produce samples close to real training members of the same subtype. There is no mechanism to learn which features distinguish membership from biology.

**Golob et al. use Mahalanobis distance** to scDesign2's per-cell-type copula parameters. This explicitly uses cell-type structure as the attack signal. Cells from training donors are expected to be closer to the fitted copula than non-member cells, but this could equally reflect that the copula captured the cell-type distribution well, not that it memorized specific donors.

**My approach trains an MLP attacker** on 69-dimensional feature vectors from the VAE's latent space (mu, logvar, reconstruction loss, KL divergence, ELBO, L2 norms). The attacker is trained with binary cross-entropy on member vs non-member labels. This gives the attacker the capacity to learn which combinations of features are informative for membership specifically, rather than for subtype identity. The evidence that this works: retrain AUC = 0.523 on PBMC (near chance), meaning the attacker does not detect biology when the model was not trained on the forget set. The off-the-shelf distance methods cannot make this distinction.

However, the trained attacker is still partially confounded (~30% of signal) because the matched negatives are from different cell types. Within-subtype matching reduces this further. A potential better approach would combine a trained attacker with within-subtype matching and a calibrated reference (retrain model), which I haven't found any existing paper on.

---

## Open Questions

1. **Is there a better MIA?** A calibrated likelihood ratio using the retrain model as reference (Carlini et al. 2022 style) would isolate memorization without requiring matched negatives. A per-sample reconstruction gap (recon_retrain - recon_baseline) is even simpler. Neither has been tried.

2. **Can architecture changes help?** The conditional VAE experiment (NB29) showed that cluster-conditional decoder columns achieve zero Fisher overlap, but the shared encoder dominates the MIA features (64/69 features are mu+logvar). A fully disentangled architecture might enable unlearning.

3. **Is patient-level memorization ever detectable?** On TCGA-BRCA (925 training samples, 4.2M parameters), no patient-level memorization was detected. This may be because KL regularization and dropout prevent individual-level overfitting. Testing on larger cohorts or less-regularized models might reveal patient-level memorization.

4. **Target audience?** Nature Methods vs TMLR vs IEEE (security/privacy) vs NeurIPS. The biology confound finding has more immediate impact in biology venues where potentially flawed evaluations are actively being published. I'm unsure how the privacy/security evaulation would rate for IEEE.

---

## References

### Core citations

- Feldman, V. (2020). Does learning require memorization? A short tale about a long tail. *STOC 2020*. arXiv:1906.05271.
- Golatkar, A., Achille, A., & Soatto, S. (2020). Eternal sunshine of the spotless net: Selective forgetting in deep networks. *CVPR 2020*.
- Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053-1058.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy risk in machine learning: Analyzing the connection to overfitting. *CSF 2018*.
- Hayes, J., Mahloujifar, S., & Balle, B. (2024). Inexact unlearning needs more careful evaluations to avoid a false sense of privacy. *SaTML 2024*. arXiv:2403.01218.
- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F. (2022). Membership inference attacks from first principles. *IEEE S&P 2022*.
- Cancer Genome Atlas Network. (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490, 61-70.
- Subramanian, A., et al. (2017). A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. *Cell*, 171(6), 1437-1452.
- Love, M. I., Huber, W., & Anders, S. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15, 550.
- Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15.
- Moon, S., et al. (2024). Feature unlearning for pre-trained GANs and VAEs. *AAAI 2024*. arXiv:2402.05224.

### New citations

- Ozturk, H., et al. (2026). Towards useful and private synthetic omics: Community benchmarking of generative models for transcriptomics data. *bioRxiv* 2026.03.02.707794.
- Golob, S., et al. (2026). Privacy vulnerabilities in synthetic single-cell RNA-sequence data. *bioRxiv* 2026.01.22.701160.
- Goldman, M. J., et al. (2020). Visualizing and interpreting cancer genomics data via the Xena platform. *Nature Biotechnology*, 38, 675-678.
- Walker, C., et al. (2024). Private information leakage from single-cell count matrices. *Cell*, 187(22), 6237-6252.
- Pawelczyk, M., et al. (2025). Machine unlearning fails to remove data poisoning attacks. *ICLR 2025*. arXiv:2406.17216.
- Sun, Z., et al. (2026). Statistical MIA: Rethinking membership inference attack for reliable unlearning auditing. arXiv:2602.01150.
- Naderloui, M., et al. (2025). Rectifying privacy and efficacy measurements in machine unlearning. *USENIX Security 2025*.
- George, A., et al. (2025). The illusion of unlearning: The unstable nature of machine unlearning in text-to-image diffusion models. *CVPR 2025*.
- Shi, Y., et al. (2024). Assessing membership inference attacks under distribution shifts. *IEEE BigData 2024*.
- Chen, D., Yu, N., Zhang, Y., & Fritz, M. (2020). GAN-leaks: A taxonomy of membership inference attacks against generative models. *CCS 2020*.
- Tabula Muris Consortium. (2018). Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. *Nature*, 562, 367-372.
- Kunstner, F., Hennig, P., & Balles, L. (2019). Limitations of the empirical Fisher approximation for natural gradient descent. *NeurIPS 2019*.
- "Machine unlearning under retain-forget entanglement." ICLR 2026 submission, OpenReview.
- EAGLE: Entanglement-aware unlearning. arXiv:2508.20443 (2025).
- "A geometric framework for understanding memorization in generative models." arXiv:2411.00113 (2024).
- "Generalization in VAE and diffusion models: A unified information-theoretic analysis." arXiv:2506.00849 (2025).
- "Position: MIA should move on to distributional statistics." ICML 2025. arXiv:2502.02970.
