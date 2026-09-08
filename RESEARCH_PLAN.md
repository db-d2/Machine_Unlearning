# Research Plan: Why VAE Unlearning Methods Fail

## Executive Summary

Multiple unlearning methods fail for VAEs, likely due to a common underlying cause. This document outlines a research plan to identify, characterize, and validate that cause for a publishable paper.

**Core Hypothesis:** VAE decoder parameters are fundamentally entangled across all data, making surgical unlearning impossible without collateral damage.

---

## Part 1: The Pattern Across Failures

### Summary of Results

| Method | AUC | Status | Apparent Failure Mode |
|--------|-----|--------|----------------------|
| Baseline (no unlearning) | 0.769 | LEAK | VAE memorizes training data |
| Frozen single critic λ=5 | 0.997 | FAIL | VAE exploits fixed decision boundary |
| Frozen multi critic λ=10 | 0.992 | FAIL | VAE exploits fixed decision boundaries |
| Extra-gradient λ=5 | 0.382 | FAIL | Over-unlearning (Streisand effect) |
| Extra-gradient λ=10 | 0.482 | SUCCESS? | Only works with early stopping |
| Fisher (structured) | 0.814 | FAIL | Posterior collapse |
| Fisher (scattered) | 0.499 | SUCCESS? | But utility degrades 39% |
| Retrain (gold standard) | 0.481 | TARGET | True removal, but expensive |

### Key Observations

1. **Frozen critics make things worse** (AUC > 0.98 vs baseline 0.77)
2. **Extra-gradient only works at specific λ and with early stopping**
3. **Fisher fails on structured sets but works on scattered**
4. **Even "successful" methods have issues** (utility loss, early stopping requirement)
5. **The structured vs. scattered distinction is consistent**

---

## Part 2: The Common Cause Hypothesis

### Primary Hypothesis: Parameter Entanglement

**VAE decoder parameters are fundamentally entangled across all data, making surgical unlearning impossible.**

Unlike classifiers where each class has relatively independent parameters, VAE decoders learn **shared generative mappings**. The parameters that encode rare cell cluster characteristics also contribute to reconstructing common cell types.

#### Why This Explains Each Failure

**1. Frozen critics fail (AUC > 0.98)**

The VAE can shift encoder representations to fool the frozen critics without modifying the decoder. The membership signal is stored in the decoder's ability to reconstruct specific samples, but the critic only sees encoder outputs. The VAE learns to "hide" the signal from the critic without removing it.

**2. Fisher fails on structured sets (AUC = 0.81)**

When forget samples form a coherent cluster (e.g., all megakaryocytes), the decoder has learned cluster-specific mappings. These parameters are entangled with general reconstruction capacity. Scrubbing them causes "posterior collapse" where the model produces generic outputs for forgotten cells—which itself becomes a detectable signature.

**3. Fisher "works" on scattered but destroys utility**

Scattered forget samples have no shared structure, so there's no coherent set of parameters to scrub. But the aggressive parameter modification still damages the shared generative capacity, causing the 39% utility degradation observed.

**4. Extra-gradient requires early stopping**

The min-max game has an equilibrium where the VAE has found ways to encode membership information that fools the current critics. Early stopping (50 epochs) captures a transient state before this equilibrium. This is exploiting training dynamics, not achieving true unlearning.

**5. Over-unlearning at λ=5 (AUC = 0.38)**

Aggressive privacy pressure damages the shared mappings so severely that forget samples become detectably "broken"—they look *less* like training data than truly unseen samples. This is the Streisand effect noted by Hayes et al.

### Supporting Hypothesis: Generative vs. Discriminative Models

**Classifiers** learn decision boundaries. Forgetting means shifting boundaries—a relatively local operation in parameter space.

**VAEs** learn to reconstruct the entire data distribution. The loss function explicitly rewards memorization:
- Reconstruction loss rewards remembering each sample
- Negative binomial likelihood models sample-specific overdispersion
- KL divergence encourages smooth latent space (but doesn't prevent memorization)

Unlearning fights the core training objective of a VAE.

### Supporting Hypothesis: Information Locality

The structured vs. scattered distinction suggests the fundamental issue is **information locality**:

- **Structured forget sets**: Correlated samples share features, dedicated decoder parameters, and a coherent "direction" in latent space. Information is concentrated.

- **Scattered forget sets**: Independent samples with no shared structure, no dedicated parameters, no coherent latent direction. Information is diffuse.

When information is concentrated, removing it causes visible damage. When it's diffuse, it's harder to detect but also harder to target.

---

## Part 3: Research Plan to Validate the Hypothesis

### Experiment 1: Information Localization Analysis

**Question:** Where is membership information stored—encoder, decoder, or both?

**Rationale:** If we know where memorization lives, we can understand why methods targeting different components fail.

**Protocol:**
```python
# 1. Extract features from encoder only
encoder_features = [mu, logvar, z, ||mu||, ||logvar||]

# 2. Extract features from decoder only
decoder_features = [reconstruction_loss, per_gene_loss_variance]

# 3. Extract combined features (current approach)
combined_features = [mu, logvar, recon_loss, kl_div, elbo, norms]

# 4. Train separate MIA attackers on each feature set
attacker_encoder = train_attacker(encoder_features)
attacker_decoder = train_attacker(decoder_features)
attacker_combined = train_attacker(combined_features)

# 5. Compare AUC
```

**Predictions:**
| Scenario | Implication |
|----------|-------------|
| Encoder AUC high, Decoder AUC low | Memorization in latent mapping |
| Encoder AUC low, Decoder AUC high | Memorization in reconstruction |
| Both high | Entangled throughout |
| Combined >> Either alone | Information distributed |

**Follow-up:** If encoder-only AUC is high, test encoder-focused unlearning. If decoder-only AUC is high, test decoder-focused unlearning.

**Estimated time:** 1-2 days

---

### Experiment 2: Parameter Entanglement Quantification

**Question:** How entangled are the parameters across forget/retain sets?

**Rationale:** Directly measures whether the core hypothesis is true.

**Protocol:**
```python
# 1. Compute diagonal Fisher information for forget set
F_forget = compute_fisher_diagonal(model, forget_loader)

# 2. Compute diagonal Fisher information for retain set
F_retain = compute_fisher_diagonal(model, retain_loader)

# 3. Measure overlap metrics
overlap_correlation = pearsonr(F_forget.flatten(), F_retain.flatten())
overlap_cosine = cosine_similarity(F_forget, F_retain)
overlap_top_k = jaccard(top_k_params(F_forget), top_k_params(F_retain))

# 4. Compare structured vs. scattered
structured_overlap = compute_overlap(F_structured_forget, F_retain)
scattered_overlap = compute_overlap(F_scattered_forget, F_retain)

# 5. Analyze by layer
for layer in model.decoder.layers:
    layer_overlap = compute_overlap(F_forget[layer], F_retain[layer])
```

**Predictions:**
- Structured forget sets → high overlap (parameters are shared)
- Scattered forget sets → low overlap (parameters are independent)
- Deeper decoder layers → higher overlap (more shared representations)

**Visualization:**
- Heatmap of Fisher overlap by layer
- Scatter plot of F_forget vs F_retain values
- Bar chart comparing structured vs scattered overlap

**Estimated time:** 1-2 days

---

### Experiment 3: Relearning Vulnerability Test

**Question:** Is information truly removed, or just obfuscated?

**Rationale:** If "unlearned" models can quickly relearn the forget set's signature, the information was hidden, not deleted. This directly addresses the concern about extra-gradient exploiting dynamics rather than achieving true unlearning.

**Protocol:**
```python
# 1. Start with unlearned model (extra-gradient λ=10)
model_unlearned = load_model("extragradient_unlearned.pt")

# 2. Fine-tune on small amounts of RETAIN data (not forget data)
for n_samples in [10, 50, 100, 500]:
    model_finetuned = finetune(model_unlearned, retain_samples[:n_samples], epochs=5)

    # 3. Re-evaluate MIA
    auc_after = evaluate_mia(model_finetuned, forget_set, unseen_set)

    # 4. Track recovery
    print(f"After {n_samples} retain samples: AUC = {auc_after}")

# 5. Compare to relearning on forget data directly
model_relearned = finetune(model_unlearned, forget_samples, epochs=5)
auc_relearned = evaluate_mia(model_relearned, forget_set, unseen_set)
```

**Predictions:**
| Outcome | Interpretation |
|---------|----------------|
| AUC stays ~0.5 after fine-tuning | True unlearning achieved |
| AUC rises with retain fine-tuning | Information was obfuscated, not removed |
| AUC rises only with forget fine-tuning | Information removed but relearnable |

**Estimated time:** 1 day

---

### Experiment 4: Latent Space Geometry Analysis

**Question:** What happens to forget sample representations during unlearning?

**Rationale:** Visualizes what unlearning actually does to the learned representations.

**Protocol:**
```python
# 1. Extract latent codes before unlearning
z_forget_before = encode(model_baseline, forget_samples)
z_retain_before = encode(model_baseline, retain_samples)
z_unseen_before = encode(model_baseline, unseen_samples)

# 2. Extract latent codes after each unlearning method
for method in ['extragradient', 'fisher', 'retrain']:
    z_forget_after = encode(model_unlearned[method], forget_samples)
    z_retain_after = encode(model_unlearned[method], retain_samples)

    # 3. Measure movement
    forget_displacement = ||z_forget_after - z_forget_before||
    retain_displacement = ||z_retain_after - z_retain_before||

    # 4. Measure overlap with unseen
    forget_to_unseen_distance_before = distance(z_forget_before, z_unseen_before)
    forget_to_unseen_distance_after = distance(z_forget_after, z_unseen_after)

# 5. Visualize with UMAP
plot_umap(z_forget, z_retain, z_unseen, labels=['forget', 'retain', 'unseen'])
```

**Key Questions:**
- Do forget samples move toward the "unseen" region?
- Do retain samples of the same cell type also move? (collateral damage)
- Is the geometry preserved or scrambled?

**Visualization:**
- UMAP before/after unlearning (side by side)
- Displacement vectors for forget samples
- Distribution of distances to nearest unseen neighbor

**Estimated time:** 1-2 days

---

### Experiment 5: Utility-Privacy Frontier Mapping

**Question:** Is there a Pareto frontier, or a fundamental impossibility region?

**Rationale:** Maps the tradeoff space to understand if there's a sweet spot or if utility and privacy are fundamentally at odds.

**Protocol:**
```python
# 1. Sweep extra-gradient λ
results = []
for lambda_val in [1, 2, 5, 7, 10, 15, 20, 30, 50]:
    model = train_extragradient(lambda_privacy=lambda_val)
    auc = evaluate_mia(model)
    utility = evaluate_utility(model)  # MSE, classification acc
    results.append({'lambda': lambda_val, 'auc': auc, 'utility': utility})

# 2. Sweep Fisher intensity
for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
    for steps in [50, 100, 200, 500]:
        model = train_fisher(scrub_lr=alpha, scrub_steps=steps)
        auc = evaluate_mia(model)
        utility = evaluate_utility(model)
        results.append({'alpha': alpha, 'steps': steps, 'auc': auc, 'utility': utility})

# 3. Plot Pareto frontier
plot_frontier(results, x='utility', y='privacy')
```

**Key Questions:**
- Is there a Pareto frontier (can't improve one without hurting the other)?
- Is there an "impossibility region" where no configuration works?
- Where does retraining sit on this frontier?

**Estimated time:** 2-3 days (many configurations to run)

---

### Experiment 6: Cross-Dataset Validation (Tabula Muris)

**Question:** Is this a VAE architecture problem or a PBMC dataset problem?

**Rationale:** Confirms findings generalize and aren't artifacts of the specific dataset.

**Protocol:**
```
1. Prepare Tabula Muris dataset
   - Download from https://tabula-muris.ds.czbiohub.org/
   - Apply same preprocessing (QC, 2000 HVGs, clustering)
   - Identify rare clusters for structured forget set

2. Replicate baseline experiments
   - Train VAE (same architecture)
   - Measure baseline MIA AUC
   - Compare memorization level to PBMC

3. Replicate unlearning experiments
   - Extra-gradient λ=10 on structured
   - Fisher on structured and scattered
   - Document success/failure pattern

4. Replicate analysis experiments
   - Information localization
   - Parameter entanglement
   - Latent space geometry

5. Compare results to PBMC
   - Same failure modes?
   - Same entanglement pattern?
   - Same information localization?
```

**Predictions:**
- If patterns match → findings are about VAE architecture
- If patterns differ → dataset properties matter (which ones?)

**Estimated time:** 3-4 days

---

## Part 4: Paper Structure

### Proposed Title

**"Why Machine Unlearning Fails for Variational Autoencoders: The Parameter Entanglement Problem"**

Alternative titles:
- "The Entanglement Problem: Why Classifer Unlearning Methods Fail for VAEs"
- "Surgical Forgetting is Impossible: Parameter Entanglement in Generative Models"

### Proposed Structure

**Abstract**
- Machine unlearning works for classifiers but fails for VAEs
- We test multiple methods and find they all fail for related reasons
- Core finding: VAE parameters are entangled, surgical removal impossible
- Implication: New approaches needed for generative model unlearning

**1. Introduction**
- Privacy motivation (GDPR, single-cell data sensitivity)
- Unlearning as alternative to retraining
- Why VAEs? (scRNA-seq, generative models are widespread)
- Research question: Do classifier unlearning methods transfer?

**2. Background**
- Machine unlearning definitions
- Membership inference attacks
- Fisher information scrubbing
- Adversarial training for privacy
- VAE architecture and training

**3. Experimental Setup**
- PBMC-33k dataset
- VAE architecture (scVI-style)
- MIA evaluation protocol (matched negatives)
- Methods tested (frozen critics, extra-gradient, Fisher)

**4. Results: Multiple Methods Fail**
- Table of all method results
- Structured vs. scattered distinction
- The λ tradeoff for extra-gradient
- Fisher posterior collapse

**5. Analysis: The Common Cause**
- 5.1 Information Localization
  - Where is membership signal stored?
  - Encoder vs. decoder contributions
- 5.2 Parameter Entanglement
  - Fisher information overlap
  - Structured vs. scattered comparison
- 5.3 Latent Space Geometry
  - What changes during unlearning?
  - Visualization of representations
- 5.4 Relearning Vulnerability
  - Is information removed or hidden?
  - Fine-tuning recovery test

**6. Cross-Dataset Validation**
- Tabula Muris results
- Confirmation of entanglement pattern
- Generalization of findings

**7. Discussion**
- Why VAEs are fundamentally different from classifiers
- The generative model challenge
- Implications for privacy in deep generative models
- Why retraining remains the gold standard

**8. Future Directions**
- Architecture modifications (separate decoders?)
- Latent space interventions (remove "cluster direction")
- Formal privacy guarantees (differential privacy integration)
- Larger models where retraining is prohibitive

**9. Conclusion**
- Summary of findings
- Key message: Classifier unlearning ≠ VAE unlearning
- Call for new methods designed for generative models

---

## Part 5: Priority and Timeline

### Recommended Order

| Priority | Experiment | Time | Why First |
|----------|-----------|------|-----------|
| 1 | Information Localization | 1-2 days | Quick, high insight |
| 2 | Parameter Entanglement | 1-2 days | Directly tests core hypothesis |
| 3 | Relearning Vulnerability | 1 day | Addresses key methodological concern |
| 4 | Tabula Muris Validation | 3-4 days | Strengthens generalization claims |
| 5 | Latent Space Geometry | 1-2 days | Visualization for paper |
| 6 | Utility-Privacy Frontier | 2-3 days | Comprehensive but time-intensive |

**Total estimated time:** 10-14 days

### Minimum Viable Paper

If time is limited, the minimum experiments needed:

1. **Information Localization** - Shows where memorization lives
2. **Parameter Entanglement** - Proves the core hypothesis
3. **Tabula Muris** - Confirms generalization

This would support a paper titled "Parameter Entanglement Explains VAE Unlearning Failures" with the other experiments as future work.

### What We Already Have

- All baseline results documented
- All method failures documented
- Fisher code ready for entanglement analysis
- MIA evaluation pipeline ready
- PBMC data preprocessed

---

## Part 6: Key Questions to Answer

### For the Core Hypothesis

1. Is membership information in the encoder, decoder, or both?
2. How much do Fisher matrices overlap between forget/retain sets?
3. Does overlap correlate with unlearning difficulty?
4. Is the entanglement worse for structured vs. scattered?

### For Methodological Validity

5. Does fine-tuning on retain data recover "unlearned" membership signal?
6. Is extra-gradient finding true minima or exploiting transient dynamics?
7. Would stronger per-example attacks break methods that "succeed"?

### For Generalization

8. Does Tabula Muris show the same entanglement pattern?
9. Are failure modes consistent across datasets?
10. Is this about VAE architecture or single-cell biology?

### For Future Work

11. Would separate decoders per cluster reduce entanglement?
12. Can latent space interventions (removing directions) achieve unlearning?
13. At what model scale does unlearning become faster than retraining?

---

## Appendix: Literature Context

### Key References Supporting This Analysis

**Hayes et al. (2024)** - "Inexact Unlearning Needs More Careful Evaluations"
- Per-example attacks are stronger than population attacks
- Average-case metrics overestimate privacy protection
- Supports our concern about evaluation methodology

**Golatkar et al. (2020)** - "Eternal Sunshine of the Spotless Net"
- Original Fisher scrubbing paper
- Designed for classifiers, not generative models
- Assumes parameter influence is separable

**Nasr et al. (2018)** - "Adversarial Regularization for Privacy"
- Original min-max game formulation
- No formal privacy guarantees
- May fail against whitebox attacks

**Chavdarova et al. (2019)** - "Extragradient for GANs"
- Dampens oscillations in min-max games
- Applied to GAN training, not unlearning
- We repurpose for VAE-attacker game

### What's Novel in Our Contribution

1. **First systematic study of unlearning for VAEs** (vs. classifiers)
2. **Identification of parameter entanglement as root cause**
3. **Explanation of why classifier methods don't transfer**
4. **Cross-dataset validation of failure modes**
5. **Information localization analysis for VAEs**
