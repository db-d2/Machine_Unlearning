# PROMPT.md - Useful Prompts for Extending This Project

Copy-paste these prompts to quickly start common extension tasks.

## Research Framing

**Current stage:** Paper draft complete. Addressing publication blockers from `study_suggestions.md`.

**Success criterion (use everywhere):**
> Robust unlearning is achieved when the worst-case attack advantage across all attacks is at or below the upper bound of the retrain advantage 95% CI.

Where: `advantage = 2 × |AUC - 0.5|` (direction-agnostic, avoids AUC < 0.5 confusion)

**Key findings now established:**
- Memorization concentrates in structured (biologically coherent) subpopulations
- Extra-gradient λ=10 works only on PBMC single-cluster n=30 (does not generalize)
- Fisher provides marginal improvement on scattered sets (where baseline memorization is weak)
- Fisher fails on structured sets via posterior collapse (consistent across datasets)

**What makes results publishable:**
- Evaluation protocol with attack diversity (low/high assumption) and biological matching
- Empirical finding about where memorization concentrates
- Failure mode characterization (posterior collapse, over-unlearning, dataset dependence)
- Honest negative results with cross-dataset validation

---

## Phase 1: Evaluation Framework Hardening

### 1.1 Implement Advantage Metric

**When to use:** First step in evaluation reform.

```
Add the advantage metric to the evaluation pipeline.

1. Edit `src/attacker_eval.py` to add:
   - attack_advantage = 2 * abs(auc - 0.5)
   - attack_success = max(auc, 1 - auc)

2. Update `compute_attack_metrics()` to return both alongside raw AUC.

3. Create `scripts/recompute_all_metrics.py`:
   - Walk outputs/ for model checkpoints
   - For each: load model, compute AUC + advantage on forget vs matched negatives
   - Save unified results to outputs/publication_metrics.json

4. Update the bootstrap CI function to also compute CIs for advantage.

The advantage metric fixes the AUC symmetry problem: AUC=0.38 becomes advantage=0.24
(still distinguishable), not "successful over-unlearning."
```

---

### 1.2 Implement Loss-Based Attack (Low-Assumption)

**When to use:** First additional attack to add. Simplest possible attack.

```
Create `src/threshold_attack.py` implementing a loss-based MIA.

For AUC computation, no threshold tuning is needed:
1. Score each sample: score(x) = reconstruction_loss(x) (or -ELBO, or KL)
2. Compute ROC AUC directly from scores (lower loss → more likely member)

Optionally, for reporting TPR@FPR thresholds:
3. Find threshold via Youden's J on calibration split
4. Report TPR@FPR=0.01 and TPR@FPR=0.05

Required functions:
- loss_attack_scores(model, samples) → scores (just reconstruction loss)
- loss_attack_auc(model, forget_set, unseen_set) → AUC, advantage

This is a LOW-ASSUMPTION attack: only requires query access to the model.
No training data, no auxiliary models, no learned parameters.

~30 lines of code. Much simpler than originally planned.

Reference: Yeom et al. "Privacy Risk in Machine Learning"
```

---

### 1.3 Implement Likelihood-Ratio Attack (High-Assumption)

**When to use:** After loss-based attack. Requires auxiliary model.

```
Create `src/likelihood_ratio_attack.py` implementing a likelihood-ratio MIA.

For generative models, this attack compares likelihoods under two models:
- Target model (the one we're attacking)
- Reference model (e.g., retrained model, or fresh random init)

Score = log P(x | target) - log P(x | reference)

Higher score → sample more likely under target → more likely member.

For VAEs:
- P(x | model) ≈ ELBO (lower bound on log-likelihood)
- Or use importance-weighted ELBO for tighter bound

**ASSUMPTION:** Attacker has access to an auxiliary reference model.
Document this explicitly in the writeup when reporting results.

This is a HIGH-ASSUMPTION attack. Group separately from loss-based attack.

Reference: Carlini et al. "Membership Inference Attacks From First Principles"
```

---

### 1.4 Implement k-NN Latent Attack (High-Assumption)

**When to use:** Non-parametric attack complement to MLP. Requires training set access.

```
Create `src/knn_attack.py` implementing a k-NN attack in latent space.

The attack works as follows:
1. Encode all training samples to get reference latent codes
2. For query sample, compute distance to k nearest training latents
3. Score = average distance (lower → more likely member)

This is non-parametric and uses only latent representations, making it
conceptually distinct from both the MLP (parametric) and loss-based (uses reconstruction).

Implementation:
- Use sklearn.neighbors.NearestNeighbors with training set latent codes
- Score = mean distance to k=10 nearest neighbors
- Convert scores to AUC (lower score → member, so negate for ROC)

**ASSUMPTION:** Attacker has access to latent embeddings of training samples (or proxy set).
Document this explicitly in the writeup when reporting results.

This is a HIGH-ASSUMPTION attack. Group separately from loss-based attack.

Reference: Carlini et al. "Extracting Training Data from Diffusion Models"
```

---

### 1.5 Create Attack Suite Runner

**When to use:** After implementing all attacks.

```
Create `scripts/run_attack_suite.py` that runs all attacks on all checkpoints.

For each model checkpoint in outputs/:
1. Run loss-based attack (low-assumption)
2. Run MLP attack (high-assumption, existing)
3. Run likelihood-ratio attack (high-assumption)
4. Run k-NN latent attack (high-assumption)

Output format for each model:
{
    "low_assumption": {
        "loss_recon": {"auc": 0.55, "advantage": 0.10},
        "loss_elbo": {"auc": 0.52, "advantage": 0.04},
        "worst_case": {"advantage": 0.10, "attack": "loss_recon"}
    },
    "high_assumption": {
        "mlp_v1": {"auc": 0.48, "advantage": 0.04},
        "likelihood_ratio": {"auc": 0.50, "advantage": 0.00},
        "knn_latent": {"auc": 0.51, "advantage": 0.02},
        "worst_case": {"advantage": 0.04, "attack": "mlp_v1"}
    },
    "overall_worst_case": {"advantage": 0.10, "attack": "loss_recon"}
}

Save to outputs/attack_suite_results.json.

**Success criterion:** overall worst-case advantage ≤ upper bound of retrain advantage 95% CI
```

---

### 1.6 Derive Principled Target Band

**When to use:** Replace arbitrary ±0.03 with data-driven CI.

```
Create `scripts/run_retrain_bootstrap.py` to derive a principled target band.

1. Run retrain 10 times with different seeds
2. For each retrained model, evaluate with full attack suite
3. Compute 95% bootstrap CI for retrain AUC and retrain advantage
4. The retrain CI becomes the target band for "successful" unlearning

Save to outputs/retrain_bootstrap.json:
{
    "n_seeds": 10,
    "auc_mean": 0.481,
    "auc_std": 0.015,
    "auc_ci_95": [0.455, 0.507],
    "advantage_mean": 0.038,
    "advantage_ci_95": [0.012, 0.056]
}

New success criterion: worst-case advantage ≤ retrain advantage CI upper bound.
```

---

### 1.7 Run Cluster-Conditioned Evaluation

**When to use:** Validate that matched negatives control for biology.

```
Run the cluster-conditioned evaluation code that already exists in attacker_eval.py.

**IMPORTANT:** Use cluster labels from original preprocessing (Leiden clustering on raw data).
Do NOT re-cluster per model. Freeze labels once and reuse for all evaluations.
This prevents reviewers from arguing you changed the target between methods.

The code has two methods:
- residualized_evaluation(): fits logistic regression with cluster as covariate
- stratified_evaluation(): computes AUC within each cluster, then averages

Run on: baseline, retrain, EG λ=10, Fisher structured, Fisher scattered

Questions to answer:
1. Is conditioned retrain advantage closer to 0 than raw retrain advantage?
   - If yes: matched negatives have residual biology confound
   - If no: matched negatives are working correctly

2. Does conditioning change which methods "succeed"?
   - If EG λ=10 still meets success criterion after conditioning → robust result
   - If EG λ=10 fails after conditioning → biology was masking leakage

Create `scripts/run_conditioned_eval.py` to run this systematically.
```

---

## Phase 2: Additional Unlearning Methods

### 2.1 Retain-Only Fine-tuning

**When to use:** Simplest unlearning baseline.

```
Create `src/train_retain_finetune.py` implementing the simplest "unlearning" baseline.

This method does NOT use the forget set at all:
1. Load baseline checkpoint
2. Fine-tune on retain set only (standard ELBO loss)
3. Early stopping based on retain validation loss

Hyperparameters:
- Same optimizer as baseline (Adam, lr=1e-3)
- Same batch size (128)
- 50 epochs max with early stopping

This baseline answers: "Does simply continuing training on retain data help?"

Run on PBMC and TM (structured + scattered), 5 seeds each.
Evaluate with full attack suite.
```

---

### 2.2 Gradient Ascent Unlearning

**When to use:** Standard unlearning baseline from literature.

```
Create `src/train_gradient_ascent.py` implementing gradient ascent + fine-tuning.

Phase 1 - Forgetting:
- Gradient ASCENT on forget set (maximize reconstruction loss)
- N steps with learning rate α

Phase 2 - Recovery:
- Fine-tune on retain set (minimize ELBO)
- M epochs with early stopping

Hyperparameter grid:
- ascent_steps: [10, 50, 100]
- ascent_lr: [1e-4, 1e-3]
- finetune_epochs: 20 (fixed)

Run grid search on PBMC structured, pick best config, then run:
- PBMC (structured + scattered), 5 seeds
- TM (structured + scattered), 5 seeds

Evaluate with full attack suite.

Reference: Golatkar et al. "Forgetting Outside the Class" uses similar approach.
```

---

## Phase 3: Utility Expansion

### 3.1 Implement Utility Suite

**When to use:** After methods are run, before final evaluation.

**Goal:** Show Fisher damage in multiple metrics; prove "identical MSE" was too coarse.

```
Create `scripts/compute_utility_suite.py` with domain-relevant scRNA-seq metrics.

Metrics to compute:

1. Held-out NB log-likelihood / ELBO (proper generative metric)
   - Compute on held-out test cells
   - More informative than MSE for count data

2. Silhouette score
   - Encode all cells to latent space
   - Compute silhouette score vs Leiden cluster labels
   - **Use frozen labels from preprocessing, not per-model re-clustering**
   - Measures how well latent space preserves cell type structure

3. Adjusted Rand Index (ARI)
   - Cluster the unlearned latent space (k-means with k = original cluster count)
   - Compare to original (frozen) cluster labels
   - Measures cluster preservation

4. Marker gene correlation
   - For known marker genes (CD4, CD8A, MS4A1, etc. for PBMC)
   - Compute Pearson r between true and reconstructed expression
   - Domain-specific measure of reconstruction quality

Optional for Tabula Muris (if batch correction is part of narrative):
5. Batch mixing metric (LISI or kBET)
   - Only include if batch correction is relevant to your claims

Output JSON with all metrics + error bars across seeds.
```

---

## Phase 4: Statistical Rigor

### 4.1 Increase Seeds

**When to use:** Last phase before final writeup.

```
Increase seed count for key results:

Extra-gradient structured PBMC:
- Current: 4 seeds
- Target: 10 seeds
- Run: seeds [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

Extra-gradient structured TM:
- Current: ~1 seed per lambda
- Target: 10 seeds for λ=10
- Note: all lambdas fail, so 10 seeds on λ=10 suffices

New methods (retain-finetune, gradient-ascent):
- Target: 5-10 seeds each

For each run, save checkpoint and evaluation results.
Compute bootstrap CIs (1000 resamples) for all metrics.

Report format in tables: mean ± std [CI_low, CI_high]
```

---

## Phase 5: Writeup Revision

### 5.1 Reframe as Subpopulation Unlearning

**When to use:** Begin writeup revision.

```
The current writeup motivates with "patient data" and GDPR "right to be forgotten,"
but the experiments delete cell type clusters, not patients/donors.

Revisions needed:

1. Introduction:
   - Acknowledge distinction between subpopulation deletion and individual sample deletion
   - Reframe as: "subpopulation unlearning" or "distributional unlearning"
   - Keep privacy motivation but scope it honestly

2. Related Work:
   - Add discussion of distributional unlearning literature
   - Distinguish from sample-level deletion papers

3. Dataset section:
   - Explicitly state forget sets are cell type clusters
   - Note that donor-level experiments would require multi-donor datasets

4. Discussion:
   - Add paragraph on scope limitations
   - Acknowledge donor-level forgetting as future work

5. Future Work:
   - Explicitly list donor-level experiments as priority extension
```

---

### 5.2 Add Contribution Bullets

**When to use:** After reframing, add explicit contributions.

```
Add explicit contribution bullets after the introduction. Example:

"Our contributions are:

1. An evaluation protocol for membership inference attacks in single-cell VAEs
   with biological matching (matched negatives) and attack diversity (MLP,
   threshold, likelihood-ratio, k-NN attacks).

2. An empirical finding that memorization in single-cell VAEs concentrates in
   structured (biologically coherent) subpopulations, while scattered (random)
   cells are weakly memorized regardless of unlearning intervention.

3. Characterization of failure modes: posterior collapse for Fisher on structured
   sets, critic exploitation for frozen adversarial methods, over-unlearning
   (Streisand effect) for aggressive privacy pressure, and dataset dependence
   for extra-gradient.

4. Cross-dataset evidence that extra-gradient unlearning is brittle: it succeeds
   on PBMC single-cluster n=30 but fails on Tabula Muris and multi-cluster
   compositions, with honest reporting of these negative results."
```

---

### 5.3 Update Tables with New Metrics

**When to use:** After all experiments complete.

```
Update all result tables to include:

Privacy columns:
- Raw AUC
- Advantage = 2|AUC - 0.5|
- 95% CI
- Worst-case attack (from attack suite)

Utility columns:
- Held-out ELBO
- Silhouette score
- ARI
- (Keep classification accuracy and MSE)

New rows:
- Retain-only fine-tuning
- Gradient ascent

New conditional columns (where available):
- Conditioned AUC (from residualized evaluation)

Create new figures:
- Attack suite comparison (bar chart grouped by method)
- Utility-privacy tradeoff (scatter: x=utility, y=advantage)
```

---

## Phase 6: Fisher Overlap Analysis Prompts

### 6.1 Write Fisher Overlap Script

```
Create `scripts/fisher_overlap_analysis.py` that computes Fisher information overlap
between forget and retain sets for the VAE.

Reuse `compute_fisher_diagonal()` from `src/train_fisher_unlearn.py`. Call it twice:
once with a forget-set DataLoader, once with a retain-set DataLoader.

Compute these metrics (globally and per-layer):
- Cosine similarity between F_forget and F_retain vectors
- Effective rank: exp(entropy(F / sum(F))) for each Fisher
- Top-k overlap: |topK(F_forget) intersect topK(F_retain)| / K for K=100,1000,10000
- Per-layer breakdown of all metrics

Load data from data/adata_processed.h5ad, splits from outputs/p1/split_structured.json,
matched negatives from outputs/p1.5/s1_matched_negatives.json,
baseline checkpoint from outputs/p1/baseline/best_model.pt.

Save results to outputs/p6/fisher_overlap.json.

VAE architecture: 7.35M params. Key layers:
- encoder.network.0.weight [1024, 2000] = 2.05M (input layer)
- decoder.fc_mean.0.weight [2000, 1024] = 2.05M (output mean)
- decoder.fc_dispersion.weight [2000, 1024] = 2.05M (output dispersion)
```

### 6.2 Train Classifier and Compare Fisher

```
Create `scripts/train_classifier_probe.py`:

1. Load baseline VAE checkpoint and freeze it
2. Encode all training cells to get 32-dim latent means
3. Train a logistic regression classifier (14 Leiden clusters)
4. Compute classifier Fisher diagonal on forget set and retain set
5. Compute same overlap metrics as 6.1
6. Save to outputs/p6/classifier_fisher.json

Expected: classifier Fisher overlap << VAE Fisher overlap because
the classifier has class-specific output weights (14-dim softmax)
while the VAE shares decoder weights across all 2000 output genes.
```

### 6.3 Create Fisher Figures

```
Create figures for the Fisher overlap analysis:

1. figures/fisher_overlap_heatmap.png
   - Per-layer cosine similarity for VAE and classifier side by side
   - Layers on y-axis, model type on x-axis

2. figures/fisher_effective_rank.png
   - Bar chart: effective rank of forget-set Fisher by layer
   - VAE bars vs classifier bars

3. figures/fisher_scatter.png
   - Per-parameter scatter: x = log(F_retain), y = log(F_forget)
   - Color by layer, one panel for VAE, one for classifier
   - High correlation = high overlap = hard to selectively perturb

Load data from outputs/p6/fisher_overlap.json and outputs/p6/classifier_fisher.json.
Use matplotlib, same style as existing figures in figures/.
```

### 6.4 Write Proposition

```
Add a theory section to Writeup.tex (Section 5 or between current Sections 4 and 5).

Proposition 1: For a decoder f_theta mapping latent z to d output dimensions through
shared hidden layers, the cosine similarity between the forget-set and retain-set
diagonal Fisher approximations satisfies:

  cos(F_forget, F_retain) >= 1 - O(d_diff / d)

where d_diff is the number of output dimensions on which forget and retain distributions
differ, and d is the total output dimensionality.

Proof sketch:
- Shared decoder weights contribute to Fisher for ALL output dimensions
- If forget and retain sets share similar expression on (d - d_diff) genes,
  those genes contribute identical Fisher entries for shared parameters
- Only d_diff/d fraction of output dimensions produce different Fisher contributions
- For PBMC: d_diff = 337, d = 2000, so bound gives cos >= 1 - 0.17 = 0.83

Contrast with classifier:
- Output layer has class-specific weights
- Forget class (1/14 = 7%) contributes Fisher only to its own output weights
- Fisher overlap bounded by class proportion, not output dimensionality

Validate with empirical measurements from outputs/p6/.

Keep the proposition informal (no epsilon-delta). This is an empirical paper
with theoretical insight, not a theory paper.
```

---

## Quick Reference Commands

```bash
# Activate environment
pyenv activate stat4243

# Run Fisher overlap analysis
PYTHONPATH=src python scripts/fisher_overlap_analysis.py

# Train classifier probe
PYTHONPATH=src python scripts/train_classifier_probe.py

# Run attack suite on a checkpoint
PYTHONPATH=src python scripts/run_attack_suite.py \
    --checkpoint outputs/p2/extragradient/lambda10/best_model.pt \
    --forget_path outputs/p1/split_structured.json \
    --output outputs/attack_suite_results.json

# Compute utility suite
PYTHONPATH=src python scripts/compute_utility_suite.py \
    --checkpoint outputs/p2/extragradient/lambda10/best_model.pt \
    --data_path data/adata_processed.h5ad
```

---

## Checklist Before Submission (TMLR)

Evaluation (all COMPLETE):
- [x] Advantage metric in all tables; raw AUC secondary
- [x] Success = worst-case advantage <= retrain advantage CI upper bound
- [x] 5 attack types grouped by assumption level
- [x] 8 unlearning methods + DP-SGD baseline
- [x] Domain-relevant utility (ELBO, silhouette, ARI, marker corr)
- [x] Multi-seed with bootstrap CIs
- [x] Subpopulation framing, explicit contributions
- [x] References validated, humanizer pass complete

Theory (Phase 6, IN PROGRESS):
- [ ] Fisher overlap computed (VAE forget vs retain)
- [ ] Classifier comparison (low overlap expected)
- [ ] Proposition 1 + proof sketch
- [ ] 3 Fisher analysis figures
- [ ] 10-15 new references
- [ ] Abstract, contributions, conclusion updated
