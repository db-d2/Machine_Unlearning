# Plan: Synthetic Matched Negatives + Representation-Space Unlearning

Two experiments that address the paper's two remaining weaknesses:
1. Small within-subtype evaluation samples (5 megakaryocytes on PBMC)
2. No constructive unlearning solution

Both use the same mechanism: generating synthetic samples from the retrain model.

---

## Experiment A: Synthetic Matched Negatives for Evaluation

### Problem
Within-subtype evaluation on PBMC uses only 5 unseen megakaryocytes as negatives.
AUC estimates are noisy and the retrain-higher-than-baseline result (0.609 vs 0.527)
may be small-sample artifact.

### Idea
The retrain model never saw the forget set. It learned megakaryocyte biology from
the retain set (which includes no cluster-13 cells) and the data distribution generally.
If we encode real megakaryocytes through the retrain model and decode sampled latent
codes, we get synthetic cells that have the right biology with zero memorization.

### Implementation (NB37)

```
NB37: Synthetic matched negatives for within-subtype evaluation
```

**Step 1: Generate synthetic megakaryocyte-like samples from retrain model**

Two generation strategies to compare:

Strategy A — Posterior sampling:
- Take the 5 real unseen megakaryocytes
- Encode through retrain model to get mu, logvar
- Sample N=200 z values from each posterior: z ~ N(mu_i, sigma_i)
- Decode each z with appropriate library_size
- This produces 200 synthetic cells per real cell, all biologically similar
- Drawback: anchored to 5 seed cells

Strategy B — Prior sampling with subtype conditioning:
- Encode ALL retain-set cells through retrain model
- Fit a GMM or kernel density estimate in retrain latent space
- Sample z from the region of latent space corresponding to cluster 13's location
  (use retain model's encoding of the 5 unseen megakaryocytes to locate the region)
- Decode with library_size drawn from empirical distribution
- Produces more diverse synthetic cells

Strategy C — Reconstruction perturbation:
- Take the 5 unseen megakaryocytes
- Encode, decode through retrain model to get reconstructions
- Add small Gaussian noise to the reconstructions (in expression space)
- These are "reconstruction-perturbed" negatives
- Simplest approach

Recommend trying A first (simplest, well-motivated), then B if more diversity needed.

**Step 2: Validate synthetic negatives are biologically plausible**
- PCA/UMAP: synthetic cells should cluster with real megakaryocytes
- Gene expression correlation: synthetic vs real megakaryocyte marker genes
- Library size distribution: should match real cells

**Step 3: Re-evaluate with synthetic matched negatives**
- For baseline model: extract 69-dim features from forget set + synthetic negatives
- Train fresh MLP attacker (per mia-evaluation skill)
- Compare AUC to:
  - Original cross-cluster matching (0.769)
  - Within-cluster holdout with 5 real cells (0.527)
  - This new synthetic evaluation
- If synthetic AUC is near 0.5, confirms no memorization beyond biology
- If synthetic AUC is elevated, the genuine memorization signal is stable

**Step 4: Permutation test for statistical significance**
- Shuffle member/non-member labels 1000 times
- Compute AUC each time to build null distribution
- Report p-value for the observed AUC
- Valid with any sample size

### Expected output
- Synthetic matched negatives (200-500 cells)
- Stable AUC estimate with permutation p-value
- Comparison figure: cross-cluster vs within-cluster vs synthetic negatives
- Resolves the small-n ambiguity

### Files to create/modify
- `notebooks/37_synthetic_matched_negatives.ipynb` (new)
- No src/ changes needed (VAE decode() already supports generation)

---

## Experiment B: Representation-Space Unlearning

### Problem
All 8 existing unlearning methods operate in parameter space (modify weights).
Fisher overlap shows this is structurally limited for shared-decoder architectures.
No method provides a constructive solution.

### Idea
Instead of modifying parameters, modify the MODEL'S BEHAVIOR on forget samples
by fine-tuning to match a "generic subtype" representation. Use the retrain model
as the reference for what forget-subtype samples should look like without memorization.

The key insight: the retrain model provides the counterfactual — "what would this
model represent for these samples if it had never seen them?" We align the baseline
model's latent representations toward the retrain model's representations.

This bypasses the Fisher overlap bottleneck because we're optimizing in representation
space (latent codes), not parameter space (weights). The loss function targets specific
samples' representations, not global parameter perturbations.

### Method: Representation Alignment Unlearning (RAU)

```
For each forget sample x_i:
  mu_baseline, logvar_baseline = baseline_model.encode(x_i)
  mu_retrain, logvar_retrain = retrain_model.encode(x_i)

  # Alignment loss: push baseline's representation toward retrain's
  L_align = KL(N(mu_baseline, sigma_baseline) || N(mu_retrain, sigma_retrain))

For each retain batch x_r:
  # Utility loss: maintain reconstruction quality
  L_utility = ELBO(baseline_model, x_r)

Total loss = L_utility + lambda * L_align
```

This is related to SCRUB but with a critical difference:
- SCRUB uses the baseline model as teacher → the teacher memorized the data
- RAU uses the retrain model as reference → the reference has zero memorization
- SCRUB tries to make the student forget → RAU tries to make the model's
  representation of forget samples match a non-memorizing reference

### Implementation (NB38)

```
NB38: Representation-space unlearning via retrain alignment
```

**Step 1: Load baseline and retrain models**
- Baseline: trained on all data (memorized forget set)
- Retrain: trained without forget set (no memorization, same biology)
- Both are frozen for feature extraction; only the "student" (copy of baseline) trains

**Step 2: Implement RAU loss**
```python
def rau_loss(student, retrain_teacher, x_forget, x_retain, lib_f, lib_r, lambda_align):
    # Utility: student reconstructs retain set well
    output_r = student(x_retain, library_size=lib_r)
    loss_utility, recon, kl = vae_loss(x_retain, output_r, likelihood='nb')

    # Alignment: student's posterior on forget matches retrain's posterior
    mu_s, logvar_s = student.encode(x_forget)
    with torch.no_grad():
        mu_t, logvar_t = retrain_teacher.encode(x_forget)

    # KL between two Gaussians (per-sample, summed over latent dims)
    kl_align = 0.5 * (
        logvar_t - logvar_s
        + (logvar_s.exp() + (mu_s - mu_t)**2) / logvar_t.exp()
        - 1
    ).sum(dim=1).mean()

    return loss_utility + lambda_align * kl_align
```

**Step 3: Train with lambda sweep**
- lambda_align in {0.1, 1.0, 10.0, 100.0}
- 50 epochs, lr=1e-4, batch_size=128
- Early stopping on retain ELBO

**Step 4: Evaluate per mia-evaluation skill**
- Fresh attacker, 69-dim v1 features, matched negatives
- Compare to all 8 existing methods
- Also evaluate with within-subtype matching and synthetic negatives (from Exp A)

**Step 5: Fisher overlap analysis of the RAU model**
- Compute Fisher overlap between RAU model's forget/retain sets
- Compare to baseline and retrain
- If RAU reduces Fisher overlap, it found parameter-space changes that are selective
- If Fisher overlap is unchanged but MIA AUC drops, the method works in representation
  space despite the parameter-space limit

### Why this might work when other methods fail
- Fisher overlap is about parameter-space geometry. It says: "you can't find a
  direction in weight space that affects forget samples without affecting retain."
- RAU doesn't need such a direction. It optimizes a loss that explicitly targets
  forget samples' representations (via the alignment term) while protecting retain
  samples' representations (via the utility term).
- The retrain model provides the EXACT target: what the representation SHOULD look
  like without memorization.
- The optimizer finds whatever weight changes achieve both objectives simultaneously,
  even if those changes aren't "selective" in the Fisher sense.

### Why this might NOT work
- The alignment loss pushes forget representations toward retrain representations,
  but the same weight changes may push retain representations away (Fisher overlap
  means the gradients are correlated)
- The utility loss protects retain quality, creating a tug-of-war that may prevent
  convergence
- The method requires the retrain model, which is expensive to compute — if you can
  retrain, why not just use the retrain model directly?

### Counterargument to "just use retrain"
The retrain model is a REFERENCE, not a DEPLOYMENT. In practice:
- Full retraining is expensive (hours/days for large models)
- RAU fine-tunes for 50 epochs (minutes)
- The retrain model could be computed once and shared as a "privacy reference"
- Multiple forget sets could be handled sequentially with RAU, each using the same
  retrain reference

### Expected output
- RAU unlearned models for each lambda value
- AUC comparison table: RAU vs 8 existing methods
- Fisher overlap analysis of RAU model
- Figure: AUC vs lambda, with retrain floor and existing method baselines

### Files to create/modify
- `notebooks/38_representation_unlearning.ipynb` (new)
- Possibly `src/train_rau.py` if the method is reusable (but notebook-first)

---

## Execution Order

1. **NB37 first** (Experiment A) — generates synthetic negatives needed by both experiments
   - Also includes permutation test
   - Quick to run (no training, just generation + evaluation)

2. **NB38 second** (Experiment B) — the constructive contribution
   - Uses synthetic negatives from NB37 for clean evaluation
   - Lambda sweep requires multiple training runs

3. Update NB36 cross-domain comparison with new results

---

## Success Criteria

**Experiment A succeeds if:**
- Synthetic negatives produce stable AUC estimates (low variance across seeds)
- The AUC is consistent with within-cluster holdout (both near 0.5 or both elevated)
- Permutation test gives a clear p-value

**Experiment B succeeds if ANY of:**
- RAU achieves lower advantage than all 8 existing methods (even partially)
- RAU achieves advantage within retrain CI (full success, unlikely)
- RAU shows that representation-space optimization can reduce MIA AUC more than
  parameter-space methods, even if it doesn't reach retrain level
  (partial success, still publishable as evidence that the Fisher overlap
  bottleneck can be partially bypassed)

**Experiment B fails if:**
- RAU converges to the same AUC as retain fine-tuning (the alignment loss has no
  effect because Fisher overlap prevents selective changes)
- The utility loss degrades faster than the alignment loss improves

Even failure is informative: if RAU also fails, it strengthens the conclusion
that unlearning is structurally impossible for shared-decoder generative models,
even in representation space.

---

## Datasets

Run on PBMC-33k first (where genuine memorization exists).
If successful, replicate on TCGA-BRCA (where no memorization exists — should be trivial).

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Synthetic negatives don't look realistic | Exp A unusable | Validate with UMAP + marker genes before using |
| RAU diverges (utility collapses) | Exp B fails | Lambda sweep, gradient clipping, early stopping |
| RAU = retain fine-tune (no improvement) | Exp B uninformative | Still publishable as evidence of structural limit |
| Retrain model unavailable | Both fail | We already have retrain models for both datasets |
| Permutation test shows p>0.05 | Exp A inconclusive | Report honestly; larger synthetic negative sets may help |
