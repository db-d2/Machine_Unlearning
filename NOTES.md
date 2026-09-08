The Pattern Across Failures
  ┌─────────────────────┬────────────┬────────────────────────────────────────┐
  │       Method        │   Result   │         Apparent Failure Mode          │
  ├─────────────────────┼────────────┼────────────────────────────────────────┤
  │ Frozen critics      │ AUC > 0.98 │ VAE exploits fixed decision boundaries │
  ├─────────────────────┼────────────┼────────────────────────────────────────┤
  │ Extra-gradient λ=5  │ AUC = 0.38 │ Over-unlearning (Streisand effect)     │
  ├─────────────────────┼────────────┼────────────────────────────────────────┤
  │ Extra-gradient λ=10 │ AUC = 0.48 │ Only works with early stopping         │
  ├─────────────────────┼────────────┼────────────────────────────────────────┤
  │ Fisher (structured) │ AUC = 0.81 │ Posterior collapse                     │
  ├─────────────────────┼────────────┼────────────────────────────────────────┤
  │ Fisher (scattered)  │ AUC = 0.50 │ Utility degrades 39%                   │
  └─────────────────────┴────────────┴────────────────────────────────────────┘
  The Common Cause Hypothesis

  VAE decoder parameters are fundamentally entangled across all data, making surgical unlearning impossible.

  Unlike classifiers where each class has relatively independent parameters, VAE decoders learn shared generative mappings. The parameters that
  encode rare cell cluster characteristics also contribute to reconstructing common cell types. You cannot remove one without damaging the other.

  This explains every failure:

  1. Frozen critics fail → VAE shifts encoder representations to fool critics, but decoder still "remembers" how to reconstruct forget samples. The
   membership signal is in the decoder, not where the critic is looking.
  2. Fisher fails on structured sets → Parameters important for the rare cluster are also important for general reconstruction. Scrubbing causes
  collateral damage (posterior collapse).
  3. Fisher "works" on scattered but destroys utility → No shared structure to cause entanglement, but the aggressive parameter modification still
  damages the shared generative capacity.
  4. Extra-gradient requires early stopping → At equilibrium, the VAE finds ways to encode membership information that fools the current critics.
  Early stopping captures a transient state, not true unlearning.
  5. Over-unlearning (λ=5) → Aggressive privacy pressure damages the shared mappings so much that forget samples become detectably "broken."

  The Fundamental Problem: Generative vs. Discriminative

  Classifiers learn decision boundaries. Forgetting means shifting boundaries—a relatively local operation.

  VAEs learn to reconstruct the entire data distribution. The loss function explicitly rewards memorization. The negative binomial likelihood
  models sample-specific overdispersion. Unlearning fights the core training objective.

  ---
  Research Plan: Highlighting the Common Cause

  Phase 1: Information Localization Analysis

  Question: Where is membership information stored—encoder, decoder, or both?

  Experiments:
  1. Train MIA attacker using ONLY encoder outputs (μ, logσ²)
  2. Train MIA attacker using ONLY decoder outputs (reconstruction loss)
  3. Train MIA attacker using ONLY latent samples (z)
  4. Compare AUC for each

  Predictions:
  - If encoder-only AUC is high → memorization in how samples map to z
  - If decoder-only AUC is high → memorization in how z maps to reconstruction
  - If both high → entangled throughout

  Then test: Does targeting the right component improve unlearning?

  Phase 2: Parameter Entanglement Quantification

  Question: How entangled are the parameters across forget/retain sets?

  Experiments:
  1. Compute diagonal Fisher information F_forget for forget set
  2. Compute diagonal Fisher information F_retain for retain set
  3. Measure overlap: correlation(F_forget, F_retain)
  4. Compare structured vs. scattered forget sets

  Predictions:
  - Structured sets → high overlap (entangled)
  - Scattered sets → low overlap (independent)
  - Overlap correlates with unlearning difficulty

  Visualization: Heatmap of Fisher information overlap by layer.

  Phase 3: Latent Space Geometry Analysis

  Question: What happens to forget sample representations during unlearning?

  Experiments:
  1. Extract latent codes before/after each unlearning method
  2. Visualize with UMAP/t-SNE
  3. Measure: Do forget samples move? Where to?
  4. Measure: Do similar retain samples also move? (collateral damage)

  Key test for extra-gradient:
  - Are forget samples moved to "unseen" region of latent space?
  - Or are they just scrambled in a way that confuses the attacker?

  Phase 4: Relearning Vulnerability Test

  Question: Is information truly removed, or just obfuscated?

  Experiments:
  1. After unlearning, fine-tune on 10/50/100 retain samples
  2. Re-evaluate MIA AUC
  3. Does membership signal recover?
  4. How quickly?

  Predictions:
  - If signal recovers quickly → information was obfuscated, not removed
  - If signal stays low → true removal (unlikely)

  This directly tests the user's original concern about extra-gradient "bouncing over minima."

  Phase 5: Utility-Privacy Frontier Mapping

  Question: Is there a Pareto frontier, or a fundamental impossibility?

  Experiments:
  1. Sweep λ from 1 to 50 for extra-gradient
  2. Sweep scrub intensity for Fisher
  3. For each configuration, measure (utility, privacy)
  4. Plot frontier

  Key question: Does the frontier hit an asymptote where you can't improve privacy without destroying utility?

  Phase 6: Cross-Dataset Validation (Tabula Muris)

  Question: Is this a VAE problem or a PBMC problem?

  Experiments:
  - Replicate all analyses on Tabula Muris
  - Confirm entanglement pattern persists
  - Confirm Fisher failure mode matches

  This strengthens the claim from "we observed this on one dataset" to "this is fundamental to VAE architecture."

  ---
  Paper Structure

  Title: "Why Machine Unlearning Fails for Variational Autoencoders: The Parameter Entanglement Problem"

  1. Introduction
    - Unlearning works for classifiers, but VAEs are different
    - Privacy concern for single-cell data motivates the work
  2. Background
    - Existing unlearning methods (Fisher, adversarial)
    - VAE architecture and training
  3. Methods Tested
    - Frozen critics, extra-gradient, Fisher
    - Evaluation protocol (MIA, matched negatives)
  4. Results: Multiple Methods Fail
    - Document each failure with metrics
    - Note the structured vs. scattered distinction
  5. Analysis: The Common Cause
    - Information localization results
    - Parameter entanglement quantification
    - Latent space geometry changes
    - Relearning vulnerability
  6. Key Finding
    - VAE parameters are fundamentally entangled
    - Generative models resist surgical unlearning
    - The decoder stores shared mappings that cannot be separated
  7. Implications
    - Existing unlearning methods designed for classifiers don't transfer
    - Future work needs architecture-aware approaches
    - Retraining remains most practical for now
  8. Cross-Dataset Validation
    - Tabula Muris confirms patterns
    - Not dataset-specific, architecture-specific

  ---
  Recommended Priority

  1. Information localization analysis (1-2 days)
    - Quick experiments, high insight value
    - Immediately tests where memorization lives
  2. Parameter entanglement quantification (1-2 days)
    - Fisher computation already in codebase
    - Directly tests the core hypothesis
  3. Relearning vulnerability (1 day)
    - Simple experiment, critical result
    - Addresses the "obfuscation vs. deletion" question
  4. Tabula Muris validation (2-3 days)
    - Strengthens generalization claims
    - Confirms findings aren't dataset artifacts
  5. Latent space geometry (1-2 days)
    - Visualization for the paper
    - Intuition-building for readers
