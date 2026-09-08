Your question of interest

  You trained a VAE on gene expression data. The VAE compresses a cell's 2,000-gene expression profile into 32 latent dimensions, then reconstructs it. Its real output is the latent space — that's
  what downstream users use for clustering, cell-type annotation, trajectory analysis. The question: if someone's data needs to be removed from this model, can you do it without retraining, and can
  you verify it actually worked?

  What you found

  The evaluation is broken first. Standard MIA on biological data confuses "this is a megakaryocyte" with "this was in training." Controlling for cell type (within-subtype matching) drops the
  apparent signal by 30–90%. On TCGA-BRCA, there's zero detectable patient-level memorization once you control for Basal-subtype identity.

  After fixing the evaluation, genuine memorization exists on rare subpopulations and nothing removes it. Eight post-hoc methods fail in three ways: do nothing (4 methods), make it worse (2
  methods), or work unreliably on one dataset (1 method). Fisher overlap in the shared decoder explains why — the forget set's influence lives in the same parameters as everything else.

  How synthetic data failed

  You generated synthetic megakaryocytes from 5 unseen seeds and trained a VAE on retain + synthetic (forget set excluded). The model never saw the forget cells. But within-subtype MIA AUC was
  0.82–1.00 — worse than baseline. The synthetic generation process imprinted the 5 seeds' specific characteristics into the model's representation. The leak isn't "the forget cells were in
  training" (they weren't). The leak is "the generative process that created the training data was shaped by specific real cells, and the model's behavior reveals which ones." You replaced
  memorization of 30 cells with memorization of 5 seeds.

  How MIA succeeded / failed

  Succeeded: Detecting genuine memorization in rare subpopulations after within-subtype correction. Detecting synthetic data leakage. Detecting Streisand effects from unlearning attempts.

  Failed: On any evaluation without subtype controls. On Tabula Muris entirely (retrain scores higher than baseline — the attacker is measuring biology, period). High-assumption attacks (k-NN,
  likelihood ratio) detect biology regardless of training history and are useless for privacy evaluation.

  One sentence for a peer

  "Standard privacy evaluation on gene-expression VAEs overestimates memorization by 30–90% because it conflates cell-type identity with training membership; after correcting for this, genuine
  memorization exists only in rare subpopulations, no approximate method removes it, synthetic augmentation leaks through the generative process, and the structural cause is Fisher overlap in the
  shared decoder."
