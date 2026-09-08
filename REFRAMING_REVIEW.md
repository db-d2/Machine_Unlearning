# Review of REFRAMING_PLAN.md

## Overall verdict

The Q1+Q2 framing is exactly right. It matches what the data actually supports and unifies the two findings (biology confound + unlearning failure) under one coherent story. The through-line sentence is strong. The rewrite is a writing task, not an experiment task.

---

## The Q1+Q2 framing works because it's what the data says

The paper has two clear findings that belong together:
1. MIA evaluation on biological data is confounded (Q1 — methodology finding)
2. Unlearning fails even after controlling for the confound (Q2 — empirical + theoretical finding)

The current paper buries #1 in a subsection (§4.4) and leads with #2. The reframe promotes #1 to the headline, which is correct because:
- It's the most novel finding (nobody else in scRNA privacy has quantified the confound)
- It's the most actionable (within-subtype matching is immediately usable by other groups)
- It makes #2 a CONSEQUENCE of honest evaluation rather than a standalone negative result

## Synthetic augmentation leakage deserves its own subsection

The plan says "promote" but doesn't give it much space. From our session's analysis: the within-subtype AUC of 0.82–1.00 on the augmented model is direct evidence that generative processes leak sample-level information even when the forget set is excluded from training. This connects to Ozturk's ρ=0.92 in a way nobody else has demonstrated empirically. It belongs as a subsection within §4 (evidence for Q1), not as a throwaway in §5.

## The new §2 (Background and Threat Model) is the most important structural addition

The current paper has NO formal threat model section. The explicit scope statement ("this paper does not address poisoning, adversarial extraction, or reconstruction attacks") directly addresses the framing confusion that prompted this reframing exercise.

## Promoting biology confound to its own §4 is the right call

This IS the paper's most novel finding. It should not be a subsection of "Experiments."

## What carries forward from this session

All of it:
- Multi-seed retrain (5 seeds, canonical protocol, patched retrain.py)
- Nested bootstrap + Welch's t-test + Holm-Bonferroni (src/stats_utils.py)
- Confound fraction CIs with validity flags (NB40)
- NB39 larger-cluster evaluation (PBMC c7, TM c28)
- TCGA overclaim fixed ("no baseline-vs-retrain gap")
- TM reframed as confound demonstration
- All retrain numbers: advantage 0.148, nested CI [0.070, 0.229]
- Writeup.tex language from Phases A/D will be overwritten by the rewrite, but the numbers and statistical methodology are permanent

## The mia-evaluation skill needs updating

The success criterion still says "Retrain advantage: 0.046, CI: [0.004, 0.266]." Should be 0.148 and [0.070, 0.229], with Welch t-test instead of CI exclusion.

## Bottom line

The plan is focused and achievable. 2-3 days of writing. No new experiments needed for a coherent, submittable paper. The content is all there — it just needs the narrative wrapper to match what the data actually says.

---

## Notes for explaining this paper to a peer

### The setup
A VAE trained on gene expression data. It compresses a cell's 2,000-gene expression profile into 32 latent dimensions, then reconstructs it. The real output is the latent space — used for clustering, cell-type annotation, trajectory analysis. The VAE is a function from high-dimensional noisy expression to low-dimensional biological structure.

### The question
If someone's data needs to be removed from this model, can you do it without retraining, and can you verify it worked?

### What was found
1. **The standard way to measure privacy leakage on biological data is broken.** MIA confuses cell-type identity with training membership. Within-subtype matching drops the apparent signal by 30–90%. On TCGA-BRCA, there's zero detectable patient-level memorization once you control for Basal-subtype identity.

2. **Even after fixing the evaluation, genuine memorization exists on rare subpopulations and nothing removes it.** Eight post-hoc methods fail in three ways: do nothing (4 methods), make it worse (2 methods), or work unreliably on one dataset (1 method). Two constructive approaches also fail.

3. **Fisher overlap in the shared decoder explains why.** The forget set's influence lives in the same parameters as everything else. The 17× gap between VAE (0.306) and classifier (0.018) is structural.

### How synthetic data failed
Generated synthetic megakaryocytes from 5 unseen seeds, trained a VAE on retain + synthetic (forget set excluded). The model never saw the forget cells. But within-subtype AUC was 0.82–1.00 — worse than baseline. The generative process imprinted the seeds' characteristics into the model. Replaced one memorization pattern with another. The leak is through the generative process itself.

### How MIA succeeded and failed
**Succeeded:** Detecting genuine memorization in rare subpopulations after within-subtype correction. Detecting synthetic data leakage. Detecting Streisand effects from unlearning attempts.

**Failed:** On any evaluation without subtype controls. On Tabula Muris entirely (retrain AUC > baseline — measuring biology, not membership). High-assumption attacks (k-NN, likelihood ratio) detect biology regardless of training history.

### One sentence for a peer
"Standard privacy evaluation on gene-expression VAEs overestimates memorization by 30–90% because it conflates cell-type identity with training membership; after correcting for this, genuine memorization exists only in rare subpopulations, no approximate method removes it, synthetic augmentation leaks through the generative process, and the structural cause is Fisher overlap in the shared decoder."
