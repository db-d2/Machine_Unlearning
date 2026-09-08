# Paper Reframing Plan: Fishing for Privacy

## The problem
The writeup mixes GDPR-style deletion, adversarial MIA, and empirical benchmarking without committing to any one framing.

## The through-line (one sentence)
*Prior MIA evaluation on biological generative models is measuring a mix of biology and membership; when you separate them, the membership signal is small, and it is structurally unreachable by the post-hoc unlearning methods the field has proposed.*

## Two headline questions
1. **Q1 (methodological).** When prior work reports MIA advantage on biological generative models, how much is actual membership memorization vs. biological subtype identity?
2. **Q2 (structural).** Can post-hoc unlearning reduce the residual signal on structured forget sets — and if not, is that a defect of specific methods or a property of the model class?

## What to cut from the current framing
- "Right to be forgotten" / GDPR language in §1
- Any implication that biological VAEs have an urgent individual-level memorization problem
- The implicit claim that this paper proposes a working unlearning method

## What to promote
- Biology confound as a positive methodological contribution, not a caveat
- Multi-seed retrain reference + within-subtype matching + attack diversity as a *corrected evaluation protocol*
- Synthetic augmentation leakage as evidence FOR Q1 (generative processes leak sample-level info through the synthetic intermediary)
- Proposition 1 + Corollary 2 as the structural spine

---

## The rewrite (section by section)

No new experiments. No new figures. No new numbers. Reorganization and reframing of existing content.

### Abstract
Three beats:
1. MIA evaluation on biological generative models is confounded by biology; prior work overestimates memorization by 30–90%.
2. Under corrected evaluation, eight post-hoc methods + two constructive approaches all fail. Synthetic augmentation leaks through the generative process.
3. Fisher overlap in shared decoders explains why. The gap scales as 1 − O(M/D) for generative models vs. 1/√C for classifiers.

### §1 Introduction
Lead with the questions, not the threat model:
- Context: MIA + unlearning have been applied to biological generative models; results have been reported; we ask whether the evaluations are measuring what they claim.
- Q1 and Q2 stated as questions.
- Summary of answers.
- Contributions list reordered: corrected protocol first, empirical re-evaluation second, synthetic leakage third, structural theory fourth.

### §2 Background and Threat Model (NEW — short)
Define MIA, define unlearning, define the threat model (curious attacker with query access trying to determine training membership). Be explicit: this paper does not address poisoning, adversarial extraction, or reconstruction attacks.

### §3 Datasets and Evaluation Protocol
Merge current datasets section with protocol material currently scattered across methods and experiments. Present the corrected evaluation protocol as a named deliverable.

### §4 The Biology Confound (Q1) — PROMOTED from current §4.4
Dedicated section:
- Cross-subtype inflation on PBMC, TCGA-BRCA, Tabula Muris
- Multi-seed retrain correction (single-seed underestimates by ~3×)
- Confound fraction CIs
- Larger-cluster validation (NB39)
- **Synthetic data leakage subsection.** Augmented model (forget set excluded from training) shows within-subtype AUC 0.82–1.00. The generative process imprints seed-cell characteristics. Evidence that even without direct training, the model leaks through the synthetic intermediary.

### §5 Unlearning Under Corrected Evaluation (Q2 empirical)
Current method results merged. Eight-method sweep + constructive approaches + utility. Summary table with Welch t-test results and Cohen's d.

### §6 Structural Explanation (Q2 theoretical)
Current Fisher section, tightened. 17× gap → Proposition 1 → Corollary 2 → verification → controls.

### §7 Discussion
- Three failure modes (keep)
- Scope of claims: what the paper does and does not show
- Connection to Ozturk/Golob
- Practical recommendations

### §8 Conclusion
Q1 answer. Q2 answer. No new claims.

---

## Timeline
- **Days 1-2:** Rewrite intro, abstract, conclusion, section reorder
- **Day 3:** Professor review of the reframing
