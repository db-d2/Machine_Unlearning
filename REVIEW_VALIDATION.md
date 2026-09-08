# Validation of critical review claims

Investigation method: read all 13 src/ modules, 5 key notebooks (NB02, NB03, NB15, NB25, NB26), Writeup.tex, and references.bib. Grepped the full codebase for loss functions, split handling, and evaluation pipelines.

---

## Claim 1: "The model uses MSE loss despite claiming negative binomial likelihood"

**Verdict: VALID. This is real and pervasive.**

The paper states in Section 2.1 (line 96): "The VAE follows scVI design principles with a negative binomial likelihood for count data overdispersion." It says the decoder maps to "negative binomial parameters."

The actual implementation in `src/vae.py` (lines 208-229):

```python
def nb_loss(x, mean, dispersion, eps=1e-8, per_cell=False):
    """Negative Binomial negative log-likelihood (simplified MSE for log-normalized data)."""
    mse = F.mse_loss(mean, x, reduction='none')
```

The dispersion parameter is computed by the decoder but never used in the loss. Every reconstruction loss call in the codebase uses `F.mse_loss`:
- `src/vae.py` nb_loss (training)
- `src/train_fisher_unlearn.py` line 90 (Fisher scrubbing)
- `notebooks/26_fisher_overlap_analysis.ipynb` cell 5 (Fisher diagonal computation)
- `notebooks/04_adversarial_unlearning.ipynb` (extra-gradient training)
- `notebooks/30_proposition_test_and_cvae_scrub.ipynb` (proposition verification)

The paper even references "the negative binomial likelihood couples output dimensions" in Section 6 (line 392) as though it's a feature of the actual implementation. In reality, MSE does not couple output dimensions.

**Mitigation:** MSE on log-normalized counts is a defensible modeling choice (it's equivalent to Gaussian likelihood on log-transformed data, which is standard). The docstring acknowledges this: "simplified MSE for log-normalized data." The problem is the paper's description, not the model itself. The decoder does compute dispersion through a separate head (`fc_dispersion`), suggesting the architecture was designed for NB but the loss was simplified. This should be disclosed.

**Impact on results:** Likely small. MSE on log1p-normalized data is well-behaved for scRNA-seq. But it means the Fisher analysis in Section 6 is computing Fisher under MSE, not NB, which actually simplifies the theory (Proposition 1 assumes MSE). The paper's line about "NB likelihood couples output dimensions, departing from the linear-MSE assumptions" is misleading since the implementation IS MSE.

---

## Claim 2: "Preprocessing (normalize_total + log1p) is inconsistent with NB"

**Verdict: VALID, but moot given Claim 1.**

`src/data.py` lines 112-115 apply `sc.pp.normalize_total(target_sum=1e4)` then `sc.pp.log1p()`. This transforms raw UMI counts into log-normalized continuous values. A true NB model would operate on raw counts (as scVI does). But since the actual loss is MSE (Claim 1), this preprocessing is consistent with the actual loss function. The inconsistency is between the paper's NB claim and the preprocessing, not between the preprocessing and the code.

---

## Claim 3: "Unseen indices overlap with retain_test, confounding evaluation"

**Verdict: INVALID for the canonical pipeline. Valid only for an unused early script.**

The critical review likely found this in `src/train_attacker.py` line 172-173:
```python
unseen_indices = retain_test_indices  # will overlap with retain_test, which is fine
```

But this script is NOT used in the paper's results. The canonical pipeline works differently:

- **NB02** creates and saves `split_structured.json` with separate `unseen_indices` (4,964 held-out cells that were never in training)
- **NB03** loads from this file and uses k-NN on unseen cells to find 194 matched negatives, saved to `outputs/p1.5/s1_matched_negatives.json`
- **NB25** (consolidated evaluation) loads these persisted matched negatives and trains fresh attackers on F (30 cells) vs matched neg (194 cells) from the baseline model
- `src/attacker_eval.py` `matched_negative_evaluation()` is the canonical evaluation function

The unseen set is completely separate from the retain set. The matched negatives are a k-NN subset of the unseen set selected to control for biological confounds.

---

## Claim 4: "Data splits are not persisted, so experiments may use different splits"

**Verdict: INVALID for the canonical pipeline. Valid only for `src/train.py` standalone.**

`src/train.py` does create a random 85/15 split without saving indices. But this is an early standalone script. The actual pipeline:

1. NB01 preprocesses data and saves `data/adata_processed.h5ad`
2. NB02 creates structured/scattered splits with `seed=42`, saves to `outputs/p1/split_structured.json` and `split_scattered.json`
3. All downstream notebooks (NB03-NB30) and scripts load from these persisted JSON files
4. `src/train_fisher_unlearn.py` accepts a `split_path` argument and loads from the JSON

Split files contain explicit index lists (`forget_indices`, `retain_indices`, `unseen_indices`) ensuring exact reproducibility.

---

## Claim 5: "Causal language overstates what the evidence shows"

**Verdict: PARTIALLY VALID -- worth reviewing but not a fatal flaw.**

The paper uses phrases like "Fisher overlap explains why parameter-space methods fail." The Fisher analysis (Section 6) demonstrates a strong correlation: VAE cosine 0.306 vs classifier 0.018 (17x), with controlled experiments (deep MLP, z=8, conditional VAE) ruling out confounds. This is a structural argument backed by multiple controls, not just a correlation.

That said, the paper doesn't prove that overlap is the sole cause of failure. Other factors (optimization landscape, gradient noise, sample size) could contribute. The 11% error in Proposition 1's prediction (0.41 predicted vs 0.37 measured) and the fact that the conditional VAE experiment didn't improve scrubbing (despite reducing fc_mean cluster-column cosine to ~0) suggest the story is more nuanced than "overlap causes failure."

The paper does acknowledge some of this. Section 6 notes that "encoder overlap dominates because 64/69 MIA features are mu+logvar" and that the conditional decoder's shared hidden representation still has cosine 0.222. But the framing could be more cautious.

---

## Claim 6: "Statistical power is insufficient (30 forget cells, 194 matched negatives)"

**Verdict: PARTIALLY VALID.**

30 forget cells is small. The bootstrap CIs on advantage are wide (retrain CI: [~0, 0.266]). The paper does use bootstrap CIs and reports them. 10 seeds for extra-gradient helps, but some methods only have 3 seeds.

However: 30 cells IS the entire cluster 13. You can't make it bigger without changing the biological question. The paper also tests at sizes 10, 30, 50, 100, 500 (NB09 ablation), and a second dataset (Tabula Muris, cluster 33 = 82 cells). The small sample size is a real limitation but it's inherent to the problem (rare cell populations are small), and the paper handles it about as well as you can with bootstrap CIs and multi-seed evaluation.

---

## Claim 7: "Tabula Muris results may be confounded by dataset differences"

**Verdict: PARTIALLY VALID -- acknowledged in the paper.**

The paper notes (NB19, Appendix C) that matched negatives from different clusters inflate baseline AUC. The within-cluster holdout drops baseline from 0.769 to 0.527. Tabula Muris has even stronger biological structure (12 tissues, 35 clusters), so the confound is likely stronger there.

The paper does acknowledge this: "higher structured memorization than PBMC" is partly biological separability. The relative method rankings hold across the within-cluster test (Appendix C), which is the relevant point for the paper's conclusion that all methods fail.

---

## Claim 8: "Attack features detect distribution shift, not memorization"

**Verdict: PARTIALLY VALID and explicitly discussed in the paper.**

NB15 found that k-NN and likelihood-ratio attacks achieve advantage >= 0.73 even on the retrain model. The paper reports this in Table 4 and Section 5.3, calling them "high-assumption attacks" that "pick up cell-type identity rather than membership." The paper deliberately separates low-assumption attacks (threshold on loss components) from high-assumption attacks, and uses the MLP attacker as the canonical metric.

The MLP attacker trains on F vs matched negatives (cells selected to be biologically similar via k-NN). This design controls for distribution shift to the extent possible without same-cluster holdout. NB19 tests the within-cluster holdout and shows the MLP still works (baseline drops from 0.769 to 0.527 but doesn't hit 0.5).

The residual signal (0.527 vs chance 0.5) after within-cluster matching suggests some genuine memorization signal, but it's smaller than the paper's headline numbers imply. This is a real limitation that the paper acknowledges but could emphasize more.

---

## Claim 9: "The success criterion (advantage <= retrain CI upper bound) is arbitrary"

**Verdict: PARTIALLY VALID but standard in the literature.**

The criterion "unlearning succeeds iff worst-case advantage falls within the retrain model's 95% CI" follows Hayes et al. 2024 and is the standard in the unlearning evaluation literature. It's not arbitrary -- it asks whether the unlearned model is statistically distinguishable from a model that never saw the data. You could argue for tighter or looser criteria, but this is the accepted benchmark.

---

## Claim 10: "Method naming is misleading (e.g., 'Fisher scrubbing')"

**Verdict: MINOR -- names follow their source papers.**

"Fisher scrubbing" comes from Golatkar et al. 2020. "SSD" from Foster et al. 2024. "SCRUB" from Kurmanji et al. 2023. The paper uses standard names from the literature. Whether those names are good is a question for their original authors.

---

## Claim 11: "Hyperparameter tuning is unfair across methods"

**Verdict: PARTIALLY VALID.**

Extra-gradient gets a lambda sweep (5, 10, 15, 20) and 10 seeds. Fisher gets 3 seeds. SSD, contrastive, SCRUB get 3 seeds each. DP-SGD gets 3 seeds. The effort is not equal.

However, the paper's conclusion is negative (everything fails), so unfair tuning works against the paper's thesis. If anything, more tuning for other methods would strengthen the negative result. The one method that got the most tuning (extra-gradient) still fails on multi-seed evaluation. For a "nothing works" paper, extensive tuning of each method isn't strictly necessary -- you'd need to show at least one succeeding.

---

## Summary

| Claim | Verdict | Severity |
|-------|---------|----------|
| 1. NB vs MSE | **VALID** | High -- paper misrepresents the loss function |
| 2. Preprocessing vs NB | **VALID** (moot given #1) | Medium |
| 3. Unseen == retain_test | **INVALID** for canonical pipeline | N/A |
| 4. Splits not persisted | **INVALID** for canonical pipeline | N/A |
| 5. Causal language | **PARTIALLY VALID** | Medium |
| 6. Statistical power | **PARTIALLY VALID** (inherent limitation) | Medium |
| 7. Tabula Muris confound | **PARTIALLY VALID** (acknowledged) | Low |
| 8. Distribution shift vs memorization | **PARTIALLY VALID** (discussed) | Medium |
| 9. Arbitrary criterion | **PARTIALLY VALID** (standard) | Low |
| 10. Method naming | **MINOR** | Low |
| 11. Tuning fairness | **PARTIALLY VALID** (works against thesis) | Low |

## The one thing to fix

**Claim 1 is the only claim that requires a code or paper change.** The paper should either:

(a) Describe the loss as Gaussian / MSE on log-normalized data (which is what it actually is), or
(b) Implement actual NB loss and retrain

Option (a) is simpler and honest. The model IS a VAE with Gaussian likelihood on log-normalized counts. The decoder architecture has an NB-style parameterization (mean + dispersion heads), but the dispersion head is unused in the loss. Calling this "negative binomial likelihood" is inaccurate. The Section 6 statement about "the negative binomial likelihood couples output dimensions, departing from the linear-MSE assumptions" should be revised since the implementation IS linear-MSE.

Everything else in the critical review either targets non-canonical code paths (claims 3-4), identifies real but acknowledged limitations (claims 7-8), or flags issues that don't substantially affect the conclusions (claims 9-11).
