# Stage 1 results and what they do to the paper (2026-09-08)

Stage 1 re-derived every method checkpoint from one baseline (July 4) with the fixed code,
replaced the leaky retrain reference with five clean retain-split retrains, re-scored
everything with one CPU attacker, recomputed utility for every checkpoint, re-ran the
Tabula Muris, TCGA-BRCA, and cluster-12 pipelines, fixed the attack suite, and added a
reference-model likelihood-ratio attack. `CLAIMS_AUDIT.md` gives the claim-by-claim
verdicts; this document gives the picture and the decision it forces.

## The picture in one table

PBMC-33k, cluster 13 (30 megakaryocytes), forget cells versus 194 matched negatives.
MLP is the canonical attacker (advantage 2|AUC - 0.5|, nested 95 percent CI). LR is the
new reference-model likelihood-ratio attack (score = negELBO under clean retrain seed 42
minus negELBO under the target; AUC near 1 means the forget cells are still fitted better
than a never-trained model would fit them, near 0 means they are now fitted worse). Utility
is held-out ELBO and marker gene correlation.

| Model | MLP AUC | MLP adv [CI] | LR AUC | ELBO | marker r |
|---|---|---|---|---|---|
| Baseline | 0.801 | 0.601 | 0.996 | 363.9 | 0.831 |
| Retrain, clean (5 seeds) | 0.529 | 0.057 [0.044, 0.182] | 0.37 (retrain vs retrain, 0.24 to 0.52) | 364.2 | 0.831 |
| Retain-only FT (5) | 0.669 | 0.339 [0.24, 0.44] | 0.991 | 363.0 | 0.832 |
| Gradient ascent (5) | 0.706 | 0.412 [0.31, 0.51] | 0.995 | 363.2 | 0.831 |
| SSD (3) | 0.701 | 0.402 [0.28, 0.53] | 0.995 | 363.6 | 0.831 |
| SCRUB (3) | 0.716 | 0.431 [0.30, 0.56] | 0.991 | 363.5 | 0.831 |
| Moon (3) | 0.740 | 0.480 [0.35, 0.60] | 0.996 | 364.5 | 0.831 |
| Fisher, fixed (3) | 0.673 | 0.345 [0.22, 0.47] | 0.991 | 363.6 | 0.832 |
| Fisher, step 3e-3 (1) | 0.312 | 0.376 | 0.126 | 363.8 | 0.831 |
| Fisher, step 1e-2 (1) | 0.445 | 0.110 | 0.256 | 370.7 | 0.830 |
| Extra-gradient lambda=10 (10) | 0.411 | 0.225 [0.15, 0.33] | 0.218 (7 seeds 0.007 to 0.15; 3 seeds 0.58 to 0.74) | 363.8 | 0.832 |
| Extra-gradient lambda=1 (10) | 0.336 | 0.327 [0.20, 0.48] | 0.049 (all 10 seeds 0.006 to 0.16) | 363.3 | 0.831 |
| Contrastive (3) | 0.170 | 0.661 [0.56, 0.75] | 0.004 | 363.7 | 0.831 |
| DP-SGD (3) | 0.466 | 0.068 [0.03, 0.19] | 0.502 | 402.9 | 0.789 |

Success threshold 0.182. Every post-hoc method's difference from retrain has a nested CI
that excludes zero (extra-gradient lambda=10: [0.015, 0.243], P = 0.987, Holm-corrected
reject). DP-SGD's does not ([-0.11, 0.11]). The refit control gives the clean retrain
advantage 0.035 [-0.03, 0.10] against a baseline positive control of 0.797.

## What the re-derived evidence says

**The membership signal on cluster 13 is large and cleanly measured.** With a reference
that never saw the negatives, the retrain sits at chance under both the transferred and the
refit attacker and at 0.37 under the reference-LR audit. The baseline sits at 0.80 and
0.99. The genuine gap is 0.54 in advantage, not 0.43, and the paper's sentence that the
retrain reference sits above chance because of biology is withdrawn for PBMC. Biology is
what the cross-class protocol measures on the other cohorts (TCGA retrain 0.830 above
baseline 0.803; Tabula Muris 0.943 above 0.903; cluster 7; cluster 12 clean retrain 0.814
against baseline 0.869), and within-subtype matching or a clean retrain removes it. That
contribution stands, and cluster 13 is now its clean positive control.

**No post-hoc method lands where retraining lands, and the failure has two modes.** The
utility-preserving methods (retain-FT, gradient ascent, SSD, SCRUB, Moon, Fisher at the
paper's step size) leave the forget cells fitted almost as well as the baseline fits them
(reference-LR 0.991 to 0.996 against the baseline's 0.996), which the MLP attacker reads as advantage 0.34
to 0.48. The methods that push harder (extra-gradient at either weight, contrastive, Fisher
at larger steps) overshoot: the forget cells end up fitted worse than a never-trained model
fits them (reference-LR 0.004 to 0.26, MLP AUC 0.17 to 0.45; frozen critics 0.003 to 0.008). Extra-gradient at lambda=10
is bimodal across seeds, seven over-correct and three under-correct, and none of its ten
seeds lands in the retrain band. Only the from-scratch models, retrain and DP-SGD, do.

**Nothing costs utility any more.** Every post-hoc method sits at baseline ELBO (363 to
364) and marker correlation (0.831 to 0.832). The paper's "posterior collapse" (Fisher) and
the 40-point ELBO cost of extra-gradient were the loss-weighting bug. Only DP-SGD pays a
utility price. The privacy-utility trade-off the paper describes among post-hoc methods
does not exist in the re-derived data; what exists is a calibration failure. Methods can
move the forget cells' treatment freely, in either direction, without disturbing the
retain cells, and none of them can aim at the retrain point because none has access to it.

**The Fisher-overlap section loses its stated role and gains a narrower one.** The
overlap measurements are correct and unchanged (0.35 to 0.49 at every layer against a
classifier's 0.02, the factorization, the controls). The section's claim that the overlap
is why parameter-space edits cannot reduce forget-set loss without proportionally
disturbing the retain set is not what the re-derived methods show: they move the forget
cells' fit by any amount asked (reference-LR from 0.99 to 0.01) with held-out ELBO
unchanged to the first decimal. The phase audit in the Stage 1b section below shows what
the overlap does cause: the edit damages the retain set along shared directions, and the
retain fine-tune that repairs it undoes the edit. That is the mechanism of the
signal-intact mode. It does not explain the over-corrected mode.

**The attack suite table is rewritten.** With the library size restored, the
reconstruction threshold is a genuine membership attack (baseline 0.78, clean retrain 0.54),
the KL and k-NN attacks are biology (they score the clean retrain at advantage 0.74 and
0.96), and the old likelihood-ratio row was one statistic reported twice with opposite
signs. The reference-LR audit replaces it and is the sharpest instrument in the paper, with
a retrain-versus-retrain control (0.24 to 0.52 across seeds) as its floor. Every likelihood is the deterministic estimator of `src/likelihood_audit.py` (eval mode, 16 seeded z samples); the unseeded single-sample forward pass the scripts first used moved a 30-cell AUC by up to 0.14 between runs.

## Stage 1b: the four re-derived families and the phase audit

Everything the paper had relegated to appendices or single seeds was re-derived from the
July baseline with the fixed code: frozen critics (four configurations, three seeds), the
structured extra-gradient size ablation (three sizes, three seeds, with per-size clean
retrains), the scattered Fisher size ablation (five sizes, ten seeds, with per-size clean
retrains), synthetic augmentation, and RAU, both against the clean retrain. A new phase
audit records what the two-phase methods do at each phase. Tables are in
`CLAIMS_AUDIT_TABLES.md`.

**The phase audit ties the proposition to the outcome.** Negative ELBO per cell (CPU run,
deterministic estimator), where the clean retrain puts the forget cells at 334 to 337, the
level of never-seen matched cells (333), and the baseline puts them at 270.

| Stage | forget | retain | held-out | MLP AUC | reference-LR |
|---|---|---|---|---|---|
| Baseline | 270 | 363 | 363 | 0.793 | 0.996 |
| Fisher step 1e-3, after scrub | 5,569 | 1,250 | 1,284 | 0.016 | 0.015 |
| Fisher step 1e-3, after fine-tune | 308 | 362 | 362 | 0.666 | 0.987 |
| Fisher step 3e-3, after fine-tune | 377 | 363 | 363 | 0.386 | 0.173 |
| Fisher step 1e-2, after fine-tune | 723 | 370 | 369 | 0.376 | 0.155 |
| Gradient ascent, after ascent | 270 | 363 | 363 | 0.802 | 0.996 |
| Gradient ascent, after fine-tune | 296 | 362 | 362 | 0.716 | 0.996 |

The scrub is not selective: it raises the forget loss 21-fold and the retain and held-out
losses 3.4-fold, which is the shared damage Proposition 1 predicts. The retain fine-tune
that repairs the retain set restores the forget set with it, undoing 99 percent of the
displacement through the shared parameters (5,569 back to 308 against a target of 334 to
337). At larger steps the fine-tune restores the retain set as before but cannot bring the
forget set back, and it ends over-corrected (the step 1e-2 endpoint is chaotic: 518 and
1,076 in two MPS runs, 723 on CPU, over-corrected in all three). The gradient-ascent
phase at the paper's setting does nothing (269.5 before and after); what that method
reports is the retain fine-tune's own drift, which on its own moves the forget cells from
270 to 296 with the audit still at 0.996. So the sharper version of
the proposition holds and is the mechanism of the "signal intact" mode: entangled
parameters do not block the edit, they make the retain fine-tune undo it. Where a method
lands on the axis from 270 to 333 and beyond is set by how hard it pushed before the
repair, and nothing in any method encodes 333.

**Frozen critics flip direction.** With the corrected objective the outcome is AUC 0.003
to 0.008 across all four configurations and three seeds (the paper had 0.992), reference-LR
0.002 to 0.007. The forget cells become the least member-like cells in the data. The
advantage is the same 0.99 in both versions; the direction is opposite, which is the
direction-agnostic metric's blind spot once more. Single-critic runs keep baseline
utility; the three-critic runs are the only post-hoc runs in the study with a utility
cost (ELBO 403 to 406, marker r intact).

**Structured size ablation, with the right reference.** The paper compared every size to
the n=30 retrain CI. With per-size clean retrains, the reference itself moves: 0.454 at
n=10, 0.565 at n=50, and 0.668 at n=100, so the multi-cluster forget sets are
confound-affected like cluster 12. Extra-gradient over-corrects at every size (AUC 0.326,
0.374, 0.411; reference-LR 0.19, 0.12, 0.13) and its n=100 advantage (0.178) sits below
the per-size retrain's (0.335) only because it has crossed to the other side. The old
reading of "over-unlearns at n=10, under-unlearns at n=100" does not survive.

**Scattered size ablation, with the right reference.** At every size from 10 to 500 the
baseline, the clean retrain, and Fisher coincide within 0.02 AUC (n=30: 0.588, 0.581,
0.585; n=500: 0.517, 0.510, 0.515). Under the reference-LR the baseline shows per-cell
memorization of scattered cells (0.69 to 0.79 against a retrain floor of 0.50 to 0.53 for n
at least 30) that Fisher reduces only partly (0.44 to 0.65) and that the attacker cannot
see. "Fisher works on scattered sets" becomes "the attacker finds nothing on scattered
sets, and what the audit finds Fisher only partly removes".

**RAU: a posterior target is not a likelihood target.** RAU matches the retrain's
posterior on the forget cells to within 0.03 to 0.23 nats (from 180) at every weight, yet its
cross-cluster AUC is 0.28 to 0.35 against the retrain's 0.548 and its reference-LR is 0.007
to 0.018. The forget cells' latents sit where the retrain's do; their likelihood does not.
RAU is the one method that was handed a target, and it still over-corrects because the
target was stated in the wrong quantity. On TCGA it is a no-op, as before.

**Augmentation: the instruments measure subpopulation knowledge.** Models trained on
retain cells plus synthetic megakaryocytes bootstrapped from 5 unseen cells, and never on
the 30 forget cells, score 0.744 and 0.795 under the MLP attacker (baseline 0.803, retrain
0.551) and 0.985 and 0.984 under the reference-LR. Against the 5 seed cells they are
perfectly separable (1.000 for the 100-cell version). A model that never saw the forget
cells is scored as if it had memorized them, because it learned megakaryocytes. For a
forget set that is the entire training subpopulation, membership of the individuals and
knowledge of the subpopulation are one quantity, and no membership evaluation, including
the reference audit, can separate them. That is the paper's own confound thesis stated
at the level of the individual, and it belongs in the measurement section as the stated
limit of what "unlearning a subpopulation" can mean.

## Stage 1c: the likelihood-targeted test (NB41, 2026-09-08)

The option-3 experiment was run. Likelihood-targeted unlearning fine-tunes on the retain set with a
two-sided penalty on the gap between each forget cell's negative ELBO and a target, ten seeds per
target, the deterministic estimator throughout (`outputs/p7/targeted_unlearning.json`).

| Target | forget negELBO | MLP AUC | advantage [CI] | audit | ELBO / marker r |
|---|---|---|---|---|---|
| each forget cell's 10 nearest evaluation negatives | 309 | 0.75 | 0.503 | 0.62 | 364.7 / 0.830 |
| mean over the 194 evaluation negatives (335, the retrain's level here) | 337 | 0.58 | 0.166 [0.111, 0.212] | 0.41 | 365.0 / 0.830 |
| 88 held-out look-alikes the evaluation never uses (ranks 11 to 30) | 296 | 0.83 | 0.668 [0.630, 0.700] | 0.70 | 364.7 / 0.830 |

Given the retrain's level, a post-hoc method lands inside the audit's retrain band (0.24 to 0.52)
and brings the attacker's advantage to the threshold (0.182; six of ten seeds at or below it), at
baseline utility. So the calibration failure is the whole of the second mode: aim the likelihood at
the right level and most of the attacker's signal goes with it (the residual 0.05 AUC above the
retrain is the latent-geometry remainder). But that level is not observable. The clean retrain fits
the forget megakaryocytes at 334 to 342 and their nearest never-seen neighbours at 292 to 301, the
same as the baseline does, because it knows the neighbours' types and not the forget cells' type;
calibrating on look-alikes therefore lands 40 nats short, back at the baseline's treatment. The
cohort-average target worked only because the 194 evaluation negatives happen to average to the
retrain's level on this cohort, and it read that level off the evaluation's own negatives, so its
result is an oracle reading. "Nowhere to aim" becomes "the target exists and cannot be read off any
cell the model knows", which is the individual-level form of the subpopulation-knowledge limit (A8).

## The decision this forces

The spine question survives with a cleaner answer. Can a gene-expression VAE selectively
forget a rare subpopulation without retraining? No. Not because the methods destroy the
model, but because post-hoc methods cannot aim at the point retraining reaches. On the
axis of the forget cells' negative ELBO, the baseline sits at 270, the retrain at 333, and
every post-hoc method lands short of 333 or past it, at no cost to the rest of the model.
The two modes now each have a mechanism. Methods that edit and then fine-tune on the
retain set land short because the fine-tune undoes the edit through shared parameters,
which is Proposition 1's implication observed directly. Methods that push with a
continuous forget-set objective (critics, contrastive, alignment, Fisher at larger steps)
land past because nothing in the objective encodes the retrain level, and RAU shows that
even a target stated as a posterior does not calibrate the likelihood. Only training from
scratch (retrain, DP-SGD) lands on the target.

The measurement contribution is the strongest part of the paper and gets stronger: the
clean-retrain reference, within-subtype matching, the confound on four cohorts, and the
reference audit as the per-sample instrument, with the augmentation result as the honest
statement of its limit. The Fisher-overlap section keeps a real role, narrower than
before: it is the mechanism of the signal-intact mode, with the phase audit as its
empirical test, and it no longer claims to explain the over-corrected mode.

Three ways to shape that, from least to most change.

1. **Keep the structure, rewrite the middle.** Sections 4 to 6 from the new numbers, the
   attack table replaced by the reference audit with corrected thresholds, the two-mode
   description replacing collapse-or-Streisand, and Section 7 rescoped to the entangled-
   repair mechanism with the phase audit as its evidence.

2. **Promote the measurement and the calibration finding.** Lead with the corrected
   measurement as the round-4 plan intended, make the clean-retrain protocol and the
   reference audit the main instrument, present the calibration failure and its two
   mechanisms as the unlearning finding, and keep Section 7 as the mechanism section it
   now is. This is the version the evidence supports best.

3. **Rebuild around calibration.** Treat the missing target as the hypothesis and test
   whether supplying it in the right quantity fixes a method (a matched-negative critic,
   or early stopping on the forget cells' likelihood against a held-out reference). RAU is
   the first data point and a negative one; a likelihood-targeted variant would be the
   test. New experimental work, roughly a week.

Recommendation: option 2, with the likelihood-targeted test (Stage 1c) as the experiment
that turns "nowhere to aim" from a diagnosis into a tested claim, oracle caveat included.

## Questions for the author

- Section 7 as the mechanism section (entangled repair, phase audit) rather than as the
  explanation of all failure: agreed?
- The reference audit as a main instrument, presented as an auditor's tool with the
  augmentation result as its stated limit: agreed?
- Extra-gradient: report both weights, or lambda=10 as the paper's regime with lambda=1
  as an ablation?
- The likelihood-targeted variant (option 3): run (Stage 1c above). Present it as the test of the second mode, with the oracle caveat on the cohort-average target?

## Residuals

- Reference audit for Tabula Muris and TCGA-BRCA (`scripts/attack_lr_reference_multi.py`,
  `outputs/attack_lr_reference_multi.json`, single retrain as reference, no floor): Tabula
  Muris baseline 0.877, extra-gradient 0.947, Fisher 0.346; TCGA-BRCA baseline 0.833, Fisher
  0.191, retain-FT 0.457, gradient ascent 0.397, SSD 0.454, SCRUB 0.533, contrastive 0.519.
  On TCGA the five methods that the MLP attacker cannot tell from the retrain (0.83 cross-
  subtype) sit at or below the retrain's own fit of the Basal patients; only the baseline is
  above it, so the retrain-level MLP score there is not the forget patients being fitted like
  never-seen ones, it is the cross-subtype confound. Tabula Muris extra-gradient (0.947 above
  the baseline's 0.877) fits the forget cells better than the baseline does while its MLP AUC
  falls to 0.723; that pair needs a look before the rewrite quotes it.
- The rules files, skills, both `CLAUDE.md` files, `README.md`, and `index.md` were swept to
  these numbers on 2026-09-08; `Writeup.md` carries a superseded banner until the rewrite.
- Number hygiene: the numbers in this document were re-checked against the JSONs on
  2026-09-08 (the canonical attacker pass is the one of 09:07, in which the baseline scores
  0.801; the earlier pass gave 0.792, the instance drift of A6).

## Files

Stage 1b adds `outputs/p2/frozen_critics/`, `outputs/p4/size_ablations_v2.json`, `outputs/p4/extragradient_size/{n*,retrain_clean}`, `outputs/p4/size_ablation/{fisher_v2,retrain_clean}`, `outputs/p6/phase_audit.json`, `outputs/rau/rau_results.json`, `outputs/augmentation/augmentation_results.json`, and `CLAIMS_AUDIT_TABLES.md`.

`outputs/p4/multiseed/nested_ci_all.json`, `persample/*.json`, `utility_all.json`,
`retrain_refit.json`, `outputs/attack_lr_reference.json`, `outputs/attack_suite_results.json`,
`outputs/p1_c12/scores_c12_v2.json`, `outputs/tabula_muris/unlearning_results.json`,
`outputs/tcga_brca/unlearning/unlearning_results.json`,
`outputs/tcga_brca/within_subtype_eval/within_subtype_results.json`,
`outputs/checkpoint_manifest.json`, `outputs/claims_audit.json`; superseded inputs in
`outputs/_superseded_20260908/`.
