# Round 5 review (2026-09-08)

Full re-review of the paper, the 40 notebooks, the source under `src/` and `scripts/`, the
result JSONs, and the checkpoints on disk. Rounds 1 to 4 verified that the paper's numbers
trace to JSON files. This round verified the JSON files against the code that produced them
and the checkpoints they describe. That is where the problems are.

## Verdict

Not submittable as it stands. Three defects in the evidence base sit underneath the text and
invalidate most of the headline numbers. None of them is visible from the paper or from the
JSONs, which is why four review rounds passed them. Each is established below from files on
disk, and each has a bounded repair.

| Defect | What it breaks | Repair cost |
|---|---|---|
| B1. The retrain reference trained on 82 to 88 percent of its own matched negatives | Retrain floor, success threshold, refit control, cluster-12 reference, PBMC confound fractions, every "not separable from retrain" claim | Five clean seeds trained today (90 s each); re-score |
| B2. The baseline checkpoint was regenerated on July 4, after most method checkpoints were derived from the old one | Table 2 mixes two parents; the utility columns for SSD, SCRUB, and contrastive describe checkpoints that no longer exist | Re-derive four method families from one baseline; re-score |
| B3. Fisher scrubbing and extra-gradient, as implemented, are not the methods the paper describes | Every Fisher-scrubbing and extra-gradient number; the "posterior collapse" and "unreliable across seeds" narratives; the attack-suite table | Fix two loss functions and one training loop; re-run |

What survives is real and worth keeping. The biology-confound measurement on TCGA-BRCA and on
PBMC cluster 7 is clean. The Fisher-overlap analysis of the current baseline is internally
consistent. The SSD, SCRUB, contrastive, Moon, and DP-SGD privacy rows descend from the
current baseline with correct loss functions and stand as reported.

## B1. The retrain reference trained on its negatives

`src/retrain.py` reads only `forget_indices` from the split file, removes those cells from
the full dataset, shuffles the remaining 33,058 cells with the run seed, and trains on the
first 85 percent. The baseline's unseen split, from which the 194 matched negatives were
drawn, is therefore inside the retrain training set at the 85 percent rate. Reconstructing
the shuffle for each seed gives the exact overlap.

| Retrain seed | Train size | Matched negatives in training set | Unseen cells in training set |
|---|---|---|---|
| 42 | 28,099 | 159 / 194 | 4,030 / 4,964 |
| 43 | 28,099 | 165 / 194 | 4,200 / 4,964 |
| 44 | 28,099 | 166 / 194 | 4,252 / 4,964 |
| 45 | 28,099 | 165 / 194 | 4,201 / 4,964 |
| 46 | 28,099 | 170 / 194 | 4,173 / 4,964 |

The train size of 28,099 recorded in each `metadata.json` is 85 percent of 33,058 and
confirms the reconstruction. The forget set is correctly excluded in all five. The
cluster-12 retrains in `outputs/p2_c12/retrain_s4*` come from the same script (train size
28,083, which is 85 percent of 33,039) and share the defect. The TCGA-BRCA retrain (notebook
32), the Tabula Muris retrain (notebook 12), and the cluster-7 and cluster-28 retrains
(notebook 39) train on the retain split and are clean.

Under the canonical CPU attacker, the four PBMC reference models score as follows.

| Model | Training set | AUC | Advantage |
|---|---|---|---|
| Baseline, July 4 checkpoint | train split | 0.793 | 0.585 |
| Baseline, December checkpoint | train split | 0.785 | 0.569 |
| Retrain, notebook 3 (December, beta 1, no warm-up) | retain split | 0.506 | 0.012 |
| Retrain seed 42, `multiseed_retrain` (April) | 85 percent of all non-forget cells | 0.570 | 0.140 |
| Retrain, clean recipe-matched, 5 seeds (today) | retain split | 0.521 (per seed 0.553, 0.499, 0.522, 0.538, 0.493) | 0.049, nested 95% CI [0.042, 0.173] |

The paper's retrain floor (advantage 0.156, nested CI [0.082, 0.258]), its success
threshold of 0.258, the refit control in Table 8 (which refits on the same five leaky
seeds), the cluster-12 reference advantage of 0.64, the PBMC rows of the confound-fraction
appendix, and the sentence "the retrain reference itself sits only marginally above chance,
consistent with the biology confound" all rest on models that had seen most of the cells
they were supposed to have never seen.

The clean reference changes the reading of PBMC-33k in three ways. The success threshold
falls from 0.258 to 0.173, so every post-hoc method still fails, extra-gradient (0.281) is
now clearly above the bar rather than borderline, and DP-SGD (0.045) still sits inside it.
The clean retrain's mean advantage of 0.049 lies below the estimator's own permutation
floor (0.091), so on PBMC the matched-negative protocol leaves no residual signal in a model
that never saw the megakaryocytes; the claim that the retrain sits above chance because of
biology is withdrawn for PBMC, while the cluster-7 and TCGA-BRCA confound results, which
use clean retrains, stand. The genuine membership signal on cluster 13 is the baseline's
0.585 against 0.049, larger than the paper states.

## B2. Two baselines

`outputs/p1/baseline/best_model.pt` is dated July 4, 01:21. It was written by the audit's
re-execution of notebook 2 and differs from the July 4 snapshot copy (December 7) by a mean
absolute weight change of 7.4e-3, which is the same order as what an unlearning method
changes. Each method checkpoint sits close to exactly one of the two.

| Checkpoint | Written | Distance to December baseline | Distance to July baseline | Parent |
|---|---|---|---|---|
| Extra-gradient seeds 42, 456 | Feb 8 | 1.3e-4, 4.5e-4 | 7.4e-3 | December |
| Retain-only fine-tune seed 42 | Feb 8 | 1.0e-2 | 1.5e-2 | December |
| Gradient ascent seed 42 | Feb 8 | 7.7e-3 | 1.3e-2 | December |
| Fisher scrubbing | Dec 8 | 3.2e-3 | 8.6e-3 | December |
| Frozen critics (adversarial sweep) | Dec 9 | 4.7e-4 | 7.4e-3 | December |
| SSD seed 42 | Jul 4 | 1.4e-2 | 8.1e-3 | July |
| SCRUB seed 42 | Jul 4 | 9.8e-3 | 4.0e-3 | July |
| Contrastive seed 42 | Jul 4 | 1.1e-2 | 5.1e-3 | July |
| Moon seed 42 | Jul 5 | 1.2e-2 | 5.9e-3, encoder identical | July |
| DP-SGD, retrain seeds | from scratch | | | none |

Consequences. The canonical attacker was trained on the July baseline and applied to
checkpoints whose parent is the December baseline, so four of the nine method rows in Table
2 are scored against a model that is not their parent. The utility table draws SSD, SCRUB,
and contrastive utility from `consolidated_method_comparison.json` (February), but those
February checkpoints were overwritten on July 4 by notebooks 21 to 24, so the privacy and
utility columns of those rows describe different models. The Tabula Muris baseline and
retrain on disk were also re-trained on July 4 (baseline AUC 0.897 in
`baseline_results.json`) while the paper's Tabula Muris numbers come from the January
`unlearning_results.json` (0.891); the January method checkpoints are still on disk and no
longer have their parent. TCGA-BRCA was re-derived end to end on July 4 and is
self-consistent. Notebooks 26 to 30 analyze the July baseline and are self-consistent.

Magnitude. The two baselines score 0.785 and 0.793 under the same attacker, so the anchor
row barely moves, and the method rows are unlikely to move far. But provenance has to be one
model, and today's re-score of the July baseline gave 0.7926 against the cached 0.7912, so
the "CPU is bit-reproducible" statement in the paper should read "reproducible to about
0.01 AUC".

## B3. Two methods are not what the paper says they are

**Fisher scrubbing** (`src/train_fisher_unlearn.py`). Three departures from the text.

1. The paper says the diagonal Fisher is taken on the forget set. The code computes it on
   the retain set, in train mode with dropout active, from batch-summed gradients.
2. The scrub step is `clamp(lr * grad / F, -0.001, 0.001)` for 100 steps. With damping 0.1
   the clamp saturates for most parameters, so the update is a signed ascent of 0.001 per
   step regardless of the Fisher weighting. The recorded forget loss rises from 283 to
   218,000 during scrubbing and the post-scrub AUC is exactly 0.5, which is a destroyed
   model, not a scrubbed one (`outputs/p2/fisher_unlearn/fisher_results.json`).
3. The retain fine-tune that follows averages the reconstruction error over genes
   (`mse_loss(reduction='mean')`) while summing KL over latent dimensions. The training
   objective sums reconstruction over the 2,000 genes. The fine-tune therefore weights KL
   2,000 times more heavily than training did, which forces the posterior toward the prior.
   The "posterior collapse" reported for Fisher (KL 10.55 to 0.007, marker r 0.628, ELBO
   490) is the signature of this objective, not of Fisher overlap. Every other method
   fine-tunes with `vae_loss` and keeps KL near 10.

Every Fisher-scrubbing number inherits this (PBMC 0.808 / 0.615, TCGA within-subtype and
cross-subtype 0.500, Tabula Muris 0.946, cluster-12 marker r 0.21, the conditional-VAE scrub
in Appendix, the scattered-set table). The Fisher-overlap analysis in Section 7 uses a
separate per-sample estimator in `fisher_utils.py` and is not affected, but the paper's link
from that analysis to "this is why Fisher scrubbing collapses the posterior" is not supported
by the method as run.

**Extra-gradient co-training** (`src/train_unlearn_extragradient.py`). Four departures.

1. There is no extragradient step. The critic update is one backward pass and a plain SGD
   step at 1e-5; the Adam critic optimizer is constructed and never stepped. The name and
   the citation to Chavdarova et al. do not describe the code.
2. The critics are trained to separate forget cells from retain cells, which is cluster
   identity, not membership. The paper's min-max objective leaves the attacker's negative
   class unspecified.
3. The utility term has the same mean-over-genes reconstruction as the Fisher fine-tune, so
   its KL is over-weighted 2,000 times, and it is multiplied by lambda 10. The recorded
   privacy loss is about 0.8 per step against a utility term of about 110, so the VAE update
   is essentially a step down the KL gradient.
4. An "epoch" is one minibatch (30 forget cells, 256 retain cells) and one optimizer step.
   The paper's "50 epochs" never happened. The shadow-attacker early-stopping rule fires
   whenever a noisy AUC rises twice in a row, and the ten seeds stopped after 3, 4, 10, 10,
   11, 11, 11, 12, 18, and 19 steps. The checkpoints sit 1.3e-4 to 4.5e-4 mean absolute
   weight change from their parent, and the seed AUC tracks the step count (3 to 4 steps
   give 0.65 to 0.72; 10 to 12 steps give 0.31 to 0.37; 18 to 19 steps give 0.46 to 0.48).

The "high variance across seeds" that the paper builds the extra-gradient narrative on is
the variance of a random stopping time. The size ablation (n=50 seed 42 stopped after 3
steps) and the cluster-12 result share it.

**Attack suite** (`src/threshold_attack.py`, `src/likelihood_ratio_attack.py`). Both decode
with `model.decode(z)` and no library size, so the "reconstruction" they score is a softmax
probability vector against log-normalized counts. That is why Table 3 reports the same
reconstruction-threshold advantage for every model. With the library size restored, the
reconstruction threshold scores 0.780 on the baseline, 0.565 on the leaky retrain, and
0.501 on the clean December retrain, so reconstruction does carry membership signal and the
sentence "reconstruction quality is preserved, so the membership signal lives in the KL and
latent geometry" is reversed. The likelihood-ratio row is one statistic reported twice with
opposite sign (baseline AUC 0.08 against retrain as reference, retrain AUC 0.92 against
baseline as reference), and the k-NN row's advantage of 0.986 comes from AUCs of 0.01 to
0.05, an attack whose membership predictions are exactly inverted.

## What holds

- **The confound measurement.** TCGA-BRCA cross-subtype 0.803 / 0.830 against within-subtype
  0.580 / 0.572 (CPU re-run today; the previous run's exact tie at 0.576 was a single-run
  coincidence and is gone). PBMC cluster 7, 80 held-out cells, cross 0.665 / 0.713 against
  within 0.527 / 0.480 with clean retrains. These carry the paper's first contribution.
- **The Fisher-overlap analysis** of the July baseline. Per-sample estimator, global 0.209,
  decoder output 0.485 against the classifier's 0.018, the deep-MLP and z=8 controls, the
  conditional-VAE decomposition, and the Proposition 1 check all trace and are consistent.
- **SSD, SCRUB, contrastive, Moon, and DP-SGD privacy rows.** Correct loss, July parent,
  scored by an attacker trained on their parent.
- **Retain-only fine-tune and gradient ascent** use the correct loss and only need
  re-deriving from the July baseline.
- **The measurement and reframing plan** from the round-4 handoff (elevate the corrected
  measurement as the lead finding, cite the calibration literature, retrain as the floor
  rather than the confound) is still the right shape, and B1 strengthens it. With a clean
  floor near chance, the genuine PBMC signal is larger than the paper states and the claim
  that the retrain reference sits above chance because of biology is withdrawn for PBMC.

## Secondary findings

Text and number mismatches that survive the repairs above.

- Section 5 says all six TCGA methods converge to 0.830. Fisher scrubbing collapses to
  0.500 (`unlearning_results.json`). The other five sit at 0.829 to 0.831.
- Table 1 lists 121 TCGA matched negatives. The file has 89.
- The TCGA within-subtype numbers are now 0.580 (baseline) and 0.572 (retrain); patient
  level n=5 0.360 / 0.327, n=10 0.443 / 0.417, n=20 0.422 / 0.415.
- Table 6's effective-rank ratios do not trace. Global is 0.0126 in the JSON, not 0.010.
- Section 5 and Appendix C say separate within-cluster runs give 0.53 to 0.60. Only 0.596
  exists on disk; the 0.527 was a typed constant in notebook 40 copied from an earlier
  run that was overwritten.
- Appendix "Cluster-conditional decoder" compares the conditional VAE against a standard
  VAE advantage of 0.54 (single-seed, December). The canonical value is 0.58.
- Section 6 "Constructive approaches" quotes a within-cluster baseline of 0.63 that does
  not trace to any file.
- The reproducibility statements should name the July 4 baseline as the single parent
  once B2 is repaired, and drop "bit-reproducible".
- `.claude/rules/results-reference.md` presents single-seed MPS-era numbers as
  "Established Results" without labeling the regime.
- The round-4 handoff items (four calibration citations, retitle, positioning paragraphs,
  Corollary 2 sigma-bar definition, abstract closer) are unchanged and still open.

## What was fixed today

- `src/canonical.py` is a single loader module for every canonical number, keyed to the
  JSON its experiment wrote.
- Eleven notebooks (12, 14, 21 to 24, 30, 34b, 35, 36, 40) no longer carry typed-in
  numbers. Notebooks 14, 34b, 35, 36, and 40 were re-executed (34b pinned to CPU);
  notebook 30's two reference cells and the new reference cells in 21 to 24 were executed
  in isolation and their outputs transplanted, because re-running those notebooks whole
  would retrain models. A pre-existing MPS-to-numpy crash in notebook 35 was fixed.
- `figures/make_result_figures.py` loads every value through the loaders; the three figure
  PNGs and the confound figure were regenerated and the paper's copy of the confound figure
  (an April 9 snapshot that matched no notebook output) replaced.
- `outputs/p2/moon/utility.json` now exists (`scripts/score_moon_utility.py`), so Moon's
  ELBO 364.5 and marker r 0.831 trace.
- `scripts/retrain_clean.py` trains a retrain reference on the retain split with notebook
  2's recipe; five seeds are in `outputs/p1/retrain_clean/`, scored by
  `scripts/score_retrain_clean.py` into `outputs/p4/multiseed/persample/retrain_clean.json`.

## Repair plan

Stage 1, evidence base (about one working day of compute and scripting).

1. Adopt the July 4 baseline as the single parent. Re-derive retain-only fine-tune (5
   seeds) and gradient ascent (5 seeds) from it with the existing scripts.
2. Fix `train_fisher_unlearn.py` (forget-set per-sample Fisher as the text says, a step
   size that does not saturate the clamp, `vae_loss` for the fine-tune) and re-run three
   seeds. If the posterior still collapses, the narrative stands; if not, it goes.
3. Fix `train_unlearn_extragradient.py` (either implement the extragradient step or rename
   the method; `vae_loss` for utility; a fixed step budget with no shadow early stop;
   define an epoch as a pass over the retain set) and re-run ten seeds.
4. Fix the two attack files (pass the library size) and re-run notebook 15.
5. Re-score every checkpoint, including `retrain_clean`, with one CPU attacker
   (`save_persample_scores.py`, `nested_bootstrap_all.py`) and recompute utility for every
   current checkpoint in one pass. Regenerate tables and figures through the loaders.
6. Add a checkpoint manifest with hashes and parents, and stop re-executing training
   notebooks in audits.

Stage 2, text. Rewrite Sections 4 to 6 and the appendices from the re-scored JSON; then the
round-4 reframing (measurement as lead finding, calibration citations, retrain as floor,
title).

## Method

Every claim above was checked against a file. Weight distances are mean absolute
differences over all floating-point parameters. Overlaps were reconstructed by replaying
`set_global_seed` and the shuffle in `retrain.py` for each seed. AUCs in this document were
produced by `scripts/eval_multiseed.py` machinery pinned to CPU with the attacker seed fixed
at 42.
