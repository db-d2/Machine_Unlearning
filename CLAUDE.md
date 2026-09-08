# Machine Unlearning for Gene-Expression VAEs

## Project Overview

Empirical study of machine unlearning for VAEs trained on gene-expression data (scRNA-seq: PBMC-33k, Tabula Muris; bulk RNA-seq: TCGA-BRCA). Paper working title "Fishing for Privacy" (`Writeup.tex`, TMLR target), being rewritten after the Stage 1 re-derivation of 2026-09-08. Resume doc: `HANDOFF_PEER_REVIEW.md`.

What the re-derived evidence says (picture in `STAGE1_ANALYSIS.md`, claim-by-claim verdicts in `CLAIMS_AUDIT.md`):
- On PBMC cluster 13 the clean retrain sits at chance (advantage 0.057, AUC 0.529) and the baseline at 0.601 (AUC 0.801), so the membership signal is real and large. On Tabula Muris, TCGA-BRCA, PBMC cluster 12, and PBMC cluster 7 the retrain matches or exceeds the baseline, so what the cross-class protocol measures there is biology; within-subtype matching or a clean retrain removes it.
- No post-hoc method reaches the retrain band. Two failure modes, neither with a utility cost. Edit-then-fine-tune methods (retain-FT, gradient ascent, SSD, SCRUB, Moon, Fisher) leave the forget cells fitted as well as the baseline does (reference-LR 0.99 against the baseline's 0.996). Continuous forget-objective methods (extra-gradient, contrastive, frozen critics, RAU, Fisher at larger steps) push them past the retrain level (reference-LR 0.004 to 0.26; retrain-versus-retrain floor 0.24 to 0.52). Only from-scratch models (retrain, DP-SGD) land on target.
- Fisher overlap (0.35 to 0.49 at every VAE layer vs 0.018 for a classifier's class-specific weights) is the mechanism of the first mode: the edit damages retain and forget together, and the retain fine-tune undoes it (phase audit). It does not explain the second mode, which is a missing target.
- The paper's "posterior collapse" and its privacy-utility trade-off were a loss-weighting bug (`LOSS_WEIGHTING_BUG.md`); the old retrain reference had trained on most of the matched negatives.
- NB41 (likelihood-targeted unlearning, `src/train_targeted_unlearn.py`, `outputs/p7/targeted_unlearning.json`) tests the missing target directly. Given the retrain's level, the method lands in the audit's retrain band (0.41; band 0.24 to 0.52) with attacker advantage 0.166 [0.111, 0.212] against the threshold 0.182, at baseline utility. Given the level of held-out look-alikes, the only substitute a controller can observe, it lands 40 nats short (attacker 0.75 to 0.83), because the clean retrain fits those look-alikes at 292 to 301 and the forget cells at 334 to 342. The target is the fit of a type the model does not know, and it is not observable without a retrain.

## Environment

```bash
pyenv activate stat4243          # Python 3.12, PyTorch 2.11, MPS on Apple Silicon (~/.pyenv/versions/stat4243/bin/python)
PYTHONPATH=src python scripts/run_notebook.py 41_targeted_unlearning   # execute a notebook in place
```
Training may use MPS; every reported score is computed on CPU (MPS is not bit-reproducible). DP-SGD runs on CPU only.

## Repository Structure

```
src/                    # vae.py, attacker.py, attacker_eval.py, train_*.py, fisher_utils.py, stats_utils.py,
                        # canonical.py (the only way numbers enter notebooks, figures, and documents)
notebooks/              # 01-41, run in order
scripts/                # retrain_clean.py, save_persample_scores.py, nested_bootstrap_all.py, score_utility_all.py,
                        # attack_lr_reference{,_multi}.py, fisher_phase_audit.py, checkpoint_manifest.py, claims_audit.py,
                        # stage1_rederive.py, stage1b_rederive.py, run_notebook.py
outputs/                # p1 baseline + clean retrains, p2 methods, p4 multiseed scores + ablations, p6 Fisher, p7 NB41,
                        # tabula_muris/, tcga_brca/, p1_c12/, rau/, augmentation/, _superseded_20260908/ (not citable)
data/                   # adata_processed.h5ad, tabula_muris_processed.h5ad, tcga_brca/
Writeup.tex             # Paper source (Overleaf-synced). Writeup.md and index.md are mirrors and lag it.
```

## Key Numbers (PBMC cluster 13; CPU one-attacker pass of 2026-09-08)

- Baseline AUC 0.801, advantage 0.601. Clean retrain (5 seeds) AUC 0.529, advantage 0.057, nested CI [0.044, 0.182].
- Success criterion: advantage at or below 0.182. Null floor of the estimator 0.090 [0.004, 0.253].
- Every post-hoc method fails with a difference-from-retrain CI that excludes zero: retain-FT 0.339, gradient ascent 0.412, SSD 0.402, SCRUB 0.431, Moon 0.480, Fisher 0.345, extra-gradient lambda=10 0.225 (AUC 0.411, diff CI [0.015, 0.243]), lambda=1 0.327, contrastive 0.661 (AUC 0.170), frozen critics 0.985 to 0.995 (AUC 0.003 to 0.008). DP-SGD 0.068 (retrain level, trained from scratch).
- Utility: every post-hoc method at baseline ELBO (363.0 to 364.5) and marker r (0.831 to 0.832); DP-SGD 402.9 / 0.789; three-critic frozen runs 403 to 406.
- Canonical files: `outputs/p4/multiseed/nested_ci_all.json`, `persample/*.json`, `utility_all.json`, `retrain_refit.json`, `outputs/attack_lr_reference.json`. Loaders: `src/canonical.py`. Full tables: `.claude/rules/results-reference.md`. Every likelihood instrument uses `src/likelihood_audit.py` (eval mode, 16 seeded z samples), which is deterministic; the unseeded VAE forward pass is not.

## Datasets

| Dataset | Type | Samples | Genes | Forget set | Matched negatives |
|---|---|---|---|---|---|
| PBMC-33k | scRNA | 33,088 (28,094 retain / 30 forget / 4,964 unseen) | 2,000 | Cluster 13, 30 megakaryocytes; cluster 12 (49 cells) as a second set | 194 (k=10 in baseline latent space) |
| Tabula Muris | scRNA | 41,647 | 2,000 | Cluster 33, 82 cardiac muscle cells | 135 |
| TCGA-BRCA | bulk RNA | 1,089 patients (767 retain / 158 forget / 164 unseen) | 978 (L1000) | 158 Basal patients; patient-level n = 5, 10, 20 | 89 cross-subtype, 30 within-subtype |

## VAE Architecture

- Encoder input -> [1024, 512, 128] -> z=32; decoder z -> [128, 512, 1024] -> softmax x library size; MSE on log-normalized counts (Gaussian observation model; the dispersion head is vestigial).
- LayerNorm + dropout 0.1; Adam 1e-4; 100 epochs; KL warm-up 20; free bits 0.03. 7,349,984 params on PBMC, 4,208,356 on TCGA-BRCA.
- `vae_loss` sums reconstruction over genes and KL over latent dims; never reweight one term (`LOSS_WEIGHTING_BUG.md`).

## Current Priority

Stage 2: rewrite the paper around the corrected measurement and the calibration finding (option 2 in `STAGE1_ANALYSIS.md`), with NB41 as the deciding experiment, then re-sync `Writeup.md` and `index.md` from the new `.tex`.

## Coding Conventions

- `pyenv activate stat4243`; `PYTHONPATH=src` from the project root; PEP8; seeds set explicitly; results to JSON.
- Notebooks are generated with nbformat scripts, show DataFrames and figures rather than prints, and load numbers via `src/canonical.py`. New notebooks continue the sequence (42+).
- Every method checkpoint descends from `outputs/p1/baseline/best_model.pt`; run `scripts/checkpoint_manifest.py` after training anything.

## Writeup Conventions

- No "we" pronouns (passive voice); no colons in prose; no em-dashes; bold only for run-in paragraph headers.
- Advantage 2|AUC - 0.5|, with both AUC and advantage in every table; the AUC is needed because advantage cannot tell signal-intact from over-corrected.
- Every number traces to an output JSON; no bold percentage claims.

## Prohibitions

- NEVER modify raw data in `data/raw/` or `data/filtered_gene_bc_matrices/`.
- NEVER load pre-trained attackers for a reported MIA number (train fresh on the baseline; mia-evaluation skill) and never mix attacker passes.
- NEVER claim a negative binomial likelihood (the loss is MSE).
- NEVER cite anything from `outputs/_superseded_20260908/` or the pre-Stage-1 numbers listed at the end of `.claude/rules/results-reference.md`.
- Do not commit checkpoints or large data.

## Key Files

| Purpose | File |
|---|---|
| Baseline checkpoint (July 4) | `outputs/p1/baseline/best_model.pt` |
| Clean retrain reference | `outputs/p1/retrain_clean/seed_{42..46}/best_model.pt` |
| Structured split / matched negatives | `outputs/p1/split_structured.json`, `outputs/p1.5/s1_matched_negatives.json` |
| Canonical PBMC scores | `outputs/p4/multiseed/nested_ci_all.json` (+ `persample/`, `utility_all.json`, `retrain_refit.json`) |
| Reference-LR audit | `outputs/attack_lr_reference.json`, `outputs/attack_lr_reference_multi.json` |
| Fisher overlap / proposition / phase audit | `outputs/p6/fisher_overlap_results.json`, `nb30_results.json`, `phase_audit.json` |
| Ablations and constructive runs | `outputs/p4/size_ablations_v2.json`, `outputs/p2/frozen_critics/`, `outputs/rau/rau_results.json`, `outputs/augmentation/augmentation_results.json` |
| Provenance and audit | `outputs/checkpoint_manifest.json`, `outputs/claims_audit.json`, `CLAIMS_AUDIT.md`, `CLAIMS_AUDIT_TABLES.md`, `STAGE1_ANALYSIS.md`, `LOSS_WEIGHTING_BUG.md` |

## On-Demand Reference

Results tables, architecture and method defaults, design rationale, and troubleshooting are in `.claude/rules/` (loaded when the matching files are open). Skills for experiment-runner, fisher-analysis, and mia-evaluation are in `.claude/skills/`.
