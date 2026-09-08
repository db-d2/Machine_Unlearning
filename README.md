# Fishing for Privacy: Biological Confounds and the Limits of Membership Inference in Gene-Expression VAEs

David Benson, Columbia University

**Status (2026-09-08).** The paper is being rewritten after a full re-derivation of every experiment (Stage 1). The numbers below are the re-derived ones and trace to JSON files under `outputs/`. `Writeup.tex`, `Writeup.md`, and `index.md` still carry the pre-audit text and are marked as such until the rewrite lands. `STAGE1_ANALYSIS.md` gives the picture, `CLAIMS_AUDIT.md` the claim-by-claim verdicts, and `LOSS_WEIGHTING_BUG.md` the bug that produced the old "posterior collapse" and utility-cost results.

## Overview

Nine post-hoc machine unlearning methods, two constructive approaches (training-time synthetic augmentation and representation alignment), DP-SGD, and a likelihood-targeted variant are evaluated on VAEs trained on gene-expression data across two modalities, scRNA-seq (PBMC-33k, Tabula Muris) and bulk RNA-seq (TCGA-BRCA). The forget sets are rare, biologically coherent subpopulations. Every model is scored by one membership inference attacker trained on the baseline model and by a reference-model likelihood-ratio audit, against five clean retrains that never saw the forget set or the matched negatives.

Three findings survive the re-derivation.

1. **Measurement.** On PBMC cluster 13 the clean retrain sits at chance (advantage 0.057) and the baseline at 0.601, so the membership signal is genuine. On Tabula Muris, TCGA-BRCA, and two further PBMC clusters the retrain matches or exceeds the baseline under cross-class matching, so what that protocol measures is subtype biology. Within-subtype matching or a clean retrain removes it. For a forget set that is an entire training subpopulation, a model trained only on synthetic look-alikes of the forget cells is scored as if it had memorized them, which is the stated limit of any membership evaluation here.
2. **No post-hoc method lands where retraining lands, and none pays for it in utility.** Methods that edit the model and then fine-tune on the retain set leave the forget cells fitted almost as well as the baseline does. Methods with a continuous forget-set objective push them past the retrain level. Every post-hoc method keeps baseline held-out ELBO and marker-gene correlation. Only models trained from scratch (retrain, DP-SGD) land on target.
3. **Mechanism.** Forget and retain Fisher information overlap at every VAE layer (cosine 0.35 to 0.49, against 0.018 for a classifier's class-specific output weights). A phase audit shows what that does. The edit damages the retain set along shared directions, and the retain fine-tune that repairs it undoes the edit. That explains the first failure mode. The second is a missing target, and the likelihood-targeted variant (notebook 41) tests it directly. Told the retrain's level, it lands inside the retrain band under the audit and brings the attacker's advantage down to the paper's threshold at baseline utility. Told instead the level of held-out look-alikes, the only substitute a data controller can observe, it lands 40 nats short, because a model without the subpopulation fits an unseen member of it worse than that member's nearest known-type neighbours. The target exists, and it is not observable without a retrain.

## Setup

```bash
pyenv activate stat4243  # or: pip install -r requirements.txt
cd data/filtered_gene_bc_matrices && unzip hg19.zip && cd ../..
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Project structure

```
src/
├── vae.py                          # VAE (MSE on log-normalized counts, z=32, hidden=[1024,512,128])
├── vae_conditional.py              # Cluster-conditional VAE variant
├── attacker.py                     # MIA attacker (spectral norm, 70-dim features)
├── attacker_eval.py                # Post-hoc attacker evaluation
├── canonical.py                    # Loaders for every canonical result JSON (no number is typed by hand)
├── stats_utils.py                  # Nested bootstrap CI, Welch, Cohen's d
├── fisher_utils.py                 # Per-sample diagonal Fisher, datasets, metrics (NB26-30, NB35)
├── augmentation_utils.py           # Synthetic augmentation helpers (NB37)
├── synthetic_utils.py              # Synthetic feature helpers (NB37 variants)
├── train.py                        # Baseline VAE training
├── train_unlearn_extragradient.py  # Extra-gradient co-training (ExtraAdam, 3 co-trained critics)
├── train_unlearn.py                # Frozen-critic co-training
├── train_fisher_unlearn.py         # Fisher scrubbing (per-sample Fisher, global step cap)
├── train_ssd.py                    # Selective Synaptic Dampening
├── train_scrub.py                  # SCRUB teacher-student
├── train_contrastive_unlearn.py    # Contrastive latent unlearning
├── train_retain_finetune.py        # Retain-only fine-tuning
├── train_gradient_ascent.py        # Gradient ascent + fine-tune
├── train_dp.py                     # DP-SGD via Opacus
├── train_rau.py                    # Representation alignment unlearning (NB38)
├── train_targeted_unlearn.py       # Likelihood-targeted unlearning (NB41)
├── retrain.py                      # Retrain on the retain split (leak fixed 2026-09-08)
└── utility_metrics.py              # ELBO, silhouette, ARI, marker correlation

notebooks/                          # Experiments (01-41, run in order)
scripts/                            # Clean retrains, one-pass scoring, nested bootstrap, audits, provenance
outputs/
├── p1/                             # Baseline, splits, clean retrains (retrain_clean/seed_42..46)
├── p2/                             # Method checkpoints (incl. frozen_critics/)
├── p4/                             # Multi-seed scores, nested CIs, utility, size ablations
├── p6/                             # Fisher overlap, proposition test, phase audit
├── p7/                             # Likelihood-targeted unlearning (NB41)
├── tabula_muris/, tcga_brca/       # Cohort results
├── p1_c12/, larger_cluster_eval/   # Second PBMC forget set; cluster-7 / TM cluster-28 confound checks
├── rau/, augmentation/             # Constructive approaches
└── _superseded_20260908/           # Pre-audit files, kept for the claims audit only

data/
├── filtered_gene_bc_matrices/      # Raw 10x PBMC data (unzip hg19.zip)
├── adata_processed.h5ad            # Generated by notebook 01
├── tabula_muris_processed.h5ad     # Generated by notebook 11
└── tcga_brca/                      # TCGA-BRCA processed data (NB31)

Writeup.tex                         # Paper (LaTeX); pre-audit text, rewrite in progress
references.bib                      # Bibliography
figures/                            # Generated figures (make_result_figures.py regenerates the data figures)
```

## Reproducing results

Run notebooks in numerical order:

```bash
jupyter notebook notebooks/          # or: PYTHONPATH=src python scripts/run_notebook.py <name>
```

- **NB01-10**: Data prep, baseline training, initial unlearning experiments
- **NB11-25**: Additional methods (SSD, SCRUB, contrastive, DP-SGD), cross-dataset validation, ablations, attack diversity
- **NB26**: Canonical Fisher overlap (VAE vs classifier, per-sample estimator)
- **NB27**: Deep MLP classifier (fair capacity comparison)
- **NB28**: VAE z=8 (architecture generalization)
- **NB29**: Conditional VAE (cluster-specific output columns)
- **NB30**: Proposition 1 verification + conditional VAE scrubbing
- **NB31-36**: TCGA-BRCA data prep, baseline, unlearning, within-subtype evaluation, Fisher overlap, cross-domain comparison
- **NB37**: Training-time synthetic augmentation (and synthetic matched negatives)
- **NB38**: Representation alignment unlearning (RAU)
- **NB39**: Larger-cluster within-subtype validation (PBMC c7, TM c28)
- **NB40**: Confound fraction CIs (parametric bootstrap)
- **NB41**: Calibration axis for every checkpoint and likelihood-targeted unlearning

The canonical PBMC pipeline (clean retrains, one attacker for every checkpoint, nested bootstrap, utility, reference audit, provenance, claims audit):

```bash
for SEED in 42 43 44 45 46; do
    PYTHONPATH=src python scripts/retrain_clean.py --seed $SEED --output_dir outputs/p1/retrain_clean/seed_$SEED
done
PYTHONPATH=src python scripts/save_persample_scores.py
PYTHONPATH=src python scripts/nested_bootstrap_all.py
PYTHONPATH=src python scripts/score_utility_all.py
PYTHONPATH=src python scripts/attack_lr_reference.py
PYTHONPATH=src python scripts/checkpoint_manifest.py
PYTHONPATH=src python scripts/claims_audit.py
```

## Key results

PBMC-33k structured forget set (cluster 13, n=30 megakaryocytes) against 194 matched negatives. Advantage = 2|AUC - 0.5|. One CPU attacker scores every model. The reference is five clean retrains trained on the retain split only. A method fails when its advantage exceeds the retrain nested 95% CI upper bound (0.182) and its difference from retrain has a CI that excludes zero. The audit column is the reference-model likelihood-ratio AUC (1 = forget cells still fitted better than a never-trained model fits them, 0 = fitted worse; deterministic estimator, eval mode with sixteen seeded z samples).

| Method | Seeds | AUC | Advantage [95% CI] | Audit | ELBO | Marker r | Status |
|---|---|---|---|---|---|---|---|
| Baseline | 1 | 0.801 | 0.601 | 0.996 | 363.9 | 0.831 | anchor |
| Retain-only fine-tune | 5 | 0.669 | 0.339 [0.24, 0.44] | 0.991 | 363.0 | 0.832 | FAIL, signal intact |
| Gradient ascent | 5 | 0.706 | 0.412 [0.31, 0.51] | 0.995 | 363.2 | 0.831 | FAIL, signal intact |
| SSD | 3 | 0.701 | 0.402 [0.28, 0.53] | 0.995 | 363.6 | 0.831 | FAIL, signal intact |
| SCRUB | 3 | 0.716 | 0.431 [0.30, 0.56] | 0.991 | 363.5 | 0.831 | FAIL, signal intact |
| Moon feature unlearning | 3 | 0.740 | 0.480 [0.35, 0.60] | 0.996 | 364.5 | 0.831 | FAIL, signal intact |
| Fisher scrubbing | 3 | 0.673 | 0.345 [0.22, 0.47] | 0.991 | 363.6 | 0.832 | FAIL, signal intact |
| Extra-gradient (lambda=10) | 10 | 0.411 | 0.225 [0.15, 0.33] | 0.218 | 363.8 | 0.832 | FAIL, over-corrected |
| Extra-gradient (lambda=1) | 10 | 0.336 | 0.327 [0.20, 0.48] | 0.049 | 363.3 | 0.831 | FAIL, over-corrected |
| Contrastive latent | 3 | 0.170 | 0.661 [0.56, 0.75] | 0.004 | 363.7 | 0.831 | FAIL, over-corrected |
| Frozen critics (4 configs) | 3 each | 0.003 to 0.008 | 0.985 to 0.995 | 0.003 to 0.006 | 364 to 406 | 0.831 | FAIL, over-corrected |
| DP-SGD (epsilon=10) | 3 | 0.466 | 0.068 [0.03, 0.19] | 0.502 | 402.9 | 0.789 | retrain level (from scratch) |
| **Clean retrain** | **5** | **0.529** | **0.057 [0.04, 0.18]** | **0.24 to 0.52 (seed vs seed)** | **364.2** | **0.831** | **reference** |

Extra-gradient at lambda=10 is bimodal across seeds (seven over-correct, three keep part of the signal) and none of its ten seeds lands in the retrain band. A refit attacker trained on each model's own features puts the clean retrain at advantage 0.035 [-0.03, 0.10] while recovering 0.595 on the baseline, so the near-chance retrain score is a property of the model.

Likelihood-targeted variants (notebook 41; scored by the notebook's own attacker instance, under which the baseline is 0.802): the look-alike target lands at advantage 0.503 (AUC 0.75, audit 0.62); the cohort-average target, which coincides with the retrain's level, at 0.166 [0.11, 0.21] (AUC 0.58, audit 0.41, inside the retrain band 0.24 to 0.52) with utility intact; the disjoint look-alike target at 0.668 (AUC 0.83, audit 0.70).

Cross-cohort: Tabula Muris cluster 33 baseline 0.903 vs retrain 0.943; TCGA-BRCA Basal cross-subtype 0.803 vs 0.830, within-subtype 0.580 vs 0.572; PBMC cluster 12 baseline advantage 0.737 vs clean retrain 0.628. In each case the retrain matches or exceeds the baseline, so the cross-class signal is biology.

Fisher overlap (per-sample estimator): PBMC VAE encoder 0.345, bottleneck 0.428, decoder hidden 0.351, decoder output 0.493 (global 0.209) vs linear classifier 0.018; TCGA-BRCA global 0.753. Phase audit: the Fisher scrub raises the forget-cell negative ELBO from 270 to 5,569 and the retain cells' from 363 to 1,250; the retain fine-tune brings them back to 308 and 362, while a clean retrain fits the forget cells at 334 to 337.

## Citation

Code: <https://github.com/db-d2/Machine_Unlearning>
