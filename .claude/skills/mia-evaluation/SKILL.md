---
name: mia-evaluation
description: Evaluate membership inference attack success on VAE models. Use when training MIA attackers, computing AUC/advantage metrics, scoring unlearning methods against the retrain reference, or running the reference-model likelihood-ratio audit.
---

# MIA Evaluation Pipeline

## CRITICAL: Fresh Attacker, One Pass, CPU

NEVER load pre-trained attackers for a reported number. Train a FRESH attacker:
- Train on forget features (positive) vs matched-negative features (negative), extracted from the BASELINE model.
- Canonical features: 70 dims = v1 (69) + mean k-NN distance (k=5) to a 5,000-cell retain sample. Attacker: spectral norm, [256, 256], dropout 0.3.
- Apply that one attacker to every checkpoint in the same pass. `scripts/save_persample_scores.py` writes `outputs/p4/multiseed/persample/*.json`; `scripts/nested_bootstrap_all.py` writes `nested_ci_all.json` (nested CIs, difference-from-retrain CIs, Holm-corrected tests, permutation null floor).
- Pin scoring to CPU. MPS is not bit-reproducible, and even on CPU a fresh attacker instance drifts 0.001 to 0.02 AUC (the baseline scored 0.792 in one pass and 0.801 in the next). Numbers from different passes are not comparable.
- Bug found 2026-02-08: early scripts loaded attackers trained on forget vs the 28k retain set, which inflated AUC. Bug found 2026-09-08: the April retrain reference had trained on 159 to 170 of the 194 matched negatives (`retrain_leaky`); the reference is now `outputs/p1/retrain_clean/seed_{42..46}`.

## Matched Negatives

Unseen cells closest to the forget set in the baseline model's latent space, k=10 neighbours per forget cell, deduplicated. PBMC cluster 13: 194; PBMC cluster 12: 57; Tabula Muris cluster 33: 135; TCGA-BRCA Basal: 89 cross-subtype, 30 within-subtype.
```python
K = 10
nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(unseen_z)
distances, indices_knn = nbrs.kneighbors(forget_z)
matched_indices = unseen_indices[np.unique(indices_knn.flatten())]
```
Within-subtype matching (negatives from the forget set's own cluster or subtype) is the control for the biology confound; on TCGA-BRCA it moves baseline / retrain from 0.803 / 0.830 to 0.580 / 0.572, and on PBMC cluster 7 from 0.665 to 0.527.

## Feature Extraction

```python
from attacker import extract_vae_features, build_attack_features, compute_knn_distances
vae_feats = extract_vae_features(vae, x, lib_size, device, requires_grad=False)   # tensors come back on CPU
knn = compute_knn_distances(vae_feats['z'].numpy(), retain_z, k=5)
attack_feats = build_attack_features(vae_feats, knn_dist_retain=knn, variant='v1')  # 70-dim
```
`src/train_targeted_unlearn.py::attack_features` wraps exactly this.

## Metrics

- Advantage = 2|AUC - 0.5| (Yeom 2018), direction-agnostic; computed per seed, then averaged.
- Nested 95% CI: outer over seeds, inner per-sample bootstrap on the 30 vs 194 pool (`src/stats_utils.py`). The estimator's null floor at this size is 0.090 [0.004, 0.253], so advantages near 0.1 are chance.
- Success: method advantage at or below the clean-retrain nested-CI upper bound, 0.182 (retrain advantage 0.057 [0.044, 0.182], AUC 0.529). Also report the nested CI of the advantage difference from retrain (`hypothesis_tests` in `nested_ci_all.json`).
- Direction blind spot: advantage cannot distinguish signal-intact (AUC 0.67) from over-corrected (AUC 0.33). Always report the AUC alongside it and pair it with the reference audit below.

## Reference-model likelihood-ratio audit (auditor's instrument)

`scripts/attack_lr_reference.py` (PBMC) and `scripts/attack_lr_reference_multi.py` (Tabula Muris, TCGA-BRCA). Score per cell = negELBO under the clean retrain (seed 42) minus negELBO under the target, CPU; AUC of forget vs matched. The negELBO is `src/likelihood_audit.py::neg_elbo_per_cell` (eval mode, 16 z samples from a seeded generator), which two runs reproduce exactly; the VAE's unseeded forward pass moved a 30-cell AUC by up to 0.14 between runs and must not be used. Near 1: forget cells still fitted better than a never-trained model fits them. Near 0: fitted worse (over-corrected). Floor: retrain seeds 43 to 46 against seed 42 give 0.24 to 0.52. Limit: models trained on synthetic look-alikes of the forget cells and never on the cells themselves score 0.984 to 0.985 here (and 0.74 to 0.80 under the MLP), so for a whole-subpopulation forget set the instruments measure subpopulation knowledge, not only individual membership.

## Refit control

`scripts/retrain_refit_attacker.py`: an attacker trained on each model's own features, out-of-fold 5-fold, 20 random splits. Baseline CV AUC 0.797 (the attacker has power at n=30); clean retrain advantage 0.035 [-0.029, 0.100]. The near-chance retrain score is a property of the model, not of a weak transferred attacker.

## Reporting

Table captions state the seed count and CI type ("mean across seeds; nested 95% CI"). Report both AUC and advantage in every table. Load every number through `src/canonical.py`.
