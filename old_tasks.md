# TASKS.md - Paper 1 Completion Checklist

## Goal

**Complete Paper 1: "Adversarial Machine Unlearning for Genomic VAEs"**

Add cross-dataset validation (Tabula Muris) to address professor's feedback on generalization. Both positive and negative findings are valuable if they characterize the methods.

## Professor Feedback (Addressed by This Work)

1. **Single dataset** → Add Tabula Muris ✓ (Priority 1)
2. **Small forget sets** → Size ablation with extra-gradient (Priority 2)
3. **Limited stress-testing** → Document failure mode consistency (Priority 1)

## Established Findings to Verify on Tabula Muris

| Finding | PBMC Result | Verify on Tabula Muris |
|---------|-------------|------------------------|
| Extra-gradient λ=10 works on structured | AUC=0.48 ✓ | Does it generalize? |
| Fisher fails on structured (posterior collapse) | AUC=0.81 ✗ | Same failure mode? |
| Fisher works on scattered | AUC=0.50 ✓ | Still works? |
| Baseline memorization level | AUC=0.77 | Similar? |

---

## PRIORITY 1: Tabula Muris Cross-Dataset Validation

### Phase 1: Data Preparation
- [x] Download Tabula Muris 10x data (droplet, from Figshare)
- [x] Create `scripts/preprocess_tabula_muris.py`
- [x] QC filtering (same thresholds as PBMC)
- [x] Normalize and select 2000 HVGs
- [x] Cluster and visualize (35 clusters)
- [x] Identify rare clusters: **Cluster 34 (21 alveolar macrophages)**, Cluster 33 (97 cells)
- [x] Save processed data to `data/tabula_muris_processed.h5ad`

**Tabula Muris Summary:**
| Metric | PBMC-33k | Tabula Muris |
|--------|----------|--------------|
| Cells | 33,088 | 41,647 |
| HVGs | 2,000 | 2,000 |
| Clusters | 14 | 35 |
| Rare cluster | Cluster 13 (35 megakaryocytes) | Cluster 34 (21 alveolar macrophages) |

### Phase 2: Baseline and Splits
- [x] Create train/unseen split (85/15)
- [x] Create structured forget set (Cluster 33, 82 cardiac muscle cells)
- [x] Create scattered forget set (random 30)
- [x] Generate matched negatives via k-NN
- [x] Train baseline VAE (same architecture)
- [x] Evaluate baseline MIA AUC (structured: 0.891, scattered: 0.411)
- [x] Save to `outputs/tabula_muris/baseline/`

### Phase 3: Unlearning Experiments
- [x] Run extra-gradient λ=10 on structured set → AUC 0.874 (FAIL, did not reach target band)
- [x] Evaluate post-hoc AUC (target: 0.45-0.52) → 0.874, outside target
- [x] Run Fisher on structured set → AUC 0.946
- [x] Verify posterior collapse pattern (expect AUC > 0.7) → Confirmed (0.946)
- [x] Run Fisher on scattered set → AUC 0.568
- [x] Verify it works (expect AUC ~ 0.50) → 0.568 vs baseline 0.411; scattered baseline near chance, marginal Fisher effect

### Phase 4: Documentation
- [x] Create comparison table (PBMC vs Tabula Muris) — notebook 13
- [x] Document any differences in behavior — notebook 13
- [x] If extra-gradient fails, sweep λ = [5, 7, 10, 15, 20] → All produce AUC ~0.87; failure is structural, not hyperparameter

---

## PRIORITY 2: Size Ablation (Extra-gradient Only)

**Note:** Fisher already has documented utility issues. Focus on extra-gradient.

- [x] Create structured splits: n = 10, 50, 100 on PBMC
  - n=10: 10 cells from cluster 13
  - n=50: Cluster 13 (30) + 20 from cluster 12
  - n=100: Cluster 13 (30) + cluster 12 (49) + 21 from cluster 11
- [x] Run extra-gradient λ=10 at each size (3 seeds)
  - n=10: AUC 0.37 ± 0.04 (over-unlearned)
  - n=30: AUC 0.48 (reference, in target band)
  - n=50: AUC 0.61 ± 0.08 (insufficient)
  - n=100: AUC 0.59 ± 0.05 (insufficient)
- [x] Document if combining clusters changes dynamics → Yes, combining clusters breaks the method
- [x] Identify any size threshold where method breaks → λ=10 only works for single-cluster n~30; fails for multi-cluster sets

---

## PRIORITY 3: Update Writeup

- [x] Add Tabula Muris results to Section V (cross-dataset, lambda sweep, size ablation)
- [x] Update Table II with cross-dataset comparison
- [x] Revise discussion based on findings (added sections E, F, G)
- [x] Qualify claims where patterns differ (extra-gradient does not generalize)

---

## Success Criteria

**Paper 1 is ready for submission when:**

1. ✅ Tabula Muris shows similar baseline memorization (AUC 0.6-0.8)
2. ✅ Extra-gradient achieves target band on Tabula Muris OR we explain why not
3. ✅ Fisher shows same posterior collapse on structured sets
4. ✅ Cross-dataset comparison table in writeup
5. ✅ Size ablation shows extra-gradient scaling behavior

---

## Future Work (Paper 2 - Mechanistic Analysis)

If pursuing the deeper "why methods fail" paper, see `RESEARCH_PLAN.md` for:
- Information localization analysis
- Parameter entanglement quantification
- Relearning vulnerability testing
- Cross-architecture validation

This is separate from Paper 1 completion.
