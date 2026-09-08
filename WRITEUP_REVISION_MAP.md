# Writeup Revision Map

Backup: Writeup_copy.tex (578 lines, unchanged)

## Section-by-section changes

### Abstract (lines 23-25)
- ADD: third dataset (TCGA-BRCA bulk RNA-seq)
- ADD: biology confound finding (30-90% of MIA signal is subtype identity)
- ADD: within-subtype matching as methodology
- ADD: synthetic augmentation and RAU both fail
- KEEP: 8 methods fail, Fisher overlap, retrain-only conclusion
- CHANGE: "two datasets" -> "three datasets"

### Introduction (lines 27-48)
- CHANGE line 31: "Donor-level or patient-level deletion would require donor
  annotations, which neither dataset provides" -> TCGA-BRCA HAS donor annotations,
  update to reflect patient-level experiments
- ADD: contribution about biology confound quantification
- ADD: contribution about TCGA-BRCA cross-modality evidence
- ADD: contribution about synthetic augmentation / RAU (negative but informative)
- REORDER contributions to lead with evaluation methodology

### Related work (lines 50-64)
- ADD: Ozturk et al. (2026) — CAMDA benchmark, rho=0.92
- ADD: Golob et al. (2026) — scMAMA-MIA, cell-type-dependent distances
- ADD: Walker et al. (2024) — single-cell privacy (Cell)
- ADD: Cheng et al. (ICLR 2026) — retain-forget entanglement
- ADD: Pawelczyk et al. (2025) — unlearning fails for data poisoning
- KEEP all existing citations

### Datasets and setup (lines 66-91)
- ADD: TCGA-BRCA to dataset table (1089 patients, 978 genes, Basal forget set)
- ADD: citation to Ozturk for data source, Goldman for Xena, TCGA Network for BRCA
- ADD: mention patient-level forget sets as new capability
- KEEP: PBMC and Tabula Muris descriptions unchanged

### Methods (lines 92-155)
- KEEP: VAE architecture, all 8 methods, attack suite — unchanged
- ADD: brief mention of RAU method (representation alignment) in methods section
- ADD: brief mention of synthetic augmentation approach
- NOTE: these already have the NB->MSE fix from earlier this session

### Experiments (lines 153-310)
- KEEP: all existing PBMC and Tabula Muris results unchanged
- ADD subsection: "TCGA-BRCA results" with cross-subtype and within-subtype matching
- ADD subsection: "Biology confound quantification" with the 3-dataset comparison
  - Attack diversity table (retrain AUC by attack type)
  - Cross-sub vs within-sub AUC comparison
  - Confound fraction (90% PBMC, 76% TCGA)
- ADD subsection: "Synthetic augmentation" (NB37 result — shifts bias)
- ADD subsection: "Representation alignment" (NB38 result — Streisand effect)
- ADD: patient-level results from NB34b (zero memorization on TCGA-BRCA)

### Section 6: Fisher theory (lines 311-412)
- ADD: TCGA-BRCA Fisher numbers (global 0.905, per-layer)
- KEEP: Proposition 1, Corollary 2, all proofs unchanged
- ADD: one paragraph connecting higher TCGA-BRCA overlap to larger forget fraction
- KEEP: deep MLP classifier comparison, z=8 variant, conditional decoder

### Discussion (lines 413-425)
- EXPAND "Biology vs membership" from one paragraph to full subsection
- ADD: threat model discussion (Kerckhoffs's principle)
- ADD: trained attacker vs off-the-shelf distance methods (Ozturk/Golob comparison)
- ADD: synthetic augmentation failure as evidence of structural limit
- ADD: RAU failure as evidence structural limit extends to representation space
- REVISE: "Limitations" to update re: TCGA-BRCA donor annotations now available

### Conclusion (lines 427-437)
- REVISE: lead with evaluation methodology finding
- ADD: "memorization is real but overestimated by 30-90%"
- ADD: TCGA-BRCA confirms across modalities
- ADD: constructive attempts (augmentation, RAU) also fail
- KEEP: Fisher overlap as structural explanation
- REVISE future work: remove "donor-level forget sets" (now done),
  add calibrated MIA as open direction

### References
- ADD ~12 new citations (Ozturk, Golob, Walker, Goldman, TCGA Network,
  Subramanian, Pawelczyk, Cheng, George, Sun, Chen/GAN-leaks, Kunstner)

## Estimated size change
- Current: 578 lines
- Expected: ~650-700 lines (+70-120 lines)
- Main additions: TCGA-BRCA results, biology confound subsection, RAU/augmentation
