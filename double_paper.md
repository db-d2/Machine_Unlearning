Can This Be Two Separate Publications?

  Yes, but with important caveats.

  Paper 1: Applied Methods Paper

  "Adversarial Machine Unlearning for Genomic VAEs"
  ┌──────────────────────────┬──────────────────────────────────────────────┬────────────────────────┐
  │          Aspect          │                Current State                 │ Needed for Publication │
  ├──────────────────────────┼──────────────────────────────────────────────┼────────────────────────┤
  │ Core contribution        │ Extra-gradient λ=10 works on structured sets │ Already demonstrated   │
  ├──────────────────────────┼──────────────────────────────────────────────┼────────────────────────┤
  │ Cross-dataset validation │ PBMC-33k only                                │ Add Tabula Muris       │
  ├──────────────────────────┼──────────────────────────────────────────────┼────────────────────────┤
  │ Negative results         │ Fisher fails (posterior collapse)            │ Already documented     │
  ├──────────────────────────┼──────────────────────────────────────────────┼────────────────────────┤
  │ Practical guidelines     │ Hyperparameter sensitivity                   │ Already in writeup     │
  └──────────────────────────┴──────────────────────────────────────────────┴────────────────────────┘
  Venue fit: Applied ML workshop (NeurIPS ML4Health, ICML CompBio) or Bioinformatics journal

  Gap: Needs Tabula Muris to address "single dataset" limitation. This is ~2-3 days of work.

  ---
  Paper 2: Mechanistic Analysis Paper

  "Why Machine Unlearning Fails in VAEs: The Parameter Entanglement Problem"
  ┌───────────────────────────────┬──────────────────────┬─────────────────────────────────┐
  │            Aspect             │    Current State     │     Needed for Publication      │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────┤
  │ Core hypothesis               │ Decoder entanglement │ Articulated in RESEARCH_PLAN.md │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────┤
  │ Localization analysis         │ Not done             │ Experiment 1 (~2 days)          │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────┤
  │ Entanglement quantification   │ Not done             │ Experiment 2 (~3 days)          │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────┤
  │ Relearning vulnerability      │ Not done             │ Experiment 3 (~2 days)          │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────┤
  │ Cross-architecture validation │ Not done             │ Experiment 4 (~3 days)          │
  └───────────────────────────────┴──────────────────────┴─────────────────────────────────┘
  Venue fit: Main track ML conference (ICML, NeurIPS, ICLR) or TMLR

  Gap: This requires 10-14 days of new experimental work.

  ---
  My Recommendation

  One strong paper is the better strategy for several reasons:

  1. Completeness: The empirical results (what works) + mechanistic analysis (why things fail) tell a complete story. Splitting them leaves each
  paper feeling incomplete.
  2. The posterior collapse finding is the key insight. It bridges both papers. Including it in Paper 1 without the deeper analysis feels
  superficial; putting it only in Paper 2 leaves Paper 1 without an interesting negative result.
  3. Tabula Muris validation is needed for both papers. You'd be duplicating this work.
  4. Timeline reality: The professor wants generalization results. Tabula Muris + size ablation (for extra-gradient) is achievable. The full
  mechanistic analysis is a larger undertaking.

  ---
  Suggested Path Forward

  Short-term (for professor meeting):
  - Add Tabula Muris dataset
  - Run extra-gradient on Tabula Muris structured/scattered sets
  - Verify Fisher shows same posterior collapse pattern
  - This strengthens Paper 1 enough for publication

  Medium-term (if pursuing deeper research):
  - Run Experiments 1-2 from RESEARCH_PLAN.md (localization + entanglement)
  - If results support hypothesis, integrate into Paper 1 as a "Why It Fails" section
  - This transforms Paper 1 from "applied methods" to "methods + insights" - more impactful

  Alternative (if ambitious):
  - Submit Paper 1 to a workshop now (after Tabula Muris)
  - Develop Paper 2 over 2-3 months for main conference
  - Paper 2 can cite Paper 1 for empirical grounding

  ---
  Bottom Line

  You have one solid paper with the current work + Tabula Muris generalization. The research plan outlines work for a second, more theoretical
  paper that would require significant additional experiments.

  The pragmatic path: finish Paper 1 with Tabula Muris, then decide if the mechanistic work is worth pursuing for Paper 2
