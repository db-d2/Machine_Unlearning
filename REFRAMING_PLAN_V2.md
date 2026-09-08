# Analysis: Paper Framing — Finding the Clean Question

## Context

The user wants a paper with the structure of Adebayo et al. or Ozturk et al.: **one clear question, one evaluation framework, systematic results.** Not a research journey narrative. The other agent (which DID see the reframing plan and Q1+Q2 structure) suggested a narrative approach. The user isn't convinced that will fly.

## Core Problem

The current paper (and the Q1+Q2 reframing plan) has TWO questions, which makes it structurally less clean than either reference paper. Adebayo has ONE: "which explanations work for which bugs?" Ozturk has ONE: "to what extent can generative models balance utility and privacy?" The user's paper needs ONE.

---

## What the Reference Papers Do

**Adebayo et al. — "Debugging Tests for Model Explanations"**
- ONE question: *"Which explanation methods are effective for which classes of model bugs?"*
- Framework: Categorize bugs (data, model, test-time contamination)
- Method: Test explanation methods against each bug class
- Structure: Framework → methods → systematic results by category → insights
- The paper doesn't propose a new explanation method. It evaluates existing ones and finds partial failures. Still published and highly cited.

**Ozturk et al. — "Towards Useful and Private Synthetic Omics"**
- ONE question: *"To what extent can generative models preserve biological signals while remaining resilient to privacy attacks?"*
- Framework: Evaluate across 4 dimensions (fidelity, utility, biological plausibility, privacy)
- Method: Benchmark 11 generative methods on 2 TCGA cohorts
- Structure: Framework → methods → results by dimension → trade-offs
- Also doesn't propose a new method. Evaluates and maps a landscape.

**Common pattern:** One question → one evaluation framework → systematic application → findings (including negative ones). No journey, no narrative arc. The question is stated as if it were always the question, even though we know both teams iterated to get there.

---

## The Problem With Your Current Paper

Your paper currently has **two questions** (Q1: biology confound, Q2: unlearning failure). That's structurally messier than either reference. But more importantly, the two aren't really independent — Q1 is something you had to solve in order to get an honest answer to Q2. The biology confound is a **prerequisite**, not a peer finding.

The other agent's suggestion to frame it as a journey is one way to unify them (Q1 came first, then Q2). But that's not how Adebayo or Ozturk do it. They'd unify them under one question and present Q1 as part of the evaluation framework.

---

## Three Candidate Single-Question Framings

### Option A: Evaluation-centered (Adebayo-style)
**Question:** *"How well do approximate unlearning methods protect privacy in biological generative models?"*

- Framework: Corrected evaluation protocol (within-subtype matching + trained attacker + retrain baseline)
- The biology confound becomes **part of the framework** — you had to build it to answer the question honestly
- Unlearning failure is the finding, explained by Fisher overlap theory
- Structure:
  - §2 Background + Threat Model
  - §3 Datasets + Corrected Evaluation Protocol (biology confound as methodology, not finding)
  - §4 Results: 10 methods × 3 datasets (the answer: none work on structured sets)
  - §5 Why: Fisher overlap theory
  - §6 Discussion

**Pros:** Clean single question, very standard structure. Biology confound is embedded naturally as "we had to fix evaluation first."
**Cons:** Slightly undersells the biology confound as a standalone contribution. If the confound is the most novel finding (and it is), burying it in §3 methodology risks reviewers missing it.

### Option B: Confound-centered (most novel finding leads)
**Question:** *"What does MIA actually measure on biological generative models, and what are the consequences for machine unlearning?"*

- Framework: Multi-layer evaluation (cross-subtype → within-subtype → attack diversity → retrain baseline)
- Biology confound is the **headline finding** (§4, dedicated section)
- Unlearning failure is the **consequence** of honest evaluation (§5)
- Fisher overlap is the **structural explanation** (§6)
- Structure:
  - §2 Background + Threat Model
  - §3 Datasets + Evaluation Protocol
  - §4 The Biology Confound (confound quantification, 30-90%, synthetic leakage, retrain table showing only MLP works)
  - §5 Unlearning Under Corrected Evaluation (10 methods, all fail)
  - §6 Structural Explanation (Fisher overlap)
  - §7 Discussion

**Pros:** Leads with the most novel, most citable finding. Makes a stronger individual contribution case. This is essentially the REFRAMING_PLAN.md structure, but unified under one question instead of two.
**Cons:** The question has two parts ("what does MIA measure" + "consequences for unlearning"), so it's not quite as crisp as Adebayo's single question. A reviewer might say "these are two papers."

### Option C: Benchmark-style (Ozturk-parallel) — REVISED
**Question:** *"Can approximate unlearning erase training-sample memorization from biological VAEs?"*

Important correction: The VAE is trained on the FULL dataset. Rare subpopulations are the evaluation target, not a training choice. The finding is that memorization *concentrates* in rare, biologically coherent subpopulations — a natural phenomenon (Feldman 2020), not a design decision.

- Framework: Corrected privacy evaluation protocol for biological generative models
- Parallels Ozturk: they benchmark generative methods on privacy; you benchmark unlearning methods on privacy
- Parallels Adebayo: they ask "which explanations work for which bugs?" after building debugging tests; you ask "which unlearning methods work?" after building corrected privacy evaluation
- Biology confound is presented as **the corrected evaluation framework you had to build** (§3-4)
- Unlearning results are the **main evaluation** (§5)
- Fisher overlap is the **explanation** (§6)

**Pros:** Clean single question. A biologist can understand it. Positions naturally against Ozturk/Golob. Doesn't imply training on rare subpopulations (which isn't what happened). The memorization-concentrates-in-rare-biology finding emerges naturally when you describe WHERE you look.
**Cons:** The biology confound finding is slightly subordinated to the unlearning evaluation, though it appears naturally as "we had to fix evaluation first."

---

## My Recommendation: Option C (Revised)

**Proposed one-sentence through-line:**
*"Approximate unlearning cannot erase the memorization that biological VAEs exhibit for rare subpopulations. Cross-class matched-negative MIA protocols, as used in recent benchmarks, can conflate subtype identity with training membership; corrected evaluation reveals a smaller but genuine memorization signal on PBMC that no tested method removes."*

**Proposed contributions list (Adebayo-style):**
1. **Corrected evaluation protocol** for MIA on biological generative models: within-subtype matching, trained attacker with member/non-member labels, multi-seed retrain baseline, attack diversity analysis. Cross-class matched-negative protocols, as used in e.g. Ozturk 2026 and partly Golob 2026, can conflate subtype identity with training membership. The magnitude of this confound varies by dataset and is documented per-dataset in §4.
2. **Systematic unlearning evaluation** across 10 approaches (8 approximate + 2 constructive), 3 datasets (PBMC, Tabula Muris, TCGA-BRCA), 2 modalities (scRNA, bulk RNA). Under corrected evaluation, all 10 fail on structured forget sets. On TCGA-BRCA, no genuine memorization is detectable under our attack suite after within-subtype correction, so unlearning is moot. Synthetic augmentation shifts memorization bias rather than removing it.
3. **Structural explanation** via Fisher information overlap. The VAE's shared decoder creates 17× higher Fisher alignment between forget and retain sets than a classifier (0.306 vs 0.018 on PBMC; 0.905 on TCGA-BRCA). Proposition 1 formalizes this for linear decoders, with dimensional scaling bounds showing generative-model overlap grows as 1 - O(M/D) while classifier overlap scales as 1/√C.

Three contributions, one question, clean structure. Clean, reviewer-friendly, citable.

### Calibrations applied:
1. **Dropped "30-90%"** from abstract/contributions. TM c28 n=50 breaks the direction (within AUC > cross AUC), so a flat range across datasets isn't supportable. Per-dataset numbers appear in §4 with appropriate scoping.
2. **Scoped TCGA null claim** to "under our attack suite." A LiRA-strength attack might find patient-level memorization we missed.
3. **Narrowed "standard MIA"** to "cross-class matched-negative protocols, as used in e.g. Ozturk 2026 and partly Golob 2026." This is specific and defensible without overgeneralizing.

---

## Revised Recommendation: Use the Original Question

The user's original question was always the right question:

**"Can a trained gene-expression VAE be made to forget specific training samples without retraining?"**

This is the question the user actually asked. Everything in the paper exists because of this question. The biology confound, the method sweep, the Fisher theory, the synthetic data attempt — all emerged while trying to answer it.

### Why this works better than Q1+Q2 or a narrative:
- **One question, not two.** Q1 (biology confound) and Q2 (unlearning failure) are both things found while answering the original question, not independent research directions.
- **No journey structure needed.** The findings are organized by topic (evaluation → methods → theory), not by chronology. Standard paper structure.
- **No bold claims required.** The biology confound is presented as "the first obstacle we encountered" rather than "prior work overestimates by 30-90%." The percentage can appear in the results without being the headline.
- **Standard format.** Clear question, systematic evaluation, definitive answer.
- **Honest.** This is literally what happened.

### Refined question (v3):
**"Can a gene-expression VAE selectively forget a rare biological subpopulation without retraining and without destroying model utility?"**

### Proposed through-line:
*"A gene-expression VAE cannot selectively forget a rare biological subpopulation without retraining. Random (scattered) samples are not meaningfully memorized. Structured subpopulations are, and every approach either fails to reduce the memorization signal or succeeds but destroys the model. Standard MIA evaluation conflates biology with memorization, requiring corrected evaluation to see this clearly. The underlying cause is Fisher information overlap in the shared decoder, which makes selective forgetting a structural impossibility for this model class."*

### Proposed contributions:
1. **A corrected evaluation protocol** for MIA on biological generative models. Cross-class matched-negative protocols, as used in e.g. Ozturk 2026 and partly Golob 2026, can conflate cell-type identity with training membership. Within-subtype matching, a trained attacker, and multi-seed retrain baseline comparison isolate the genuine memorization signal, which concentrates in biologically coherent subpopulations. The confound magnitude varies by dataset and is documented per-dataset in §4.
2. **A systematic evaluation** of 10 unlearning approaches across 3 datasets and 2 modalities. On structured forget sets, methods either fail to reduce the memorization signal (retain fine-tune, gradient ascent, SSD, SCRUB) or reduce it but destroy model utility (Fisher scrubbing → posterior collapse, contrastive → Streisand effect, extra-gradient → seed-dependent basin-hopping). Synthetic augmentation leaks through the generative process. On scattered forget sets, memorization is negligible and unlearning is unnecessary.
3. **A structural explanation** via Fisher information overlap. The VAE's shared decoder means the parameters encoding the rare subpopulation overlap heavily with parameters encoding everything else (cosine 0.306 vs 0.018 for classifiers). Proposition 1 formalizes this for linear decoders with dimensional scaling bounds. This is why the privacy-utility tradeoff is not a tuning problem but a structural limit.

### Why this resolves the tension:
- The question matches what the user actually set out to do
- The scattered-vs-structured distinction is a finding, not an assumption
- Utility preservation is explicitly part of the question, not just privacy
- The biology confound appears naturally as "the first thing we had to solve to measure this honestly"
- No bold percentage claims required — the confound magnitude is documented per-dataset in §4, not as a headline range
- The Fisher theory explains the dilemma (can't have both privacy AND utility), not just failure

---

## Proposed Paper Structure

### Core question:
**"Can a gene-expression VAE selectively forget a rare biological subpopulation without retraining and without destroying model utility?"**

### Section outline:

**Abstract** (~250 words)
- The question: can biological VAEs forget rare subpopulations?
- To answer this, we built a corrected evaluation protocol (trained attacker + within-subtype matching + retrain baseline) because cross-class matched-negative MIA protocols can conflate biology with memorization
- Under corrected evaluation: all 10 approaches fail on structured forget sets — either privacy doesn't improve or utility collapses
- The structural explanation: Fisher overlap in shared decoders
- No flat percentage claim — magnitude varies by dataset

**§1 Introduction**
- Motivate: gene-expression VAEs memorize training data, rare subpopulations are the critical case (Feldman 2020), clinical relevance
- State the question clearly
- Summary of what was found
- Contributions (3 items):
  1. Corrected evaluation protocol for MIA on biological generative models (most transferable contribution, prominent but motivated by the question)
  2. Systematic evaluation of 10 approaches across 3 datasets — structured fail, scattered trivial
  3. Structural explanation via Fisher overlap + formal proposition

**§2 Background and Threat Model**
- Define MIA, machine unlearning, advantage metric
- **Name the threat model principle:** A realistic adversary trains a dedicated attacker on data they control, targeting the specific model. Off-the-shelf distance-based MIA (GAN-leaks, Monte Carlo) detects biology, not membership. Only trained attackers isolate memorization. (Content from NARRATIVE.md lines 44-56)
- Scope: subpopulation unlearning (rare cell types, disease subtypes), not individual sample deletion

**§3 Datasets and Evaluation Protocol**
- Three datasets: PBMC-33k, Tabula Muris, TCGA-BRCA (table from current Writeup.tex)
- Structured vs scattered forget sets — the structured sets are the hard case
- **Corrected evaluation protocol** as a named deliverable:
  - Within-subtype matching (vs cross-subtype)
  - Trained MLP attacker (vs distance-based)
  - Multi-seed retrain baseline with Welch t-test
  - Attack diversity analysis (table showing only MLP attacker gives near-chance on retrain)
- This section establishes HOW you measure before showing WHAT you measure

**§4 The Biology Confound**
- Dedicated section (promoted from current §4.4)
- Present findings WITHOUT leading with a specific percentage:
  - Cross-subtype vs within-subtype comparison on each dataset
  - Attack diversity table: off-the-shelf MIA detects biology on retrain model
  - TCGA-BRCA null result: no genuine memorization detectable under our attack suite after within-subtype correction
  - Scattered vs structured: memorization concentrates in biologically coherent subpopulations
  - Synthetic data leakage subsection: augmented model leaks via the generative process
- Frame as: "the first obstacle we encountered" and "what other evaluations may be missing"

**§5 Unlearning Results**
- 10 methods × 3 datasets under corrected evaluation
- Summary table with Welch t-test + Holm-Bonferroni
- Failure mode characterization:
  - Do nothing: retain-FT, gradient ascent, SSD, SCRUB
  - Make it worse: Fisher → posterior collapse, contrastive → Streisand effect
  - Unreliable: extra-gradient → seed-dependent basin-hopping
  - Not unlearning: DP-SGD (trains from scratch)
  - Constructive failures: synthetic augmentation (shifts memorization), RAU (Streisand)
- TCGA null result: methods "succeed" trivially when there's nothing to unlearn
- Scattered: unlearning is unnecessary — memorization is negligible

**§6 Structural Explanation**
- Fisher overlap analysis: 17× gap (VAE 0.306 vs classifier 0.018)
- Why shared decoders create overlap
- Proposition 1 + Corollary 2 (linear decoder factorization, dimensional scaling)
- Controls: deep classifier (shared layers 0.262, class-specific 0.010), z=8 (0.846, worse), conditional VAE (shared 0.222, cluster 0.000)
- Connection to failure modes: this is why the privacy-utility tradeoff is a dilemma, not a tuning problem

**§7 Discussion**
- Scope of claims: what the paper shows and doesn't show
- Connection to Ozturk/Golob (complementary, not critique)
- Practical recommendations (use corrected protocol; retrain if privacy matters)
- Limitations (single architecture family, domain-specific confound, diagonal Fisher approximation)

**§8 Conclusion**
- Answer the question directly: no, approximate unlearning cannot selectively forget rare biology from a gene-expression VAE while maintaining utility
- The corrected evaluation protocol is independently useful
- Retraining remains the only dependable option

### Content sources:
- All section content exists across Writeup.tex, NARRATIVE.md, and notebook outputs
- No new experiments needed
- This is a reorganization and rewrite

---

## Verification Plan

- All numbers trace to existing JSON output files (verified in prior sessions)
- Proposition 1 verified in NB30 (11% error, rank-1 approximation)
- Statistical tests in src/stats_utils.py
- Run `pdflatex Writeup.tex` + `bibtex Writeup` to verify compilation after rewrite
- Verify scattered-set selection methodology (user flagged this as needing verification)

---

## Open Item: Scattered Set Selection

User noted: need to verify how the scattered set was selected. This matters because the claim "scattered cells are not memorized" depends on how they were chosen. If they were selected to be non-distinctive, the finding is less interesting. If they were truly random, the finding supports the structured-memorization thesis.

Check: `outputs/p1/split_scattered.json` and the code that generates it (likely `src/define_forget_sets.py` or NB01/02).
