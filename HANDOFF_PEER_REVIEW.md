# HANDOFF — sweep done, NB41 run, estimators pinned (2026-09-08, evening)

Authoritative resume doc. Everything below the next heading is history. Nothing has been pushed; see
the commit made at the end of this session (working tree was committed after the sweep, checkpoints
and data excluded).

## To resume
Say **"read HANDOFF_PEER_REVIEW.md and continue"**. Next step is Stage 2, the paper rewrite (option 2
in `STAGE1_ANALYSIS.md`), to be done WITH the user (framing decisions by discussion, not multiple
choice): title, abstract, structure Q -> A -> R, measurement section first, the two-mode calibration
finding with NB41 as its test (oracle caveat on the cohort-average target), Section 7 as the mechanism
of the signal-intact mode, Corollary 2 / Cheng to an appendix, round-4 items (calibration citations in
`references.bib`). Then regenerate `Writeup.md` and `index.md` from the new `.tex`.

## What this session did
- **Sweep (user's request, before committing).** `.claude/rules/*.md` (4 files rewritten), `.claude/skills/*`
  (3 rewritten/patched), `CLAUDE.md` (rewritten), `~/mygit/CLAUDE.md` (trimmed to stable facts, defers
  numbers), `README.md` and `index.md` (rewritten to the re-derived numbers), `Writeup.md` (superseded
  banner with a headline old-vs-new table; body untouched until the rewrite). Every number checked
  against the JSONs by two scripts (token-level and by-label table check; scratch `verify_numbers.py`).
  Discrepancies found in the old mirrors are logged as CLAIMS_AUDIT section I (I1-I10).
- **Estimator bug found and fixed.** `attack_lr_reference.py`, `fisher_phase_audit.py`,
  `score_size_ablations.py` computed the negative ELBO through the VAE's unseeded single-sample
  forward pass; two runs moved a 30-cell audit AUC by up to 0.14. New `src/likelihood_audit.py`
  (eval mode, 16 seeded z samples, CPU; two runs agree exactly) is used by all three, by
  `attack_lr_reference_multi.py` (TM + TCGA audits, new), and by NB41. All three JSONs re-generated;
  all docs re-synced. Phase audit now runs on CPU (`PHASE_AUDIT_DEVICE`, default cpu) because the
  step-1e-2 endpoint differs between MPS runs (518 / 1,076; CPU 723).
- **Deterministic audit numbers (PBMC c13):** baseline 0.996; retrain-vs-retrain 0.24-0.52 (mean 0.37);
  retain-FT/GA/SSD/SCRUB/Moon/Fisher 0.991-0.996 (signal fully intact); EG lambda10 0.218 (7 seeds
  0.007-0.15, 3 seeds 0.58-0.74), lambda1 0.049; contrastive 0.004; frozen 0.002-0.007; RAU 0.007-0.018;
  augmentation 0.984-0.985; DP-SGD 0.502; leaky retrain 0.80 (the audit reads its training on the
  negatives). TM: baseline 0.877, EG 0.947, Fisher 0.346. TCGA: baseline 0.833, Fisher 0.191, others
  0.40-0.53. Scattered size ablation: audit baseline 0.69-0.79, retrain ~0.5, Fisher 0.44-0.65 (Fisher
  partly removes a per-cell signal the attacker cannot see).
- **Phase audit (CPU):** baseline forget 269.5 -> Fisher 1e-3 scrub 5,569 (retain 1,250) -> fine-tune 308
  (audit 0.987); 3e-3 -> 377 (0.173); 1e-2 -> 723 (0.155); GA ascent no-op (269.5), fine-tune 296 (0.996).
- **NB41 (new, `notebooks/41_targeted_unlearning.ipynb`, generator `scripts/make_nb41.py`, method
  `src/train_targeted_unlearn.py`, outputs `outputs/p7/`).** Part A: every checkpoint on the forget-negELBO
  axis (baseline 270, retrain 337, matched 333). Parts B/C/D: likelihood-targeted unlearning with three
  targets, 10 seeds each, curves + tables + JSON. Look-alike target (10-NN of evaluation negatives): forget
  309, MLP 0.75 (adv 0.503), audit 0.62. Cohort-average target (mean of the 194 negatives = 335 = retrain's
  level): forget 337, MLP 0.58, **adv 0.166 [0.111, 0.212] vs threshold 0.182 (6/10 seeds pass), audit
  0.41 in band, utility intact** (oracle-ish: level read off evaluation negatives). Disjoint look-alikes
  (88 unseen cells ranks 11-30): fitted at 292 by baseline AND retrain, forget lands 296, MLP 0.83, audit
  0.70. Reading: the second failure mode is the missing level; the level is the fit of a type the model
  does not know (retrain: forget 334-339, its look-alikes 292-301) and cannot be read off any cell the
  model knows. CLAIMS_AUDIT section J; STAGE1_ANALYSIS "Stage 1c".
- Re-ran NB36 (its confound figure now uses the clean retrain, but its PBMC decomposition panel still
  rests on the 5-cell within-cluster control; not used in index.md; CLAIMS I9). `figures/make_result_figures.py`
  re-run (method comparison, privacy-utility from the 09:07 pass). NB41 figures copied to `figures/`.
- **Canonical attacker pass = 09:07 (baseline 0.801/0.601).** The earlier pass gave 0.792/0.584 (A6 drift).

## Open items for the user
- `outputs/` is gitignored, so the JSONs every number traces to are not in the repo; decide whether to
  un-ignore `outputs/**/*.json` (small) so the trace is checkable from a clone.
- `data/`, `cache/`, and the paper PDFs at the root are untracked and not ignored; `git add -A` would
  commit gigabytes. `notebooks/cache/*.h5ad` and `data/filtered_gene_bc_matrices/hg19.zip` are already
  tracked (81 MB).
- The old `figures/confound_comparison.png` and NB36's decomposition need the retrain-based version.
- Writeup.tex still has the pre-audit numbers (matched negatives 137/121 -> 135/89, etc.; CLAIMS I1-I8).

# HANDOFF (earlier today) — round-5 review: code and checkpoints audited (2026-09-08)

Authoritative resume doc. Supersedes the round-4 section below. Everything is in the working tree and
**NOT committed**. Full review: `PEER_REVIEW_ROUND5.md` (source) and
https://claude.ai/code/artifact/635f6daa-1e24-4ee0-b02a-86a389e36ef2 (styled).

## To resume
Say **"read HANDOFF_PEER_REVIEW.md and continue"**. The round-4 positioning question is now downstream of
the Stage-1 repairs in `PEER_REVIEW_ROUND5.md`; do not edit paper prose until the evidence base is
re-derived. First decision needed from the user: approve Stage 1 (adopt the July-4 baseline as the single
parent; re-derive retain-FT/GA/Fisher/extra-gradient; fix the two loss functions and the attack files;
re-score everything with `retrain_clean` as the reference).

## Three blocking defects (all verified against files; see the review for tables)
- **B1 retrain leak.** `src/retrain.py` splits 85/15 over ALL non-forget cells, so the 5 canonical retrain
  seeds (`outputs/p1/multiseed_retrain`) trained on 159-170 of the 194 matched negatives (and the c12
  retrains likewise). Clean recipe-matched retrains (`scripts/retrain_clean.py`, 5 seeds in
  `outputs/p1/retrain_clean/`, scored by `scripts/score_retrain_clean.py` into
  `outputs/p4/multiseed/persample/retrain_clean.json`) give **adv 0.049, nested CI [0.042, 0.173]**
  (per-seed AUC 0.553/0.499/0.522/0.538/0.493) vs the paper's leaky 0.156 [0.082, 0.258]. Threshold becomes
  0.173; DP-SGD passes, extra-gradient (0.281) clearly fails; "retrain above chance = biology" withdrawn for
  PBMC. TCGA (NB32), TM (NB12), NB39 retrains are clean.
- **B2 two baselines.** `outputs/p1/baseline/best_model.pt` was regenerated 2026-07-04 01:21 by the audit's
  NB02 re-run. Parents: December baseline (snapshot `../final_proj_staging_ckpt_snapshot_20260704/p1/
  baseline/best_model.pt`) -> EG (10 seeds), retain-FT, grad-ascent, Fisher, frozen critics; July baseline ->
  SSD, SCRUB, contrastive, Moon, NB26-30. TM baseline+retrain also re-trained Jul 4 (paper's TM numbers are
  January's). Old vs new baseline under the same attacker: 0.785 vs 0.793. CPU re-score drifts ~0.001-0.007
  (not bit-reproducible).
- **B3 methods differ from the text.** Fisher: Fisher computed on RETAIN set (paper says forget), clamp
  saturates (forget loss 283 -> 218,000, post-scrub AUC exactly 0.5), fine-tune uses mean-over-genes MSE so KL
  is over-weighted 2000x -> the "posterior collapse" is the objective. Extra-gradient: no extragradient step,
  critics trained forget-vs-RETAIN, same 2000x KL over-weighting x lambda, and the 10 seeds early-stopped after
  3-19 single-minibatch steps (not 50 epochs); AUC tracks step count. Attack suite: threshold + LR attacks
  decode without library size (recon threshold 0.719 for every model; with library size baseline 0.780 /
  clean retrain 0.501, reversing the Table 3 reading).

## DONE 2026-09-08
- `src/canonical.py` loaders; 11 notebooks (12, 14, 21-24, 30, 34b, 35, 36, 40) load from JSON; 14/34b/35/36/40
  re-executed (34b pinned CPU: TCGA within-subtype now baseline 0.580 / retrain 0.572, tie gone; patient-level
  n=5 0.360/0.327, n=10 0.443/0.417, n=20 0.422/0.415); NB30 + NB21-24 reference cells surgically re-executed;
  NB35 MPS->numpy crash fixed. `figures/make_result_figures.py` loads from JSON; 3 PNGs + confound figure
  regenerated. `outputs/p2/moon/utility.json` (Moon ELBO 364.5 / marker_r 0.831 now trace).
- Memory: `session_2026_09_08_round5.md`.

## STAGE 1b DONE (2026-09-08 09:40). Frozen critics, both size ablations (with per-size clean retrains),
augmentation, RAU re-derived; phase audit added (`outputs/p6/phase_audit.json`). Analysis + decision in
`STAGE1_ANALYSIS.md` (Stage 1b section), verdicts `CLAIMS_AUDIT.md`, tables `CLAIMS_AUDIT_TABLES.md`.
**Phase audit (first, stochastic run; CPU rerun above supersedes):** Fisher scrub (step 1e-3) raises forget negELBO 270->5,652 and retain 363->1,254 (shared damage);
retain fine-tune restores retain to 362 AND forget to 301 (target = clean retrain's 333) -> "entangled repair"
is the mechanism of the signal-intact mode; larger steps end over-corrected (384, 518). GA ascent is a no-op.
**Frozen critics (fixed objective):** AUC 0.003-0.008, LR 0.005 (extreme over-correction; paper had 0.992).
**Size ablations:** structured per-size retrains 0.452/0.574/0.670 (n=10/50/100 confound-affected), EG overshoots
at every size; scattered: baseline≈retrain≈Fisher within 0.02 at all sizes ("Fisher works on scattered" FAILS).
**RAU:** posterior matched (KL 0.03) but LR 0.01-0.08, AUC 0.28-0.35 (over-corrected). **Augmentation:** models
never trained on forget cells score LR 0.96-0.98 / MLP 0.74-0.80 -> instruments measure subpopulation knowledge
(A8 in CLAIMS_AUDIT). Recommendation: option 2 + likelihood-targeted variant. AWAITING framing decision.
Also fixed today: `src/train_unlearn.py` utility (per_cell 60x KL), `src/synthetic_utils.py` device handling.
`LOSS_WEIGHTING_BUG.md` written for the user.

## STAGE 1 DONE (2026-09-08 03:15). Analysis: `STAGE1_ANALYSIS.md` + `CLAIMS_AUDIT.md` (artifact: see MEMORY).
**Headline numbers (PBMC c13, one CPU attacker, clean retrain reference):** baseline AUC 0.792 / adv 0.584;
clean retrain 0.529 / 0.057 [0.044, 0.182] (threshold 0.182; refit 0.035 at chance); retain-FT 0.339, GA 0.412,
SSD 0.402, SCRUB 0.431, Moon 0.480, **Fisher (fixed) 0.345 with utility intact (no collapse)**, contrastive 0.661,
EG lambda10 0.225 [0.15,0.33] (bimodal: 7 seeds overshoot / 3 undershoot; diff CI excludes 0), EG lambda1 0.327,
DP-SGD 0.068 (pass). **All post-hoc methods at baseline utility** (ELBO 363-364, marker 0.831); only DP-SGD
degrades (403/0.789). **Reference-LR audit** (`outputs/attack_lr_reference.json`): baseline 0.993, retrain-vs-
retrain 0.44, utility-preserving methods 0.90-0.98 (signal intact), EG/contrastive/Fisher-large-step 0.01-0.27
(over-corrected). TCGA: five methods 0.83, Fisher 0.784 (within 0.543). TM: baseline 0.903, retrain 0.943, EG
0.723, Fisher 0.944. c12: clean retrain 0.628 vs baseline 0.737 (confound holds), Fisher 0.717 (no collapse),
EG bimodal. **FAILS:** posterior collapse; "retrain above chance = biology" on PBMC; "signal-reducing methods
destroy the model"; Fisher overlap as the CAUSE of failure (methods change forget-set fit freely with retain
utility untouched). **Recommendation:** option 2 in STAGE1_ANALYSIS.md (promote clean-retrain + reference-LR
audit; two-mode calibration failure; Fisher section as characterization). AWAITING user's framing decision.

## (superseded) STAGE 1 IN PROGRESS note
Pipeline: `scripts/stage1_rederive.py --phase A B C` (log `outputs/stage1_rederive.log`, per-step
`outputs/stage1_steps.log`). Superseded checkpoints/results moved to `outputs/_superseded_20260908/`.
Code fixes applied: `src/train_fisher_unlearn.py` (per-sample retain Fisher, global trust-region step,
vae_loss fine-tune), `src/train_unlearn_extragradient.py` (ExtraAdam, vae_loss/n_genes utility,
`--lambda_privacy`, epoch = retain pass, no early stop), `src/threshold_attack.py` +
`src/likelihood_ratio_attack.py` (library size), `src/retrain.py` (train on split retain_indices).
Critics re-trained on July baseline (`outputs/p2/attackers`). Pilots: fixed Fisher seed42 AUC 0.676 /
adv 0.351, ELBO 363.6, KL 10.7, marker 0.831 (NO collapse). EG pilots in
`outputs/p4/multiseed/extragradient_pilot/lambda{10,1}_seed42`.
New scripts: `score_checkpoints.py`, `score_utility_all.py`, `score_persample_c12.py`,
`checkpoint_manifest.py`, `run_notebook.py`, `claims_audit.py` (old vs new tables -> `outputs/claims_audit.json`).
Phase C writes: persample (retrain = retrain_clean, retrain_leaky kept), nested_ci_all.json, retrain_refit.json
(clean), utility_all.json, p1_c12/scores_c12_v2.json, attack_suite_results.json (NB15 patched), TM
unlearning_results.json (one attacker incl. baseline/retrain), TCGA NB33->34b->36->40, figures, manifest.
After C: run `claims_audit.py`, write the claims analysis, then Stage 2 (paper text).

## Open decisions (after Stage 1)
1. Framing option 1/2/3 (STAGE1_ANALYSIS.md; recommended 2). 2. Fisher section: body characterization or appendix.
3. Reference-LR audit as main instrument? 4. Keep/re-derive/drop: frozen critics, augmentation, RAU, size ablations.
5. EG: both lambdas or lambda10 main + lambda1 ablation. 6. Then Stage 2 text + round-4 items (title, citations).

---
(Round-4 handoff preserved below.)

# HANDOFF — round-4 review + measurement-contribution reshape (2026-07-12)

Authoritative resume doc. Supersedes the round-3 content below the line. Everything is in the working tree
and **NOT committed** (user has not asked to commit).

## To resume
Say **"read HANDOFF_PEER_REVIEW.md and continue"**. The immediate next action is the OPEN QUESTION at the
top of "IN PROGRESS" (approve the positioning wording), then execute the approved plan
`~/.claude/plans/spec-it-out-and-lazy-pearl.md`.

## State of the paper
`Writeup.tex` -> `Writeup.pdf` builds clean, **20pp, 0 undefined refs**. Mirrors `Writeup.md`, `index.md`,
`README.md`. Build: `export PATH="/usr/local/bin:$PATH"; cd final_proj_staging; pdflatex -interaction=nonstopmode Writeup.tex; bibtex Writeup; pdflatex ...x2`.

## DONE this session (all uncommitted)
1. **Title -> hybrid** everywhere (tex + 3 mirrors): "Fishing for Privacy: Biological Confounds and the
   Limits of Membership Inference in Gene-Expression VAEs". NOTE round-4 flagged "Limits of Membership
   Inference" as imprecise (the real limit is on *unlearning*); title is OPEN again (Workstream 4).
2. **Retrain-refit attacker experiment (closes round-4 issue #11).** `scripts/retrain_refit_attacker.py`
   -> `outputs/p4/multiseed/retrain_refit.json`. Fresh attacker trained on each model's OWN features,
   out-of-fold, **repeated over 20 fold splits** (hardened after review finding F1). Result: baseline
   positive control adv 0.60 / AUC 0.80 (protocol has power at n=30); retrain adv **0.134 / AUC
   0.567 +/- 0.015**, nested adv CI [0.05, 0.22] (< 0.258 threshold), split SD <= 0.011. In paper
   Section 4 + Appendix "Refit attacker control" (Table 8, `app:refit`) + both mirrors.
3. **Theory reframe APPLIED** (Section 7 + abstract + contribution 3 + conclusion + both mirrors): headline
   is now "a VAE has no separable weights to target at ANY layer" (encoder 0.345, bottleneck 0.428,
   decoder-hidden 0.351, decoder-output 0.493, all 19-27x the classifier's 0.018). The 27x is scoped to
   "the layer the Fisher methods edit"; the cluster-conditional decoder is promoted to the *confirming*
   test (zero its output overlap, MIA barely moves 0.76->0.71, because the signal is encoder-borne). Jargon
   "locus"/"logit" purged per user. PBMC 0.485/0.018 regression intact. This discharged the round-4 theory
   MAJOR finding. (The old abstract "0.48" typo is gone, superseded by the reframe.)
4. **Non-VAE generality: left hedged** (decided; no experiment).
5. **MIA-calibration literature investigated + PDFs downloaded** to `refs/` (gitignored):
   `carlini2022_lira.pdf` (S&P 2022, arXiv 2112.03570), `watson2022_difficulty_calibration.pdf`
   (ICLR 2022, arXiv 2111.08440), `jayaraman2021_realistic_assumptions.pdf` (PoPETs 2021, arXiv
   2005.10881), `rezaei2021_difficulty_of_mia.pdf` (CVPR 2021, pp.7892-7900). **Novelty boundary verified
   against the actual PDFs:** all four are per-example / base-rate calibration for DISCRIMINATIVE
   classifiers; none cover generative models, group-level subpopulation confounds, or omics. Closest
   precedent = Watson 2022 p.1 ("a non-member sample may have a high membership score simply because it is
   over-represented in the data distribution"). Rezaei & Liu p.7893 explicitly EXCLUDE generative models.
   `references.bib` is MISSING all four (verified).

## Round-4 peer review (4 adversarial reviewers) — status
Overall MAJOR revision (2 majors). **FIXED:** theory-locus (item 3 above); refit fold-split variance
(item 2, 20-split CV). Verified: every Table 8 cell + all headline numbers trace to JSON.
**STILL-OPEN fixable batch (not yet applied):**
- "All six TCGA methods converge to 0.830" is FALSE -> Fisher scrubbing collapses to AUC **0.500** (fix
  prose in Section 5; reinforces narrative). Traceable to `outputs/tcga_brca/unlearning/unlearning_results.json`.
- Table 1 TCGA matched-negatives **121 -> 89** (actual `n_matched`).
- Cross-subtype TCGA inconsistency: prose 0.803/0.830 vs confound-figure 0.821/0.860 -- **NEEDS USER: which
  pair is canonical?**
- TCGA within-subtype "indistinguishable" rests on an in-sample estimator + a coincidental exact tie ->
  relabel "both near chance", lean on cluster-7.
- Table 6 "Eff. rank ratio (F)" values don't trace to `outputs/p6/fisher_overlap_results.json` -> regenerate.
- Figure 1 caption add "on PBMC-33k"; abstract closer "If privacy actually matters, retrain." soften +
  include DP-SGD.
- Size-ablation n=30 row uses superseded MPS numbers -> CPU; augmentation "0.63" -> 0.57/0.61; success
  threshold prose 0.26 -> 0.258.
- Corollary 2: define `\bar{sigma}`; soften the "1-O(M/D) vs 1/sqrt(C)" to a scaling heuristic.

## APPROVED PLAN (this session) — `~/.claude/plans/spec-it-out-and-lazy-pearl.md`
Reshape driven by 3-reviewer adversarial pass on the "elevate the MIA protocol" idea. Direction (DECIDED):
- **Workstream 1 — elevate the measurement result as a FINDING, not a general protocol.** Fuse the
  protocol + biology finding into the LEAD contribution ("corrected measurement -> flagship transcriptomics
  signal is subtype identity, not membership; real signal in 1/3 cohorts; corrects Ozturk/Golob"). Keep the
  unlearning question as the titled spine. **Cite the 4 calibration papers**, concede the principle is
  known, claim only the domain + group-level + generative novelty. **Fix retrain framing**: retrain =
  achievable-privacy FLOOR / unlearning target, NOT "the confound"; the within-subtype baseline is the
  confound estimate. Keep the existing nested-bootstrap-on-difference estimator (do NOT add a folded
  baseline-minus-retrain subtraction -- it is wrong; TCGA 0.830>0.803 would give negative membership).
  Generality = ONE Discussion sentence; keep Section 5 titled "The biology confound".
- **Workstream 2 — subordinate Fisher EXPOSITION, keep the CLAIM.** Prop 1 + overlap table +
  cluster-conditional falsification stay in main text; move Corollary 2 + the Cheng conjecture to appendix.
- **Workstream 3 — Texas-100: bounded gated Fisher-overlap appendix row OR drop.** STILL TO CONFIRM. NOT
  the MIA suite (all 3 reviewers rejected that as "doing too much"). If kept: staged, Phase-0 licensing
  first, Bernoulli head, decoder-output go/no-go pilot with cos(sigma)/cos(nu) factorization, controls,
  pre-registered support/null/void rule; confined to 1 Table 6 row + a few Section 7 sentences.
- **Workstream 4 — retitle** to restore "unlearning" (spine) + keep "biological" (punch) + subordinate
  "structural"; drop the "Fishing for Privacy" pun (conflicts with anti-narrative Q->A->R preference).
  Reviewer pick: "Nothing to Forget: Biological Confounds and the Structural Limits of Unlearning in
  Gene-Expression VAEs". Offer a plainer variant too.

## IN PROGRESS — Workstream 1, first step
**OPEN QUESTION (answer on resume):** proposed the concrete positioning text below and am AWAITING the
user's OK on the wording before editing the paper. Once approved: (1) add 4 BibTeX entries to
`references.bib`, (2) weave in the two passages, (3) fix retrain-as-floor wording, (4) sync mirrors +
rebuild + confirm PBMC numbers unchanged, then Workstream 2, then Workstream 4 (title options).

Proposed **reshaped lead contribution** (paper voice, passive, no colons/em-dashes):
> A corrected measurement overturns the premise. Evaluated against a leave-the-subpopulation-out retrain
> reference, with within-subpopulation matched negatives and a trained attacker on model-internal features,
> the apparent membership signal in gene-expression VAEs is largely biological subtype identity rather than
> training membership, and genuine removable signal survives in only one of three cohorts. This corrects
> two recent transcriptomics MIA benchmarks that report high leakage from distance-based attacks on
> synthetic outputs.

Proposed **prior-art positioning** (Related Work / Section 5):
> For discriminative classifiers, apparent membership is known to reflect example typicality or base rate
> rather than training, corrected per-example against IN/OUT reference models~\citep{carlini2022} or by
> difficulty calibration~\citep{watson2022}, or through base-rate-aware metrics~\citep{jayaraman2021}. These
> results are established only for classifiers, and~\citet{rezaei2021} explicitly decline to extend them to
> generative models. The correction used here lifts that principle to a generative VAE, where the signal is
> reconstruction and latent fidelity, and to a group-level subpopulation confound removed by retraining
> without an entire biological subtype rather than by calibrating each point.

## Open decisions
1. Positioning wording (awaiting nod) — then Workstream 1.
2. Texas-100: bounded gated Fisher row vs drop.
3. Title (Workstream 4 + round-4 open item).
4. TCGA cross-subtype canonical pair (0.803/0.830 vs 0.821/0.860).

## Memory
`session_2026_07_09_peer_review.md` (round-3/round-4 history), `feedback_ask_vs_discuss.md`,
`feedback_verify_references.md`. RESEARCH_PLAN.md = the OLD plan for THIS paper, NOT a "Paper 2".

---
(Older round-3 handoff content is preserved in git history / the memory files; this doc is the current
resume anchor.)
