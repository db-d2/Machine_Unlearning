# HANDOFF — Moon run, Cheng rebuttal, cluster-7 (2026-07-05)

State-save before a context clear. This is the authoritative resume doc for the
"run Moon, argue Cheng, add a 2nd PBMC forget set" work (user-approved via AskUserQuestion).

## TL;DR / where we are

- The user asked to: **(A4) run Moon 2024 + argue Cheng 2026 away**, and **(A1) add a second
  PBMC forget set (cluster 7)** — the two items the independent re-review left open.
- **Moon 2024 is implemented and run** (`src/train_moon_unlearn.py`, 3 seeds in
  `outputs/p2/moon/seed{42,43,44}/`). It **FAILS** (advantage 0.480) — thesis-consistent
  (it fine-tunes the decoder; 64/69 MIA features come from the untouched encoder).
- **A methodology upgrade fell out of this:** scoring is now **CPU-deterministic and
  reproducible**, and **all methods + retrain + Moon are scored in ONE pass with ONE
  attacker** (`scripts/save_persample_scores.py`, now pinned to CPU + folds in retrain via
  `load_retrain_model`). MPS was not reproducible run-to-run. New nested CIs in
  `outputs/p4/multiseed/nested_ci_all.json`.
- **This shifts ~12 numbers in the paper** (the CPU attacker is slightly stronger: baseline
  AUC 0.783→0.791). A **full Table 2 / prose / figure re-sweep is PENDING** — all new numbers
  are pre-computed below so it is mechanical.

## Already DONE and verified (do not redo)

Edits committed to `Writeup.tex` this session (all build clean, 17pp before the re-sweep):
- **fa quick fixes:** A5 (DP-SGD no longer erased from conclusion, line ~302; Fig1 caption
  line 49), A2 (extra-grad 0.300→0.282 in Table 7 line ~356 and Table 3 footnote line ~182;
  folding-aggregation disclosure added to line ~109), A3 (global 11× stated beside 27× at
  lines ~43 and ~251), B1 (decoder cosine 0.485 vs 0.493 reconciled at line ~251; factorization
  Fisher-marginal 0.50 vs data-direct 0.53 labeled at lines ~288 and ~388).
- **A6 FIXED (was factually wrong):** verified against `notebooks/03_privacy_audit.ipynb`
  (cells 9–12 train ONE attacker on the BASELINE model; cells 20–22 apply it unchanged to
  retrain and baseline). The method is **train-on-baseline, transfer to every target**.
  `train_fresh_attacker` (src eval_multiseed lines 153–174) replicates this. Line ~107 now
  describes transfer accurately and frames it as conservative (fixed baseline-tuned rule is
  more lenient than a per-model refit, yet every post-hoc method still fails).
- **Cheng 2026 bib FIXED** (`references.bib` ~line 269): real paper — Jingpu Cheng, Ping Liu,
  Qianxiao Li, Chi Zhang; ICLR 2026; arXiv:2603.26569. Method = two-phase augmented-Lagrangian
  forget-loss increase + Wasserstein-2-regularized gradient projection (for classifiers).
- **Peer-review v2 artifact published:** `PEER_REVIEW.html` →
  https://claude.ai/code/artifact/979b47d4-e913-43e3-8b3d-e7ffb26634aa (v2 review saved at
  `scratchpad/peer_review_v2.md`).

## NEW canonical numbers (CPU, one attacker, single pass) — SOURCE OF TRUTH

From `outputs/p4/multiseed/nested_ci_all.json` (regenerate via the two commands in "How to
reproduce"). Format: meanAUC | advantage [nested95] | diff-from-retrain [CI] P(>retrain) | n.

```
baseline          0.791 | 0.582 (anchor)
retain_finetune   0.666 | 0.333 [0.233,0.429] | +0.167 [+0.034,+0.294] P=0.994 | n=5
gradient_ascent   0.698 | 0.396 [0.296,0.491] | +0.231 [+0.098,+0.356] P=1.000 | n=5
ssd               0.718 | 0.435 [0.310,0.557] | +0.268 [+0.115,+0.418] P=1.000 | n=3
scrub             0.706 | 0.411 [0.280,0.540] | +0.246 [+0.089,+0.397] P=0.998 | n=3
contrastive       0.164 | 0.673 [0.570,0.755] | +0.506 [+0.375,+0.627] P=1.000 | n=3
fisher            0.808 | 0.615 [0.448,0.765] | +0.451 [+0.256,+0.620] P=1.000 | n=1
extragradient     0.433 | 0.281 [0.197,0.371] | +0.120 [-0.006,+0.240] P=0.969 | n=10
moon (NEW)        0.740 | 0.480 [0.355,0.599] | +0.314 [+0.163,+0.460] P=1.000 | n=3
dp_sgd            0.478 | 0.045 [0.026,0.176] | -0.076 [-0.188,+0.042] P=0.099 | n=3
retrain (TARGET)  0.578 | 0.156 [0.082,0.258]   <-- CI upper 0.258 is the success threshold
null floor              | 0.091 [0.003,0.253]
```

### OLD (MPS, current paper) → NEW (CPU) mapping for the re-sweep
- Success threshold (retrain CI upper): **0.229 → 0.258**
- retrain advantage: **0.13/0.132/0.148 → 0.156** ; retrain AUC **0.566/0.574 → 0.578** ;
  retrain CI **[0.069,0.233]/[0.07,0.23] → [0.082,0.258]**
- baseline: AUC **0.783 → 0.791**, advantage **0.565/0.566 → 0.582**
- retain_finetune adv **0.302 → 0.333** (AUC 0.651→0.666)
- gradient_ascent adv **0.371 → 0.396** (AUC 0.685→0.698)
- ssd adv **0.399 → 0.435** (AUC 0.699→0.718)
- scrub adv **0.398 → 0.411** (AUC 0.699→0.706)
- contrastive adv **0.605 → 0.673** (AUC 0.198→0.164)
- fisher adv **0.618 → 0.615** (AUC 0.809→0.808) [≈, minor]
- extragradient adv **0.282 → 0.281** (AUC 0.428→0.433) — STABLE. BUT diff-CI now
  **[0.02,0.25] (excludes 0) → [-0.006,0.24] (INCLUDES 0)**, P 0.988→0.969. Narrative:
  extra-grad's separation from retrain is now not statistically resolved — the honest,
  instance-independent statement is "ambiguous/not robust" (point adv 0.281 is above the
  0.258 threshold, but its difference from retrain is not significant).
- dp_sgd adv **0.073 → 0.045** (AUC 0.464→0.478), diff CI [-0.16,0.07]→[-0.188,0.042]; still
  ≈retrain (includes 0), status holds.
- moon: NEW row — AUC 0.740, adv 0.480 [0.36,0.60], diff +0.314 [0.16,0.46] P=1.000, n=3, FAIL.
  **marker_r TODO** (compute; see below).
- null floor 0.091 [0.003,0.254] → [0.003,0.253] (≈ same, can leave).

## Exact Writeup.tex edit sites (line numbers approximate — grep to confirm)
- **49** Fig1 caption: `advantage $\leq 0.229$` → `0.258`.
- **57** threat model: retrain advantage `0.13` → `0.16` (the `\geq 0.43` distance-based is
  from Table 3 / NB15, UNCHANGED — do not touch it).
- **109** stats setup: `advantage $0.13$` → `0.16`; `[0.07, 0.23]` → `[0.08, 0.26]`;
  `AUC $= 0.566$` → `0.578`; extra-grad clause `$0.282$ against $0.144$ ... mean AUC of $0.428$`
  → `$0.281$ against $0.134$ ... mean AUC of $0.433$`. Null `[0.003, 0.254]` → `[0.003, 0.253]`.
- **130** Table 2 caption: the single-run note (baseline 0.769 / retrain 0.481) is the SINGLE-SEED
  NB03 values — UNCHANGED, keep. Multi-seed means it references are now 0.791/0.578.
  Consider adding a sentence: scoring is CPU-deterministic; all rows use one attacker instance.
- **136–148** Table 2 rows — replace per the mapping above. **INSERT a Moon row** (put it near
  the post-hoc block, e.g. after extra-grad or with the utility-preserving group). Suggested:
  `Moon feature-unlearn & 3 & 0.740 & 0.480 [0.36, 0.60] & <marker_r> & FAIL \\`. Update the
  "eight post-hoc" count → "nine post-hoc" in the abstract/intro/Fig1/Table-2 caption if Moon
  is counted as post-hoc (it is). Grep `eight post-hoc` and `Ten unlearning`→ becomes eleven.
- **155** "advantages of $0.30$ to $0.40$" → "$0.33$ to $0.44$".
- **157** contrastive `0.198 (advantage $= 0.605$)` → `0.164 (advantage $= 0.673$)`;
  fisher `advantage rises to $0.618$` → `0.615`.
- **159** extra-grad `mean advantage to $0.282$` → `0.281`; rewrite the difference clause:
  now `nested $95\%$ CI $[-0.01, 0.24]$` INCLUDES zero, so say its separation from retrain is
  not statistically resolved (P above retrain 0.97) and it is the one ambiguous method.
  dp_sgd `advantage $= 0.073$ ... $[-0.16, 0.07]$` → `0.045 ... [-0.19, 0.04]`.
- **357/363** size ablation prose: retrain CI `[0.069, 0.233]` → `[0.082, 0.258]` (2 spots).
  n=50 adv 0.203 is still inside → "borderline" holds. (Table 7 point values are a SEPARATE
  size-ablation experiment, NOT the persample re-score — do NOT change 0.474/0.203/0.222; the
  n=30 row 0.282 ≈ new 0.281, fine.)
- **410** cross-dataset caption: `baseline $= 0.783$, retrain $= 0.574$` → `0.791`, `0.578`.
  (The single-seed `0.769`/`0.481` in the same caption are UNCHANGED.)
- **457** Fig5 caption: `mean ($0.132$)` → `0.156`; `[0.069, 0.233]` → `[0.082, 0.258]`;
  `DP-SGD's advantage ($0.073$)` → `0.045`.
- Grep the abstract/intro for any retrain-advantage or baseline-advantage numbers and update.

## Figures to regenerate (numbers hardcoded in the scripts)
- `figures/make_result_figures.py`: set RETRAIN_MEAN=0.156, CI_LO=0.082, CI_HI=0.258,
  BASELINE=0.582; update the `methods` list advantages (retain_ft 0.333, grad 0.396, ssd 0.435,
  scrub 0.411, contrastive 0.673, fisher 0.615, extragradient 0.281, dp_sgd 0.045) and ADD
  moon 0.480; update the privacy-utility `pts` (advantages + add moon). Threshold label
  "0.229" → "0.258". Re-run: `PYTHONPATH=src ~/.pyenv/versions/stat4243/bin/python figures/make_result_figures.py`.
- `figures/make_eval_pipeline.py` (Fig1 SVG): if it shows the 0.229 threshold, update to 0.258.

## REMAINING TASKS (in order)
1. **Moon marker_r** for the Table 2 row: compute utility (marker gene correlation) on the 3
   moon checkpoints. Look at how `outputs/utility_suite_results.json` / `src/utility_metrics.py`
   were run (NB18). Expected ~0.83 (L_recon preserves retain) but VERIFY. Until computed, the
   Moon row marker_r is a placeholder.
2. **fc3 — Cheng rebuttal paragraph** (number-independent, can do anytime). Add a technical,
   Fisher-grounded argument (Section 7 discussion, near line ~292 "Implication", or strengthen
   related-work line ~59): Cheng's Wasserstein-2 gradient projection protects semantically-
   related RETAIN samples by steering the update off directions that would degrade them; but in
   the shared decoder output the forget and retain Fisher directions are nearly aligned
   (cosine 0.485), so any projection that spares retain also spares forget — the projection has
   no separating direction to exploit. This is exactly why Cheng works for classifiers
   (class-specific outputs, cosine 0.018, near-orthogonal) and is predicted to fail in the
   generative shared-decoder regime. Tie it to Moon's empirical failure (decoder-only fine-tune,
   encoder-borne signal survives) as the two faces of the same mechanism.
3. **fc2 — cluster-7 second forget set** (NOT STARTED, the biggest remaining piece). Goal:
   show the PBMC failure is not a one-forget-set artifact. Steps: define cluster-7 forget set
   (CD14+ monocytes; ~80 unseen cells exist per line 118) + within/cross matched negatives
   (`src/define_forget_sets.py`), **retrain a cluster-7-excluded model** (retrain.py), run a
   representative method subset (extra-grad + fisher + one utility-preserving + retrain) with
   cluster 7 as forget, score with a fresh attacker. Verify baseline>retrain (signal present)
   and methods fail. Report as a compact table or paragraph. This needs its own compute + care.
4. **fc4 — integrate + rebuild + doc/memory sync.** After 1–3: rebuild
   (`bibtex` + `pdflatex`×3 via TinyTeX PATH `$HOME/.TinyTeX/bin/universal-darwin`), verify 0
   undefined refs and no stale old numbers (grep 0.229, 0.132, 0.783, 0.302, etc.). Sync
   README.md, Writeup.md, index.md, `.claude/rules/results-reference.md`,
   `.claude/rules/design-rationale.md`, both CLAUDE.md files, and memory.

## DECISION POINT for the next instance (confirm with user)
**Recommended: adopt the CPU numbers** (reproducible; Moon on equal footing; all pre-computed
above → mechanical). **Alternative (lower churn): revert to MPS** — restore
`outputs/p4/multiseed/persample_mps_backup/` + `nested_ci_all_mps_backup.json`, keep the 8
methods' validated numbers, and add ONLY the Moon row scored under its own attacker with a
footnote (Moon fails by a wide margin so its verdict is attacker-instance-insensitive). The
CPU path changes ~12 numbers + the 0.258 threshold everywhere; the MPS path changes ~2 sites
+ a Moon row. Pick based on how much number-churn is acceptable.

## How to reproduce the numbers
```
cd /Users/dbenson/mygit/final_proj_staging
export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHONPATH=src ~/.pyenv/versions/stat4243/bin/python scripts/save_persample_scores.py   # all methods+retrain+moon, CPU, one attacker
PYTHONPATH=src ~/.pyenv/versions/stat4243/bin/python scripts/nested_bootstrap_all.py     # -> outputs/p4/multiseed/nested_ci_all.json
```
Moon training (already done, 26s/seed on MPS):
```
for s in 42 43 44; do PYTHONPATH=src ~/.pyenv/versions/stat4243/bin/python src/train_moon_unlearn.py \
  --baseline_checkpoint outputs/p1/baseline/best_model.pt --data_path data/adata_processed.h5ad \
  --split_path outputs/p1/split_structured.json --output_dir outputs/p2/moon/seed$s --seed $s; done
```

## Gotchas / verified facts
- Canonical MIA = train ONE attacker on baseline features, transfer to all targets
  (`03_privacy_audit.ipynb` cells 9–12, 20–22; `train_fresh_attacker` seeds torch/np with 42).
- MPS is NOT bit-reproducible; scoring pinned to CPU via `EM.DEVICE='cpu'` in both
  save_persample_scores.py and score_retrain_supplement.py. Even CPU differs slightly across
  SEPARATE script runs (retain-sample RNG order) — so score everything in ONE run.
- retrain checkpoints lack `input_dim` in config + use BatchNorm → load via
  `load_retrain_model` (nested_bootstrap_retrain.py), not `load_vae_model`.
- Backups: `outputs/p4/multiseed/persample_mps_backup/`, `nested_ci_all_mps_backup.json`.
- Env: `~/.pyenv/versions/stat4243/bin/python`, `PYTHONPATH=src`, torch 2.11, MPS available.
- Writeup conventions: no "we", no prose colons, no em-dashes, bold only run-in headers,
  advantage = 2|AUC-0.5|, report both AUC and advantage.
```
