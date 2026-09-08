# Notebook Rewrite Plan

All 5 notebooks need: fewer prints, DataFrames for tables, matplotlib figures, interpretation after data.
Results are already computed and saved in JSON — these rewrites are presentation only, no re-execution needed.

## NB33: TCGA-BRCA Unlearning Methods

**Current:** 26 print statements, 0 plots.

**Rewrite:**
- Keep: computation cells (Fisher, retain-FT, GA, SSD, SCRUB, contrastive) — they work and produced correct results
- Fix: Replace print-formatted results summary with `pd.DataFrame`
- Add Figure 1: **Bar chart** — AUC by method with retrain floor line, color-coded pass/fail
- Add Figure 2: **Advantage comparison** — horizontal bar chart, advantage per method vs retrain CI upper bound
- Remove: print-heavy progress that duplicates what Timer already shows
- Move: interpretation text to markdown cells AFTER the figures

## NB34b: Within-Subtype Evaluation (PIVOTAL notebook)

**Current:** 33 print statements, 0 plots. This is the most important notebook for Narrative 3.

**Rewrite:**
- Add Figure 1: **Side-by-side bar chart** — cross-subtype vs within-subtype AUC for structured, baseline + retrain + all methods. This is THE key figure showing the confound.
- Add Figure 2: **Patient-level panel** — 3 groups (n=5,10,20), baseline vs retrain vs methods within-subtype AUC. Shows no memorization signal.
- Add Figure 3: **Confound magnitude** — stacked/grouped bar showing biology fraction vs memorization fraction for PBMC and TCGA-BRCA
- Replace: all print-formatted tables with `pd.DataFrame.style`
- Key insight after Figure 1: "baseline and retrain are indistinguishable with within-subtype matching"

## NB35: Fisher Overlap

**Current:** 37 print statements, 0 plots.

**Rewrite:**
- Add Figure 1: **Per-layer cosine bar chart** — TCGA-BRCA layers (Encoder, Bottleneck, Decoder hidden, Decoder output) with PBMC values overlaid for comparison
- Add Figure 2: **Gene-level cosine histogram** — distribution of per-gene cosines, with PBMC histogram overlaid
- Add Figure 3: **Log-Fisher scatter** — forget vs retain Fisher (log scale), colored by layer category
- Replace: print summaries with DataFrame
- Reference numbers against PBMC in figure annotations, not just print statements

## NB36: Cross-Domain Comparison

**Current:** 58 print statements, 0 plots. Most print-heavy notebook.

**Rewrite:**
- Add Figure 1: **Biology confound figure** — grouped bar chart showing cross-sub vs within-sub AUC for PBMC and TCGA-BRCA, with retrain overlaid. This is the paper's main evidence figure.
- Add Figure 2: **Fisher overlap comparison** — grouped bar chart, PBMC vs TCGA-BRCA per layer category
- Add Figure 3: **Cross-dataset summary heatmap** — the big comparison table as a styled heatmap
- Replace: ALL print-formatted tables with DataFrames
- Remove: the massive print block at the end — replace with a clean DataFrame and one-paragraph markdown interpretation

## Execution Order

1. NB34b first (pivotal figure for Narrative 3)
2. NB35 (Fisher figures)
3. NB36 (cross-domain master figures)
4. NB33 (unlearning comparison — least urgent, results are straightforward)

## Technical Notes

- All results already in JSON — load and visualize, no re-training needed
- Use `plt.style.use('seaborn-v0_8-whitegrid')` for consistent style
- Save all figures to respective OUTPUT_DIR
- Each notebook should be re-executable from JSON without running the training cells
- Generate notebooks via Python script (not Write tool) to avoid newline issues
