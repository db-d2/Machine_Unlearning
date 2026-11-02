---
layout: default
title: Home
---

# Fishing for Privacy: Unlearning a Rare Cell Type from a Gene-Expression VAE Is Not Reached, Approached, and Not Verifiable

**Status (2026-11-01).** Every experiment was re-derived from one baseline with corrected code and clean retrain references (September 2026), the paper was rewritten around that evidence, and a second single-cell cohort (PBMC-68k, two rare clusters with within-type negatives) was added. This page is generated from the paper source by `scripts/make_index_md.py`.

David Benson, Columbia University

[View Code](https://github.com/db-d2/Machine_Unlearning) | [Writeup (Markdown)](./Writeup.md) | [Writeup (PDF)](./Writeup.pdf)

## Abstract

Can a gene-expression VAE undergo unlearning of a rare cell type without losing utility? Nine post-hoc unlearning methods and DP-SGD are tested on scVI-style VAEs trained on single-cell and bulk RNA-seq, against models retrained without the forget cells. A method has unlearned the cells when the strongest available membership attack separates them from cells the model never saw no better than it does on the retrained models, whether or not the model still represents their type. No method meets that standard at the budget its authors use. Where the cells can be tested, every method on one forget set and four of seven on the other, two of them after a multiplicity correction, leave them more identifiable than the retrained models do; the rest are not resolved, on intervals too wide to certify equivalence. Where only the type can be tested, the retrained level is not reached, and the methods miss it in two kinds. Those that edit and then fine-tune leave the cells more identifiable than a retrained model does. Those that push them with a continuous objective leave them less identifiable, fitted worse than a model that never saw them wherever the audit resolves them.

Retain-only fine-tuning run to 200 epochs, four times the published retain fine-tune’s budget, erodes the memorized fit with held-out utility inside its margins. On the PBMC-68k megakaryocytes it brings them into the spread of the retrained models, though the design cannot certify that it arrives. Nothing a controller can observe marks arrival, so the result cannot be verified from the model alone. Two measurements explain why. A retain fine-tune reverts the part of an edit that lies in the directions the retain data constrain and compensates the rest. In the shared decoder on PBMC-33k almost every direction the forget cells use is one the retain cells use too, so these methods reduce to retain fine-tuning. The level a retrained model gives the forget cells is the fit of a type it does not know, which no cell it knows supplies. The usual evaluation, which scores the forget cells against cells of other types, measures how well the model fits the type, not whether it trained on the cells.

## Contributions

1.  The answer. At published budgets no method is shown to unlearn a rare cell type from a gene-expression VAE, by a paired comparison with retrained models on the same cells. On PBMC-33k, where the design can certify equivalence, the retrained level is not reached; on PBMC-68k no interval is narrow enough to certify it. The misses come in two kinds that the usual advantage cannot distinguish. Long retain-only fine-tuning approaches the retrained models without a utility cost the margins detect, and the design cannot certify that it arrives. None of the observable quantities tried, including a calibration on held-back members, marks arrival, so the result is not verifiable without a retrained model.

2.  The reasons. A retain fine-tune reverts the part of an edit that lies in the directions the retain data constrain (Proposition <a href="#prop:restoration" data-reference-type="ref" data-reference="prop:restoration">[prop:restoration]</a>), so what a retain fine-tune leaves of an edit in the weights is its forget Fisher mass on parameters the retain set does not use. At ten units of the proposition’s own time that share orders the settings, 0.02 in the shared decoder on PBMC-33k, 0.22 where a rare type is routed through hidden units the retain set does not use, and 0.77 in a classifier. The appended fine-tunes’ ten epochs are a fraction of one unit. An intervention that confines an edit to those parameters confirms it under gradient descent, and finds Adam reverting those parameters too. The forget cells’ fit is nonetheless largely restored around an edit that survives. A linear-response calculation (Proposition <a href="#prop:response" data-reference-type="ref" data-reference="prop:response">[prop:response]</a>) separates reversal of an edit from compensation of its damage. The saved fine-tunes show two routes to compensation, a counterweight the repair builds through parameters the edit did not touch and the retain fine-tune’s own erosion of the forget type. The retain-null mass orders the residual fits without bounding them.

    The level the overshooting methods lack is the fit of a type the model does not know, which targeted fine-tuning shows is not observable. Matching that level is not sufficient either. The per-point signal survives a fine-tune given the oracle level in seven of eight simulation configurations, and on PBMC-33k the attacker verdict for such a fine-tune is not resolved.

3.  The measurement (the paper). A reference retrained on exactly the retain split, held-out cells of the forget type, an attacker refit on standardized features, and a reference-free likelihood audit, without which an attacker’s score on these data reads biology.

## Main figures

### Evaluation pipeline

![Evaluation pipeline](figures/eval_pipeline.svg)

### Where every PBMC-33k checkpoint lands relative to the retrained models

![Calibration axis](figures/calibration_axis.png)

### Long retain-only fine-tuning on PBMC-33k

![Long fine-tune](figures/pbmc_long_finetune.png)

### Erosion by support in the simulation

![Erosion by support](figures/erosion_by_support.png)

### Fisher overlap across datasets

![Fisher overlap](figures/fisher_overlap_cross_dataset.png)

## Key results

Every number below is produced by `scripts/make_tex_tables.py` from the result files under `outputs/` (the same tables the paper prints). Advantage is 2|AUC - 0.5| with its nested 95 percent interval; the audit is the likelihood-ratio AUC against a model retrained on exactly the retain split.

### PBMC-33k cluster 13 (30 megakaryocytes against 194 matched negatives)

Every method on PBMC-33k cluster 13 (30 megakaryocytes against 194 matched negatives). One attacker, trained on the baseline, scores every checkpoint. The paired difference compares each arm with the five retrained models on the shared cells and is read as the paper describes, at the margin of 0.107. The audit is the reference-free audit against the mean of the retrained models with its paired verdict; ELBO and marker $`r`$ are held-out utility. Intervals resample the cells once for every model and the seeds within each; the two Fisher step caps are one seed each. The advantage is the mean over seeds of each seed’s $`2|\mathrm{AUC}-0.5|`$; for extra-gradient at $`\lambda=10`$ it exceeds $`2|\mathrm{AUC}-0.5|`$ of the arm’s mean AUC because its seeds fall on both sides of 0.5. Generated from `paired_diff_33k.json`, `audit_reference_free.json`, and `utility_intervals.json`.

| Method | Seeds | Attacker AUC \[nested 95% CI\] | Advantage | Paired difference \[95% CI\] | Attacker | Reference-free audit \[95% CI\] | Audit | ELBO | Marker $`r`$ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Baseline (no unlearning) | 1 | 0.801 \[0.70, 0.89\] | 0.601 | +0.272 \[0.21, 0.33\] | trained on the cells | 0.995 \[0.99, 1.00\] | trained on the cells | 364.0 | 0.831 |
| Clean retrain | 5 | 0.529 \[0.42, 0.63\] | 0.057 | – | reference | 0.507 \[0.40, 0.60\] | reference | 364.2 | 0.831 |
| Retain-only fine-tune | 5 | 0.669 \[0.56, 0.77\] | 0.339 | +0.141 \[0.10, 0.19\] | more identifiable | 0.988 \[0.93, 1.00\] | fitted better | 363.0 | 0.832 |
| Gradient ascent + fine-tune | 5 | 0.706 \[0.59, 0.81\] | 0.412 | +0.177 \[0.14, 0.22\] | more identifiable | 0.995 \[0.99, 1.00\] | fitted better | 363.2 | 0.831 |
| SSD | 3 | 0.701 \[0.59, 0.80\] | 0.402 | +0.172 \[0.13, 0.22\] | more identifiable | 0.995 \[0.99, 1.00\] | fitted better | 363.6 | 0.831 |
| SCRUB | 3 | 0.716 \[0.60, 0.82\] | 0.431 | +0.187 \[0.14, 0.24\] | more identifiable | 0.980 \[0.92, 1.00\] | fitted better | 363.6 | 0.831 |
| Moon feature unlearning | 3 | 0.740 \[0.63, 0.84\] | 0.480 | +0.211 \[0.16, 0.26\] | more identifiable | 0.996 \[0.99, 1.00\] | fitted better | 364.5 | 0.831 |
| Fisher scrubbing | 3 | 0.673 \[0.56, 0.77\] | 0.345 | +0.144 \[0.10, 0.19\] | more identifiable | 0.992 \[0.98, 1.00\] | fitted better | 363.6 | 0.832 |
| Fisher, step cap $`3\times 10^{-3}`$ | 1 | 0.312 \[0.21, 0.42\] | 0.376 | -0.217 \[-0.30, -0.14\] | less identifiable | 0.141 \[0.03, 0.27\] | fitted worse | 363.8 | 0.831 |
| Fisher, step cap $`10^{-2}`$ | 1 | 0.445 \[0.33, 0.56\] | 0.111 | -0.084 \[-0.14, -0.02\] | less identifiable | 0.288 \[0.15, 0.45\] | fitted worse | 370.8 | 0.830 |
| Extra-gradient ($`\lambda=10`$) | 10 | 0.411 \[0.31, 0.52\] | 0.225 | -0.117 \[-0.19, -0.05\] | less identifiable | 0.250 \[0.001, 0.91\] | not resolved | 363.8 | 0.832 |
| Extra-gradient ($`\lambda=1`$) | 10 | 0.336 \[0.23, 0.44\] | 0.327 | -0.192 \[-0.28, -0.12\] | less identifiable | 0.044 \[0.0003, 0.20\] | fitted worse | 363.2 | 0.831 |
| Contrastive latent | 3 | 0.170 \[0.11, 0.24\] | 0.661 | -0.359 \[-0.44, -0.28\] | less identifiable | 0.003 \[0.00, 0.01\] | fitted worse | 363.7 | 0.831 |
| Frozen critics (4 configurations) | 3 each | 0.003 to 0.008 | 0.985 to 0.995 | -0.526 to -0.521 | less identifiable | 0.003 to 0.006 | fitted worse | 363.7 to 404.4 | 0.831 to 0.832 |
| DP-SGD ($`\varepsilon=10`$, from scratch) | 3 | 0.466 \[0.37, 0.56\] | 0.068 | -0.063 \[-0.12, -0.01\] | less identifiable | 0.533 \[0.43, 0.64\] | as retrained | 402.9 | 0.789 |

### PBMC-68k, two rare clusters with within-type negatives

Every method on the two PBMC-68k forget sets. One cross-type attacker, trained on the baseline, scores every checkpoint in one CPU pass. The forget-cell fit is the negative ELBO per cell, ranged over seeds. The audit is the cross-type likelihood-ratio AUC against clean retrain seed 42 with a nested interval, the reference row being the range the other seven seeds span. The per-cell columns are the standardized logistic refit against the within-type negatives, out of fold, with a nested interval over seeds and the shared cells, and its difference from the eight retrained models paired on those cells. The status applies the criterion of the paper. Generated from `results_mk.json`, `results_pdc.json`, and `percell_paired_{mk,pdc}.json`.

<table>
<thead>
<tr>
<th style="text-align: left;">Method</th>
<th style="text-align: center;">Seeds</th>
<th style="text-align: center;">Forget median</th>
<th style="text-align: center;">Per-cell AUC, within [nested 95% CI]</th>
<th style="text-align: center;">Paired difference [95% CI]</th>
<th style="text-align: center;">Cells</th>
<th style="text-align: center;">Reference-free audit [95% CI]</th>
<th style="text-align: center;">Median gap [95% CI]</th>
<th style="text-align: center;">Type</th>
<th style="text-align: center;">ELBO</th>
<th style="text-align: center;">Marker <span class="math inline"><em>r</em></span></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11" style="text-align: left;"><em>Cluster 9, megakaryocytes (107 forget cells, 71 within-type and 143 cross-type negatives; margin 0.044)</em></td>
</tr>
<tr>
<td style="text-align: left;">Baseline (no unlearning)</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">491</td>
<td style="text-align: center;">0.683 [0.61, 0.76]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">trained on the cells</td>
<td style="text-align: center;">1.000 [1.00, 1.00]</td>
<td style="text-align: center;">-585 [-628, -548]</td>
<td style="text-align: center;">trained on the cells</td>
<td style="text-align: center;">484.5</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Clean retrain</td>
<td style="text-align: center;">8</td>
<td style="text-align: center;">1,014 to 1,154</td>
<td style="text-align: center;">0.472 to 0.538 [0.45, 0.57]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">reference</td>
<td style="text-align: center;">0.592 [0.41, 0.66]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">reference</td>
<td style="text-align: center;">488.0</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">Retain-only fine-tune</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">665 to 697</td>
<td style="text-align: center;">0.542 to 0.564 [0.48, 0.63]</td>
<td style="text-align: center;">+0.041 [-0.03, 0.11]</td>
<td style="text-align: center;">not resolved</td>
<td style="text-align: center;">1.000 [1.00, 1.00]</td>
<td style="text-align: center;">-395 [-441, -354]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.9</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">Gradient ascent + fine-tune</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">615 to 634</td>
<td style="text-align: center;">0.568 to 0.606 [0.51, 0.65]</td>
<td style="text-align: center;">+0.066 [-0.01, 0.14]</td>
<td style="text-align: center;">not resolved</td>
<td style="text-align: center;">1.000 [1.00, 1.00]</td>
<td style="text-align: center;">-452 [-496, -415]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.6</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">SSD</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">607 to 614</td>
<td style="text-align: center;">0.635 to 0.664 [0.57, 0.72]</td>
<td style="text-align: center;">+0.132 [0.06, 0.20]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">1.000 [1.00, 1.00]</td>
<td style="text-align: center;">-464 [-503, -431]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.8</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">SCRUB</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">572 to 622</td>
<td style="text-align: center;">0.596 to 0.609 [0.53, 0.67]</td>
<td style="text-align: center;">+0.086 [0.02, 0.15]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.994 [0.97, 1.00]</td>
<td style="text-align: center;">-480 [-530, -424]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.8</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Fisher scrubbing</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">814 to 831</td>
<td style="text-align: center;">0.591 to 0.622 [0.53, 0.68]</td>
<td style="text-align: center;">+0.088 [0.02, 0.16]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.653 [0.53, 0.76]</td>
<td style="text-align: center;">-251 [-356, -56]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">487.0</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Extra-gradient (<span class="math inline"><em>λ</em> = 10</span>)</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">475 to 489</td>
<td style="text-align: center;">0.840 to 0.914 [0.81, 0.93]</td>
<td style="text-align: center;">+0.361 [0.29, 0.44]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.999 [1.00, 1.00]</td>
<td style="text-align: center;">-594 [-635, -556]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">486.0</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Contrastive latent</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">663 to 755</td>
<td style="text-align: center;">0.549 to 0.564 [0.48, 0.63]</td>
<td style="text-align: center;">+0.043 [-0.03, 0.11]</td>
<td style="text-align: center;">not resolved</td>
<td style="text-align: center;">0.971 [0.92, 1.00]</td>
<td style="text-align: center;">-365 [-444, -295]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">485.7</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">Retain-only fine-tune, 200 epochs</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">820 to 846</td>
<td style="text-align: center;">0.506 to 0.539 [0.45, 0.59]</td>
<td style="text-align: center;">+0.008 [-0.06, 0.08]</td>
<td style="text-align: center;">not resolved</td>
<td style="text-align: center;">1.000 [1.00, 1.00]</td>
<td style="text-align: center;">-247 [-290, -205]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">486.7</td>
<td style="text-align: center;">0.756</td>
</tr>
<tr>
<td colspan="11" style="text-align: left;"><em>Cluster 8, plasmacytoid dendritic cells (302 forget cells, 201 within-type and 445 cross-type negatives; margin 0.038)</em></td>
</tr>
<tr>
<td style="text-align: left;">Baseline (no unlearning)</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">532</td>
<td style="text-align: center;">0.603 [0.56, 0.65]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">trained on the cells</td>
<td style="text-align: center;">0.999 [1.00, 1.00]</td>
<td style="text-align: center;">-148 [-160, -131]</td>
<td style="text-align: center;">trained on the cells</td>
<td style="text-align: center;">484.5</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Clean retrain</td>
<td style="text-align: center;">8</td>
<td style="text-align: center;">672 to 684</td>
<td style="text-align: center;">0.450 to 0.503 [0.45, 0.52]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">reference</td>
<td style="text-align: center;">0.510 [0.48, 0.53]</td>
<td style="text-align: center;">–</td>
<td style="text-align: center;">reference</td>
<td style="text-align: center;">486.7</td>
<td style="text-align: center;">0.756</td>
</tr>
<tr>
<td style="text-align: left;">Retain-only fine-tune</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">580 to 585</td>
<td style="text-align: center;">0.542 to 0.558 [0.50, 0.59]</td>
<td style="text-align: center;">+0.062 [0.02, 0.10]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.994 [0.99, 1.00]</td>
<td style="text-align: center;">-97 [-113, -84]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.7</td>
<td style="text-align: center;">0.760</td>
</tr>
<tr>
<td style="text-align: left;">Gradient ascent + fine-tune</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">568 to 571</td>
<td style="text-align: center;">0.550 to 0.584 [0.52, 0.61]</td>
<td style="text-align: center;">+0.079 [0.04, 0.12]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.997 [0.99, 1.00]</td>
<td style="text-align: center;">-110 [-123, -96]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.6</td>
<td style="text-align: center;">0.760</td>
</tr>
<tr>
<td style="text-align: left;">SSD</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">561 to 572</td>
<td style="text-align: center;">0.560 to 0.582 [0.53, 0.62]</td>
<td style="text-align: center;">+0.087 [0.04, 0.13]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.999 [1.00, 1.00]</td>
<td style="text-align: center;">-112 [-130, -99]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.7</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">SCRUB</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">555 to 557</td>
<td style="text-align: center;">0.540 to 0.602 [0.52, 0.62]</td>
<td style="text-align: center;">+0.084 [0.03, 0.13]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.999 [1.00, 1.00]</td>
<td style="text-align: center;">-124 [-137, -111]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">484.5</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Fisher scrubbing</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">697 to 812</td>
<td style="text-align: center;">0.583 to 0.593 [0.54, 0.63]</td>
<td style="text-align: center;">+0.102 [0.06, 0.14]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.489 [0.39, 0.59]</td>
<td style="text-align: center;">+60 [-6, +226]</td>
<td style="text-align: center;">not resolved</td>
<td style="text-align: center;">493.9</td>
<td style="text-align: center;">0.761</td>
</tr>
<tr>
<td style="text-align: left;">Extra-gradient (<span class="math inline"><em>λ</em> = 10</span>)</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">492 to 508</td>
<td style="text-align: center;">0.703 to 0.828 [0.70, 0.83]</td>
<td style="text-align: center;">+0.283 [0.21, 0.35]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.929 [0.80, 1.00]</td>
<td style="text-align: center;">-180 [-196, -160]</td>
<td style="text-align: center;">fitted better</td>
<td style="text-align: center;">486.4</td>
<td style="text-align: center;">0.762</td>
</tr>
<tr>
<td style="text-align: left;">Contrastive latent</td>
<td style="text-align: center;">3</td>
<td style="text-align: center;">747 to 777</td>
<td style="text-align: center;">0.630 to 0.633 [0.58, 0.68]</td>
<td style="text-align: center;">+0.145 [0.10, 0.19]</td>
<td style="text-align: center;">more identifiable</td>
<td style="text-align: center;">0.011 [0.00004, 0.04]</td>
<td style="text-align: center;">+87 [+61, +106]</td>
<td style="text-align: center;">fitted worse</td>
<td style="text-align: center;">487.9</td>
<td style="text-align: center;">0.760</td>
</tr>
</tbody>
</table>

### The confound across cohorts

Attacker AUC of the baseline and the retrained model on every cohort, each with its advantage $`2|\mathrm{AUC}-0.5|`$ in parentheses, under cross-type negatives and, where a held-out set of the same type exists, within-type negatives (the standardized logistic refit). An inverted AUC yields a large advantage, which is why the AUC is reported beside it; the retrained models’ large advantages here are their worse fit of the forget type, not leakage. Generated from the per-cohort result files.

| Cohort and forget set | Baseline, cross | Retrained, cross | Baseline, within | Retrained, within |
|:---|:--:|:--:|:--:|:--:|
| PBMC-33k cluster 13 (30 cells; 194 cross) | 0.801 (0.60) | 0.529 (0.06) | – | – |
| PBMC-33k cluster 12 (49 cells; 57 cross) | 0.869 (0.74) | 0.814 (0.63) | – | – |
| PBMC-33k cluster 7 (50 cells; 80 held out within) | 0.665 (0.33) | 0.655 (0.31) | 0.466 (0.07) | 0.414 (0.17) |
| PBMC-68k cluster 9 (107 cells; 143 cross, 71 held out within) | 0.965 (0.93) | 0.011 (0.98) | 0.683 (0.37) | 0.515 (0.03) |
| PBMC-68k cluster 8 (302 cells; 445 cross, 201 held out within) | 0.581 (0.16) | 0.229 (0.54) | 0.603 (0.21) | 0.487 (0.03) |
| Tabula Muris cluster 33 (82 cells; 135 cross) | 0.903 (0.81) | 0.943 (0.89) | – | – |
| Tabula Muris cluster 28 (50 cells; 67 held out within) | 0.400 (0.20) | 0.379 (0.24) | 0.487 (0.03) | 0.550 (0.10) |
| TCGA-BRCA Basal (158 patients; 89 cross, 30 within) | 0.803 (0.61) | 0.830 (0.66) | 0.442 (0.12) | 0.404 (0.19) |

### Long retain-only fine-tuning

Long retain-only fine-tuning at the methods’ learning rate, ranges over three seeds, on PBMC-33k cluster 13 and TCGA-BRCA Basal, with NB42’s own per-epoch instruments: mean negative ELBO per cell, the notebook’s own instance of the attacker trained on the baseline, and the likelihood-ratio audit against clean retrain seed 42. Under the paper’s instruments the 200-epoch PBMC checkpoints score 0.568 with the canonical attacker (the paired test of the paper, `paired_diff_33k.json`) and 0.688 \[0.48, 0.88\] under the reference-free audit, not resolved (the paper). The TCGA run overfits its 767-patient retain set after epoch 100 and every group is fitted better again, so its first 100 epochs are the informative ones. Generated from `long_finetune.json` under `outputs/p8`.

| Retain epochs | PBMC forget negELBO, mean | PBMC attacker (the run’s own instance) | PBMC audit vs. retrain seed 42 \[nested 95% CI at 200\] | TCGA forget negELBO, mean | TCGA unseen Basal, mean | TCGA audit vs. the TCGA retrain \[95% CI at 200\] |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 269.5 | 0.795 to 0.799 | 0.996 | 1116.5 | 1086.0 | 0.833 |
| 10 | 285.0 to 286.7 | 0.734 to 0.750 | 0.995 to 0.996 | 1253.1 to 1277.8 | 1220.4 to 1247.7 | 0.338 to 0.691 |
| 50 | 303.6 to 306.8 | 0.669 to 0.688 | 0.992 to 0.994 | 1256.9 to 1257.5 | 1225.5 to 1228.3 | 0.482 to 0.503 |
| 100 | 317.6 to 320.3 | 0.580 to 0.615 | 0.944 to 0.971 | 1216.2 to 1257.1 | 1190.7 to 1224.4 | 0.419 to 0.494 |
| 200 | 327.7 to 331.3 | 0.552 to 0.561 | 0.519 to 0.662 \[0.46, 0.68\] | 938.2 to 1023.8 | 866.6 to 952.9 | 0.638 to 0.671 \[0.61, 0.71\] |
| Retrained models | 337.2 | 0.529 (canonical instance) | 0.24 to 0.52 | 1277.4 | 1246.4 | – |

Long retain-only fine-tuning on PBMC-68k cluster 9 (`scripts/long_finetune_pbmc68k.py`, the run whose 200-epoch checkpoints the paper scores), ranges over three seeds. Fits are mean negative ELBO per cell; the attacker is the script’s own instance of the cross-type attacker trained on the baseline, and the audit the cross-type likelihood-ratio AUC against clean retrain seed 42, with the nested interval at 200 epochs. The last row gives the eight clean retrains’ ranges, their attacker range from the paper, and the range the audit spans over every ordered pair of them. Generated from `long_finetune.json` under `outputs/pbmc68k`.

| Retain epochs | Forget cells, mean | Held-out megakaryocytes, mean | Retain sample, mean | Held-out cells, mean | Fit-excess attacker, cross | Audit vs. retrain seed 42, cross \[nested 95% CI at 200\] |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| 0 | 488.3 | 521.7 | 481.7 | 484.4 | 0.964 to 0.967 | 1.000 |
| 10 | 543.0 to 558.1 | 559.9 to 573.2 | 481.0 to 481.1 | 484.0 to 484.1 | 0.945 to 0.951 | 1.000 |
| 50 | 659.0 to 678.5 | 658.4 to 675.3 | 479.3 to 479.4 | 483.9 to 483.9 | 0.740 to 0.802 | 1.000 |
| 100 | 728.0 to 757.1 | 722.6 to 751.9 | 477.6 to 477.7 | 483.9 to 484.0 | 0.467 to 0.598 | 1.000 |
| 200 | 790.7 to 813.4 | 788.4 to 808.6 | 474.4 to 474.5 | 484.7 to 484.9 | 0.312 to 0.380 | 1.000 \[1.00, 1.00\] |
| Retrained models | 981.2 to 1106.3 | 975.1 to 1108.8 | 481.2 to 482.3 | 483.9 to 485.0 | 0.007 to 0.017 | 0.03 to 0.97 (every ordered pair) |

## Reproducing

See `README.md` for the environment and the order of the notebooks (01 to 44) and scripts; `HANDOFF_PEER_REVIEW.md` records the state of the work and `CLAIMS_AUDIT.md` the verdict on every claim.
