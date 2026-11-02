"""Generate notebooks/44_pbmc68k_cohort.ipynb (nbformat, per project convention).

The notebook displays the PBMC-68k cohort (second scRNA-seq cohort with two rare-cluster forget
sets) from the JSONs written by scripts/prepare_pbmc68k.py, scripts/run_pbmc68k.py and
scripts/score_pbmc68k.py. Every number comes through src/canonical.py; nothing is typed by hand.
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md("""# 44 - PBMC-68k: a second single-cell cohort with two rare-cluster forget sets

PBMC-33k contributes one rare forget set of 30 megakaryocytes (cluster 13). This notebook repeats the
whole protocol on the 68k PBMC set of Zheng et al. (2017), a second donor processed with the same
recipe as notebook 01 (QC, 2,000 highly variable genes, PCA, Leiden at resolution 0.5) and trained
with the notebook 02 recipe (`scripts/train_cohort.py`). Two rare clusters are the forget sets, each
with 40 percent of its cells held out of training so that within-type negatives exist by design.

**Cluster 9, megakaryocytes (178 cells; 107 forget, 71 held out).** PPBP, PF4 and GP9 in nearly
every cell. The published labels of Zheng et al. call these cells CD34+ because their label set
has no megakaryocyte class. This is the direct replicate of PBMC-33k cluster 13.

**Cluster 8, plasmacytoid dendritic cells (503 cells; 302 forget, 201 held out).** GZMB, MZB1,
LILRA4 and CLEC4C; the cluster also absorbs some B and CD34+ cells at this resolution, so it is a
larger and less pure rare cluster than the megakaryocytes.

For each forget set the pipeline is the paper's: one baseline (seed 42), eight clean retrains on the
retain split only (seeds 42 to 49; 42 is the audit reference), seven post-hoc methods at their PBMC-33k settings, three seeds
each, all descending from the baseline. Two matched-negative sets, cross-type (ten nearest held-out
cells of other clusters per forget cell in the baseline's latent space) and within-type (the held-out
cells of the forget cluster), each with its own attacker trained on the baseline and applied to every
checkpoint in one CPU pass (`scripts/score_pbmc68k.py`). The audit is the reference likelihood ratio
against clean retrain seed 42, with the other seven seeds giving the range a retrained model itself scores.

The two questions are the paper's. Does the cross-type protocol measure biology here as well
(retrained models above chance), and does within-type matching remove it? And do the post-hoc
methods reach the clean retrains' level on a forget set three to ten times the size of cluster 13?""")

code("""import sys
sys.path.insert(0, '../src')

import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

import canonical as C

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
OUT = Path('../outputs/pbmc68k')
FS = {'mk': 'cluster 9, megakaryocytes', 'pdc': 'cluster 8, plasmacytoid dendritic cells'}
NAMES = {'baseline': 'Baseline', 'retrain': 'Clean retrain', 'retain_finetune': 'Retain-only fine-tune',
         'gradient_ascent': 'Gradient ascent + fine-tune', 'ssd': 'SSD', 'scrub': 'SCRUB', 'fisher': 'Fisher scrubbing',
         'contrastive': 'Contrastive latent', 'extragradient': 'Extra-gradient (lambda=10)', 'long_finetune': 'Retain-only fine-tune, 200 epochs'}""")

md("""## The cohort and the two forget clusters""")

code("""adata = sc.read_h5ad('../data/pbmc68k_processed.h5ad')
sizes = adata.obs['leiden'].value_counts().sort_index()
comp = pd.crosstab(adata.obs['leiden'], adata.obs['published_label'])
print(f'{adata.n_obs:,} cells x {adata.n_vars:,} genes; {len(sizes)} Leiden clusters')
comp.loc[['9', '8']].T[lambda d: d.sum(1) > 0]""")

code("""MARKERS = {'9': ['PPBP', 'PF4', 'GP9', 'ITGA2B'], '8': ['GZMB', 'MZB1', 'LILRA4', 'CLEC4C', 'CD79A', 'CD34']}
rows = []
for cl, genes in MARKERS.items():
    genes = [g for g in genes if g in adata.var_names]
    inside = adata[adata.obs['leiden'] == cl][:, genes].X.toarray()
    outside = adata[adata.obs['leiden'] != cl][:, genes].X.toarray()
    for g, i, o in zip(genes, inside.T, outside.T):
        rows.append({'cluster': cl, 'gene': g, 'fraction expressing, cluster': (i > 0).mean(), 'fraction expressing, rest': (o > 0).mean(),
                     'mean log expression, cluster': i.mean(), 'mean log expression, rest': o.mean()})
pd.DataFrame(rows).set_index(['cluster', 'gene']).round(3)""")

code("""split_rows = []
for fs, label in FS.items():
    s = C.pbmc68k_split(fs)
    split_rows.append({'forget set': label, 'cluster size': int(sizes[s['forget_cluster']]), 'forget (trained on)': s['n_forget'],
                       'within-type held out': s['n_within_type_unseen'], 'retain': s['n_retain'], 'held out (all)': s['n_unseen'],
                       'cross-type negatives': C.pbmc68k_n_matched(fs), 'forget share of training (%)': round(100 * s['n_forget'] / (s['n_forget'] + s['n_retain']), 2)})
pd.DataFrame(split_rows).set_index('forget set')""")

md("""## Results

Advantage is 2|AUC - 0.5| with a nested bootstrap interval over seeds and cells for the transferred
cross-type attacker. `diff CI` is the nested bootstrap interval of the method's advantage minus the
clean retrains'. `audit` is the reference likelihood-ratio AUC against clean retrain seed 42 (near 1
means the forget cells are fitted as the baseline fits them; inside the retrained band means fitted as
a retrained model fits them; near 0 means fitted worse than any retrained model). The per-cell columns
are a logistic regression on standardized features refit on each checkpoint against the within-type
negatives and scored out of fold (20 fold splits), compared with the arm of eight clean retrains on the
cells every checkpoint shares (`scripts/percell_paired_test.py`; outer bootstrap over seeds, inner over the
shared cells). A method is at the retrained level per cell when the upper end of the paired interval lies
below a margin of half the baseline's excess, fixed before these scores existed; identifiable when the lower
end lies above zero; not resolved otherwise. Each checkpoint's own label-permutation null (1,000
permutations) is shown as the instrument's resolution, never as a threshold; the paper's raw-feature MLP reads little beyond the
reconstruction term and scored the baseline at 0.449 out of fold where this instrument finds 0.69
(`scripts/check_pbmc68k_extremes.py`, `check_feature_groups.py`). The transferred within-type attacker
appears in the per-seed detail only. Negative ELBO is per cell, lower is a better fit. ELBO, marker
correlation and ARI are the utility metrics on the held-out cells.""")

code("""def table(fs):
    frames = []
    for kind in ('cross',):
        S = C.pbmc68k_summary(fs, kind)
        d = pd.DataFrame({NAMES[m]: {'seeds': r['n_seeds'], f'AUC {kind}': r['auc'], f'adv {kind}': r['mean'],
                                     f'adv CI {kind}': f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                                     f'diff CI {kind}': '--' if r['diff_ci'] is None else f"[{r['diff_ci'][0]:.3f}, {r['diff_ci'][1]:.3f}]",
                                     f'audit {kind}': r['audit']} for m, r in S.items()}).T
        frames.append(d)
    PC = C.pbmc68k_refit_std(fs, 'within'); RES = C.pbmc68k_results(fs)['per_checkpoint']; PP = C.pbmc68k_percell_paired(fs, 'within')['arms']
    def nci(m, key):
        v = RES[m].get(key)
        return f"[{v['nested_ci_low']:.3f}, {v['nested_ci_high']:.3f}]" if v else '--'
    frames.append(pd.DataFrame({NAMES[m]: {'audit cross nested 95% CI': nci(m, 'audit_cross_nested'),
                                          'per-cell AUC (refit, within)': PP[m]['mean'], 'per-cell nested 95% CI': '[{:.3f}, {:.3f}]'.format(*PP[m]['nested_ci']),
                                          'per-cell AUC per seed': ', '.join(f'{a:.3f}' for a in PP[m]['per_seed_auc']),
                                          'per-cell advantage': r['advantage'],
                                          'paired diff from retrains': PP[m].get('diff', np.nan),
                                          'paired diff 95% CI': '[{:.3f}, {:.3f}]'.format(*PP[m]['diff_ci']) if 'diff_ci' in PP[m] else '--',
                                          'per-cell verdict': PP[m].get('verdict', 'reference'),
                                          'own permutation null p97.5': ', '.join(f'{x:.3f}' for x in PP[m]['perm_null_p975'])} for m, r in PC.items()}).T)
    S = C.pbmc68k_summary(fs, 'cross')
    fit = pd.DataFrame({NAMES[m]: {'forget negELBO': r['forget_neg_elbo'], 'within-type negELBO': r['within_neg_elbo'], 'cross-type negELBO': r['cross_neg_elbo'],
                                   'ELBO': r['elbo'], 'marker r': r['marker_r'], 'ARI': r['ari']} for m, r in S.items()}).T
    out = pd.concat(frames + [fit], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out

def context(fs):
    d = C.pbmc68k_results(fs)
    for kind in ('cross', 'within'):
        thr = C.pbmc68k_threshold(fs, kind); band = C.pbmc68k_audit_band(fs, kind); nf = C.pbmc68k_null_floor(fs, kind)
        print(f"{fs} / {kind} (transferred attacker): retrains' nested CI upper {thr:.3f}; retrained-vs-retrained audit band {band[0]:.2f} to {band[1]:.2f}; "
              f"permutation null of the estimator {nf['mean']:.3f} [{nf['ci_low']:.3f}, {nf['ci_high']:.3f}]")
    P = C.pbmc68k_percell_paired(fs, 'within'); A = P['arms']; nl = C.percell_refit_null(fs)
    print(f"{fs} / per-cell (standardized logistic refit, within-type, out of fold): baseline {A['baseline']['mean']:.3f}, retrains "
          f"{P['retrain_range'][0]:.3f} to {P['retrain_range'][1]:.3f} (arm mean {P['retrain_mean']:.3f}); paired non-inferiority margin "
          f"{P['margin']:.3f} (half the baseline's excess); each checkpoint's own label-permutation null has its 97.5th percentile at "
          f"{nl['p975_range'][0]:.3f} to {nl['p975_range'][1]:.3f}")""")

md("""### Cluster 9, megakaryocytes""")
code("""context('mk')
table('mk').round(3)""")

md("""### Cluster 8, plasmacytoid dendritic cells""")
code("""context('pdc')
table('pdc').round(3)""")

md("""## Per-seed detail""")
code("""def per_seed(fs):
    d = C.pbmc68k_results(fs)['per_checkpoint']
    rows = []
    for m, v in d.items():
        for r in v['per_seed']:
            rows.append({'method': NAMES[m], **{k: r[k] for k in ('seed', 'attacker_cross', 'attacker_within', 'audit_cross', 'audit_within',
                                                                 'forget_neg_elbo', 'within_neg_elbo', 'cross_neg_elbo', 'elbo', 'marker_r', 'ari')}})
    return pd.DataFrame(rows).set_index(['method', 'seed']).round(3)
per_seed('mk')""")
code("""per_seed('pdc')""")

md("""## The forget cells on the calibration axis

Each point is one checkpoint. The horizontal axis is how well the model fits the forget cells; the
vertical axis is how well it fits the held-out cells of the same cluster, which the model never
trained on. Retrained models sit where the two are equal (they know neither set); the baseline
sits to the left (forget cells fitted better than their held-out siblings). A post-hoc method
succeeds only if it moves the forget cells to the retrained models' level without moving the
held-out siblings.""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {'baseline': '#222222', 'retrain': '#2E7D5B', 'retain_finetune': '#5C6B7A', 'gradient_ascent': '#7B3FA0', 'ssd': '#B7791F',
          'scrub': '#2F5D8A', 'fisher': '#B3261E', 'contrastive': '#C2185B', 'extragradient': '#00838F', 'long_finetune': '#8D6E63'}
for ax, fs in zip(axes, FS):
    d = C.pbmc68k_results(fs)['per_checkpoint']
    for m, v in d.items():
        xs = [r['forget_neg_elbo'] for r in v['per_seed']]; ys = [r['within_neg_elbo'] for r in v['per_seed']]
        ax.scatter(xs, ys, s=70 if m in ('baseline', 'retrain') else 40, color=colors[m], marker='s' if m == 'retrain' else ('*' if m == 'baseline' else 'o'),
                   label=NAMES[m], alpha=0.9, edgecolor='white', linewidth=0.5, zorder=3)
    lo = min(min(r['forget_neg_elbo'], r['within_neg_elbo']) for v in d.values() for r in v['per_seed'])
    hi = max(max(r['forget_neg_elbo'], r['within_neg_elbo']) for v in d.values() for r in v['per_seed'])
    ax.plot([lo, hi], [lo, hi], color='k', lw=0.8, ls=':', label='forget fit = held-out fit')
    ax.set_xlabel('negative ELBO per cell, forget cells (lower = better fit)'); ax.set_ylabel('negative ELBO per cell, held-out cells of the same cluster')
    ax.set_title(FS[fs])
    ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(OUT / 'calibration_axis_68k.png', dpi=150, bbox_inches='tight')
plt.show()""")

md("""## Side by side with PBMC-33k cluster 13

The same protocol on the first cohort, for the same quantities (canonical one-attacker CPU pass;
cluster 13 has 194 cross-type negatives and no within-type negatives beyond the five unseen
megakaryocytes, which is why the 68k splits hold 40 percent of each rare cluster out).""")

code("""rows = [{'cohort / forget set': 'PBMC-33k cluster 13 (30 megakaryocytes)', 'negatives': 'cross (194)', 'baseline AUC': C.pbmc_auc('baseline'),
         'retrain AUC': C.pbmc_auc('retrain'), 'retrain advantage': C.pbmc_advantage('retrain')['mean'], 'threshold': C.pbmc_success_threshold(),
         'best post-hoc advantage': min(v['mean'] for k, v in C.pbmc_summary().items() if k not in ('baseline', 'retrain', 'retrain_leaky', 'dp_sgd'))}]
for fs, label in FS.items():
    for kind in ('cross', 'within'):
        S = C.pbmc68k_summary(fs, kind)
        rows.append({'cohort / forget set': f'PBMC-68k {label} ({C.pbmc68k_split(fs)[\"n_forget\"]} cells)',
                     'negatives': f'{kind} ({C.pbmc68k_n_matched(fs) if kind == \"cross\" else C.pbmc68k_split(fs)[\"n_within_type_unseen\"]})',
                     'baseline AUC': S['baseline']['auc'], 'retrain AUC': S['retrain']['auc'], 'retrain advantage': S['retrain']['mean'],
                     'threshold': C.pbmc68k_threshold(fs, kind),
                     'best post-hoc advantage': min(v['mean'] for k, v in S.items() if k not in ('baseline', 'retrain', 'long_finetune'))})
pd.DataFrame(rows).set_index(['cohort / forget set', 'negatives']).round(3)""")

md("""## Out-of-fold refit attackers

The transferred attacker above is trained on the baseline and scored in-sample on the baseline. For the
baseline, the clean retrains, and the long fine-tune below, an attacker is also refit on each checkpoint's
own features and scored out of fold (stratified 5-fold, 20 fold splits; the refit control of the PBMC-33k
paper). This is the honest reading of how much per-cell membership signal each model carries under
within-type negatives, and of the type signal under cross-type negatives.""")

code("""rows = []
for fs, label in FS.items():
    for kind in ('cross', 'within'):
        for m, r in C.pbmc68k_refit(fs, kind).items():
            rows.append({'forget set': label, 'negatives': kind, 'model': NAMES[m], 'seeds': r['n_seeds'], 'refit AUC (out of fold)': r['cv_auc_mean'],
                         'per-seed': ', '.join(f'{a:.3f}' for a in r['cv_auc_per_seed']), 'sd over fold splits': r['cv_auc_split_sd']})
pd.DataFrame(rows).set_index(['forget set', 'negatives', 'model']).round(3)""")

md("""## Long retain-only fine-tuning on the megakaryocytes

Notebook 42's experiment on this cohort (`scripts/long_finetune_pbmc68k.py`): 200 epochs of the methods'
own retain fine-tune from the baseline, three seeds, measured every five epochs. The retrained models fit
every megakaryocyte, trained on or held out, at the same level, because nothing in the retain set
resembles them; the baseline fits the ones it trained on better than their held-out siblings.""")

code("""lf = C.pbmc68k_long_finetune()
lv = lf['levels']; H = {sd: pd.DataFrame(h) for sd, h in lf['history'].items()}
retr = {k: np.mean([v[k] for kk, v in lv.items() if kk.startswith('retrain_clean')]) for k in ('forget', 'within', 'cross', 'retain_sample', 'unseen')}
res_mk = C.pbmc68k_results('mk'); retr_auc = np.mean(res_mk['summary']['cross']['retrain']['per_seed_auc']); band = C.pbmc68k_audit_band('mk', 'cross')
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
first = list(H)[0]
for sd, d in H.items():
    lab = sd == first
    axes[0].plot(d.epoch, d.forget_neg_elbo, color='#B3261E', lw=1.2, alpha=0.85, label='forget megakaryocytes' if lab else None)
    axes[0].plot(d.epoch, d.within_neg_elbo, color='#7B3FA0', lw=1.2, alpha=0.85, ls='-.', label='held-out megakaryocytes' if lab else None)
    axes[0].plot(d.epoch, d.matched_neg_elbo, color='#2E7D5B', lw=1.2, alpha=0.85, ls='--', label='cross-type negatives' if lab else None)
    axes[1].plot(d.epoch, d.retain_neg_elbo, color='#5C6B7A', lw=1.2, alpha=0.85, label='retain sample' if lab else None)
    axes[1].plot(d.epoch, d.unseen_neg_elbo, color='#2F5D8A', lw=1.2, alpha=0.85, ls='--', label='held-out cells' if lab else None)
    axes[2].plot(d.epoch, d.mlp_auc, color='#2F5D8A', lw=1.2, alpha=0.85, label='cross-type attacker' if lab else None)
    axes[2].plot(d.epoch, d.lr_auc, color='#B7791F', lw=1.2, alpha=0.85, label='reference audit' if lab else None)
axes[0].axhline(retr['forget'], color='k', lw=1, ls=':', label='retrained models, all megakaryocytes')
axes[2].axhline(retr_auc, color='k', lw=1, ls=':', label='retrained models, attacker')
axes[2].axhspan(band[0], band[1], color='#B7791F', alpha=0.12, label='retrained models, audit range')
axes[2].axhline(0.5, color='grey', lw=0.8, ls=':')
axes[0].set_title('megakaryocyte fit (negative ELBO per cell)'); axes[1].set_title('retain and held-out fit'); axes[2].set_title('attacker and audit AUC')
for a in axes:
    a.set_xlabel('retain fine-tune epoch'); a.legend(fontsize=8)
plt.suptitle('PBMC-68k cluster 9: 200 epochs of retain-only fine-tuning, three seeds', fontsize=13)
plt.tight_layout()
plt.savefig(OUT / 'long_finetune_68k.png', dpi=150, bbox_inches='tight')
plt.show()""")

code("""rows = []
for sd, d in H.items():
    d = d.set_index('epoch')
    for e in (0, 10, 50, 100, 150, 200):
        rows.append({'seed': sd, 'epoch': e, **{c: d.loc[e, c] for c in ('forget_neg_elbo', 'within_neg_elbo', 'matched_neg_elbo', 'retain_neg_elbo', 'unseen_neg_elbo', 'mlp_auc', 'lr_auc')}})
tab = pd.DataFrame(rows).set_index(['seed', 'epoch']).round(3)
print('retrained models:', {k: round(v, 1) for k, v in retr.items()}, f'attacker {retr_auc:.3f}, audit range {band[0]:.2f} to {band[1]:.2f}')
tab""")

code("""# the epoch-200 models under every instrument of the one-pass scorer (both attackers, both audits, utility)
if 'long_finetune' in res_mk['per_checkpoint']:
    display(pd.DataFrame([{'seed': r['seed'], **{k: r[k] for k in ('attacker_cross', 'attacker_within', 'audit_cross', 'audit_within', 'forget_neg_elbo', 'within_neg_elbo', 'elbo', 'marker_r', 'ari')}}
                          for r in res_mk['per_checkpoint']['long_finetune']['per_seed']]).set_index('seed').round(3))""")

md("""## Findings

The criterion is the adversary's: can the strongest attacker available tell the forget cells from held-out
cells of the same type any better on this model than on a model retrained without them? Each arm is compared
with the arm of eight clean retrains, paired on the cells every checkpoint shares, and passes only when the
upper end of the 95 percent interval of the difference lies below a margin fixed before the scores existed,
half the baseline's excess over the retrains (`scripts/percell_paired_test.py`). An interval that includes
zero but reaches past the margin is not resolved: the design, 107 + 71 and 302 + 201 cells, cannot tell such
a method from a retrained model or from one that removed less than half of the signal. Each checkpoint's own
label-permutation null is shown as the instrument's resolution, never as a threshold. Whether the model still
represents the type is reported beside the criterion, from the cross-type audit and the forget-cell median.

The first version of this notebook (2026-09-22, morning) passed most methods under the rule `per-cell AUC <=
max(retrains' max, null p97.5)`, with the reference arm scored under a different random state than the
methods and a noise-feature null that no script wrote; `PEER_REVIEW_ROUND7.md` records the three defects.""")

code("""from IPython.display import Markdown
def findings(fs, label):
    P = C.pbmc68k_percell_paired(fs, 'within'); A = P['arms']; nl = C.percell_refit_null(fs)
    med = C.forget_fit_quantiles(f'pbmc68k_{fs}'); rm = list(med['retrain_median'].values())
    out = [f"**{label}.** Per-cell refit on the baseline {A['baseline']['mean']:.3f} (paired excess over the retrains "
           f"{A['baseline']['diff']:+.3f} [{A['baseline']['diff_ci'][0]:+.3f}, {A['baseline']['diff_ci'][1]:+.3f}]); eight retrains "
           f"{P['retrain_range'][0]:.3f} to {P['retrain_range'][1]:.3f}; margin {P['margin']:.3f}; each checkpoint's own null "
           f"reaches {nl['p975_range'][0]:.3f} to {nl['p975_range'][1]:.3f} at its 97.5th percentile. Retrained forget-cell medians "
           f"{min(rm):,.0f} to {max(rm):,.0f}.", '']
    for m in sorted((k for k in A if k not in ('retrain', 'baseline')), key=lambda k: A[k]['diff']):
        r = A[m]; meds = [x['forget_median'] for x in med['models'][m]]
        flag = '; more identifiable than the baseline' if r['more_identifiable_than_baseline'] else ''
        out.append(f"- {NAMES[m]}: {r['mean']:.3f}, difference {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}], "
                   f"**{r['verdict']}**{flag} (Holm over the arms: {r['verdict_holm']}); against the baseline "
                   f"[{r['diff_vs_baseline_ci'][0]:+.3f}, {r['diff_vs_baseline_ci'][1]:+.3f}]; forget median {min(meds):,.0f} to {max(meds):,.0f}")
    return '\\n'.join(out)
display(Markdown(findings('mk', 'Cluster 9, megakaryocytes')))
display(Markdown(findings('pdc', 'Cluster 8, plasmacytoid dendritic cells')))""")

nb = new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                                         'language_info': {'name': 'python'}})
nbformat.write(nb, 'notebooks/44_pbmc68k_cohort.ipynb')
print('wrote notebooks/44_pbmc68k_cohort.ipynb')
