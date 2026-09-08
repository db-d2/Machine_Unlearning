#!/usr/bin/env python3
"""Reproducible source for the three data figures that must track the canonical numbers:
Fisher cross-dataset overlap (per-sample), the MIA-advantage bar chart, and the
privacy-utility plane.

Every value is loaded through src/canonical.py from the JSON its experiment wrote. Nothing
is typed in here, so re-running this script after a re-score regenerates the PNGs with the
current numbers. Run from anywhere:

    PYTHONPATH=src python figures/make_result_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(D), "src"))
import canonical as C

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.edgecolor": "#888", "savefig.dpi": 150, "font.family": "DejaVu Sans"})
BLUE, RED, GREEN, ORANGE, GREY = "#5B9BD5", "#E15759", "#2CA02C", "#F0A030", "#7f7f7f"

LABELS = {"contrastive": "Contrastive latent", "fisher": "Fisher scrubbing",
          "moon": "Moon feature-unlearn", "ssd": "SSD", "scrub": "SCRUB",
          "gradient_ascent": "Gradient ascent", "retain_finetune": "Retain-only FT",
          "extragradient": "Extra-gradient", "dp_sgd": "DP-SGD (eps=10)", "retrain": "Retrain"}

# ---------- Fig 3: Fisher overlap, per-sample (PBMC vs TCGA) ----------
layers = ["Encoder", "Bottleneck", "Decoder hidden", "Decoder output"]
pf, tf = C.pbmc_fisher(), C.tcga_fisher()
pbmc = [pf["per_category"][l] for l in layers]
tcga = [tf["per_category"][l] for l in layers]
pbmc_g, tcga_g = pf["global_cosine"], tf["global_cosine"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.4, 1]})
x = np.arange(len(layers)); w = 0.38
a1.bar(x - w/2, pbmc, w, label="PBMC-33k", color=BLUE)
a1.bar(x + w/2, tcga, w, label="TCGA-BRCA", color=RED)
for i, (p, t) in enumerate(zip(pbmc, tcga)):
    a1.text(i - w/2, p + 0.015, f"{p:.3f}", ha="center", fontsize=10)
    a1.text(i + w/2, t + 0.015, f"{t:.3f}", ha="center", fontsize=10)
a1.set_xticks(x); a1.set_xticklabels(layers, rotation=12); a1.set_ylim(0, 1.0)
a1.set_ylabel("Fisher cosine similarity"); a1.set_title("Per-layer Fisher overlap (per-sample)"); a1.legend(frameon=False)
a2.bar([0, 1], [pbmc_g, tcga_g], color=[BLUE, RED], width=0.6)
a2.text(0, pbmc_g + 0.015, f"{pbmc_g:.3f}", ha="center", fontsize=13, fontweight="bold")
a2.text(1, tcga_g + 0.015, f"{tcga_g:.3f}", ha="center", fontsize=13, fontweight="bold")
a2.set_xticks([0, 1]); a2.set_xticklabels(["PBMC-33k (0.1%)", "TCGA-BRCA (17%)"]); a2.set_ylim(0, 1.0)
a2.set_ylabel("Global Fisher cosine"); a2.set_title("Global Fisher overlap")
plt.tight_layout(); plt.savefig(os.path.join(D, "fisher_overlap_cross_dataset.png")); plt.close()

# ---------- Fig 5: MIA advantage bar chart (PBMC structured) ----------
summary = C.pbmc_summary()
order = ["contrastive", "fisher", "moon", "ssd", "scrub", "gradient_ascent",
         "retain_finetune", "extragradient", "dp_sgd", "retrain"]
methods = [(LABELS[m], summary[m]["mean"]) for m in order]
RETRAIN_MEAN = summary["retrain"]["mean"]
CI_LO, CI_HI = summary["retrain"]["ci_low"], summary["retrain"]["ci_high"]
BASELINE = summary["baseline"]["mean"]
names = [m[0] for m in methods][::-1]; vals = [m[1] for m in methods][::-1]
cols = [GREEN if v <= CI_HI else RED for v in vals]
fig, ax = plt.subplots(figsize=(13, 6.5))
y = np.arange(len(names)); ax.barh(y, vals, color=cols)
for i, v in enumerate(vals): ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10)
ax.axvspan(CI_LO, CI_HI, color=GREEN, alpha=0.10)
ax.axvline(RETRAIN_MEAN, color=GREEN, ls="--", lw=2)
ax.text(RETRAIN_MEAN + 0.004, len(names) - 0.4, f"Retrain mean ({RETRAIN_MEAN:.3f})", color=GREEN, fontweight="bold", fontsize=11)
ax.axvline(BASELINE, color="#999", ls=":", lw=1.3)
ax.text(BASELINE + 0.006, 6.5, f"Baseline ({BASELINE:.3f})", color="#777", fontsize=10, rotation=90, va="center")
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 0.78)
ax.set_xlabel("MIA Advantage (lower = better privacy)")
ax.set_title("Unlearning method comparison: MIA advantage on PBMC structured forget set")
ax.legend(handles=[Patch(color=GREEN, label=f"Below retrain CI upper ({CI_HI:.3f})"), Patch(color=RED, label="Above")],
          loc="lower right", frameon=True)
plt.tight_layout(); plt.savefig(os.path.join(D, "method_comparison_advantage.png")); plt.close()

# ---------- Fig 4: privacy-utility plane ----------
# Fisher (ELBO ~490) is omitted as off-scale, matching the original figure.
plane = ["contrastive", "scrub", "ssd", "gradient_ascent", "retain_finetune", "moon", "extragradient", "dp_sgd"]
pts = []
for m in plane:
    u = C.pbmc_utility(m)
    pts.append((LABELS[m], u["elbo"], u["marker_r"], summary[m]["mean"]))
ub, ur = C.pbmc_utility("baseline"), C.pbmc_utility("retrain")
BASE = (ub["elbo"], ub["marker_r"], BASELINE); RET = (ur["elbo"], ur["marker_r"], RETRAIN_MEAN)
fig, (p1, p2) = plt.subplots(1, 2, figsize=(16, 6.5)); fig.suptitle("Privacy-utility tradeoff across all methods", fontsize=15)
for ax, xi, xlab, title in [(p1, 1, "ELBO (lower = better reconstruction)", "Privacy vs reconstruction quality"),
                            (p2, 2, "Marker gene correlation (higher = better biology)", "Privacy vs biological fidelity")]:
    for n, e, r, adv in pts:
        xv = e if xi == 1 else r
        ax.scatter(xv, adv, s=90, color=RED, edgecolor="k", zorder=3)
        ax.annotate(n, (xv, adv), xytext=(6, -2), textcoords="offset points", fontsize=9)
    bx = BASE[0] if xi == 1 else BASE[1]
    ax.scatter(bx, BASE[2], s=130, marker="s", color=GREY, edgecolor="k", zorder=3)
    ax.annotate("Baseline", (bx, BASE[2]), xytext=(6, -2), textcoords="offset points", fontsize=9)
    rx = RET[0] if xi == 1 else RET[1]
    ax.scatter(rx, RET[2], s=240, marker="*", color=GREEN, edgecolor="k", zorder=4)
    ax.annotate("Retrain", (rx, RET[2]), xytext=(8, -2), textcoords="offset points", fontsize=9)
    ax.axhline(RET[2], color=GREEN, ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel(xlab); ax.set_ylabel("MIA Advantage (lower = better privacy)"); ax.set_title(title); ax.set_ylim(0, 0.74)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(D, "privacy_utility_all_methods.png")); plt.close()

print("regenerated from canonical JSON: fisher_overlap_cross_dataset.png, "
      "method_comparison_advantage.png, privacy_utility_all_methods.png")
print(f"  fisher global PBMC {pbmc_g:.3f} / TCGA {tcga_g:.3f}; retrain adv {RETRAIN_MEAN:.3f} "
      f"[{CI_LO:.3f}, {CI_HI:.3f}]; baseline adv {BASELINE:.3f}")
for n, e, r, adv in pts:
    print(f"  {n:22s} ELBO={e:.1f} marker_r={r:.3f} adv={adv:.3f}")
