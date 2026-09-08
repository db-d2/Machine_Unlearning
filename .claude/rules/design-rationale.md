---
paths:
  - "src/vae.py"
  - "src/train_unlearn_extragradient.py"
  - "src/train_fisher_unlearn.py"
  - "src/train_targeted_unlearn.py"
  - "Writeup.tex"
---

# Design Rationale

**Why MSE on log-normalized counts?**
Data preprocessed with normalize_total + log1p are continuous. MSE is a Gaussian observation model on the transformed counts. The decoder keeps the scVI mean + dispersion heads, but the dispersion head receives no gradient under MSE. The loss is MSE, never negative binomial.

**Why the loss reduction matters.**
`vae_loss` sums reconstruction over 2,000 genes and KL over 32 latent dimensions. Two December-era objectives averaged reconstruction per gene while keeping the KL summed, which weighted the KL 60x to 2,000x (`LOSS_WEIGHTING_BUG.md`). That produced the "posterior collapse" attributed to Fisher scrubbing and the utility cost attributed to extra-gradient. With the reduction fixed, no post-hoc method costs utility (ELBO 363 to 364.5 for all of them).

**Why z=32?**
Balances reconstruction against regularization and matches scVI defaults. z=8 does not lower Fisher overlap (global 0.349 vs 0.209).

**Why one CPU attacker for every checkpoint?**
MPS is not bit-reproducible run to run, and even on CPU a fresh attacker instance drifts by 0.001 to 0.02 AUC. Scoring every model with the same attacker in one pass makes the comparison paired; differences between passes are attacker noise, not results.

**Why a clean retain-split retrain?**
`src/retrain.py` used to draw its own 85/15 split from all non-forget cells, which put 159 to 170 of the 194 matched negatives into the retrain's training set. A reference that trained on its own non-member negatives cannot be a membership floor (it scored 0.579 where the clean retrain scores 0.529, and the refit control puts the clean retrain at chance). `scripts/retrain_clean.py` trains on `retain_indices` only.

**Why the reference-model likelihood-ratio audit?**
The MLP attacker's advantage is direction-agnostic, so it cannot tell a method that left the signal intact (AUC 0.67) from one that over-corrected (AUC 0.33). Scoring each forget cell by negELBO(clean retrain) minus negELBO(target), with the seeded 16-sample estimator of `src/likelihood_audit.py`, places every model on one axis. Near 1 the forget cells are still fitted better than a never-trained model fits them; near 0 they are fitted worse. It requires a retrain, so it is an auditor's instrument, not a defender's.

**Why lambda=10 for extra-gradient, and why no early stopping?**
lambda=10 is the paper's regime and lambda=1 the ablation. Both over-correct (audit 0.22 and 0.05 against a retrain floor of 0.24 to 0.52). The re-derived runs make 50 full passes over the retain set with ExtraAdam; the December runs had stopped after 3 to 19 random steps, which is why their numbers could not be reproduced.

**Why three co-trained critics?**
Reduces critic variance during the min-max game. Frozen critics let the VAE drive the forget cells to the least member-like region of the data (AUC 0.003 to 0.008), the purest over-correction in the study.

**Why does Fisher scrubbing (and every edit-then-fine-tune method) fail?**
Not because the edit is blocked. The scrub raises the forget loss 21x and the retain loss 3.4x, the shared damage that Proposition 1 predicts from Fisher cosines of 0.35 to 0.49 at every layer (a classifier's class-specific weights sit at 0.018). The retain fine-tune that repairs the retain set restores the forget set with it, undoing 99% of the displacement through the shared parameters. At larger scrub steps the fine-tune cannot bring the forget set back and the model ends over-corrected. Nothing in any post-hoc objective encodes where the retrain lands (forget-cell negELBO 334 to 337, against the baseline's 270).

**Why likelihood-targeted unlearning (NB41)?**
It tests the "nowhere to aim" diagnosis directly. The objective is given the retrain-level target in the quantity the audit measures, using the matched negatives (observable without a retrain) as the reference for how a never-seen cell of the same type is fitted. RAU, which targeted the retrain's posterior instead, still over-corrected. Outcome (NB41): given the retrain's level the method lands in the audit's retrain band with attacker advantage at the threshold and utility intact; given the level of held-out look-alikes it lands 40 nats short, because the retrain fits an unseen member of an unknown type worse than that member's nearest known-type neighbours. The target is real and not observable without a retrain.
