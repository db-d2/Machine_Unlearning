# The loss-weighting bug

A note on the defect that manufactured the paper's "posterior collapse" result for Fisher
scrubbing, the utility cost attributed to extra-gradient co-training, and part of the
frozen-critic result. Found in the round-5 review (2026-09-08), fixed the same day, and
confirmed by re-derivation. Written so that the mechanism, not just the fix, is clear.

## 1. What the VAE is trained to minimize

The training objective is the negative ELBO on a batch of cells. In `src/vae.py`:

```python
def nb_loss(x, mean, dispersion, eps=1e-8, per_cell=False):
    mse = F.mse_loss(mean, x, reduction='none')
    if per_cell:
        return mse.mean()                # average over genes AND cells
    else:
        return mse.sum(dim=1).mean()     # sum over genes, average over cells

def kl_divergence(mu, logvar, free_bits=0.0, per_cell=False):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if per_cell:
        return kl_per_dim.mean()         # average over latent dims AND cells
    else:
        return torch.sum(kl_per_dim, dim=1).mean()   # sum over dims, average over cells

def vae_loss(..., per_cell=False):
    total_loss = recon_loss + beta * kl_loss
```

With the default `per_cell=False`, the two terms are on the same footing. Each cell
contributes the sum of its squared reconstruction errors over the 2,000 genes plus the
sum of its KL terms over the 32 latent dimensions. That is the actual negative ELBO of a
Gaussian observation model with unit variance, up to a constant, and it is what
`src/train.py`, notebook 2, `retrain.py`, and every correctly implemented method use.

On the PBMC baseline, per cell, the two terms are about

| Term | Value per cell | Share of objective |
|---|---|---|
| Reconstruction, summed over 2,000 genes | 353.7 | 97.1 percent |
| KL, summed over 32 dimensions | 10.55 | 2.9 percent |
| Negative ELBO | 364.2 | |

The reconstruction term is large because it is a sum over 2,000 genes; per gene it is
about 0.18. The KL term is what it is: about a third of a nat per latent dimension. The
ratio of the two, 34 to 1, is the balance the model was trained at, and it is what keeps
information flowing through the latent code. Reconstruction pulls the posterior
q(z|x) toward whatever encodes the cell well; KL pulls it toward the prior N(0, I).
The equilibrium between the two is the trained model.

## 2. What the buggy code minimized instead

Three scripts wrote their own version of the objective instead of calling `vae_loss`.

**Fisher scrubbing, retain fine-tune** (`src/train_fisher_unlearn.py`, old `retain_finetune`):

```python
recon_loss = nn.functional.mse_loss(mean, x_batch, reduction='mean')   # mean over genes and cells
kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))  # sum over dims
loss = recon_loss + kl_loss
```

**Extra-gradient co-training, utility term** (`src/train_unlearn_extragradient.py`, old):

```python
recon_loss_r = nn.functional.mse_loss(mean_r, x_r, reduction='mean')
kl_loss_r = -0.5 * torch.mean(torch.sum(1 + logvar_r - mu_r.pow(2) - logvar_r.exp(), dim=1))
utility_loss = recon_loss_r + kl_loss_r
total_loss = privacy_loss + args.lambda_retain * utility_loss     # lambda_retain = 10
```

**Frozen critics** (`src/train_unlearn.py`, `train_vae_multi_step`):

```python
utility_loss, _, _ = vae_loss(x_r, output_r, ..., per_cell=True)   # both terms averaged
```

In the first two, `reduction='mean'` averages the squared error over every element of the
batch matrix, that is over genes as well as cells, while the KL line still sums over the
32 dimensions. The reconstruction term is therefore divided by 2,000 and the KL term is
not. In the third, `per_cell=True` averages both, so reconstruction is divided by 2,000
and KL by 32.

The same numbers from the table above, as these objectives see them:

| Objective | Reconstruction term | KL term | KL share | Effective beta |
|---|---|---|---|---|
| Training (`vae_loss`) | 353.7 | 10.55 | 2.9 percent | 1 |
| Fisher fine-tune, extra-gradient utility | 0.177 | 10.55 | 98.4 percent | 2,000 |
| Frozen critics (`per_cell=True`) | 0.177 | 0.330 | 65 percent | 62.5 |

"Effective beta" is the weight on KL relative to reconstruction, in the units of the
training objective. A beta-VAE with beta = 2,000 is not a VAE that reconstructs; it is a
model whose only real incentive is to make every posterior equal to the prior. That is
what the Fisher fine-tune and the extra-gradient utility term asked for. The frozen-critic
objective, at beta 62.5, is a milder version of the same thing.

None of this was visible from the loss values printed during training, because the
printed numbers were small and decreasing. The old Fisher fine-tune log reads
292.4, 4.11, 0.55, 0.41, 0.39, 0.38, 0.36, 0.35, 0.33, 0.32 over its ten epochs. That
looks like excellent convergence. Translated back into the training objective's units,
a final value of 0.32 is a per-gene reconstruction error of about 0.2 (about 400 summed
over genes, worse than the baseline's 354) plus a KL of about 0.1 nats (against the
baseline's 10.55). The log was reporting the collapse as a success.

## 3. Why a weighting error produces posterior collapse

Two facts about optimization explain why the outcome was not merely "slightly different"
but a qualitatively different model.

**The direction of the gradient follows the relative weights, not the absolute scale.**
For any parameter theta, the gradient of the buggy objective is
(1/2000) dRecon/dtheta + dKL/dtheta, against dRecon/dtheta + dKL/dtheta at training. For
almost every encoder parameter the reconstruction gradient is the larger of the two by a
wide margin at the trained optimum (the model sits where they balance, and reconstruction
is the term with 2,000 outputs behind it). Divide it by 2,000 and the KL gradient wins
everywhere. The update direction becomes, to first order, the direction that most
reduces KL, which is the direction that shrinks mu toward zero and log-variance toward
zero for every cell.

**Adam does not rescue you.** It is tempting to think an adaptive optimizer neutralizes
loss scaling. It neutralizes the overall scale of the gradient, since each parameter's
step is roughly the learning rate times the sign of its recent gradient. It does nothing
about the relative weight of two terms inside one gradient, because that weighting fixes
the sign. Under the buggy objective the sign of nearly every encoder parameter's update
is the KL sign, so Adam takes a full learning-rate step in the collapsing direction on
every parameter at once. That is why a handful of steps did so much. The old
extra-gradient runs stopped after 3 to 19 single-batch steps at learning rate 1e-4, so
no parameter moved by more than about 2e-3, yet KL fell from 10.6 to 4.7 on average
across seeds and to 2.1 on seed 42. A coherent shift of 1e-3 across the two million
weights of the first encoder layer changes each hidden pre-activation by roughly 1e-3
times the L1 norm of the input, which for log-normalized expression is hundreds, so the
posterior parameters move by order one after a few steps. Coherence is the whole effect;
the same total movement in random directions would do nothing.

**Where it ends.** With KL driven to 0.007 nats per cell, every posterior is the prior.
The decoder receives no information about which cell it is reconstructing and its best
response is to output roughly the same profile for every cell. The reconstruction error
of that constant profile on held-out cells is 489.7 per cell, against 353.7 for the
baseline. That is the "ELBO 490" in the paper's utility table: 489.7 of reconstruction
plus 0.007 of KL. Marker-gene correlation falls to 0.628 because the decoder cannot
specialize by cell type any more, and the drop concentrates on the NK-cell markers
(GNLY 0.30, NKG7 0.52), which is what a constant-profile decoder does to the genes with
the most cell-type-specific expression. Every one of these numbers, which the paper
presented as the signature of a method failing, is the signature of the objective it was
given.

## 4. Why extra-gradient showed a partial version

The extra-gradient utility term had the same beta of 2,000 and was additionally
multiplied by lambda = 10. The recorded privacy term (the critics' mean membership
probability on the forget batch) was about 0.8 per step; the recorded utility term was
about 11, so lambda times utility was about 110. The VAE update was therefore almost
entirely a KL-descent step, with the adversarial signal contributing almost nothing to
the direction. The runs then stopped early, after 3 to 19 steps, on a noisy
shadow-attacker rule. The result was a partially collapsed posterior: KL 4.7, ELBO 404,
marker r 0.789 as the 10-seed means, and KL 2.1, ELBO 475 on the single seed the utility
table showed. The paper read this as "extra-gradient trades utility for privacy", the
cost of reducing the membership signal. It was the same collapse, interrupted.

The privacy effect the paper saw was real but had the same cause. A partially collapsed
encoder maps every cell, forget cells included, closer to the prior, so the features the
attacker was trained on shift for forget cells and matched negatives alike, and the
attacker's transferred decision rule reads the forget cells as less member-like. The
scatter across seeds (AUC 0.29 to 0.72) tracked how many steps each seed took before the
stopping rule fired, which is why the paper found extra-gradient "unreliable across
seeds". Neither the privacy gain nor the variance was the adversarial mechanism the
method is named for.

## 5. What was correct all along

`train_retain_finetune.py`, `train_gradient_ascent.py`, `train_ssd.py`, `train_scrub.py`,
`train_contrastive_unlearn.py`, `train_rau.py`, `train_dp.py`, `retrain.py`, `train.py`,
and notebook 2 all call `vae_loss` with the default reduction. That is why those methods
kept baseline utility (ELBO 363 to 364, KL 10.4 to 10.6) in the old results as well as
the new. The pattern in the paper's Table 2, five methods preserving utility and two
"destroying the model", was the pattern of which scripts called the shared loss function
and which re-implemented it. Moon fine-tunes the decoder alone with an L1 term and no KL,
so it is unaffected.

## 6. The fix

The Fisher fine-tune now calls the training objective directly:

```python
output = model(x_batch, library_size=lib_batch)
loss, _, _ = vae_loss(x_batch, output, likelihood=model.likelihood, library_size=lib_batch, beta=1.0)
```

The extra-gradient utility term and the frozen-critic utility term call the same
function and then divide the whole thing by the number of genes:

```python
utility, _, _ = vae_loss(x_r, out, likelihood=vae.likelihood, library_size=lib_r, beta=1.0)
utility = utility / n_genes
```

Dividing the entire objective by 2,000 keeps the O(1) magnitude the original authors of
those lines wanted, which matters when the term is added to a privacy loss that lives in
[0, 1], while leaving the reconstruction-to-KL ratio exactly where training set it. A
global scale changes neither the optimum nor the direction of the gradient; only the
relative weights do. The general rule is that any change of reduction has to be applied
to every term of a composite loss, or applied once to the sum.

## 7. What changed when the fix was applied

PBMC-33k, held-out utility, canonical CPU attacker throughout.

| Model | ELBO before | ELBO after | KL before | KL after | marker r before | marker r after | MLP advantage before | after |
|---|---|---|---|---|---|---|---|---|
| Baseline | 364.2 | 364.0 | 10.55 | 10.83 | 0.831 | 0.831 | 0.582 | 0.584 |
| Fisher scrubbing | 489.7 | 363.6 | 0.007 | 10.81 | 0.628 | 0.832 | 0.615 (worse than baseline) | 0.345 |
| Extra-gradient lambda=10 (10 seeds) | 403.7 | 363.8 | 4.70 | 10.78 | 0.789 | 0.832 | 0.281 | 0.225 |

Elsewhere: the TCGA-BRCA Fisher checkpoint that scored AUC 0.500 under both matchings
(a collapsed model produces the same features for every input, so the attacker's output
is constant) now scores 0.784 cross-subtype and 0.543 within-subtype; the cluster-12
Fisher run whose marker correlation fell to 0.21 now keeps 0.831; the Tabula Muris Fisher
run moves from 0.946 to 0.944 by a different route (no collapse, no privacy change).

Two of the paper's results depended on the bug and do not survive it: "Fisher
scrubbing collapses the posterior" and "the methods that reduce the membership signal
destroy the model". A third was reshaped rather than removed: extra-gradient still fails,
but with utility intact and a bimodal over- and under-correction across seeds rather
than a trade-off. The Fisher-overlap section had used the collapse as the observable
consequence of entangled parameters; with the collapse gone, that link has to be
re-established or dropped, which is what the running phase audit is testing.

## 8. How it was found, and why it was not found earlier

It was found by reading the loss lines of every unlearning script side by side and
noticing that three of them used `reduction='mean'` or `per_cell=True` where the rest
called `vae_loss`. The tell in the results was KL = 0.007. A KL that small on a model
that still reconstructs at all is not a state gradient descent on the negative ELBO
reaches; it is the state gradient descent on the KL alone reaches.

It survived four review rounds because every round checked the paper's numbers against
the JSON files that produced them, and the JSON files were faithful to the code. The
July 4 audit re-ran the same scripts and reproduced the same collapse, which read as
confirmation. The narrative around the numbers was internally coherent, with a mechanism
(Fisher overlap) ready to explain them, so nothing prompted a look at the objective
itself. The lesson is a cheap one to institutionalize: a composite loss in a fine-tuning
script should be unit-tested against the training objective on one batch, and a
reported KL below one nat per cell on a model that is supposed to still reconstruct
should be treated as a bug report until proven otherwise.

## 9. A related but separate defect in the Fisher scrub step

For completeness, since the two combined to produce the Fisher numbers. The old scrub
step was `clamp(lr * grad / F, -0.001, 0.001)` applied per parameter, with an absolute
damping of 0.1 added to the batch-summed Fisher. The clamp saturated almost everywhere,
so every parameter moved by 0.001 per step in the sign of its gradient, 100 times, and
the forget loss rose from 283 to 218,666 per cell; the model was destroyed before the
fine-tune started (post-scrub attacker AUC exactly 0.5). The fine-tune then rebuilt a
model from that wreck under the beta-2,000 objective, which is how the specific
signature of ELBO 490 and KL 0.007 arose. The fix rescales the whole update vector so
its largest element is at most the step size, which preserves the Fisher-preconditioned
direction, and uses the per-sample Fisher of the paper's own analysis with damping
relative to the Fisher's scale. Under the fixed scrub the forget loss rises 269 to
5,585, the fine-tune restores retain loss to 365, and the model ends at baseline utility.
