---
paths:
  - "src/vae.py"
  - "src/attacker.py"
  - "src/fisher_utils.py"
  - "src/vae_conditional.py"
  - "src/train_*.py"
  - "notebooks/26*"
  - "notebooks/27*"
  - "notebooks/28*"
  - "notebooks/35*"
  - "notebooks/41*"
---

# VAE Architecture and Training Reference

## Parameter Counts (PBMC, 2000 genes)

| Layer | Shape | Parameters | % |
|-------|-------|-----------|---|
| encoder.network.0 | [1024, 2000] | 2,048,000 | 27.9% |
| encoder.network.4 | [512, 1024] | 524,288 | 7.1% |
| encoder.network.8 | [128, 512] | 65,536 | 0.9% |
| encoder.fc_mu | [32, 128] | 4,096 | 0.1% |
| encoder.fc_logvar | [32, 128] | 4,096 | 0.1% |
| decoder.network.0 | [128, 32] | 4,096 | 0.1% |
| decoder.network.4 | [512, 128] | 65,536 | 0.9% |
| decoder.network.8 | [1024, 512] | 524,288 | 7.1% |
| decoder.fc_mean | [2000, 1024] | 2,048,000 | 27.9% |
| decoder.fc_dispersion | [2000, 1024] | 2,048,000 | 27.9% |
| **Total** | | **7,349,984** | |

Fisher categories: Encoder 2,642,816; Bottleneck 8,256; Decoder hidden 598,912; Decoder output 4,100,000 (fc_mean + fc_dispersion = 56% of all parameters). TCGA-BRCA (978 genes): 4,208,356 parameters.
fc_dispersion receives zero gradients under MSE (vestigial from the scVI design); its Fisher cosine (0.996) is an artifact, and fc_mean (0.485) is the decoder-output figure quoted for the shared layer.

## Loss (src/vae.py)

`vae_loss` returns reconstruction summed over genes plus KL summed over the 32 latent dimensions, per cell, averaged over the batch (`per_cell=False`). Reconstruction is MSE on log-normalized counts against softmax(decoder) x library size (`likelihood='nb'` names the decoder head, not the loss). Any objective that averages reconstruction per gene while keeping the KL summed, or that scales one term and not the other, reweights the KL by 60x to 2,000x (`LOSS_WEIGHTING_BUG.md`). Use `vae_loss` for every utility term and scale the whole loss (for example `/ n_genes`), never one term.

## MIA Features (src/attacker.py)

Canonical: 70 dims = v1 (69) + mean k-NN distance (k=5) to a 5,000-cell retain sample in latent space.
v1 (69): latent mean mu (32), log-variance (32), reconstruction loss (1), KL (1), ELBO (1), norms of mu and logvar (2).
Attacker: MLP [256, 256], spectral norm, dropout 0.3, trained on the baseline model's forget cells vs 194 matched negatives, then applied to every checkpoint in one CPU pass. `extract_vae_features` returns CPU tensors whatever device the model is on.

## Baseline and retrain training (notebook 02; scripts/retrain_clean.py)

| Parameter | Value |
|---|---|
| Hidden layers | [1024, 512, 128], LayerNorm, dropout 0.1 |
| z | 32 |
| Optimizer | Adam 1e-4 (the `train.py` CLI default of 1e-3 is not what any checkpoint used) |
| Epochs / batch | 100 / 256; KL warm-up 20 epochs; free bits 0.03; checkpoint selected by training loss, patience 10 |
| beta | 1.0 |
| Retrain training set | `split['retain_indices']` only (28,094 cells). `src/retrain.py` now does the same; retrains that drew their own split are superseded |

## Method defaults (as re-derived, 2026-09-08)

| Method | Setting |
|---|---|
| Extra-gradient (`train_unlearn_extragradient.py`) | ExtraAdam two-step updates; `--lambda_privacy 10` (1 as the ablation); VAE lr 1e-4, critic lr 1e-5 (TTUR); 3 critics initialized from `outputs/p2/attackers/attacker_{v1_seed42,v2_seed43,v3_seed44}.pt` and co-trained (2 critic steps per VAE step); utility = `vae_loss / n_genes`; privacy = max over critics of the mean sigmoid on the forget batch; `--epochs 50` passes over the retain set, no early stopping; per-epoch history of forget and retain negELBO saved |
| Fisher scrubbing (`train_fisher_unlearn.py`) | per-sample retain Fisher (`fisher_utils.compute_vae_fisher`; cache with `--fisher_cache`); relative damping `--fisher_damping 0.1` x mean(F); `--scrub_lr 1e-4` for `--scrub_steps 100` with a global per-step cap `--max_step 1e-3` (3e-3 and 1e-2 are the over-correcting ablation); `--finetune_epochs 10` at 1e-4 with `vae_loss` |
| Retain-only fine-tune, gradient ascent, SSD, SCRUB, contrastive, Moon | their `train_*.py` defaults. Gradient ascent's ascent phase (10 SGD steps at 1e-5, clipped) leaves the forget-set negELBO unchanged (269.5 before and after), so the method as run is the retain fine-tune |
| Frozen critics (`train_unlearn.py`; `scripts/run_frozen_critics.py`) | one or three frozen critics, lambda 5 or 10; utility `vae_loss(per_cell=False) / n_genes` |
| DP-SGD (`train_dp.py`) | Opacus 1.5.4, from scratch on the retain set, epsilon 10, CPU only |
| Likelihood-targeted unlearning (`train_targeted_unlearn.py`; NB41) | retain `vae_loss` + lambda x mean absolute gap between each forget cell's negELBO and a target computed under the current model without gradient: its 10 nearest matched negatives (Part B), the mean over all 194 matched negatives (Part C), or 88 held-out look-alikes disjoint from the evaluation (Part D, `eval_neighbors` keeps the evaluator on the canonical negatives); Adam 1e-4; at most 50 epochs; stop once the mean gap stays within 5 nats for 3 epochs; return the smallest-gap state |

## Checkpoint Format

```python
torch.save({'model_state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
            'train_loss': loss}, output_dir / 'best_model.pt')
```
Load with the six config keys only (see troubleshooting). Every new checkpoint must start from `outputs/p1/baseline/best_model.pt` and be registered by `scripts/checkpoint_manifest.py`.
