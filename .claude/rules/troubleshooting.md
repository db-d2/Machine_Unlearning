---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "notebooks/**/*.ipynb"
---

# Common Issues

1. **ModuleNotFoundError** — Use `PYTHONPATH=src python ...` from the repo root with `~/.pyenv/versions/stat4243/bin/python` (system Python lacks numpy/torch).
2. **CUDA / MPS OOM** — Reduce batch_size from 256 to 128 or 64.
3. **FileNotFoundError for a processed .h5ad** — Run NB01 (PBMC), NB11 (Tabula Muris), or NB31 (TCGA-BRCA) first.
4. **"Tensor for argument input is on cpu but expected on mps"** — A model on MPS received CPU inputs or features. Move built tensors with `.to(device)` or take the device from `next(model.parameters()).device` (fixed in `train_unlearn_extragradient.py`, `synthetic_utils.py`, `train_targeted_unlearn.py`). MPS tensors need `.cpu()` before `.numpy()`.
5. **Two runs of the same scoring give different AUCs** — Expected. MPS is not bit-reproducible, and a fresh attacker instance drifts 0.001 to 0.02 AUC even on CPU. Score everything in one pass with one attacker (`scripts/save_persample_scores.py`, then `scripts/nested_bootstrap_all.py`), pin evaluation to CPU, and never compare numbers across passes.
6. **A method's KL collapses toward 0 or ELBO jumps by 100+** — Check the loss reduction before blaming the method. Averaging reconstruction per gene while summing the KL reweights the KL up to 2,000x (`LOSS_WEIGHTING_BUG.md`). Use `vae_loss(..., per_cell=False)` and scale the whole loss, never one term.
7. **Retrain reference scores above chance** — Check what it trained on. A retrain that drew its own split contains most of the matched negatives. Use `scripts/retrain_clean.py` (trains on `split['retain_indices']` only) and confirm the training set in `outputs/checkpoint_manifest.json`.
8. **Fisher overlap numbers do not match the reference file** — `scripts/fisher_overlap_analysis.py` is the old batch-summed estimator (default damping 1e-5) and is superseded. Canonical numbers come from the per-sample `fisher_utils.compute_vae_fisher` with damping 0 (NB26 -> `outputs/p6/fisher_overlap_results.json`).
9. **Threshold or likelihood-ratio attack flat across models** — The decoder needs the library size: `model.decode(z, x.sum(1, keepdim=True))` when `likelihood == 'nb'`. Without it the reconstruction attack scored 0.438 on every model (fixed 2026-09-08).
10. **MT gene filtering removes all cells** — Use "MT-" for human, "mt-" for mouse.
11. **A likelihood or audit AUC changes between runs** — The VAE forward pass samples z without a seed. Use `likelihood_audit.neg_elbo_per_cell` (eval mode, 16 seeded z samples) for every reported likelihood; never call `model(x)` for a number that goes in a table.

## Key Gotchas

- **Never hard-code a result.** Load every number through `src/canonical.py` (which reads the JSONs under `outputs/`). Notebooks 12, 14, 15, 21 to 24, 30, 34b to 38, and 40 were patched to do this; a literal in a summary cell is a bug.
- **Provenance.** Every current checkpoint descends from the July-4 baseline; `scripts/checkpoint_manifest.py` records sha256, parent, and training set. Superseded checkpoints and results live in `outputs/_superseded_20260908/`; nothing there is citable.
- **VAE config**: checkpoints carry extra keys (lr, epochs, ...). Filter with
  `{k: v for k, v in config.items() if k in ['input_dim', 'latent_dim', 'hidden_dims', 'likelihood', 'dropout', 'use_layer_norm']}`.
- **Legacy attacker paths**: `retrain.py --track_auc` and `learning_curve.load_attacker_for_eval` load pre-trained attackers that were trained on forget vs the 28k retain set. Never use them for a reported number; train a fresh attacker on the baseline (mia-evaluation skill).
- **Classifier Fisher**: at 100% accuracy the Fisher is damping-dominated. Regularize (dropout 0.3, weight_decay 1e-2) and early-stop near 95%.
- **Opacus**: PyTorch 2.11 + Opacus 1.5.4. `PrivacyEngine(accountant='rdp')` + `make_private()`. Freeze `vae.decoder.fc_dispersion`. Run DP-SGD on CPU.
- **Notebook execution**: `PYTHONPATH=src python scripts/run_notebook.py 41_targeted_unlearning` executes in place with the stat4243 kernel. Long runs go in the background; check CPU time against wall time before killing one.
