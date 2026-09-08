"""Fisher-preconditioned scrubbing for a VAE, followed by retain-only fine-tuning.

Method. Let F be the diagonal empirical Fisher of the negative ELBO on the retain set.
Scrubbing takes ascent steps on the forget-set negative ELBO, preconditioned by
(F + lambda I)^-1, so the update moves most along parameters the retain set does not
depend on and least along parameters it does. Retain-only fine-tuning then restores
utility. This is the Fisher-preconditioned ascent variant of selective forgetting; the
Fisher noise injection of Golatkar et al. (2020) is not used.

Revision 2026-09-08 (round-5 review, defect B3). Three defects in the previous version
manufactured the "posterior collapse" reported for this method.
  1. The Fisher was batch-summed in train mode. It is now the per-sample estimator of
     fisher_utils.compute_vae_fisher (eval mode, z = mu), the same estimator used by the
     paper's overlap analysis, and can be cached to disk with --fisher_cache.
  2. The scrub step clamped each parameter's update to +/-0.001 independently. With the
     old damping that clamp saturated almost everywhere, turning the update into a signed
     ascent that ignored the Fisher and destroyed the model (forget loss 283 -> 218,000).
     The update is now rescaled globally so its largest element is at most max_step; the
     Fisher-preconditioned direction is preserved.
  3. The fine-tune averaged the reconstruction error over genes while summing KL over
     latent dimensions, weighting KL 2,000 times more than training did. It now uses the
     training objective (vae.vae_loss).
Damping is now relative: lambda = damping * mean(F), because the per-sample Fisher has a
different scale from the batch-summed one.

References:
    - Golatkar et al. (2020). Eternal Sunshine of the Spotless Net. CVPR 2020.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae import VAE, vae_loss
from fisher_utils import AnnDataDataset, compute_vae_fisher


def _neg_elbo_sum(model, x, lib):
    """Batch-summed negative ELBO (reconstruction summed over genes, KL over latent dims)."""
    output = model(x, library_size=lib)
    loss, _, _ = vae_loss(x, output, likelihood=model.likelihood, library_size=lib, beta=1.0)
    return loss * x.size(0)


def compute_fisher_diagonal(model, dataloader, device, damping=0.1, cache=None):
    """Per-sample diagonal empirical Fisher on the retain set, with relative damping.

    Args:
        model: VAE.
        dataloader: retain-set loader yielding (x, library_size).
        device: device string.
        damping: relative damping; the returned tensors are F + damping * mean(F).
        cache: optional path; if it exists the undamped Fisher is loaded from it,
            otherwise it is computed and saved there.

    Returns:
        Dict mapping parameter names to damped Fisher diagonals.
    """
    if cache is not None and Path(cache).exists():
        raw = torch.load(cache, map_location=device, weights_only=False)
        fisher = {k: v.to(device) for k, v in raw['fisher'].items()}
        n_samples = raw['n_samples']
        print(f"  Loaded per-sample Fisher over {n_samples} retain samples from {cache}")
    else:
        print("Computing per-sample Fisher diagonal on retain set...")
        fisher, n_samples = compute_vae_fisher(model, dataloader, device, damping=0.0)
        print(f"  Computed over {n_samples} retain samples")
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'fisher': {k: v.cpu() for k, v in fisher.items()}, 'n_samples': n_samples}, cache)
            print(f"  Cached to {cache}")
    flat_mean = torch.cat([v.flatten() for v in fisher.values()]).mean()
    lam = damping * flat_mean
    print(f"  mean(F) = {flat_mean.item():.3e}, damping lambda = {lam.item():.3e}")
    return {k: v + lam for k, v in fisher.items()}


def fisher_scrub(model, forget_loader, fisher, device, scrub_lr=0.0001, scrub_steps=100,
                 max_step=0.001, tracker=None, eval_fn=None, eval_interval=10):
    """Fisher-preconditioned ascent on the forget-set negative ELBO.

    Each step computes g = grad of the forget-set negative ELBO, forms the preconditioned
    direction u = scrub_lr * g / F, and rescales u globally so that max|u| <= max_step.
    The rescaling is a trust region, not an element-wise clamp, so the relative sizes of
    the parameter updates (the Fisher preconditioning) are preserved.

    Returns:
        List of per-sample forget losses recorded at each step.
    """
    model.eval()  # deterministic forward (no dropout) while scrubbing
    forget_losses = []
    n_forget = len(forget_loader.dataset)
    print(f"Fisher scrubbing for {scrub_steps} steps (max step {max_step:g})...")
    if tracker is not None and eval_fn is not None:
        auc = eval_fn(model)
        tracker.log(step=0, phase="scrub", auc=auc)

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for step in range(scrub_steps):
        model.zero_grad()
        total = 0.0
        for x_batch, lib_batch in forget_loader:
            x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
            loss = _neg_elbo_sum(model, x_batch, lib_batch)
            loss.backward()
            total += loss.item()
        with torch.no_grad():
            updates = {n: scrub_lr * p.grad / fisher[n] for n, p in params if p.grad is not None}
            largest = max(u.abs().max() for u in updates.values())
            scale = min(1.0, max_step / (largest.item() + 1e-30))
            for n, p in params:
                if n in updates:
                    p.add_(updates[n] * scale)      # ascent: increase forget loss
        forget_losses.append(total / n_forget)
        if (step + 1) % 20 == 0 or step == 0:
            print(f"  Step {step + 1}/{scrub_steps}: forget_loss={forget_losses[-1]:.2f} "
                  f"(scale {scale:.2e})")
        if tracker is not None and eval_fn is not None and (step + 1) % eval_interval == 0:
            auc = eval_fn(model)
            tracker.log(step=step + 1, phase="scrub", auc=auc, loss=forget_losses[-1])
    return forget_losses


def retain_finetune(model, retain_loader, device, epochs=10, lr=0.0001,
                    tracker=None, eval_fn=None, eval_interval=2, scrub_steps=0):
    """Fine-tune on the retain set with the training objective (vae_loss)."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    retain_losses = []
    print(f"Fine-tuning on retain set for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss, n_batches = 0.0, 0
        for x_batch, lib_batch in retain_loader:
            x_batch, lib_batch = x_batch.to(device), lib_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, library_size=lib_batch)
            loss, _, _ = vae_loss(x_batch, output, likelihood=model.likelihood,
                                  library_size=lib_batch, beta=1.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        retain_losses.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: retain_loss={retain_losses[-1]:.2f}")
        if tracker is not None and eval_fn is not None and (epoch + 1) % eval_interval == 0:
            auc = eval_fn(model)
            tracker.log(step=scrub_steps + epoch + 1, phase="finetune", auc=auc, loss=retain_losses[-1])
    return retain_losses


def main():
    parser = argparse.ArgumentParser(description='Fisher-preconditioned scrubbing for a VAE')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--scrub_lr', type=float, default=0.0001)
    parser.add_argument('--scrub_steps', type=int, default=100)
    parser.add_argument('--max_step', type=float, default=0.001,
                        help='largest per-parameter change allowed per scrub step')
    parser.add_argument('--finetune_epochs', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=0.0001)
    parser.add_argument('--fisher_damping', type=float, default=0.1,
                        help='relative damping: lambda = fisher_damping * mean(F)')
    parser.add_argument('--fisher_cache', type=str, default=None,
                        help='path to cache the undamped per-sample retain Fisher')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    adata = sc.read_h5ad(args.data_path)
    with open(args.split_path) as f:
        split_data = json.load(f)
    forget_indices = split_data['forget_indices']
    retain_indices = split_data['retain_indices']
    print(f"Forget set: {len(forget_indices)} samples; retain set: {len(retain_indices)} samples")

    forget_loader = DataLoader(AnnDataDataset(adata, forget_indices), batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(AnnDataDataset(adata, retain_indices), batch_size=args.batch_size, shuffle=True)
    fisher_loader = DataLoader(AnnDataDataset(adata, retain_indices), batch_size=128, shuffle=False)

    checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = VAE(input_dim=config['input_dim'], latent_dim=config['latent_dim'],
                hidden_dims=config['hidden_dims'], likelihood=config['likelihood'],
                dropout=config.get('dropout', 0.1), use_layer_norm=config.get('use_layer_norm', True)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    start = time.time()
    fisher = compute_fisher_diagonal(model, fisher_loader, device, damping=args.fisher_damping,
                                     cache=args.fisher_cache)
    scrub_losses = fisher_scrub(model, forget_loader, fisher, device, scrub_lr=args.scrub_lr,
                                scrub_steps=args.scrub_steps, max_step=args.max_step)
    finetune_losses = retain_finetune(model, retain_loader, device, epochs=args.finetune_epochs,
                                      lr=args.finetune_lr)
    total_time = time.time() - start

    torch.save({'model_state_dict': model.state_dict(), 'config': config, 'args': vars(args),
                'scrub_losses': scrub_losses, 'finetune_losses': finetune_losses},
               output_dir / 'unlearned_model.pt')
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({'scrub_losses': scrub_losses, 'finetune_losses': finetune_losses}, f, indent=2)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({'method_id': 'fisher_unlearning', 'method': 'fisher_preconditioned_scrub_finetune',
                   'revision': '2026-09-08 per-sample retain Fisher, global trust-region step, vae_loss fine-tune',
                   **{k: v for k, v in vars(args).items() if k not in ('data_path',)},
                   'baseline_checkpoint': args.baseline_checkpoint,
                   'total_time_seconds': total_time, 'forget_set_size': len(forget_indices),
                   'retain_set_size': len(retain_indices)}, f, indent=2)
    print(f"\nDone in {total_time:.0f}s. Forget loss {scrub_losses[0]:.1f} -> {scrub_losses[-1]:.1f}; "
          f"final finetune loss {finetune_losses[-1]:.1f}. Saved to {output_dir}")


if __name__ == '__main__':
    main()
