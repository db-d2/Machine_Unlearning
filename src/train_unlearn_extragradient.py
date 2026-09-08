"""Extra-gradient co-training for VAE unlearning.

Objective (the paper's equation):

    min_theta max_psi  L_VAE(theta; R) / D  +  lambda * P(theta, psi; F)

where L_VAE is the negative ELBO on a retain batch divided by the number of genes D (so
it is O(1)), P is the privacy term (the largest mean membership probability any critic
assigns to the forget batch, which the VAE drives toward zero), and the critics psi are
trained to separate forget cells from retain cells. Both players use the extra-gradient
update of Gidel et al. (2019) with Adam ("ExtraAdam"): a step to an extrapolated point,
the gradient there, then the update applied at the original point. Two-time-scale
learning rates (TTUR), critic 10x lower than the VAE.

Revision 2026-09-08 (round-5 review, defect B3). The previous version had no
extrapolation step, trained critics with plain SGD while an unused Adam optimizer sat
idle, averaged the reconstruction error over genes so KL was over-weighted 2,000 times
and multiplied by lambda, defined an epoch as one minibatch, and stopped on a noisy
shadow-attacker rule after 3 to 19 steps. An epoch is now one pass over the retain set,
the step budget is fixed, and there is no early stopping.

References:
    - Gidel, Berard, Vignoud, Vincent, Lacoste-Julien (2019). A Variational Inequality
      Perspective on GANs. ICLR 2019 (ExtraAdam).
    - Chavdarova et al. (2019). Reducing Noise in GAN Training with Variance Reduced
      Extragradient. NeurIPS 2019.
    - Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule. NeurIPS 2017.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from vae import VAE, vae_loss
from attacker import MLPAttacker, extract_vae_features, build_attack_features


class AnnDataDataset(Dataset):
    def __init__(self, adata, indices):
        self.adata = adata
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.adata.X[self.indices[idx]]
        x = torch.FloatTensor(row.toarray().flatten() if hasattr(row, 'toarray') else np.asarray(row).flatten())
        return x, torch.FloatTensor([x.sum().item()])


def load_attackers(attacker_paths, device):
    attackers, variants = [], []
    for path in attacker_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        att = MLPAttacker(ckpt['config']['feature_dim'], ckpt['config']['hidden_dims'],
                          dropout=ckpt['config'].get('dropout', 0.3),
                          use_spectral_norm=ckpt['config'].get('use_spectral_norm', True)).to(device)
        att.load_state_dict(ckpt['model_state_dict'])
        attackers.append(att)
        variants.append(ckpt['config'].get('variant', 'v1'))
    return attackers, variants


def extra_adam_step(params, optimizer, loss_fn):
    """One ExtraAdam step: extrapolate with grad(theta), then update theta with grad(theta~)."""
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    backup = [p.detach().clone() for p in params]
    optimizer.step()                       # theta~ = extrapolated point
    optimizer.zero_grad()
    loss_extra = loss_fn()
    loss_extra.backward()                  # gradient at theta~
    with torch.no_grad():
        for p, b in zip(params, backup):
            p.copy_(b)                     # back to theta
    optimizer.step()                       # apply grad(theta~) at theta
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Extra-gradient co-training for VAE unlearning')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--attacker_paths', nargs='+', required=True)
    parser.add_argument('--lambda_privacy', type=float, default=10.0,
                        help='weight on the privacy term (the paper\'s lambda)')
    parser.add_argument('--epochs', type=int, default=50, help='passes over the retain set')
    parser.add_argument('--lr_vae', type=float, default=0.0001)
    parser.add_argument('--lr_critic', type=float, default=0.00001)
    parser.add_argument('--critic_steps', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Extra-gradient co-training (ExtraAdam, TTUR) on {device}")
    print(f"lambda_privacy={args.lambda_privacy} lr_vae={args.lr_vae} lr_critic={args.lr_critic} "
          f"critic_steps={args.critic_steps} epochs={args.epochs}")

    adata = sc.read_h5ad(args.data_path)
    with open(args.split_path) as f:
        split = json.load(f)
    forget_idx, retain_idx = split['forget_indices'], split['retain_indices']
    n_genes = adata.n_vars
    print(f"Forget: {len(forget_idx)}, Retain: {len(retain_idx)}, genes: {n_genes}")

    forget_loader = DataLoader(AnnDataDataset(adata, forget_idx), batch_size=len(forget_idx), shuffle=True)
    retain_loader = DataLoader(AnnDataDataset(adata, retain_idx), batch_size=args.batch_size, shuffle=True, drop_last=True)
    forget_iter = iter(forget_loader)

    def next_forget():
        nonlocal forget_iter
        try:
            return next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            return next(forget_iter)

    ckpt = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    vae = VAE(input_dim=config['input_dim'], latent_dim=config['latent_dim'], hidden_dims=config['hidden_dims'],
              likelihood=config['likelihood'], dropout=config.get('dropout', 0.1),
              use_layer_norm=config.get('use_layer_norm', True)).to(device)
    vae.load_state_dict(ckpt['model_state_dict'])

    attackers, variants = load_attackers(args.attacker_paths, device)
    print(f"Loaded {len(attackers)} critics with variants {variants}")

    vae_params = [p for p in vae.parameters() if p.requires_grad]
    vae_opt = optim.Adam(vae_params, lr=args.lr_vae)
    critic_params = [p for a in attackers for p in a.parameters()]
    critic_opt = optim.Adam(critic_params, lr=args.lr_critic)
    bce = nn.functional.binary_cross_entropy_with_logits

    history = {'utility': [], 'privacy': [], 'critic_loss': [], 'forget_neg_elbo': [], 'retain_neg_elbo': []}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(args.epochs):
        sums = {'utility': 0.0, 'privacy': 0.0, 'critic': 0.0}
        n_iter = 0
        for x_r, lib_r in retain_loader:
            x_r, lib_r = x_r.to(device), lib_r.to(device)
            x_f, lib_f = next_forget()
            x_f, lib_f = x_f.to(device), lib_f.to(device)

            # --- VAE step (critics frozen) ---
            vae.train()
            for a in attackers:
                a.eval()
            last = {}

            def vae_loss_fn():
                out = vae(x_r, library_size=lib_r)
                utility, _, _ = vae_loss(x_r, out, likelihood=vae.likelihood, library_size=lib_r, beta=1.0)
                utility = utility / n_genes
                feats = extract_vae_features(vae, x_f, lib_f, device, requires_grad=True)
                probs = [torch.sigmoid(a(build_attack_features(feats, variant=v))).mean()
                         for a, v in zip(attackers, variants)]
                privacy = torch.stack(probs).max()
                last['utility'], last['privacy'] = utility.item(), privacy.item()
                return utility + args.lambda_privacy * privacy

            extra_adam_step(vae_params, vae_opt, vae_loss_fn)
            sums['utility'] += last['utility']
            sums['privacy'] += last['privacy']

            # --- critic steps (VAE frozen): forget = 1, retain = 0 ---
            vae.eval()
            with torch.no_grad():
                feats_f = extract_vae_features(vae, x_f, lib_f, device, requires_grad=False)
                feats_r = extract_vae_features(vae, x_r, lib_r, device, requires_grad=False)
                inputs = [(build_attack_features(feats_f, variant=v).to(device),
                           build_attack_features(feats_r, variant=v).to(device)) for v in variants]
            for a in attackers:
                a.train()

            def critic_loss_fn():
                total = 0.0
                for a, (in_f, in_r) in zip(attackers, inputs):
                    lf, lr_ = a(in_f), a(in_r)
                    total = total + 0.5 * (bce(lf, torch.ones_like(lf)) + bce(lr_, torch.zeros_like(lr_)))
                return total / len(attackers)

            for _ in range(args.critic_steps):
                sums['critic'] += extra_adam_step(critic_params, critic_opt, critic_loss_fn)
            n_iter += 1

        # --- epoch bookkeeping: negative ELBO on the full forget set and a retain batch ---
        vae.eval()
        with torch.no_grad():
            xf_all, lf_all = next(iter(DataLoader(AnnDataDataset(adata, forget_idx), batch_size=len(forget_idx))))
            xf_all, lf_all = xf_all.to(device), lf_all.to(device)
            f_loss, _, _ = vae_loss(xf_all, vae(xf_all, library_size=lf_all), likelihood=vae.likelihood,
                                    library_size=lf_all, beta=1.0)
            r_loss, _, _ = vae_loss(x_r, vae(x_r, library_size=lib_r), likelihood=vae.likelihood,
                                    library_size=lib_r, beta=1.0)
        history['utility'].append(sums['utility'] / n_iter)
        history['privacy'].append(sums['privacy'] / n_iter)
        history['critic_loss'].append(sums['critic'] / (n_iter * args.critic_steps))
        history['forget_neg_elbo'].append(f_loss.item())
        history['retain_neg_elbo'].append(r_loss.item())
        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} ({time.time() - start:.0f}s) | utility/gene "
                  f"{history['utility'][-1]:.4f} | privacy {history['privacy'][-1]:.3f} | critic "
                  f"{history['critic_loss'][-1]:.3f} | negELBO forget {f_loss.item():.1f} retain {r_loss.item():.1f}",
                  flush=True)

    total_time = time.time() - start
    torch.save({'model_state_dict': vae.state_dict(), 'config': config, 'args': vars(args),
                'history': history}, output_dir / 'best_model.pt')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump({'method': 'extra_gradient_unlearning',
                   'approach': 'ExtraAdam co-training, TTUR, critics forget-vs-retain',
                   'revision': '2026-09-08 real extragradient step, vae_loss/n_genes utility, epoch = retain pass, no early stop',
                   'lambda_privacy': args.lambda_privacy, 'lr_vae': args.lr_vae, 'lr_critic': args.lr_critic,
                   'critic_steps': args.critic_steps, 'epochs': args.epochs,
                   'steps_per_epoch': n_iter, 'total_vae_steps': n_iter * args.epochs,
                   'baseline_checkpoint': args.baseline_checkpoint, 'attacker_paths': args.attacker_paths,
                   'seed': args.seed, 'total_time_seconds': total_time}, f, indent=2)
    print(f"\nDone: {args.epochs} epochs x {n_iter} steps in {total_time:.0f}s -> {output_dir}")


if __name__ == '__main__':
    main()
