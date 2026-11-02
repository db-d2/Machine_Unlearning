#!/usr/bin/env python3
"""Where the retain fine-tune goes after an edit, read from the saved fine-tuned states of the SGD arms.

Companion of scripts/edit_compensation_test.py (run it first; this script adds a ``states`` block to its JSON). For
every saved state theta_T of an edit arm (restricted_scrub_test.py --save_states, plain gradient descent at 1e-3) and
the no-edit state at the same epoch, with delta the edit and r = theta_T - theta_0:

  path          rho = <r - Delta, delta> / |delta|^2 (the surviving share of the edit, drift-corrected as in the paper),
                r_perp and Delta_perp the parts of r and Delta orthogonal to delta, and the path difference
                r_perp - Delta_perp with its squared norm by layer group and its ten largest parameter tensors; the
                cosine of the path difference with the edit-induced retain gradient H_R delta (a Hessian-vector product
                by central differences of the full retain gradient over a retain sample);
  removal       the forget and held-out fit at theta_T with the surviving edit taken out (theta_T - rho delta) and with
                the whole edit taken out (theta_T - delta), beside the fit at theta_T and at the no-edit state, all in
                double precision with the deterministic estimator;
  effective     per linear layer with an edit (delta W, delta b), the mean over the forget cells and over a retain
                sample of |delta W a(x) + delta b|^2 where a(x) is the layer's input activation under theta_0, under
                the no-edit state and under theta_T: the perturbation the surviving edit inflicts on that layer's
                output, which an upstream re-routing lowers and which is fixed where the input is the data.
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model
from fisher_utils import vae_layer_category
from likelihood_audit import neg_elbo_per_cell
from retain_null_mass import SETTINGS
from edit_compensation_test import EDIT_DIR, OUT, loss_zmu

torch.set_default_dtype(torch.float64)


def det(model, xf, xu):
    nf = neg_elbo_per_cell(model, xf); nu = neg_elbo_per_cell(model, xu)
    return {'forget': float(nf.mean()), 'forget_median': float(np.median(nf)), 'heldout': float(nu.mean())}


def mean_grad(model, x):
    """Gradient of the mean per-cell loss (eval mode, z = mu) over x, as a flat vector."""
    model.eval(); model.zero_grad()
    mu, logvar = model.encode(x)
    if model.likelihood == 'nb':
        mean, _ = model.decode(mu, x.sum(1, keepdim=True)); rec = ((x - mean) ** 2).sum(1)
    else:
        rm, rlv = model.decode(mu); rec = 0.5 * (rlv + (x - rm) ** 2 / rlv.exp() + np.log(2 * np.pi)).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    (rec + kl).mean().backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters()]).detach().clone()   # the vestigial dispersion head gets no gradient


def linear_inputs(model, x):
    """Input activation of every nn.Linear in the model on x (eval mode, z = mu path), keyed by module name."""
    acts, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m, inp, out, n=name: acts.__setitem__(n, inp[0].detach())))
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
        if model.likelihood == 'nb':
            model.decode(mu, x.sum(1, keepdim=True))
        else:
            model.decode(mu)
    for h in hooks:
        h.remove()
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk']); ap.add_argument('--arms', nargs='+', default=['edits_x2_opt/unrestricted', 'edits_x2_opt/active_only', 'edits_x2_opt/null_only', 'edits_opt/null_only'])
    ap.add_argument('--optimizer', default='sgd'); ap.add_argument('--epochs', nargs='+', type=int, default=[10, 50, 200]); ap.add_argument('--retain_sample', type=int, default=10000); ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    res = json.load(open(args.out))
    for cohort in args.cohorts:
        cfg = SETTINGS[cohort]; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
        f_idx, r_idx, u_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
        X = adata.X
        def tens(idx):
            x = X[idx]; return torch.tensor(x.toarray() if hasattr(x, 'toarray') else np.asarray(x), dtype=torch.float64)
        rs = np.random.RandomState(1).choice(r_idx, min(args.retain_sample, len(r_idx)), replace=False)
        xf, xr, xu = tens(f_idx), tens(rs), tens(u_idx[:2000])
        base_f, _ = load_vae_model(str(cfg['baseline'])); model = deepcopy(base_f.float().cpu()).double()
        names = [k for k, _ in model.named_parameters()]; sizes = [p.numel() for _, p in model.named_parameters()]
        offs = np.r_[0, np.cumsum(sizes)]
        theta0 = parameters_to_vector(model.parameters()).detach().clone()
        def flat(sd):
            return torch.cat([sd[k].double().flatten() for k in names])
        def by_group(v):
            out = {}
            for i, k in enumerate(names):
                g = vae_layer_category(k); out[g] = out.get(g, 0.0) + float((v[offs[i]:offs[i + 1]] ** 2).sum())
            return out
        def top_tensors(v, n=10):
            parts = sorted(((float((v[offs[i]:offs[i + 1]] ** 2).sum()), k) for i, k in enumerate(names)), reverse=True)[:n]
            return [{'name': k, 'sq': s} for s, k in parts]
        def set_theta(v):
            vector_to_parameters(v, model.parameters()); return model
        g0 = mean_grad(set_theta(theta0.clone()), xr)
        S = res['cohorts'][cohort].setdefault('states', {})
        S['_meta'] = {'optimizer': args.optimizer, 'retain_sample': int(len(rs)), 'eps': args.eps, 'retain_grad_sq': float(g0 @ g0), 'retain_grad_sq_by_group': by_group(g0)}
        print(f'== {cohort}: |g_R(theta_0)|^2 {float(g0 @ g0):.4g} by group {S["_meta"]["retain_grad_sq_by_group"]}', flush=True)
        none_states = {e: flat(torch.load(EDIT_DIR / f'{cohort}_none_{args.optimizer}_ft{e}.pt', map_location='cpu', weights_only=False)['state_dict']) - theta0 for e in args.epochs}
        acts0_f, acts0_r = linear_inputs(set_theta(theta0.clone()), xf), linear_inputs(set_theta(theta0.clone()), xr)
        none_acts = {e: (linear_inputs(set_theta(theta0 + none_states[e]), xf), linear_inputs(set_theta(theta0 + none_states[e]), xr)) for e in args.epochs}
        none_det = {e: det(set_theta(theta0 + none_states[e]), xf, xu) for e in args.epochs}
        for arm in args.arms:
            tag, edit = arm.split('/')
            d = flat(torch.load(EDIT_DIR / f'{cohort}_{tag}_{edit}.pt', map_location='cpu', weights_only=False)['state_dict']) - theta0; dsq = float(d @ d)
            gp, gm = mean_grad(set_theta(theta0 + args.eps * d), xr), mean_grad(set_theta(theta0 - args.eps * d), xr)
            hvp = (gp - gm) / (2 * args.eps)
            g_edit = mean_grad(set_theta(theta0 + d), xr)   # the retain gradient at the edited point itself
            nonlin = g_edit - g0 - hvp                      # beyond the linear response of the gradient to the edit
            # the layers that carry the edit, with (delta W, delta b) per nn.Linear
            edit_layers = {}
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    iw = names.index(name + '.weight'); dW = d[offs[iw]:offs[iw + 1]].view_as(mod.weight)
                    db = d[offs[names.index(name + '.bias')]:offs[names.index(name + '.bias') + 1]] if name + '.bias' in names else torch.zeros(mod.out_features)
                    if float((dW ** 2).sum() + (db ** 2).sum()) > 1e-6 * dsq:
                        edit_layers[name] = (dW, db)
            def effective(acts):
                return {n: {'forget': float(((acts[0][n] @ dW.T + db) ** 2).sum(1).mean()), 'retain': float(((acts[1][n] @ dW.T + db) ** 2).sum(1).mean())} for n, (dW, db) in edit_layers.items()}
            A = {'delta_sq': dsq, 'delta_sq_by_group': by_group(d), 'hvp_sq': float(hvp @ hvp), 'hvp_sq_by_group': by_group(hvp), 'delta_H_delta': float(hvp @ d), 'edit_layers': list(edit_layers),
                 'grad_edit_sq': float(g_edit @ g_edit), 'grad_change_sq': float((g_edit - g0) @ (g_edit - g0)), 'grad_change_sq_by_group': by_group(g_edit - g0), 'grad_nonlinear_sq': float(nonlin @ nonlin), 'grad_nonlinear_sq_by_group': by_group(nonlin),
                 'cos_grad_change_hvp': float((g_edit - g0) @ hvp / ((g_edit - g0).norm() * hvp.norm())),
                 'effective_at_base': effective((acts0_f, acts0_r)), 'epochs': {}}
            print(f'  {arm}: |delta|^2 {dsq:.3f} |H_R delta|^2 {float(hvp @ hvp):.4g} (by group {A["hvp_sq_by_group"]}), delta H delta {float(hvp @ d):.4g}; layers {list(edit_layers)}', flush=True)
            print(f'       retain gradient: |g_0|^2 {float(g0 @ g0):.4g}, change at the edited point |g_edit - g_0|^2 {A["grad_change_sq"]:.4g} (by group {A["grad_change_sq_by_group"]}), beyond linear |g_edit - g_0 - H delta|^2 {A["grad_nonlinear_sq"]:.4g} (by group {A["grad_nonlinear_sq_by_group"]}), cos(change, H delta) {A["cos_grad_change_hvp"]:+.3f}', flush=True)
            for e in args.epochs:
                sv = torch.load(EDIT_DIR / f'{cohort}_{tag}_{edit}_{args.optimizer}_ft{e}.pt', map_location='cpu', weights_only=False)
                r = flat(sv['state_dict']) - theta0; D = none_states[e]
                rho = float((r @ d - D @ d) / dsq); rperp = r - (r @ d) / dsq * d; Dperp = D - (D @ d) / dsq * d; diff = rperp - Dperp
                st = {'rho': rho, 'r_sq': float(r @ r), 'r_perp_sq': float(rperp @ rperp), 'disp_perp_sq': float(Dperp @ Dperp), 'path_diff_sq': float(diff @ diff), 'path_diff_sq_by_group': by_group(diff),
                      'path_diff_top': top_tensors(diff), 'cos_rperp_Dperp': float(rperp @ Dperp / (rperp.norm() * Dperp.norm())), 'cos_path_diff_hvp': float(diff @ hvp / (diff.norm() * hvp.norm())),
                      'cos_path_diff_g0': float(diff @ g0 / (diff.norm() * g0.norm())), 'cos_path_diff_grad_change': float(diff @ (g_edit - g0) / (diff.norm() * (g_edit - g0).norm()))}
                st['fit'] = det(set_theta(theta0 + r), xf, xu); st['fit_none'] = none_det[e]
                resid = r - D; st['residual_sq'] = float(resid @ resid); st['residual_sq_by_group'] = by_group(resid); st['cos_residual_delta'] = float(resid @ d / (resid.norm() * d.norm()))
                st['fit_residual'] = det(set_theta(theta0 + resid), xf, xu)   # the residual r - Delta alone at the baseline: the quadratic model's surviving edit
                st['fit_minus_rho_delta'] = det(set_theta(theta0 + r - rho * d), xf, xu); st['fit_minus_delta'] = det(set_theta(theta0 + r - d), xf, xu)
                st['effective'] = effective((linear_inputs(set_theta(theta0 + r), xf), linear_inputs(set_theta(theta0 + r), xr))); st['effective_none'] = effective(none_acts[e])
                A['epochs'][str(e)] = st
                eff = {n: (round(v['forget'], 3), round(st['effective_none'][n]['forget'], 3), round(A['effective_at_base'][n]['forget'], 3)) for n, v in st['effective'].items()}
                print(f'     ft{e}: rho {rho:+.3f}; |r_perp|^2 {st["r_perp_sq"]:.3f} |Delta_perp|^2 {st["disp_perp_sq"]:.3f} |diff|^2 {st["path_diff_sq"]:.3f} (by group {st["path_diff_sq_by_group"]}); cos(r_perp, Delta_perp) {st["cos_rperp_Dperp"]:+.3f}, cos(diff, H delta) {st["cos_path_diff_hvp"]:+.3f}, cos(diff, g0) {st["cos_path_diff_g0"]:+.3f}', flush=True)
                print(f'            forget: edited {st["fit"]["forget"]:.1f} no-edit {st["fit_none"]["forget"]:.1f} residual alone {st["fit_residual"]["forget"]:.1f} (|resid|^2 {st["residual_sq"]:.3f}, cos with delta {st["cos_residual_delta"]:+.3f}) edited minus surviving edit {st["fit_minus_rho_delta"]["forget"]:.1f} minus whole edit {st["fit_minus_delta"]["forget"]:.1f}; held-out {st["fit"]["heldout"]:.1f} / {st["fit_none"]["heldout"]:.1f} / {st["fit_minus_rho_delta"]["heldout"]:.1f}; effective edit on forget cells (edited / none / base) {eff}', flush=True)
            S[arm] = A
            json.dump(res, open(args.out, 'w'), indent=1)
        print(f'saved {cohort} states to {args.out}', flush=True)


if __name__ == '__main__':
    main()
