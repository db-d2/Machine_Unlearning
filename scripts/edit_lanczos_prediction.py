#!/usr/bin/env python3
"""The quadratic model's prediction for what survives of an edit under retain gradient descent, with the full retain
curvature, by a Lanczos iteration on empirical-Fisher-vector products.

In the quadratic model of the retain loss around the baseline, T steps of gradient descent at step eta from the
edited point leave the residual delta_T = (I - eta H_R)^T delta beside the no-edit trajectory (Proposition 2). With
H_R the empirical Fisher of the per-cell retain loss over a sample (eval mode, z = mu, the matrix whose diagonal the caches hold), this script
computes delta_T by the Lanczos approximation of f(H_R) delta with f(lambda) = (1 - eta lambda)^T, reports its
projection on the edit (the predicted rho), its norm, the spectral weight of the edit on the eigenvalues of H_R
(Ritz values), and the forget and held-out fit of the real network at theta_0 + delta_T, the forget damage the
predicted residual carries. Companion of scripts/edit_compensation_test.py; writes into its JSON under
cohorts/<cohort>/lanczos/<tag>/<edit>.

The m-step Lanczos recurrence is the m-point Gauss rule for the spectral measure of delta on K, so rho_hat is an
m-point quadrature of f(lambda) = (1 - eta lambda)^T. For this f (positive even derivatives on the spectrum) the Gauss
rule is a LOWER bound on the quadratic model's rho_hat and the Gauss-Radau rule with a node fixed at lambda = 0 an
UPPER bound (2026-10-30, after the referee's objection that a 40-node rule puts one node below lambda = 2 while the
e-fold scale of f at 200 epochs is 1/(eta T) = 0.02 to 0.05). The script therefore stores the recurrence coefficients,
both rules at every truncation m' <= m (the convergence check), the Radau residual vector beside the Gauss one, and
the superposition test of eq. (excess): the predicted residual added to the saved no-edit SGD state of the same epoch
(the like-for-like comparison with the measured on-path excess, which the baseline-only damage share is not).
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
from retain_null_mass import SETTINGS
from edit_compensation_test import EDIT_DIR, OUT
from edit_compensation_test import loss_zmu
from edit_compensation_states import det
from torch.func import functional_call, jvp, vjp


def weighted_grad(model, x, w):
    """Gradient of mean_n w_n L_n(theta) over x, per-cell loss in eval mode with z = mu, as a flat vector."""
    model.eval(); model.zero_grad()
    mu, logvar = model.encode(x)
    if model.likelihood == 'nb':
        mean, _ = model.decode(mu, x.sum(1, keepdim=True)); rec = ((x - mean) ** 2).sum(1)
    else:
        rm, rlv = model.decode(mu); rec = 0.5 * (rlv + (x - rm) ** 2 / rlv.exp() + np.log(2 * np.pi)).sum(1)
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    ((rec + kl) * w).mean().backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters()]).detach().clone()

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk']); ap.add_argument('--arms', nargs='+', default=['edits_x2_opt/unrestricted', 'edits_x2_opt/active_only', 'edits_x2_opt/null_only', 'edits_opt/null_only'])
    ap.add_argument('--epochs', nargs='+', type=int, default=[10, 50, 200]); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--m', type=int, default=120, help='Lanczos iterations (40 before 2026-10-30)')
    ap.add_argument('--truncations', nargs='+', type=int, default=[20, 40, 60, 80, 100, 120], help='Krylov depths at which the Gauss and Radau rules are also evaluated (convergence check)')
    ap.add_argument('--no_save_residuals', action='store_true', help='do not overwrite the saved Gauss residual vectors (smoke tests)')
    ap.add_argument('--retain_sample', type=int, default=5000); ap.add_argument('--eps', type=float, default=0.01); ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--curvature', choices=['fisher', 'gn'], default='fisher', help="'fisher': the empirical Fisher mean_n g_n g_n' (the methods' matrix, JSON key lanczos); 'gn': the Gauss-Newton matrix, the model's Fisher information (key lanczos_gn)")
    args = ap.parse_args()
    res = json.load(open(args.out))
    for cohort in args.cohorts:
        cfg = SETTINGS[cohort]; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
        f_idx, r_idx, u_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
        X = adata.X
        def tens(idx):
            x = X[idx]; return torch.tensor(x.toarray() if hasattr(x, 'toarray') else np.asarray(x), dtype=torch.float64)
        rs = np.random.RandomState(2).choice(r_idx, min(args.retain_sample, len(r_idx)), replace=False)
        xf, xr, xu = tens(f_idx), tens(rs), tens(u_idx[:2000])
        base_f, _ = load_vae_model(str(cfg['baseline'])); model = deepcopy(base_f.float().cpu()).double()
        names = [k for k, _ in model.named_parameters()]; sizes = [p.numel() for _, p in model.named_parameters()]; offs = np.r_[0, np.cumsum(sizes)]
        theta0 = parameters_to_vector(model.parameters()).detach().clone(); nb = int(np.ceil(len(r_idx) / 256))
        def flat(sd):
            return torch.cat([sd[k].double().flatten() for k in names])
        def by_group(v):
            out = {}
            for i, k in enumerate(names):
                g = vae_layer_category(k); out[g] = out.get(g, 0.0) + float((v[offs[i]:offs[i + 1]] ** 2).sum())
            return out
        def set_theta(v):
            vector_to_parameters(v, model.parameters()); return model
        def hvp(v):
            """Empirical-Fisher product F_R v = mean_n g_n (g_n . v): per-cell directional derivatives a_n = g_n . v by central
            differences of the per-cell loss, then one backward pass of the a-weighted sum of per-cell losses. Positive
            semidefinite, and the same matrix whose diagonal the methods and Proposition 1 use."""
            n = float(v.norm()); u = v / n
            Lp = loss_zmu(set_theta(theta0 + args.eps * u), xr); Lm = loss_zmu(set_theta(theta0 - args.eps * u), xr)
            a = ((Lp - Lm) / (2 * args.eps)).detach()
            return weighted_grad(set_theta(theta0.clone()), xr, a) * n
        base_det = det(set_theta(theta0.clone()), xf, xu)
        # Gauss-Newton product: the model's own Fisher information of the retain cells, 2 J_mean' J_mean + J_mu' J_mu +
        # 1/2 J_lv' diag(e^lv) J_lv averaged over cells, by one forward-mode and one reverse-mode pass
        set_theta(theta0.clone()); P = {k: v.detach().clone() for k, v in model.named_parameters()}
        lib_r = xr.sum(1, keepdim=True)
        def heads(pd):
            mu, lv = functional_call(model.encoder, {k[8:]: v for k, v in pd.items() if k.startswith('encoder.')}, (xr,))
            if model.likelihood == 'nb':
                mean, _ = functional_call(model.decoder, {k[8:]: v for k, v in pd.items() if k.startswith('decoder.')}, (mu, lib_r))
            else:
                mean, _ = functional_call(model.decoder, {k[8:]: v for k, v in pd.items() if k.startswith('decoder.')}, (mu,))
            return mean, mu, lv
        def unflat(v):
            return {k: v[offs[i]:offs[i + 1]].view_as(P[k]) for i, k in enumerate(names)}
        def gn_product(v):
            model.eval()
            (mean, mu, lv), (dmean, dmu, dlv) = jvp(heads, (P,), (unflat(v),))
            n = xr.shape[0]; w = (2 * dmean / n, dmu / n, 0.5 * lv.exp() * dlv / n)
            _, vjp_fn = vjp(heads, P)
            cot = vjp_fn(w)[0]
            return torch.cat([cot[k].flatten() for k in names]).detach().clone()
        product = gn_product if args.curvature == 'gn' else hvp
        # the saved no-edit SGD states Delta_T (restricted_scrub_test.py --save_states), for the superposition test of eq. (excess)
        none_disp = {e: flat(torch.load(EDIT_DIR / f'{cohort}_none_sgd_ft{e}.pt', map_location='cpu', weights_only=False)['state_dict']) - theta0 for e in args.epochs}
        none_fit = {e: det(set_theta(theta0 + none_disp[e]), xf, xu) for e in args.epochs}
        L = res['cohorts'][cohort].setdefault('lanczos' if args.curvature == 'fisher' else 'lanczos_gn', {})
        L['_meta'] = {'curvature': args.curvature, 'm': args.m, 'truncations': args.truncations, 'retain_sample': int(len(rs)), 'eps': args.eps, 'lr': args.lr, 'batches_per_epoch': nb, 'baseline': base_det,
                      'none_fit': {str(e): none_fit[e] for e in args.epochs}, 'bounds': 'rho_pred is the m-point Gauss rule (lower bound); rho_pred_upper the Gauss-Radau rule with a node at 0 (upper bound); convergence holds both at every truncation'}
        for arm in args.arms:
            tag, edit = arm.split('/')
            d = flat(torch.load(EDIT_DIR / f'{cohort}_{tag}_{edit}.pt', map_location='cpu', weights_only=False)['state_dict']) - theta0
            dn = float(d.norm())
            # Lanczos with full reorthogonalization on H_R, started at delta / |delta|
            V = [d / dn]; alphas, betas = [], []
            w = product(V[0]); a = float(w @ V[0]); w = w - a * V[0]; alphas.append(a)
            for j in range(1, args.m):
                b = float(w.norm())
                if b < 1e-10:
                    break
                vj = w / b
                for u in V:   # full reorthogonalization
                    vj = vj - (vj @ u) * u
                vj = vj / vj.norm(); V.append(vj); betas.append(b)
                w = product(vj); a = float(w @ vj); w = w - a * vj - b * V[-2]; alphas.append(a)
                print(f'  {arm}: Lanczos {j + 1}/{args.m} alpha {a:.4g} beta {b:.4g}', flush=True)
            k = len(alphas); beta_next = float(w.norm()); v_next = w / beta_next if beta_next > 1e-10 else None

            def rules(mm):
                """Gauss rule of the Krylov depth mm (Ritz values, squared first components) and the Gauss-Radau rule with a
                node at 0 (Jacobi matrix bordered by the next beta and alpha_tilde = beta^2 (T_mm^{-1})_{mm,mm}); the Radau
                rule is skipped when T_mm is numerically singular, i.e. the Gauss rule already holds a node at 0."""
                Tm = np.diag(alphas[:mm]) + np.diag(betas[:mm - 1], 1) + np.diag(betas[:mm - 1], -1)
                ev, U = np.linalg.eigh(Tm); ev = np.clip(ev, 0.0, None); w1 = U[0, :] ** 2
                bm = betas[mm - 1] if mm < k else beta_next
                radau = None
                if ev.min() > 1e-8 * ev.max() and bm > 1e-10:
                    a_tilde = bm * bm * float((U[-1, :] ** 2 / ev).sum())
                    Tr = np.zeros((mm + 1, mm + 1)); Tr[:mm, :mm] = Tm; Tr[mm - 1, mm] = Tr[mm, mm - 1] = bm; Tr[mm, mm] = a_tilde
                    evr, Ur = np.linalg.eigh(Tr); radau = (np.clip(evr, 0.0, None), Ur, Ur[0, :] ** 2)
                return ev, U, w1, radau

            evals, evecs, w1, radau = rules(k)
            assert evals.max() < 2 / args.lr, f'top Ritz value {evals.max():.4g} exceeds 2/eta: gradient descent at this step would diverge along the edit'
            f_of = lambda ev, T: (1 - args.lr * ev) ** T                 # exact factor; |1 - eta lambda| < 1 on the spectrum by the assertion above
            A = {'delta_sq': dn * dn, 'ritz_values': evals.tolist(), 'ritz_weights': w1.tolist(), 'rayleigh': float(alphas[0]), 'alphas': [float(a) for a in alphas], 'betas': [float(b) for b in betas], 'beta_next': beta_next,
                 'weight_above': {str(c): float(w1[evals > c].sum()) for c in (0.01, 0.1, 1.0, 10.0, 100.0)},
                 'radau_values': radau[0].tolist() if radau else None, 'radau_weights': radau[2].tolist() if radau else None, 'convergence': {}, 'epochs': {}}
            for mm in sorted({m_ for m_ in args.truncations if m_ <= k} | {k}):
                ev_m, _, w_m, rd_m = rules(mm)
                A['convergence'][str(mm)] = {'rho_gauss': {str(e): float((w_m * f_of(ev_m, e * nb)).sum()) for e in args.epochs},
                                             'rho_radau': {str(e): float((rd_m[2] * f_of(rd_m[0], e * nb)).sum()) for e in args.epochs} if rd_m else None,
                                             'lowest_ritz': float(ev_m.min()), 'weight_lowest': float(w_m[ev_m.argmin()])}
            for e in args.epochs:
                T = e * nb; f = f_of(evals, T)
                coef = evecs @ (f * evecs[0, :])            # f(T_m) e_1
                dT = torch.zeros_like(d)
                for c, vj in zip(coef, V):                   # delta_T = |delta| V f(T_m) e_1, accumulated to spare memory
                    dT += float(c) * vj
                dT *= dn
                fit = det(set_theta(theta0 + dT), xf, xu); fit_sup = det(set_theta(theta0 + none_disp[e] + dT), xf, xu)
                damage = res['cohorts'][cohort]['edits'][arm]['damage']
                rec = {'steps': T, 'rho_pred': float(dT @ d / (dn * dn)), 'residual_sq': float(dT @ dT), 'residual_sq_by_group': by_group(dT), 'residual_perp_sq': float((dT - (dT @ d) / (dn * dn) * d).norm() ** 2),
                       'fit_residual_pred': fit, 'damage_share_pred': (fit['forget'] - base_det['forget']) / damage,
                       'fit_superposed_pred': fit_sup, 'share_sup_pred': (fit_sup['forget'] - none_fit[e]['forget']) / damage,
                       'rho_pred_upper': float((radau[2] * f_of(radau[0], T)).sum()) if radau else None}
                if radau and v_next is not None:
                    # the Radau residual: f of the bordered Jacobi matrix applied to e_1 in the (m + 1)-dimensional Krylov space
                    coef_r = radau[1] @ (f_of(radau[0], T) * radau[1][0, :])
                    dR = torch.zeros_like(d)
                    for c, vj in zip(coef_r, V + [v_next]):
                        dR += float(c) * vj
                    dR *= dn
                    fit_r = det(set_theta(theta0 + dR), xf, xu); fit_rs = det(set_theta(theta0 + none_disp[e] + dR), xf, xu)
                    rec['radau'] = {'rho_pred': float(dR @ d / (dn * dn)), 'residual_sq': float(dR @ dR), 'fit_residual_pred': fit_r, 'damage_share_pred': (fit_r['forget'] - base_det['forget']) / damage,
                                    'fit_superposed_pred': fit_rs, 'share_sup_pred': (fit_rs['forget'] - none_fit[e]['forget']) / damage}
                A['epochs'][str(e)] = rec
                if not args.no_save_residuals:
                    torch.save({'residual': dT.float(), 'cohort': cohort, 'tag': tag, 'edit': edit, 'epochs': e}, EDIT_DIR / f'{cohort}_{tag}_{edit}_lanczos_{args.curvature}_ft{e}.pt')
                up = f"{rec['rho_pred_upper']:.3f}" if rec['rho_pred_upper'] is not None else 'n/a'
                print(f'  [{args.curvature}] {arm} ft{e}: predicted rho {rec["rho_pred"]:.3f} (Radau upper {up}), |delta_T|^2 {float(dT @ dT):.3f} (perp {rec["residual_perp_sq"]:.3f}), forget at theta_0 + delta_T {fit["forget"]:.1f} (damage share {rec["damage_share_pred"]:+.3f}; superposed on the no-edit state {rec["share_sup_pred"]:+.3f})', flush=True)
            print(f'  {arm}: Rayleigh {alphas[0]:.4g}; spectral weight above 0.01 / 0.1 / 1 / 10 / 100: ' + ' / '.join(f"{A['weight_above'][c]:.3f}" for c in ('0.01', '0.1', '1.0', '10.0', '100.0')), flush=True)
            print('  convergence (m: rho_gauss / rho_radau at ' + ','.join(str(e) for e in args.epochs) + '): ' + '; '.join(f"{mm}: " + '/'.join(f"{v['rho_gauss'][str(e)]:.3f}" for e in args.epochs) + (' vs ' + '/'.join(f"{v['rho_radau'][str(e)]:.3f}" for e in args.epochs) if v['rho_radau'] else '') for mm, v in A['convergence'].items()), flush=True)
            L[arm] = A; json.dump(res, open(args.out, 'w'), indent=1)
            del V
        print(f'saved {cohort} lanczos to {args.out}', flush=True)


if __name__ == '__main__':
    main()
