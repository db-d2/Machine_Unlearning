#!/usr/bin/env python3
"""Linear response of the retain fine-tune to an edit, measured on the saved edits of the restoration intervention.

The quadratic model of Proposition 2 (Writeup, Section 6.1) says that gradient descent on the retain loss from an
edited baseline theta_0 + delta reaches, after T steps, the no-edit trajectory plus the decayed edit,
theta_T = theta_0 + Delta_T + delta_T with delta_T = (I - eta H_R)^T delta, and that the forget loss excess of the
edited over the no-edit fine-tune is the surviving edit's own damage plus its coupling to the fine-tune's path
through the forget curvature, g_F . delta_T + 1/2 delta_T H_F delta_T + delta_T H_F Delta_T. This script tests both
halves on the saved edits (outputs/p6/restricted_scrub_edits/) and the saved no-edit states of
scripts/restricted_scrub_test.py, with everything on the CPU in double precision:

  superposition  the forget and held-out fit at theta_0 + Delta_T + rho_T delta (the no-edit state at the same epoch
                 plus the measured surviving share rho_T of the edit, drift-corrected as in tab:restricted_rho),
                 at theta_0 + Delta_T + delta (the whole edit kept) and at theta_0 + rho_T delta (the surviving edit
                 with no fine-tune), against the edited arm's measured fit (restricted_scrub_test.json);
  curvature      the retain and forget quadratic forms along the edit direction, delta' H delta by a central second
                 difference of the mean loss and delta' F delta = mean_n (g_n . delta)^2 by per-cell directional
                 derivatives (the empirical Fisher along delta), against the diagonal form sum_j F_jj delta_j^2 the
                 proposition uses; the per-cell loss is the one of the Fisher caches (eval mode, z = mu);
  coupling       delta' H_F Delta_T by a mixed second difference on the forget cells, delta' F_F Delta_T and its
                 cosine, and the diagonal form sum_j F_F,jj delta_j Delta_T,j.

Output: outputs/p6/edit_compensation.json (loader canonical.edit_compensation()).
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
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model
from fisher_utils import AnnDataDataset, compute_vae_fisher
from likelihood_audit import neg_elbo_per_cell
from retain_null_mass import SETTINGS

EDIT_DIR = BASE / 'outputs/p6/restricted_scrub_edits'
RS_JSON = BASE / 'outputs/p6/restricted_scrub_test.json'
OUT = BASE / 'outputs/p6/edit_compensation.json'
torch.set_default_dtype(torch.float64)


def loss_zmu(model, x):
    """Per-cell loss of the Fisher caches: eval mode, z = mu, reconstruction summed over genes plus KL."""
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x)
        if model.likelihood == 'nb':
            mean, _ = model.decode(mu, x.sum(1, keepdim=True)); rec = ((x - mean) ** 2).sum(1)
        else:
            rm, rlv = model.decode(mu); rec = 0.5 * (rlv + (x - rm) ** 2 / rlv.exp() + np.log(2 * np.pi)).sum(1)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
    return rec + kl


class Probe:
    """Evaluates per-cell losses at theta_0 + displacement without copying the model."""
    def __init__(self, model, theta0):
        self.model, self.theta0 = model, theta0

    def at(self, disp):
        vector_to_parameters(self.theta0 + disp, self.model.parameters()); return self.model

    def zmu(self, disp, x):
        return loss_zmu(self.at(disp), x)

    def det(self, disp, xf, xu):
        m = self.at(disp); nf = neg_elbo_per_cell(m, xf); nu = neg_elbo_per_cell(m, xu)
        return {'forget': float(nf.mean()), 'forget_median': float(np.median(nf)), 'heldout': float(nu.mean())}


def directional(probe, u, x, eps, L0):
    """Per-cell directional derivative along the unit vector u, the mean second difference along u, and the mean
    first difference, at theta_0, by central differences of step eps."""
    Lp, Lm = probe.zmu(eps * u, x), probe.zmu(-eps * u, x)
    a = (Lp - Lm) / (2 * eps)
    return a, float(((Lp + Lm - 2 * L0).mean()) / eps ** 2), float(a.mean())


def mixed(probe, u, v, x, eps):
    """u' H v on the mean loss over x by a mixed central difference."""
    f = lambda s, t: float(probe.zmu(eps * (s * u + t * v), x).mean())
    return (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * eps ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk']); ap.add_argument('--tags', nargs='+', default=['edits_x2_opt', 'edits_opt'])
    ap.add_argument('--edits', nargs='+', default=['unrestricted', 'active_only', 'null_only']); ap.add_argument('--optimizers', nargs='+', default=['sgd:1e-3', 'adam:1e-4'])
    ap.add_argument('--epochs', nargs='+', type=int, default=[10, 50, 200]); ap.add_argument('--retain_sample', type=int, default=10000)
    ap.add_argument('--eps', nargs='+', type=float, default=[0.01, 0.02]); ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    J = json.load(open(RS_JSON))['cohorts']
    out_path = Path(args.out); res = json.load(open(out_path)) if out_path.exists() else {}
    res.update({'eps': args.eps, 'retain_sample': args.retain_sample, 'precision': 'float64, CPU', 'percell_loss': 'eval mode, z = mu (the Fisher caches); fits by likelihood_audit.neg_elbo_per_cell', 'cohorts': res.get('cohorts', {})})
    for cohort in args.cohorts:
        cfg = SETTINGS[cohort]; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
        f_idx, r_idx, u_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
        X = adata.X
        def tens(idx):
            x = X[idx]; return torch.tensor(x.toarray() if hasattr(x, 'toarray') else np.asarray(x), dtype=torch.float64)
        rs = np.random.RandomState(1).choice(r_idx, min(args.retain_sample, len(r_idx)), replace=False)
        xf, xr, xu = tens(f_idx), tens(rs), tens(u_idx[:2000])
        base_f, _ = load_vae_model(str(cfg['baseline'])); base_f = base_f.float().cpu(); base = deepcopy(base_f).double()
        names = [k for k, _ in base.named_parameters()]
        theta0 = parameters_to_vector(base.parameters()).detach().clone()
        def flat(sd):
            return torch.cat([sd[k].double().flatten() for k in names])
        base_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
        raw = torch.load(cfg['cache'], map_location='cpu', weights_only=False)['fisher']; FR = flat(raw)
        FF_d, _ = compute_vae_fisher(base_f, DataLoader(AnnDataDataset(adata, f_idx), batch_size=128, shuffle=False), 'cpu', damping=0.0); FF = flat(FF_d)
        nb = int(np.ceil(len(r_idx) / 256))
        probe = Probe(deepcopy(base), theta0)
        L0r, L0f = probe.zmu(torch.zeros_like(theta0), xr), probe.zmu(torch.zeros_like(theta0), xf)
        base_det = probe.det(torch.zeros_like(theta0), xf, xu)
        C = {'n_forget': len(f_idx), 'n_retain': len(r_idx), 'batches_per_epoch': nb, 'baseline': {**base_det, 'forget_json': J[cohort]['baseline']['forget']},
             'retain_fisher_diag': {'mean': float(FR.mean()), 'max': float(FR.max()), 'step_bound_diag': float(1.0 / FR.max())}, 'none': {}, 'edits': {}}
        print(f'== {cohort}: baseline forget {base_det["forget"]:.2f} (json {J[cohort]["baseline"]["forget"]:.2f}) held-out {base_det["heldout"]:.2f}; retain sample {len(rs)}', flush=True)
        # the no-edit fine-tune's displacement at each epoch and its directional derivatives on the forget cells
        none_json = J[cohort]['edits_x2_opt']['none']['finetune']
        D, bF = {}, {}
        for spec in args.optimizers:
            oname, lr = spec.split(':'); lr = float(lr); C['none'][spec] = {}
            for e in args.epochs:
                sv = torch.load(EDIT_DIR / f'{cohort}_none_{oname}_ft{e}.pt', map_location='cpu', weights_only=False)['state_dict']
                Dv = flat(sv) - theta0; D[(spec, e)] = Dv; v = Dv / Dv.norm()
                det = probe.det(Dv, xf, xu); meas = none_json[spec][f'after_ft{e}']
                b, hvv, gv = directional(probe, v, xf, args.eps[0], L0f); bF[(spec, e)] = b
                C['none'][spec][str(e)] = {'disp_sq': float(Dv @ Dv), 'steps': e * nb, 'lr': lr, 'forget': det['forget'], 'forget_median': det['forget_median'], 'heldout': det['heldout'],
                                           'forget_json': meas['forget'], 'heldout_json': meas['heldout'], 'forget_hess_along_unit': hvv, 'forget_grad_along_unit': gv}
                print(f'  none {spec} ft{e}: |Delta|^2 {float(Dv @ Dv):.3f} forget {det["forget"]:.2f} (json {meas["forget"]:.2f}) held-out {det["heldout"]:.2f} (json {meas["heldout"]:.2f})', flush=True)
        for tag in args.tags:
            for edit in args.edits:
                key = f'{tag}/{edit}'; rec = J[cohort][tag][edit]
                sv = torch.load(EDIT_DIR / f'{cohort}_{tag}_{edit}.pt', map_location='cpu', weights_only=False)['state_dict']
                d = flat(sv) - theta0; dn = float(d.norm()); u = d / dn; dsq = dn * dn
                edit_det = probe.det(d, xf, xu); damage = edit_det['forget'] - base_det['forget']
                E = {'delta_sq': dsq, 'after_edit': {**edit_det, 'forget_json': rec['after_edit']['forget']}, 'damage': damage, 'retain': {'diag_form': float(FR @ (d * d))}, 'forget': {'diag_form': float(FF @ (d * d))}, 'arms': {}}
                for eps in args.eps:
                    a, huu, gu = directional(probe, u, xr, eps, L0r)
                    E['retain'][f'eps{eps:g}'] = {'fisher_form': float((a * a).mean()) * dsq, 'hess_form': huu * dsq, 'grad_along': gu * dn, 'rayleigh_hess': huu, 'rayleigh_fisher': float((a * a).mean())}
                    aF, hFF, gF = directional(probe, u, xf, eps, L0f)
                    E['forget'][f'eps{eps:g}'] = {'fisher_form': float((aF * aF).mean()) * dsq, 'hess_form': hFF * dsq, 'grad_along': gF * dn, 'quad_damage': gF * dn + 0.5 * hFF * dsq}
                    if eps == args.eps[0]:
                        aF0 = aF
                r0 = E['retain'][f'eps{args.eps[0]:g}']; E['retain']['fisher_over_diag'] = r0['fisher_form'] / E['retain']['diag_form']; E['retain']['hess_over_diag'] = r0['hess_form'] / E['retain']['diag_form']
                f0 = E['forget'][f'eps{args.eps[0]:g}']; E['forget']['fisher_over_diag'] = f0['fisher_form'] / E['forget']['diag_form']; E['forget']['hess_over_diag'] = f0['hess_form'] / E['forget']['diag_form']
                print(f'  {key}: |delta|^2 {dsq:.3f} damage {damage:.1f} (json {rec["after_edit"]["forget"] - J[cohort]["baseline"]["forget"]:.1f}); retain curvature along delta: diag {E["retain"]["diag_form"]:.4g} fisher {r0["fisher_form"]:.4g} hess {r0["hess_form"]:.4g} (Rayleigh {r0["rayleigh_hess"]:.4g}); forget: diag {E["forget"]["diag_form"]:.4g} fisher {f0["fisher_form"]:.4g} hess {f0["hess_form"]:.4g} grad {f0["grad_along"]:.4g}', flush=True)
                for spec in args.optimizers:
                    oname, lr = spec.split(':'); lr = float(lr); E['arms'][spec] = {}
                    for e in args.epochs:
                        Dv = D[(spec, e)]; v = Dv / Dv.norm(); dnorm = float(Dv.norm()); steps = e * nb
                        m = rec['finetune'][spec][f'after_ft{e}']; drift = none_json[spec][f'after_ft{e}']['drift_projection'][key]['all']
                        rho = m['edit']['all']['rho'] - drift
                        sup_rho, sup_full, rev = probe.det(Dv + rho * d, xf, xu), probe.det(Dv + d, xf, xu), probe.det(rho * d, xf, xu)
                        none_f = C['none'][spec][str(e)]['forget']
                        coup_h = mixed(probe, u, v, xf, args.eps[0]) * dn * dnorm; coup_f = float((aF0 * bF[(spec, e)]).mean()) * dn * dnorm
                        cosF = float((aF0 * bF[(spec, e)]).mean() / ((aF0 * aF0).mean().sqrt() * (bF[(spec, e)] * bF[(spec, e)]).mean().sqrt()))
                        r1 = E['retain'][f'eps{args.eps[0]:g}']
                        A = {'rho': rho, 'rho_raw': m['edit']['all']['rho'], 'drift_projection': drift, 'orth_abs': m['edit']['all']['orth'] * dsq, 'disp_sq': float(Dv @ Dv), 'disp_sq_perp': float(Dv @ Dv) - (float(Dv @ d) ** 2) / dsq,
                             'forget_edited_json': m['forget'], 'forget_none_json': none_json[spec][f'after_ft{e}']['forget'], 'forget_none': none_f, 'heldout_edited_json': m['heldout'],
                             'sup_rho': sup_rho, 'sup_full': sup_full, 'rev_only': rev,
                             'excess_meas': m['forget'] - none_json[spec][f'after_ft{e}']['forget'], 'excess_sup_rho': sup_rho['forget'] - none_f, 'excess_sup_full': sup_full['forget'] - none_f, 'excess_rev_only': rev['forget'] - base_det['forget'],
                             'coupling': {'hess': coup_h, 'fisher': coup_f, 'cos_fisher': cosF, 'diag': float(FF @ (d * Dv))},
                             'pred_quad_excess': rho * f0['grad_along'] + 0.5 * rho * rho * f0['hess_form'] + rho * coup_h,
                             'pred_rho_diag': m['edit']['all'].get('pred_rho'), 'pred_rho_one_mode_hess': float(max(0.0, 1 - lr * r1['rayleigh_hess']) ** steps) if oname == 'sgd' else None,
                             'pred_rho_one_mode_fisher': float(max(0.0, 1 - lr * r1['rayleigh_fisher']) ** steps) if oname == 'sgd' else None}
                        for k in ('excess_meas', 'excess_sup_rho', 'excess_sup_full', 'excess_rev_only', 'pred_quad_excess'):
                            A['share_' + k[7:] if k.startswith('excess_') else 'share_quad'] = A[k] / damage
                        E['arms'][spec][str(e)] = A
                        print(f'     {spec} ft{e}: rho {rho:+.3f}; forget measured {m["forget"]:.1f} vs no-edit {none_f:.1f} (excess {A["excess_meas"]:+.1f}, share {A["share_meas"]:+.3f}); superposed rho {sup_rho["forget"]:.1f} (share {A["share_sup_rho"]:+.3f}), full {sup_full["forget"]:.1f} ({A["share_sup_full"]:+.3f}), surviving edit alone {rev["forget"]:.1f} ({A["share_rev_only"]:+.3f}); coupling hess {coup_h:+.4g} fisher {coup_f:+.4g} cos {cosF:+.3f} diag {A["coupling"]["diag"]:+.4g}; quad share {A["share_quad"]:+.3f}; held-out sup {sup_rho["heldout"]:.1f} vs {m["heldout"]:.1f}', flush=True)
                C['edits'][key] = E
                res['cohorts'][cohort] = C; json.dump(res, open(out_path, 'w'), indent=1)
        res['cohorts'][cohort] = C; json.dump(res, open(out_path, 'w'), indent=1)
        print(f'saved {cohort} to {out_path}', flush=True)


if __name__ == '__main__':
    main()
