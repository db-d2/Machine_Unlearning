#!/usr/bin/env python3
"""Intervention for the restoration proposition (Section 6, round 8): does the retain fine-tune undo an edit that
lies in the directions the retain set constrains, and leave one that lies in the directions it does not?

From the baseline, three edits of matched initial damage to the forget cells' fit, each followed by the methods'
retain-only fine-tune (Adam 1e-4, batch 256, 10 epochs, then continued to 50):
- unrestricted: the paper's Fisher scrub (100 steps, cap 1e-3), the published setting;
- null-only: the same scrub with the update masked to the retain-null coordinates, F_R,j < tau mean(F_R), run until
  the forget fit is damaged as much as the unrestricted scrub damaged it (cap raised to 1e-2 and up to MAX_STEPS
  steps, since these coordinates carry little forget Fisher mass on PBMC-33k), or as far as it gets;
- active-only: the same scrub masked to the complement.
The proposition predicts that the fine-tune restores the unrestricted and active-only edits at a rate set by the
retain Fisher and does not touch the null-only edit, so the forget fit after the fine-tune should sit near the
baseline's for the first two and stay damaged for the third, with the held-out fit the observable difference (the
null-only edit costs nothing the retain data can repair, and nothing they need to).

Measured after the edit and after 10 and 50 retain epochs, on the CPU with the deterministic estimator: negative
ELBO per cell for the forget cells, the matched negatives, a retain sample of 2,000 and a held-out sample of 2,000,
and the reference-free audit point (reference = mean over the clean retrains' per-cell fits, from
`audit_percell_negelbo_pbmc33k.npz` on PBMC-33k and `persample_mk.json` on PBMC-68k). Training runs on MPS (as
the methods did); the endpoint is therefore reproducible to the estimator's precision only.

Writes outputs/p6/restricted_scrub_test.json. Run: PYTHONPATH=src python scripts/restricted_scrub_test.py [--cohorts pbmc33k pbmc68k_mk] [--tau 0.01]

Optimizer arms (2026-10-13; the proposition's rates are those of gradient descent, the methods use Adam): with
``--optimizers adam:1e-4 sgd:1e-3`` every edit is saved to --edit_dir after the scrub and each fine-tune arm starts
from that same saved edit, so the arms are paired on the edit; results go under <tag>/<edit>/finetune/<spec> with the
measurements at every epoch of --ft_epochs. ``--edits none ...`` adds the no-edit fine-tune (the trajectory an edit is
restored onto). The SGD step 1e-3 is the largest that one retain epoch from the baseline leaves without a held-out
cost on either cohort (2.5e-3 and above spike; the proposition's bound 1/max F_R is 0.013, so the empirical Fisher
underestimates the top curvature by about five). Example:
  PYTHONPATH=src python scripts/restricted_scrub_test.py --cohorts pbmc33k --target_factor 2 --tag edits_x2_opt \
      --edits none unrestricted active_only null_only --optimizers adam:1e-4 sgd:1e-3 --ft_epochs 10 50 200
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src')); sys.path.insert(0, str(BASE / 'scripts'))
import eval_multiseed as EM
EM.DEVICE = 'cpu'
from eval_multiseed import load_vae_model
from fisher_utils import AnnDataDataset, compute_vae_fisher, vae_layer_category
from train_fisher_unlearn import compute_fisher_diagonal, retain_finetune, _neg_elbo_sum
from likelihood_audit import neg_elbo_per_cell, lr_auc
from retain_null_mass import SETTINGS

OUT = BASE / 'outputs/p6/restricted_scrub_test.json'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MAX_STEPS, CAP_MASKED, CAP_PUBLISHED, SCRUB_LR, STEPS_PUBLISHED = 3000, 1e-2, 1e-3, 1e-4, 100
FT_EPOCHS = (10, 50)


def masked_scrub(model, forget_loader, fisher, mask, device, target=None, max_steps=MAX_STEPS, cap=CAP_MASKED):
    """The paper's Fisher scrub with the preconditioned update multiplied by ``mask`` (None = unrestricted).
    Stops when the per-cell forget loss reaches ``target`` or after ``max_steps``."""
    model.eval(); n = len(forget_loader.dataset); params = [(k, p) for k, p in model.named_parameters() if p.requires_grad]; losses = []
    for step in range(max_steps):
        model.zero_grad(); total = 0.0
        for x, lib in forget_loader:
            x, lib = x.to(device), lib.to(device); loss = _neg_elbo_sum(model, x, lib); loss.backward(); total += loss.item()
        with torch.no_grad():
            upd = {k: SCRUB_LR * p.grad / fisher[k] for k, p in params if p.grad is not None}
            if mask is not None:
                upd = {k: u * mask[k] for k, u in upd.items()}
            largest = max(u.abs().max() for u in upd.values()); scale = min(1.0, cap / (largest.item() + 1e-30))
            for k, p in params:
                if k in upd:
                    p.add_(upd[k] * scale)
        losses.append(total / n)
        if target is not None and losses[-1] >= target:
            break
    return losses


def edit_diagnostics(model, base_sd, edit_sd, null_mask, F_F, F_R, steps, eta):
    """How much of the edit delta = theta_edit - theta_base remains in the parameters, overall, on the retain-null and
    retain-active coordinates, and by layer group. With r = theta - theta_base: rho = <r, delta> / <delta, delta> (1 =
    the edit untouched, 0 = reverted); fisher_survival = sum F_F r^2 / sum F_F delta^2 (the quadratic model's forget-loss
    excess relative to the edit's); orth = |r - rho delta|^2 / |delta|^2 (movement off the edit direction, the fine-tune's
    own drift and any compensation); pred_gd = sum F_F delta^2 (1 - eta F_R)^(2 steps) / sum F_F delta^2, the proposition's
    prediction for this edit under gradient descent (None for Adam), and pred_gd_x5 with the curvature five times the
    empirical Fisher (the step-size probe's calibration); pred_rho = sum delta^2 (1 - eta F_R)^steps / sum delta^2 is the
    same prediction for rho itself. CPU, double precision."""
    sd = {k: v.detach().cpu().double() for k, v in model.state_dict().items()}; acc = {}
    def add(key, d, r, ff, fr):
        a = acc.setdefault(key, {'dd': 0.0, 'rd': 0.0, 'rr': 0.0, 'fdd': 0.0, 'frr': 0.0, 'pred': 0.0, 'pred5': 0.0, 'prho': 0.0, 'prho5': 0.0})
        a['dd'] += float((d * d).sum()); a['rd'] += float((r * d).sum()); a['rr'] += float((r * r).sum()); a['fdd'] += float((ff * d * d).sum()); a['frr'] += float((ff * r * r).sum())
        if eta is not None:
            a['pred'] += float((ff * d * d * (1 - eta * fr).clamp(min=0.0) ** (2 * steps)).sum()); a['pred5'] += float((ff * d * d * (1 - 5 * eta * fr).clamp(min=0.0) ** (2 * steps)).sum())
            a['prho'] += float((d * d * (1 - eta * fr).clamp(min=0.0) ** steps).sum()); a['prho5'] += float((d * d * (1 - 5 * eta * fr).clamp(min=0.0) ** steps).sum())
    for k, m in null_mask.items():
        d = edit_sd[k].double() - base_sd[k].double(); r = sd[k] - base_sd[k].double(); ff = F_F[k].double(); fr = F_R[k].double(); mb = m.cpu().bool()
        add('all', d, r, ff, fr); add('null', d[mb], r[mb], ff[mb], fr[mb]); add('active', d[~mb], r[~mb], ff[~mb], fr[~mb]); add(vae_layer_category(k), d, r, ff, fr)
    out = {}
    for key, a in acc.items():
        if a['dd'] <= 0:   # the group carries no edit (a masked edit on the other coordinates)
            out[key] = {'rho': None, 'fisher_survival': None, 'orth': None, 'pred_gd': None, 'pred_gd_x5': None, 'pred_rho': None, 'pred_rho_x5': None, 'delta_sq': 0.0, 'fisher_damage': 0.0}
            continue
        rho = a['rd'] / a['dd']
        out[key] = {'rho': rho, 'fisher_survival': a['frr'] / a['fdd'] if a['fdd'] > 0 else None, 'orth': (a['rr'] - 2 * rho * a['rd'] + rho * rho * a['dd']) / a['dd'],
                    'pred_gd': (a['pred'] / a['fdd'] if a['fdd'] > 0 else None) if eta is not None else None, 'pred_gd_x5': (a['pred5'] / a['fdd'] if a['fdd'] > 0 else None) if eta is not None else None,
                    'pred_rho': a['prho'] / a['dd'] if eta is not None else None, 'pred_rho_x5': a['prho5'] / a['dd'] if eta is not None else None, 'delta_sq': a['dd'], 'fisher_damage': a['fdd']}
    return out


def parse_opt(spec):
    name, lr = spec.split(':'); return name, float(lr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohorts', nargs='+', default=['pbmc33k', 'pbmc68k_mk']); ap.add_argument('--tau', type=float, default=1e-2); ap.add_argument('--only', nargs='+', default=None, help='rerun these edits only and merge'); ap.add_argument('--cap_active', type=float, default=CAP_PUBLISHED, help='cap for the retain-active edit (its coordinates carry the mass, so the published cap reaches the target)'); ap.add_argument('--target_factor', type=float, default=None, help='stop every edit when the forget loss reaches this multiple of its starting value (a moderate, local test); default = match the published scrub'); ap.add_argument('--tag', default=None, help='JSON key for this configuration (default edits, or edits_x<factor>)')
    ap.add_argument('--optimizers', nargs='+', default=None, help='fine-tune arms as name:lr (adam:1e-4 sgd:1e-3); given, edits are saved and every arm starts from the same saved edit'); ap.add_argument('--ft_epochs', nargs='+', type=int, default=list(FT_EPOCHS), help='measure after these cumulative retain epochs'); ap.add_argument('--edits', nargs='+', default=['unrestricted', 'active_only', 'null_only'], help="edits to run, in order; 'none' = no edit"); ap.add_argument('--edit_dir', default=str(BASE / 'outputs/p6/restricted_scrub_edits')); ap.add_argument('--rescrub', action='store_true', help='ignore saved edits'); ap.add_argument('--save_states', action='store_true', help='save each edit arm\'s fine-tuned state at every measured epoch (<cohort>_<tag>_<edit>_<opt>_ft<e>.pt)'); ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    res = json.load(open(out_path)) if out_path.exists() else {}
    res.update({'tau': args.tau, 'device_train': DEV, 'ft': 'Adam 1e-4, batch 256, retain only (legacy after_ft keys); optimizer arms under finetune/<name:lr>', 'cohorts': res.get('cohorts', {})})
    for cohort in args.cohorts:
        cfg = SETTINGS[cohort]; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
        f_idx, r_idx, u_idx = split['forget_indices'], split['retain_indices'], split['unseen_indices']
        if cohort == 'pbmc33k':
            m_idx = json.load(open(BASE / 'outputs/p1.5/s1_matched_negatives.json'))['matched_indices']
            z = np.load(BASE / 'outputs/audit_percell_negelbo_pbmc33k.npz')
            ref_f = np.mean([z[k] for k in z.files if k.startswith('retrain__') and k.endswith('__forget')], 0); ref_m = np.mean([z[k] for k in z.files if k.startswith('retrain__') and k.endswith('__matched')], 0)
        else:
            m_idx = json.load(open(BASE / 'outputs/pbmc68k/matched_negatives_mk.json'))['matched_indices']
            ps = json.load(open(BASE / 'outputs/pbmc68k/persample_mk.json'))['neg_elbo']['retrain']
            ref_f = np.mean([r['forget'] for r in ps], 0); ref_m = np.mean([r['cross'] for r in ps], 0)
        X = adata.X
        def tens(idx):
            x = X[idx]; return torch.FloatTensor(x.toarray() if hasattr(x, 'toarray') else np.asarray(x))
        rs = np.random.RandomState(0).choice(r_idx, 2000, replace=False)
        xf, xm, xr, xu = tens(f_idx), tens(m_idx), tens(rs), tens(u_idx[:2000])
        base, _ = load_vae_model(str(cfg['baseline']))
        def measure(model, tag):
            m = deepcopy(model).to('cpu'); nf, nm = neg_elbo_per_cell(m, xf), neg_elbo_per_cell(m, xm)
            rec = {'forget': float(nf.mean()), 'forget_median': float(np.median(nf)), 'matched': float(nm.mean()), 'retain_sample': float(neg_elbo_per_cell(m, xr).mean()),
                   'heldout': float(neg_elbo_per_cell(m, xu).mean()), 'audit_meanref': lr_auc(ref_f, ref_m, nf, nm)}
            print(f"  {tag:48s} forget {rec['forget']:8.1f} (median {rec['forget_median']:8.1f}) matched {rec['matched']:7.1f} retain {rec['retain_sample']:7.1f} heldout {rec['heldout']:7.1f} audit {rec['audit_meanref']:.3f}", flush=True)
            return rec
        forget_loader = DataLoader(AnnDataDataset(adata, f_idx), batch_size=256, shuffle=False)
        retain_loader = DataLoader(AnnDataDataset(adata, r_idx), batch_size=256, shuffle=True)
        fisher_loader = DataLoader(AnnDataDataset(adata, r_idx), batch_size=128, shuffle=False)
        fisher = compute_fisher_diagonal(deepcopy(base).to(DEV), fisher_loader, DEV, damping=0.1, cache=str(cfg['cache']))
        raw = {k: v.to(DEV) for k, v in torch.load(cfg['cache'], map_location='cpu', weights_only=False)['fisher'].items()}
        m_r = torch.cat([v.flatten() for v in raw.values()]).mean()
        null_mask = {k: (v < args.tau * m_r).float() for k, v in raw.items()}; active_mask = {k: 1.0 - v for k, v in null_mask.items()}
        n_null = sum(int(v.sum()) for v in null_mask.values()); n_all = sum(v.numel() for v in null_mask.values())
        if args.optimizers is not None:
            F_F, _ = compute_vae_fisher(deepcopy(base).to('cpu'), DataLoader(AnnDataDataset(adata, f_idx), batch_size=128, shuffle=False), 'cpu', damping=0.0)
            F_F = {k: v.cpu() for k, v in F_F.items()}; F_R = {k: v.cpu() for k, v in raw.items()}; base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
            nb = len(retain_loader)
        tag = args.tag or ('edits' if args.target_factor is None else f'edits_x{args.target_factor:g}')
        print(f'== {cohort} [{tag}]: retain-null coordinates at tau {args.tau}: {n_null:,} of {n_all:,}; optimizers {args.optimizers}; epochs {args.ft_epochs}', flush=True)
        prev = res['cohorts'].get(cohort, {})
        out = dict(prev); out.update({'n_null_coords': n_null, 'n_coords': n_all, 'baseline': prev.get('baseline') or measure(base, 'baseline')}); out[tag] = dict(prev.get(tag, {}))
        target = out[tag]['unrestricted']['target_loss'] if 'unrestricted' in out[tag] else None
        specs = {'unrestricted': (None, CAP_PUBLISHED, STEPS_PUBLISHED), 'active_only': (active_mask, args.cap_active, MAX_STEPS), 'null_only': (null_mask, CAP_MASKED, MAX_STEPS), 'none': (None, None, 0)}
        for name in args.edits:
            if args.only and name not in args.only:
                continue
            mask, cap, steps = specs[name]
            edit_path = Path(args.edit_dir) / f'{cohort}_{tag}_{name}.pt'
            if args.optimizers is not None and edit_path.exists() and not args.rescrub:
                saved = torch.load(edit_path, map_location='cpu', weights_only=False)
                model = deepcopy(base); model.load_state_dict(saved['state_dict']); model = model.to(DEV)
                rec = dict(saved['scrub']); target = target if target is not None else rec['target_loss']
                print(f'  loaded saved edit {edit_path.name}', flush=True)
                rec['after_edit'] = measure(model, f'{name}: saved edit ({rec["scrub_steps"]} steps)')
            else:
                torch.manual_seed(42); np.random.seed(42)
                model = deepcopy(base).to(DEV)
                if name == 'none':
                    scrub = {'scrub_steps': 0, 'scrub_loss_start': None, 'scrub_loss_end': None, 'target_loss': target, 'cap': None}
                else:
                    if args.target_factor is not None:
                        first = masked_scrub(deepcopy(model), forget_loader, fisher, mask, DEV, target=None, max_steps=1, cap=cap)[0] if target is None else None
                        target = target if target is not None else args.target_factor * first
                        losses = masked_scrub(model, forget_loader, fisher, mask, DEV, target=target, max_steps=steps, cap=cap)
                    else:
                        losses = masked_scrub(model, forget_loader, fisher, mask, DEV, target=None if name == 'unrestricted' else target, max_steps=steps, cap=cap)
                        if name == 'unrestricted':
                            target = losses[-1]
                    scrub = {'scrub_steps': len(losses), 'scrub_loss_start': losses[0], 'scrub_loss_end': losses[-1], 'target_loss': target, 'cap': cap}
                rec = dict(scrub); rec['after_edit'] = measure(model, f'{name}: after edit ({scrub["scrub_steps"]} steps)')
                if args.optimizers is not None:
                    edit_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'scrub': scrub, 'cohort': cohort, 'tag': tag, 'edit': name}, edit_path)
            if args.optimizers is None:
                done = 0
                for e in args.ft_epochs:
                    retain_finetune(model.to(DEV), retain_loader, DEV, epochs=e - done, lr=1e-4); done = e
                    rec[f'after_ft{e}'] = measure(model, f'{name}: after {e} retain epochs')
            else:
                edited = {k: v.clone() for k, v in model.state_dict().items()}; edit_cpu = {k: v.detach().cpu() for k, v in edited.items()}
                rec['after_edit']['edit'] = edit_diagnostics(model, base_sd, edit_cpu, null_mask, F_F, F_R, 0, None) if name != 'none' else None
                rec['finetune'] = dict(out[tag].get(name, {}).get('finetune', {}))
                for spec in args.optimizers:
                    oname, lr = parse_opt(spec)
                    model.load_state_dict(edited); model = model.to(DEV)
                    torch.manual_seed(42); np.random.seed(42)
                    arm = {'optimizer': oname, 'lr': lr, 'retain_loss': []}; done = 0
                    for e in args.ft_epochs:
                        arm['retain_loss'] += [float(x) for x in retain_finetune(model, retain_loader, DEV, epochs=e - done, lr=lr, optimizer=oname)]; done = e
                        arm[f'after_ft{e}'] = measure(model, f'{name} / {spec}: after {e} retain epochs')
                        if name == 'none':
                            # the fine-tune's own drift projected onto every saved edit of this cohort (any tag), so that an edit
                            # arm's rho can be read relative to where the no-edit fine-tune sits at the same epoch
                            proj = {}
                            for ep in sorted(Path(args.edit_dir).glob(f'{cohort}_*.pt')):
                                sv = torch.load(ep, map_location='cpu', weights_only=False)
                                if 'scrub' not in sv or sv.get('edit') == 'none':
                                    continue
                                dg = edit_diagnostics(model, base_sd, dict(sv['state_dict']), null_mask, F_F, F_R, e * nb, None)
                                proj[f"{sv['tag']}/{sv['edit']}"] = {g: dg[g]['rho'] for g in ('all', 'null', 'active')}
                            arm[f'after_ft{e}']['drift_projection'] = proj
                            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'cohort': cohort, 'optimizer': spec, 'epochs': e}, Path(args.edit_dir) / f'{cohort}_none_{oname}_ft{e}.pt')
                            fmt = lambda x: '--' if x is None else f'{x:+.3f}'
                            print('    drift projection onto saved edits: ' + ', '.join(f"{k} {fmt(v['all'])} (null {fmt(v['null'])}, active {fmt(v['active'])})" for k, v in proj.items()), flush=True)
                        else:
                            arm[f'after_ft{e}']['edit'] = ed = edit_diagnostics(model, base_sd, edit_cpu, null_mask, F_F, F_R, e * nb, lr if oname == 'sgd' else None)
                            fmt = lambda x: '--' if x is None else f'{x:.3f}'
                            if args.save_states:
                                torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'cohort': cohort, 'tag': tag, 'edit': name, 'optimizer': spec, 'epochs': e}, Path(args.edit_dir) / f'{cohort}_{tag}_{name}_{oname}_ft{e}.pt')
                            print(f"    edit remaining rho all {fmt(ed['all']['rho'])} null {fmt(ed['null']['rho'])} active {fmt(ed['active']['rho'])}; off-edit {fmt(ed['all']['orth'])}" + (f"; GD prediction for rho all {fmt(ed['all']['pred_rho'])} null {fmt(ed['null']['pred_rho'])} active {fmt(ed['active']['pred_rho'])} (x5 {fmt(ed['all']['pred_rho_x5'])})" if ed['all']['pred_rho'] is not None else ''), flush=True)
                    rec['finetune'][spec] = arm
            out[tag][name] = rec
            res['cohorts'][cohort] = out
            json.dump(res, open(out_path, 'w'), indent=1)
        print(f'saved {cohort} to {out_path}', flush=True)


if __name__ == '__main__':
    main()
