#!/usr/bin/env python3
"""Retain-null mass: the share of the forget set's Fisher information that lies on parameters the retain set does
not constrain. This is the quantity the restoration proposition (Section 6, round 8) says decides whether an
edit-then-fine-tune method can work.

In the quadratic model around the baseline, T epochs of retain gradient descent at rate eta shrink coordinate j of
an edit by (1 - eta F_R,j)^T, so the forget-loss damage that survives a long fine-tune is the forget Fisher mass on
coordinates with F_R,j near zero. Two threshold-free and threshold readings are reported per parameter group:
- survival S(t) = sum_j F_F,j exp(-t F_R,j / mean(F_R)) / sum_j F_F,j, the share of forget mass still uncorrected
  after t units of fine-tuning time in which a typical retain coordinate has decayed by e^-t (t = 1, 10, 100, 1000);
- retain-null mass N(tau) = share of forget mass on coordinates with F_R,j < tau mean(F_R), tau = 1e-3, 1e-2, 1e-1.
Settings: PBMC-33k cluster 13, PBMC-68k clusters 9 and 8 (per-sample diagonal Fisher, damping 0, the retain Fisher
from the scrub caches and the forget Fisher computed here with the same estimator), and a linear classifier on the
PBMC-33k baseline's latents (the paper's classifier control; per-sample Fisher of the cross-entropy loss, exact).
Cosines are recomputed as a check against the paper's overlap tables.

Writes outputs/p6/retain_null_mass.json. Run: PYTHONPATH=src python scripts/retain_null_mass.py
"""
import json
import sys
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
from fisher_utils import AnnDataDataset, compute_vae_fisher, vae_layer_category, cosine_sim

OUT = BASE / 'outputs/p6/retain_null_mass.json'
TS = [1.0, 10.0, 100.0, 1000.0]
TAUS = [1e-3, 1e-2, 1e-1]
SETTINGS = {
    'pbmc33k': {'data': BASE / 'data/adata_processed.h5ad', 'split': BASE / 'outputs/p1/split_structured.json', 'baseline': BASE / 'outputs/p1/baseline/best_model.pt',
                'cache': BASE / 'outputs/p2/fisher_unlearn/fisher_retain_persample.pt'},
    'pbmc68k_mk': {'data': BASE / 'data/pbmc68k_processed.h5ad', 'split': BASE / 'outputs/pbmc68k/split_mk.json', 'baseline': BASE / 'outputs/pbmc68k/baseline/best_model.pt',
                   'cache': BASE / 'outputs/pbmc68k/mk/fisher_retain_persample.pt'},
    'pbmc68k_pdc': {'data': BASE / 'data/pbmc68k_processed.h5ad', 'split': BASE / 'outputs/pbmc68k/split_pdc.json', 'baseline': BASE / 'outputs/pbmc68k/baseline/best_model.pt',
                    'cache': BASE / 'outputs/pbmc68k/pdc/fisher_retain_persample.pt'},
}


def stats(ff, fr, m_r):
    """ff, fr: 1-d numpy arrays (forget and retain Fisher on the same coordinates); m_r: global mean retain Fisher."""
    tot = ff.sum()
    out = {'n_params': int(ff.size), 'forget_mass_share': None, 'cosine': float(cosine_sim(torch.tensor(ff), torch.tensor(fr))),
           'survival': {str(t): float((ff * np.exp(-t * fr / m_r)).sum() / tot) for t in TS},
           'retain_null_mass': {str(tau): float(ff[fr < tau * m_r].sum() / tot) for tau in TAUS},
           'retain_null_share_of_retain_mass': {str(tau): float(fr[fr < tau * m_r].sum() / fr.sum()) for tau in TAUS},
           'retain_null_share_of_params': {str(tau): float((fr < tau * m_r).mean()) for tau in TAUS}}
    return out


def vae_setting(name, cfg):
    adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
    model, _ = load_vae_model(str(cfg['baseline']))
    raw = torch.load(cfg['cache'], map_location='cpu', weights_only=False)
    F_R = {k: v.cpu().double().numpy() for k, v in raw['fisher'].items()}
    loader = DataLoader(AnnDataDataset(adata, split['forget_indices']), batch_size=128, shuffle=False)
    F_F, n = compute_vae_fisher(model, loader, 'cpu', damping=0.0)
    F_F = {k: v.cpu().double().numpy() for k, v in F_F.items()}
    m_r = np.concatenate([v.ravel() for v in F_R.values()]).mean()
    groups = {}
    for k in F_F:
        groups.setdefault(vae_layer_category(k), []).append(k)
    res = {'n_forget': int(n), 'n_retain_cache': int(raw['n_samples']), 'mean_retain_fisher': float(m_r), 'groups': {}}
    tot_f = sum(F_F[k].sum() for k in F_F)
    for g, keys in groups.items():
        ff = np.concatenate([F_F[k].ravel() for k in keys]); fr = np.concatenate([F_R[k].ravel() for k in keys])
        res['groups'][g] = stats(ff, fr, m_r); res['groups'][g]['forget_mass_share'] = float(ff.sum() / tot_f)
    ff = np.concatenate([F_F[k].ravel() for k in F_F]); fr = np.concatenate([F_R[k].ravel() for k in F_F])
    res['groups']['global'] = stats(ff, fr, m_r); res['groups']['global']['forget_mass_share'] = 1.0
    # the decoder output weight alone, by hidden-unit column: share of forget mass on columns the retain set ranks lowest
    kw = [k for k in F_F if k.endswith('fc_mean.0.weight')]
    if kw:
        W_f, W_r = F_F[kw[0]], F_R[kw[0]]; nu_f, nu_r = W_f.sum(0), W_r.sum(0)
        order = np.argsort(nu_r)
        res['decoder_output_columns'] = {'n_units': int(W_f.shape[1]),
                                         'forget_mass_on_bottom_5pct_retain_units': float(nu_f[order[:max(1, len(order) // 20)]].sum() / nu_f.sum()),
                                         'forget_mass_on_bottom_1pct_retain_units': float(nu_f[order[:max(1, len(order) // 100)]].sum() / nu_f.sum()),
                                         'retain_mass_on_top_5pct_forget_units': float(nu_r[np.argsort(-nu_f)[:max(1, len(order) // 20)]].sum() / nu_r.sum())}
    return res


def classifier_setting():
    """Linear classifier on the PBMC-33k baseline's latent means, Leiden labels, trained on retain + forget cells;
    forget = cluster 13 (the NB26 control). Per-sample Fisher of the cross-entropy loss computed exactly."""
    cfg = SETTINGS['pbmc33k']; adata = sc.read_h5ad(cfg['data']); split = json.load(open(cfg['split']))
    model, _ = load_vae_model(str(cfg['baseline']))
    idx = np.array(split['retain_indices'] + split['forget_indices']); is_forget = np.r_[np.zeros(len(split['retain_indices'])), np.ones(len(split['forget_indices']))].astype(bool)
    X = adata.X[idx]; X = torch.FloatTensor(X.toarray() if hasattr(X, 'toarray') else np.asarray(X))
    with torch.no_grad():
        mu, _ = model.encode(X)
    y = torch.tensor(adata.obs['leiden'].values[idx].astype(int)); n_classes = int(y.max()) + 1
    torch.manual_seed(0)
    clf = torch.nn.Linear(mu.shape[1], n_classes); opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-2)
    for step in range(2000):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(clf(mu), y); loss.backward(); opt.step()
        acc = (clf(mu).argmax(1) == y).float().mean().item()
        if acc >= 0.95 and step >= 100:
            break
    with torch.no_grad():
        p = torch.softmax(clf(mu), 1); e = p.clone(); e[torch.arange(len(y)), y] -= 1.0   # dL/dlogits = p - onehot
        # per-sample gradients: W: e_c z_h ; b: e_c
        gW2 = (e ** 2).unsqueeze(2) * (mu ** 2).unsqueeze(1)   # (n, C, H) squared per-sample gradients
        gb2 = e ** 2
    def fisher(mask):
        return {'weight': gW2[mask].mean(0).double().numpy(), 'bias': gb2[mask].mean(0).double().numpy()}
    F_F, F_R = fisher(torch.tensor(is_forget)), fisher(torch.tensor(~is_forget))
    m_r = np.concatenate([F_R['weight'].ravel(), F_R['bias'].ravel()]).mean()
    forget_class = int(y[is_forget][0]); rows = np.zeros(n_classes, dtype=bool); rows[forget_class] = True
    res = {'n_forget': int(is_forget.sum()), 'n_retain': int((~is_forget).sum()), 'n_classes': n_classes, 'train_accuracy': acc, 'steps': step + 1, 'mean_retain_fisher': float(m_r), 'groups': {}}
    tot_f = F_F['weight'].sum() + F_F['bias'].sum()
    for g, ff, fr in (('global', np.r_[F_F['weight'].ravel(), F_F['bias'].ravel()], np.r_[F_R['weight'].ravel(), F_R['bias'].ravel()]),
                      ('forget_class_row', np.r_[F_F['weight'][rows].ravel(), F_F['bias'][rows]], np.r_[F_R['weight'][rows].ravel(), F_R['bias'][rows]]),
                      ('other_rows', np.r_[F_F['weight'][~rows].ravel(), F_F['bias'][~rows]], np.r_[F_R['weight'][~rows].ravel(), F_R['bias'][~rows]])):
        res['groups'][g] = stats(ff, fr, m_r); res['groups'][g]['forget_mass_share'] = float(ff.sum() / tot_f)
    return res


def main():
    res = {'definitions': 'survival S(t) = sum F_F exp(-t F_R / mean F_R) / sum F_F; retain-null mass N(tau) = share of forget Fisher mass on coordinates with F_R < tau mean F_R; per-sample diagonal Fisher, damping 0', 'settings': {}}
    for name, cfg in SETTINGS.items():
        print(f'== {name}', flush=True)
        res['settings'][name] = vae_setting(name, cfg)
        for g, v in res['settings'][name]['groups'].items():
            print(f"  {g:16s} n={v['n_params']:>9,} share={v['forget_mass_share']:.3f} cos={v['cosine']:.3f} S(1)={v['survival']['1.0']:.3f} S(10)={v['survival']['10.0']:.3f} S(100)={v['survival']['100.0']:.3f} S(1000)={v['survival']['1000.0']:.3f} N(1e-3)={v['retain_null_mass']['0.001']:.3f} N(1e-2)={v['retain_null_mass']['0.01']:.3f} N(1e-1)={v['retain_null_mass']['0.1']:.3f}", flush=True)
        if 'decoder_output_columns' in res['settings'][name]:
            print('  decoder output columns:', {k: round(v, 3) if isinstance(v, float) else v for k, v in res['settings'][name]['decoder_output_columns'].items()}, flush=True)
    print('== linear classifier (PBMC-33k latents)', flush=True)
    res['settings']['classifier_linear'] = classifier_setting()
    for g, v in res['settings']['classifier_linear']['groups'].items():
        print(f"  {g:16s} n={v['n_params']:>9,} share={v['forget_mass_share']:.3f} cos={v['cosine']:.3f} S(1)={v['survival']['1.0']:.3f} S(10)={v['survival']['10.0']:.3f} S(100)={v['survival']['100.0']:.3f} S(1000)={v['survival']['1000.0']:.3f} N(1e-3)={v['retain_null_mass']['0.001']:.3f} N(1e-2)={v['retain_null_mass']['0.01']:.3f} N(1e-1)={v['retain_null_mass']['0.1']:.3f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1); print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
