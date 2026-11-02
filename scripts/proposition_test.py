#!/usr/bin/env python3
"""Proposition 1 (decoder-output Fisher factorization) and Corollary 1 (dimensional scaling) tested on
PBMC-33k cluster 13 and on both PBMC-68k forget sets with one procedure.

For each cohort the decoder output layer's per-sample diagonal Fisher (decoder.fc_mean.0.weight, [genes x hidden])
is taken from the retain cache written by train_fisher_unlearn.py (damping 0, eval mode, z = mu) and computed
fresh for the forget set with the same estimator. Reported per cohort:
  measured cosine of the two Fisher matrices; rank-one share (top singular value) of each; the factorization's two
  factors from the Fisher marginals (row sums = residual-variance profile sigma over genes, column sums = hidden
  second-moment profile nu) and from a direct forward pass; their product against the measured cosine; and the
  corollary's quantities, the concentration of sigma^F (participation ratio, genes carrying 90 percent of its
  squared norm), the number M of genes whose forget/retain residual-variance ratio exceeds V after mean
  normalization, and the corollary's lower bound (D - M) / (D - M + M V^2) for V = 2, 3, 5.
Writes outputs/p6/proposition_test.json. Run: PYTHONPATH=src python scripts/proposition_test.py
"""
import json
import sys
import time
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
from fisher_utils import AnnDataDataset, compute_vae_fisher, cosine_sim

KEY = 'decoder.fc_mean.0.weight'
COHORTS = {
    'pbmc33k_c13': dict(data='data/adata_processed.h5ad', model='outputs/p1/baseline/best_model.pt',
                        split='outputs/p1/split_structured.json', cache='outputs/p2/fisher_unlearn/fisher_retain_persample.pt'),
    'pbmc68k_mk': dict(data='data/pbmc68k_processed.h5ad', model='outputs/pbmc68k/baseline/best_model.pt',
                       split='outputs/pbmc68k/split_mk.json', cache='outputs/pbmc68k/mk/fisher_retain_persample.pt'),
    'pbmc68k_pdc': dict(data='data/pbmc68k_processed.h5ad', model='outputs/pbmc68k/baseline/best_model.pt',
                        split='outputs/pbmc68k/split_pdc.json', cache='outputs/pbmc68k/pdc/fisher_retain_persample.pt'),
}


def indices(split, kind):
    for k in (f'{kind}_indices', kind, f'{kind}_idx'):
        if k in split:
            return list(split[k])
    raise KeyError(kind)


@torch.no_grad()
def residuals_and_hidden(model, loader):
    """E[e_d^2] over genes and E[h_j^2] over the decoder's last hidden layer, eval mode, z = mu (the Fisher estimator's setting)."""
    e2, h2, n = 0.0, 0.0, 0
    for x, lib in loader:
        mu, _ = model.encode(x)
        h = model.decoder.network(mu)
        mean, _ = model.decode(mu, lib)
        e2 = e2 + ((x - mean) ** 2).sum(0); h2 = h2 + (h ** 2).sum(0); n += len(x)
    return e2 / n, h2 / n


def concentration(v):
    v = v.clamp(min=0); s2 = (v ** 2)
    pr = float(v.sum() ** 2 / s2.sum())
    cum = torch.cumsum(torch.sort(s2, descending=True).values, 0) / s2.sum()
    return {'participation_ratio': pr, 'n_genes_90pct': int((cum < 0.9).sum().item() + 1)}


def shared_support_bound(sf, sr):
    """Corollary part (i) in its general form. For any gene set S, with m^S the share of a profile's squared norm on S,
    cos(sigma^F, sigma^R) >= cos(sigma^F_S, sigma^R_S) * sqrt(m^F_S m^R_S), because all entries are nonnegative.
    S is taken as the genes inside both profiles' top-90-percent sets; the counting bound of the paper is the special
    case in which the shared genes have equal residual variance and the M others are bounded by V times it."""
    def top_set(v):
        s2 = v.clamp(min=0) ** 2; order = torch.argsort(s2, descending=True); cum = torch.cumsum(s2[order], 0) / s2.sum()
        k = int((cum < 0.9).sum().item() + 1); return set(order[:k].tolist())
    S = sorted(top_set(sf) & top_set(sr)); idx = torch.tensor(S, dtype=torch.long)
    mf = float((sf[idx] ** 2).sum() / (sf ** 2).sum()); mr = float((sr[idx] ** 2).sum() / (sr ** 2).sum())
    cS = float(cosine_sim(sf[idx], sr[idx]))
    tf = top_set(sf)
    rank = {u: i for i, u in enumerate(torch.argsort(sr, descending=True).tolist())}   # 0 = the retain set's most used unit
    top_f = sorted(tf, key=lambda u: -float(sf[u]))
    return {'n_S': len(S), 'n_top90_forget': len(tf), 'n_top90_retain': len(top_set(sr)), 'cos_S': cS,
            'mass_forget_S': mf, 'mass_retain_S': mr, 'bound': cS * (mf * mr) ** 0.5, 'n_units': len(sf),
            'retain_rank_of_top_forget_units': [rank[u] for u in top_f[:10]],
            'worst_retain_rank_in_top90_forget': max(rank[u] for u in tf), 'best_retain_rank_in_top90_forget': min(rank[u] for u in tf)}


def corollary_bound(sf, sr, D):
    sf = sf / sf.mean(); sr = sr / sr.mean()
    ratio = torch.maximum(sf / sr.clamp(min=1e-12), sr / sf.clamp(min=1e-12))
    out = {}
    for V in (2, 3, 5):
        M = int((ratio > V).sum().item())
        out[f'V{V}'] = {'M': M, 'bound': (D - M) / (D - M + M * V * V)}
    return out


def main():
    model_cache = {}
    res = {'key': KEY, 'estimator': 'per-sample diagonal empirical Fisher of the negative ELBO, eval mode, z = mu, damping 0 (fisher_utils.compute_vae_fisher); retain from the train_fisher_unlearn cache, forget computed fresh', 'cohorts': {}}
    for name, c in COHORTS.items():
        t0 = time.time()
        adata = sc.read_h5ad(BASE / c['data'])
        if c['model'] not in model_cache:
            model_cache[c['model']] = load_vae_model(str(BASE / c['model']))[0].eval()
        model = model_cache[c['model']]
        split = json.load(open(BASE / c['split'])); f_idx, r_idx = indices(split, 'forget'), indices(split, 'retain')
        cache = torch.load(BASE / c['cache'], map_location='cpu', weights_only=False)
        Fr = cache['fisher'][KEY].float(); n_r_cache = int(cache['n_samples'])
        f_loader = DataLoader(AnnDataDataset(adata, f_idx), batch_size=128, shuffle=False)
        Ff_all, n_f = compute_vae_fisher(model, f_loader, 'cpu', damping=0.0); Ff = Ff_all[KEY].float().reshape(Fr.shape)
        D, H = Fr.shape
        measured = float(cosine_sim(Ff.flatten(), Fr.flatten()))
        Sf = torch.linalg.svdvals(Ff); Sr = torch.linalg.svdvals(Fr)
        top1 = {'forget': float(Sf[0] ** 2 / (Sf ** 2).sum()), 'retain': float(Sr[0] ** 2 / (Sr ** 2).sum())}
        sig_f, sig_r = Ff.sum(1), Fr.sum(1); nu_f, nu_r = Ff.sum(0), Fr.sum(0)
        cs, cn = float(cosine_sim(sig_f, sig_r)), float(cosine_sim(nu_f, nu_r))
        r_loader = DataLoader(AnnDataDataset(adata, r_idx), batch_size=512, shuffle=False)
        sig_f_d, nu_f_d = residuals_and_hidden(model, f_loader); sig_r_d, nu_r_d = residuals_and_hidden(model, r_loader)
        csd, cnd = float(cosine_sim(sig_f_d, sig_r_d)), float(cosine_sim(nu_f_d, nu_r_d))
        entry = {
            'n_forget': int(n_f), 'n_retain_cache': n_r_cache, 'n_retain_split': len(r_idx), 'D_genes': D, 'H_hidden': H,
            'measured_cosine': measured, 'svd_top1_share': top1,
            'fisher_marginals': {'cos_sigma': cs, 'cos_nu': cn, 'product': cs * cn, 'ratio_product_over_measured': cs * cn / measured},
            'direct_data': {'cos_sigma': csd, 'cos_nu': cnd, 'product': csd * cnd, 'ratio_product_over_measured': csd * cnd / measured,
                            'marginal_vs_data_agreement': {'sigma_forget': float(cosine_sim(sig_f, sig_f_d)), 'sigma_retain': float(cosine_sim(sig_r, sig_r_d)),
                                                           'nu_forget': float(cosine_sim(nu_f, nu_f_d)), 'nu_retain': float(cosine_sim(nu_r, nu_r_d))}},
            'sigma_concentration': {'forget': concentration(sig_f), 'retain': concentration(sig_r)},
            'nu_concentration': {'forget': concentration(nu_f), 'retain': concentration(nu_r)},
            'corollary_i_counting': corollary_bound(sig_f, sig_r, D),
            'corollary_i_shared_support': shared_support_bound(sig_f, sig_r),
            'nu_shared_support': shared_support_bound(nu_f, nu_r),
            'vectors': {'sigma_forget': sig_f.tolist(), 'sigma_retain': sig_r.tolist(), 'nu_forget': nu_f.tolist(), 'nu_retain': nu_r.tolist()},
            'seconds': round(time.time() - t0, 1),
        }
        res['cohorts'][name] = entry
        print(f"[{name}] measured {measured:.3f} | marginals cos_sigma {cs:.3f} cos_nu {cn:.3f} product {cs*cn:.3f} | data {csd:.3f} x {cnd:.3f} = {csd*cnd:.3f} | "
              f"top1 {top1['forget']:.3f}/{top1['retain']:.3f} | sigma PR forget {entry['sigma_concentration']['forget']['participation_ratio']:.0f} retain {entry['sigma_concentration']['retain']['participation_ratio']:.0f} | "
              f"n90 {entry['sigma_concentration']['forget']['n_genes_90pct']}/{entry['sigma_concentration']['retain']['n_genes_90pct']} | "
              + ' '.join(f"V{V}: M={entry['corollary_i_counting'][f'V{V}']['M']} bound={entry['corollary_i_counting'][f'V{V}']['bound']:.2f}" for V in (2, 3, 5))
              + " | shared-support sigma: " + ', '.join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry['corollary_i_shared_support'].items())
              + " | nu: " + ', '.join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry['nu_shared_support'].items()) + f" ({entry['seconds']}s)", flush=True)
    out = BASE / 'outputs/p6/proposition_test.json'; json.dump(res, open(out, 'w'), indent=1); print('wrote', out)


if __name__ == '__main__':
    main()
