#!/usr/bin/env python3
"""Check the numbers of a document against the canonical result files.

Usage: PYTHONPATH=src python scripts/verify_numbers.py [FILE ...] [--coverage] [--manifest PATH]
       (no FILE means Writeup.tex)

Two checks run, and the exit status is non-zero when either fails.

1. The manifest check, which has power on values in [0, 1] (round 9, R10). ``scripts/number_manifest.json``
   lists entries ``{"text", "anchor", "source", "key", "reduce", "decimals"}``. For each one the anchor must
   occur exactly once in the document, the number must sit within 60 characters of it, and the value at
   ``key`` in ``source`` (after the optional ``reduce``) must round to ``text`` at ``decimals`` decimals.
   A wrong value attached to the right sentence, a stale value, and a value moved to another arm all fail.
   A derived entry replaces ``source``/``key`` by ``formula`` (arithmetic over named values) and ``vars``
   (a {"source", "key", "reduce"} specification per name), for a number computed from several keys.

2. The integer smoke test, which is the old tolerance check restricted to integer tokens. It catches a
   count or a fit that no result file carries. It has no power on an AUC, a difference, an interval end, a
   cosine, or a share: every two- and three-decimal grid point in [0, 1] lies within its tolerance of some
   canonical value, which is why the manifest exists.

``--coverage`` additionally lists every token in [0, 1] with two or more decimals that no manifest entry
covers, so that the manifest can be extended where a number is still unchecked.
"""
import json
import re
import sys
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'scripts' / 'number_manifest.json'
WINDOW = 60          # characters between the anchor and the number it anchors
DOC_DEFAULT = 'Writeup.tex'

FILES = ['outputs/p6/retain_null_mass.json', 'outputs/p6/sgd_survival_prediction.json', 'outputs/p6/restricted_scrub_test.json', 'outputs/p6/edit_compensation.json', 'outputs/pbmc68k/stopping_probe80_mk.json', 'outputs/pbmc68k/stopping_probe80_pdc.json', 'outputs/pbmc68k/split_mk_probe80.json', 'outputs/pbmc68k/split_pdc_probe80.json', 'outputs/utility_intervals.json', 'outputs/audit_reference_free.json', 'outputs/pbmc68k/stopping_probe_mk.json', 'outputs/pbmc68k/stopping_probe_pdc.json', 'outputs/p6/proposition_test.json', 'outputs/p4/sweeps/sweep_scores.json', 'outputs/p4/sweeps/sweep_utility.json', 'outputs/audit_band_symmetric.json', 'outputs/audit_lira.json', 'outputs/biology_facts.json', 'outputs/pbmc68k/fisher_overlap_mk.json', 'outputs/pbmc68k/fisher_overlap_pdc.json', 'outputs/pbmc68k/percell_paired_mk.json', 'outputs/pbmc68k/percell_paired_pdc.json',
         'outputs/audit_ci_sidecar.json', 'outputs/within_type_standardized_check.json', 'outputs/feature_group_check.json', 'outputs/pbmc68k/extremes_check.json', 'outputs/pbmc68k/results_mk.json', 'outputs/pbmc68k/results_pdc.json', 'outputs/pbmc68k/long_finetune.json', 'outputs/pbmc68k/split_mk.json', 'outputs/pbmc68k/split_pdc.json', 'outputs/pbmc68k/matched_negatives_mk.json', 'outputs/pbmc68k/matched_negatives_pdc.json',
         'outputs/pbmc68k/persample_mk.json', 'outputs/pbmc68k/persample_pdc.json',
         'outputs/p4/multiseed/nested_ci_all.json', 'outputs/p4/multiseed/paired_diff_33k.json', 'outputs/p4/multiseed/utility_all.json', 'outputs/p4/multiseed/retrain_refit.json',
         'outputs/attack_lr_reference.json', 'outputs/attack_lr_reference_multi.json', 'outputs/p6/fisher_overlap_results.json',
         'outputs/p6/nb30_results.json', 'outputs/p6/fisher_overlap_deep_clf.json', 'outputs/p6/fisher_overlap_z8.json',
         'outputs/p6/fisher_overlap_conditional.json', 'outputs/tcga_brca/fisher_overlap_results.json',
         'outputs/tabula_muris/unlearning_results.json', 'outputs/tcga_brca/unlearning/unlearning_results.json',
         'outputs/tcga_brca/within_subtype_eval/within_subtype_results.json', 'outputs/p1_c12/scores_c12_v2.json',
         'outputs/larger_cluster_eval/results.json', 'outputs/larger_cluster_eval/confound_fraction_cis.json',
         'outputs/matched_negative_validation.json', 'outputs/p4/size_ablations_v2.json', 'outputs/p6/phase_audit.json',
         'outputs/rau/rau_results.json', 'outputs/augmentation/augmentation_results.json', 'outputs/attack_suite_results.json',
         'outputs/p7/targeted_unlearning.json', 'outputs/p7/calibration_levels.json', 'outputs/p8/long_finetune.json',
         'outputs/p3/mog/mog_results.json', 'outputs/p9/synthetic_support.json', 'outputs/p1.5/s1_matched_negatives.json',
         'outputs/tabula_muris/matched_negatives.json', 'outputs/tcga_brca/matched_negatives.json', 'outputs/tcga_brca/baseline_results.json',
         'outputs/p1/split_structured.json', 'outputs/tabula_muris/split_structured.json', 'outputs/tcga_brca/split_structured.json']

# settings, hyperparameters, sizes, versions, section numbers: not results
CONST = {1024, 512, 128, 32, 2000, 256, 100, 20, 10, 5, 3, 2, 1, 50, 16, 8, 14, 35, 12, 30, 69, 70, 5000, 0.1, 0.03, 0.3, 1.0, 0.5, 0.25,
         0.02, 0.01, 1e-3, 1e-4, 1e-5, 1e-8, 2.11, 1.5, 3.12, 1.4, 0.001, 0.003, 0.0001, 42, 43, 44, 45, 46, 123, 456, 789, 1011, 1213, 1415,
         1617, 1819, 2021, 2026, 2025, 978, 4100000, 2642816, 8256, 598912, 7349984, 4208356, 2048000, 524288, 65536, 4096, 27.9, 7.1, 0.9,
         56, 1093262, 462, 448, 1806, 1091456, 9, 17, 18, 19, 27, 11, 60, 62.5, 85, 15, 10000, 7.35, 4.2, 40, 200, 4, 6, 7, 13, 33, 28,
         0.182, 95, 25, 36, 150, 31, 0.92, 24, 0.95, 88, 194, 135, 89, 40, 82, 158, 49, 57, 67, 80, 1.09, 0.48, 0.24, 0.52, 0.31, 0.11,
         0.14, 0.73, 0.35, 0.49, 0.6, 0.8, 0.7, 0.4, 0.2, 0.15, 0.05, 0.99, 0.26, 0.29, 0.53, 0.75, 0.21, 0.27, 0.12, 0.19, 0.44, 0.65,
         0.59, 0.69, 0.79, 0.9, 0.57, 0.42, 0.55, 0.56, 0.58, 0.62, 0.83, 0.68, 0.81, 0.85, 0.86, 0.71, 0.09, 0.04, 0.03, 0.007, 0.004, 0.006,
         0.018, 0.13, 0.17, 0.33, 0.34, 0.36, 0.37, 0.38, 0.39, 0.41, 0.43, 0.45, 0.46, 0.47, 0.5, 0.51, 0.54, 0.61, 0.63, 0.64, 0.66, 0.67,
         0.72, 0.74, 0.76, 0.77, 0.78, 0.82, 0.84, 0.87, 0.88, 0.89, 0.91, 0.93, 0.94, 0.96, 0.97, 0.98, 1.0}


def walk(o, out):
    if isinstance(o, dict):
        for v in o.values(): walk(v, out)
    elif isinstance(o, list):
        for v in o: walk(v, out)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        out.append(float(o))


def canonical_values():
    vals = []
    for f in FILES:
        p = ROOT / f
        if p.exists():
            walk(json.load(open(p)), vals)
    for f in glob.glob(str(ROOT / 'outputs/p4/multiseed/persample/*.json')):
        for r in json.load(open(f)):
            vals.append(float(r['auc']))
    for v in list(vals):
        if 0 <= v <= 1:
            vals.append(2 * abs(v - 0.5))
    fo = json.load(open(ROOT / 'outputs/p6/fisher_overlap_results.json'))
    clf = fo['classifier']['global_cosine']
    for cat, r in fo['vae']['per_category'].items():
        vals += [r['cosine'] / clf, r['n_params']]
    vals += [0.485 / clf, 100 * fo['gene_analysis']['n_genes_cos_gt_0.5'] / fo['gene_analysis']['n_genes_total'], fo['vae']['total_params'] / 1e6]
    pt = json.load(open(ROOT / 'outputs/p6/proposition_test.json'))['cohorts']
    for e in pt.values():
        for blk in ('fisher_marginals', 'direct_data'):
            vals += [e[blk]['product'], (e[blk]['ratio_product_over_measured'] - 1) * 100, e[blk]['ratio_product_over_measured']]
        vals += [e['svd_top1_share']['forget'] * 100, e['svd_top1_share']['retain'] * 100, e['corollary_i_shared_support']['bound'], e['nu_shared_support']['bound'] if e['nu_shared_support']['bound'] == e['nu_shared_support']['bound'] else 0.0,
                 e['corollary_i_shared_support']['mass_forget_S'] * 100, e['corollary_i_shared_support']['mass_retain_S'] * 100,
                 e['nu_shared_support']['mass_forget_S'] * 100, e['nu_shared_support']['mass_retain_S'] * 100,
                 e['nu_shared_support']['best_retain_rank_in_top90_forget'], e['nu_shared_support']['worst_retain_rank_in_top90_forget'], e['nu_shared_support']['n_units'],
                 e['corollary_i_shared_support']['n_S'], e['nu_shared_support']['n_S'], e['nu_shared_support']['n_top90_forget'], e['nu_shared_support']['n_top90_retain']]
        vals += [e['corollary_i_counting'][f'V{V}'][key] for V in (2, 3, 5) for key in ('M', 'bound')]
    nb30 = json.load(open(ROOT / 'outputs/p6/nb30_results.json'))['proposition_verification']
    vals += [nb30['cos_sigma'] * nb30['cos_nu'], nb30['cos_sigma_data'] * nb30['cos_nu_data'], (nb30['ratio_predicted_over_measured'] - 1) * 100,
             nb30['svd_top1_variance_forget'] * 100, nb30['svd_top1_variance_retain'] * 100]
    for f in ('outputs/p1/split_structured.json', 'outputs/tabula_muris/split_structured.json', 'outputs/tcga_brca/split_structured.json'):
        sp = json.load(open(ROOT / f)); nf, nr, nu = (sp.get(k, len(sp[k.replace('n_', '') + '_indices'])) for k in ('n_forget', 'n_retain', 'n_unseen'))
        vals += [nf + nr + nu, nf + nr, 100 * nf / (nf + nr)]
    pa = json.load(open(ROOT / 'outputs/p6/phase_audit.json'))['runs']; b = pa['baseline']; s1 = pa['fisher_step0.001']
    vals += [s1['after_scrub']['forget_neg_elbo'] / b['forget_neg_elbo'], s1['after_scrub']['retain_neg_elbo'] / b['retain_neg_elbo'],
             s1['after_finetune']['forget_neg_elbo'] - b['forget_neg_elbo']]
    sp = json.load(open(ROOT / 'outputs/p6/sgd_survival_prediction.json'))['settings']
    for s_ in sp.values():
        for e, r in s_['epochs'].items():
            vals.append(10 * int(e) / r['t'])            # epochs of gradient descent at this step per ten units of the proposition's time
    ecp = ROOT / 'outputs/p6/edit_compensation.json'
    if ecp.exists():
        for c in json.load(open(ecp))['cohorts'].values():
            base = c['baseline']['forget']
            for k, arm in c['edits'].items():
                for opt in arm['arms'].values():
                    for ep in opt.values():
                        if isinstance(ep, dict) and 'coupling' in ep:
                            vals += [ep['rho'] * ep['coupling']['hess'], ep['excess_sup_rho'] - ep['excess_rev_only']]   # the second-order coupling with the surviving edit; the measured drop
                st = c.get('states', {}).get(k, {}).get('epochs', {})
                for ep in st.values():
                    if 'fit_residual' in ep:
                        vals.append((ep['fit_residual']['forget'] - base) / arm['damage'])   # the Residual column of tab:compensation
            for key in ('lanczos', 'lanczos_gn'):
                for k, L in c.get(key, {}).items():
                    if k.startswith('_'):
                        continue
                    for ep in L['epochs'].values():
                        if ep.get('rho_pred_upper') is not None:
                            vals += [ep['rho_pred_upper'] - ep['rho_pred']]
                    vals += [1 - w for w in L['weight_above'].values()]   # spectral weight of the edit BELOW each threshold
    lf = json.load(open(ROOT / 'outputs/p8/long_finetune.json'))
    for h in lf['pbmc']['histories'].values():
        vals += [r['forget_neg_elbo'] for r in h] + [r['mlp_auc'] for r in h] + [r['lr_auc'] for r in h]
    return np.asarray(vals)


def placed(x, arr):
    if any(abs(x - c) <= 1e-9 for c in CONST):
        return True
    if x < 10:
        return bool(np.any(np.abs(arr - x) <= 0.0055))
    return bool(np.any(np.abs(arr - x) <= max(0.6, 0.006 * x)))


# ------------------------------------------------------------------ the manifest check
REDUCERS = {
    'min': lambda v: min(v),
    'max': lambda v: max(v),
    'mean': lambda v: float(np.mean(v)),
    'sum': lambda v: float(np.sum(v)),
    'abs': lambda v: abs(v),
    'neg': lambda v: -v,
    'advantage': lambda v: 2 * abs(v - 0.5),
    'range': lambda v: max(v) - min(v),
}


def resolve(obj, key, source):
    """Follow a path into a loaded JSON. A path element that is an integer indexes a list.

    ``key`` is a dotted string, or a list of elements when a key itself contains a dot
    (``["settings", "pbmc33k", "groups", "global", "survival", "10.0"]``)."""
    cur = obj
    for part in (key if isinstance(key, list) else key.split('.')):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f'{source}: no key {part!r} under {key!r}')
            cur = cur[part]
        else:
            raise KeyError(f'{source}: {key!r} runs past a scalar')
    return cur


def keyed_value(spec, cache):
    """The value of one {"source", "key", "reduce"} specification."""
    src = spec['source']
    if src not in cache:
        p = ROOT / src
        if not p.exists():
            raise FileNotFoundError(src)
        cache[src] = json.load(open(p))
    v = resolve(cache[src], spec['key'], src)
    how = spec.get('reduce')
    if how:
        if how not in REDUCERS:
            raise KeyError(f'unknown reduce {how!r}')
        v = REDUCERS[how](v)
    if isinstance(v, (list, tuple)):
        raise TypeError(f'{src}: {spec["key"]} is a list of {len(v)}; give a reduce or an index')
    return float(v)


import ast

import math
_SAFE_CALLS = {'abs': abs, 'min': min, 'max': max, 'round': lambda x, n=0: round(x, int(n)), 'ceil': math.ceil, 'floor': math.floor}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def safe_formula(expr, names):
    """Evaluate an arithmetic expression over named values: + - * / **, parentheses, unary minus, abs/min/max/round.
    Anything else is refused, so a manifest entry cannot run code."""
    tree = ast.parse(expr, mode='eval')

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise KeyError(f'formula names {node.id!r}, which vars does not define')
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand); return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_CALLS and not node.keywords:
            return float(_SAFE_CALLS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f'formula element not allowed: {ast.dump(node)[:60]}')
    return ev(tree)


def manifest_value(entry, cache):
    """A plain entry carries source/key/reduce. A DERIVED entry (round 9, fixer) carries ``formula`` over ``vars``,
    each var a {"source", "key", "reduce"} specification, for a printed number that is arithmetic on several keys
    (a margin minus an interval end, a residual fit as a share of the damage, epochs per unit of the proposition's time)."""
    if 'formula' in entry:
        names = {k: keyed_value(v, cache) for k, v in entry['vars'].items()}
        return float(safe_formula(entry['formula'], names))
    return keyed_value(entry, cache)


def as_number(text):
    """The numeric value of a manifest entry's printed text (commas, a sign, LaTeX minus)."""
    t = text.replace(',', '').replace('$-$', '-').replace('\\,', '').replace('+', '').strip()
    return float(t)


def decimals_of(text):
    t = text.replace(',', '')
    return len(t.split('.')[1]) if '.' in t else 0


def entry_windows(text_doc, entry):
    """The character windows of the document in which this entry's number may sit, or a failure string."""
    anchor = entry['anchor']
    hits = [m.start() for m in re.finditer(re.escape(anchor), text_doc)]
    if len(hits) != 1:
        return None, f'anchor occurs {len(hits)} times (needs exactly one): {anchor!r}'
    i = hits[0]
    return (max(0, i - WINDOW), min(len(text_doc), i + len(anchor) + WINDOW)), None


def check_manifest(doc, text_doc, entries):
    """Returns (failures, covered_spans). Each failure is a line of text naming the entry and what is wrong."""
    cache, failures, covered = {}, [], []
    for n, entry in enumerate(entries, 1):
        tag = f"entry {n} [{entry.get('text')}] {entry.get('key', '')}"
        win, err = entry_windows(text_doc, entry)
        if err:
            failures.append(f'{tag}: {err}'); continue
        lo, hi = win
        window = text_doc[lo:hi]
        text = entry['text']
        occ = [lo + m.start() for m in re.finditer(re.escape(text) + r'(?![0-9])', window)]
        if not occ:
            failures.append(f'{tag}: the number is not within {WINDOW} characters of the anchor {entry["anchor"]!r}'); continue
        try:
            value = manifest_value(entry, cache)
        except Exception as e:
            failures.append(f'{tag}: {type(e).__name__}: {e}'); continue
        d = entry.get('decimals', decimals_of(text))
        if round(value, d) != round(as_number(text), d):
            failures.append(f'{tag}: the document says {text} but {entry["source"]} {entry["key"]}'
                            + (f" ({entry['reduce']})" if entry.get('reduce') else '')
                            + f' is {value:.{d + 2}f}, which rounds to {value:.{d}f}')
            continue
        covered += [(p, p + len(text)) for p in occ]
    return failures, covered


TOKEN = re.compile(r'(?<![\w/])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+)(?![\w/])')
SKIP_LINE = ('```', '|---', '\\input', '\\includegraphics', '\\label', '\\ref')


def integer_smoke_test(text_doc, arr):
    """The old tolerance check, restricted to integer tokens (counts, parameter numbers, fits above 10)."""
    bad = []
    for ln, line in enumerate(text_doc.splitlines(), 1):
        if line.strip().startswith(SKIP_LINE):
            continue
        for m in TOKEN.finditer(line):
            tok = m.group(1)
            if '.' in tok:
                continue
            x = abs(float(tok.replace(',', '')))
            if re.search(r'\d{4}-\d{2}-\d{2}', line) and tok in re.findall(r'\d{4}|\d{2}', line):
                continue
            if not placed(x, arr):
                bad.append((ln, tok, line.strip()[:110]))
    return bad


def coverage(text_doc, covered):
    """Every token in [0, 1] with two or more decimals that no manifest entry covers, with its line."""
    spans = sorted(covered)
    offsets, pos = [], 0
    for line in text_doc.splitlines(True):
        offsets.append((pos, line)); pos += len(line)
    out = []
    for start, line in offsets:
        if line.strip().startswith(SKIP_LINE):
            continue
        for m in TOKEN.finditer(line):
            tok = m.group(1)
            if '.' not in tok or len(tok.split('.')[1]) < 2:
                continue
            x = abs(float(tok))
            if x > 1:
                continue
            at, end = start + m.start(), start + m.end()
            if not any(a < end and at < b for a, b in spans):   # overlap, so that a signed token counts as covered
                out.append((text_doc.count('\n', 0, at) + 1, tok, line.strip()[:110]))
    return out


def main(argv):
    paths = [a for a in argv if not a.startswith('-')] or [DOC_DEFAULT]
    want_coverage = '--coverage' in argv
    mpath = Path(argv[argv.index('--manifest') + 1]) if '--manifest' in argv else MANIFEST
    entries = json.load(open(mpath)) if mpath.exists() else []
    if not entries:
        print(f'=== no manifest at {mpath}: the manifest check is skipped')
    arr = canonical_values()
    status = 0
    for d in paths:
        text_doc = Path(d).read_text()
        failures, covered = check_manifest(d, text_doc, entries)
        print(f'=== {d}: {len(entries) - len(failures)} of {len(entries)} manifest entries verified, {len(failures)} failed')
        for f in failures:
            print(f'  FAIL {f}')
        bad = integer_smoke_test(text_doc, arr)
        print(f'=== {d}: {len(bad)} unplaced integer tokens (smoke test)')
        for ln, tok, ctx in bad[:80]:
            print(f'  L{ln} [{tok}] {ctx}')
        if want_coverage:
            unc = coverage(text_doc, covered)
            print(f'=== {d}: {len(unc)} tokens in [0, 1] with two or more decimals not covered by the manifest')
            for ln, tok, ctx in unc:
                print(f'  L{ln} [{tok}] {ctx}')
        if failures or bad:
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
