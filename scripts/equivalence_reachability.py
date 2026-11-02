#!/usr/bin/env python3
"""Can the verdict "at the retrained level" be returned at all, arm by arm?

The equivalence verdict of the paired attacker test needs the whole 95 percent interval of the
difference inside plus or minus the margin, so an interval wider than twice the margin (the window)
cannot return it whatever the data say. For every arm of every paired-test file this script records
the interval width, whether the width is below the window (``reachable``), and the smallest margin at
which the arm would have read at the retrained level (``min_margin`` = max(-lo, hi)). It also copies
the three reference-free audit margins so that the paper can print them by ``\\input``.

Reads (never writes): outputs/p4/multiseed/paired_diff_33k.json, outputs/pbmc68k/percell_paired_{mk,pdc}.json,
outputs/pbmc68k/stopping_probe_{mk,pdc}.json, outputs/pbmc68k/stopping_probe80_{mk,pdc}.json,
outputs/audit_reference_free.json.
Writes: outputs/equivalence_reachability.json and the snippets tables/reach_*.tex, tables/audit_margin_*.tex
(bare numbers or ranges, no trailing newline, like the margin snippets of make_tex_tables.py).

Run: PYTHONPATH=src python scripts/equivalence_reachability.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'src'))
try:
    import canonical as C
    OUTPUTS = Path(C.OUT)
except Exception:  # canonical.py absent: fall back to the repository layout
    OUTPUTS = BASE / 'outputs'

TABLES = BASE / 'tables'
TABLES.mkdir(exist_ok=True)


def J(rel):
    return json.load(open(OUTPUTS / rel))


def f3(x):
    return f'{x:.3f}'


def arm_record(diff, ci, margin, verdict, family=None, verdict_holm=None):
    lo, hi = float(ci[0]), float(ci[1])
    width = hi - lo
    rec = {
        'diff': float(diff), 'lo': lo, 'hi': hi, 'width': width,
        'window': 2.0 * margin, 'reachable': bool(width < 2.0 * margin),
        'min_margin': max(-lo, hi), 'verdict': verdict,
    }
    if family is not None:
        rec['family'] = family
    if verdict_holm is not None:
        rec['verdict_holm'] = verdict_holm
    return rec


def summarize(arms):
    """Range of widths and the reachable count over the method arms (every arm passed in)."""
    widths = [a['width'] for a in arms.values()]
    return {
        'n_arms': len(arms),
        'n_reachable': sum(a['reachable'] for a in arms.values()),
        'width_min': min(widths), 'width_max': max(widths),
        'too_wide': sorted(k for k, a in arms.items() if not a['reachable']),
    }


def pbmc33k():
    d = J('p4/multiseed/paired_diff_33k.json')
    margin = d['margin']
    method, control = {}, {}
    for name, a in d['arms'].items():
        if 'diff_ci' not in a:
            continue  # the retrained arm itself
        rec = arm_record(a['diff'], a['diff_ci'], margin, a['verdict'], a.get('family'), a.get('verdict_holm'))
        (control if a.get('family') == 'reference' else method)[name] = rec
    return {
        'source': 'outputs/p4/multiseed/paired_diff_33k.json', 'statistic': d['statistic'],
        'margin': margin, 'window': 2.0 * margin, 'arms': method, 'controls': control,
        'summary': summarize(method),
    }


def pbmc68k(fs):
    d = J(f'pbmc68k/percell_paired_{fs}.json')
    w = d['kinds']['within']
    margin = w['margin']
    method, control = {}, {}
    for name, a in w['arms'].items():
        if 'diff_ci' not in a:
            continue
        rec = arm_record(a['diff'], a['diff_ci'], margin, a['verdict'], verdict_holm=a.get('verdict_holm'))
        (control if name == 'baseline' else method)[name] = rec
    return {
        'source': f'outputs/pbmc68k/percell_paired_{fs}.json', 'statistic': d['statistic'],
        'n_forget': w['n_forget'], 'n_negatives': w['n_negatives'],
        'margin': margin, 'window': 2.0 * margin, 'arms': method, 'controls': control,
        'summary': summarize(method),
    }


def probe(fs):
    d = J(f'pbmc68k/stopping_probe_{fs}.json')
    margin = d['margin']['within']
    arms = {}
    for name, a in d['arms'].items():
        r = a['refit_std_within']
        if name == 'baseline' or 'diff_ci' not in r:
            continue
        arms[name] = arm_record(r['diff_vs_retrain'], r['diff_ci'], margin, r['verdict'])
    return {'source': f'outputs/pbmc68k/stopping_probe_{fs}.json', 'margin': margin,
            'window': 2.0 * margin, 'arms': arms, 'summary': summarize(arms)}


def probe80(fs):
    d = J(f'pbmc68k/stopping_probe80_{fs}.json')
    margin = d['margin']
    arms = {}
    for name, a in d['arms'].items():
        r = a.get('refit_within', {})
        if name in ('baseline', 'retrain') or 'diff_ci' not in r:
            continue
        arms[name] = arm_record(r['diff_vs_retrain'], r['diff_ci'], margin, r['verdict'])
    return {'source': f'outputs/pbmc68k/stopping_probe80_{fs}.json', 'margin': margin,
            'window': 2.0 * margin, 'arms': arms, 'summary': summarize(arms)}


def main():
    out = {
        'definition': ('width = hi - lo of the two-sided 95 percent interval of the paired difference; '
                       'window = 2 x margin; reachable = width < window (the equivalence verdict needs the '
                       'whole interval within plus or minus the margin); min_margin = max(-lo, hi), the '
                       'smallest margin at which the arm would read at the retrained level'),
        'cohorts': {
            'pbmc33k': pbmc33k(),
            'pbmc68k_mk': pbmc68k('mk'),
            'pbmc68k_pdc': pbmc68k('pdc'),
            'probe_mk': probe('mk'),
            'probe_pdc': probe('pdc'),
            'probe80_mk': probe80('mk'),
            'probe80_pdc': probe80('pdc'),
        },
    }
    rf = J('audit_reference_free.json')['cohorts']
    out['audit_margins'] = {k: rf[k]['margins']['audit_cross'] for k in ('pbmc33k', 'pbmc68k_mk', 'pbmc68k_pdc')}
    out['audit_margin_rule'] = rf['pbmc33k']['margin_rule']

    (OUTPUTS / 'equivalence_reachability.json').write_text(json.dumps(out, indent=1))

    co = out['cohorts']

    def rng(s):
        return f"{f3(s['width_min'])} to {f3(s['width_max'])}"

    probe_all = {**co['probe_mk']['arms'], **{'pdc_' + k: v for k, v in co['probe_pdc']['arms'].items()}}
    probe80_all = {**co['probe80_mk']['arms'], **{'pdc_' + k: v for k, v in co['probe80_pdc']['arms'].items()}}
    snippets = {
        'reach_33k_window': f3(co['pbmc33k']['window']),
        'reach_mk_window': f3(co['pbmc68k_mk']['window']),
        'reach_pdc_window': f3(co['pbmc68k_pdc']['window']),
        'reach_33k_widths': rng(co['pbmc33k']['summary']),
        'reach_mk_widths': rng(co['pbmc68k_mk']['summary']),
        'reach_pdc_widths': rng(co['pbmc68k_pdc']['summary']),
        'reach_probe_widths': rng(summarize(probe_all)),
        'reach_probe80_widths': rng(summarize(probe80_all)),
        'reach_probe80_mk_window': f3(co['probe80_mk']['window']),
        'reach_probe80_pdc_window': f3(co['probe80_pdc']['window']),
        'reach_33k_fit': f"{co['pbmc33k']['summary']['n_reachable']} of {co['pbmc33k']['summary']['n_arms']}",
        'reach_mk_longft_minmargin': f3(co['pbmc68k_mk']['arms']['long_finetune']['min_margin']),
        'reach_mk_longft_width': f3(co['pbmc68k_mk']['arms']['long_finetune']['width']),
        'reach_33k_cap2e3_minmargin': f3(co['pbmc33k']['arms']['sweep_fisher_step0.002']['min_margin']),
        'reach_33k_cap25e3_minmargin': f3(co['pbmc33k']['arms']['sweep_fisher_step0.0025']['min_margin']),
        'reach_33k_longft_minmargin': f3(co['pbmc33k']['arms']['long_finetune_200']['min_margin']),
        'audit_margin_33k': f3(out['audit_margins']['pbmc33k']),
        'audit_margin_mk': f3(out['audit_margins']['pbmc68k_mk']),
        'audit_margin_pdc': f3(out['audit_margins']['pbmc68k_pdc']),
    }
    for name, text in snippets.items():
        (TABLES / f'{name}.tex').write_text(text + '%')   # the trailing % keeps \input from adding a space before the next character

    for k, c in co.items():
        s = c['summary']
        print(f"{k}: margin {c['margin']:.4f} window {c['window']:.4f} arms {s['n_arms']} reachable "
              f"{s['n_reachable']} widths {s['width_min']:.4f} to {s['width_max']:.4f} too wide: {s['too_wide']}")
    for name, text in snippets.items():
        print(f"  {name}.tex = {text}")


if __name__ == '__main__':
    main()
