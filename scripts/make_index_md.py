#!/usr/bin/env python3
"""Regenerate index.md (site landing page) and the key-results block of README.md from Writeup.tex
and the script-generated tables, so that neither mirror can drift from the paper.

Run from the project root after scripts/make_tex_tables.py: python scripts/make_index_md.py
"""
import re
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).parent.parent
TEX = (BASE / 'Writeup.tex').read_text()
TABLES = BASE / 'tables'


def latex_to_md(fragment: str) -> str:
    """Convert a LaTeX fragment (no preamble) to GitHub Markdown with pandoc."""
    frag = re.sub(r'\\resizebox\{\\textwidth\}\{!\}\{(\\input\{[^}]*\})\}', r'\1', fragment)
    frag = re.sub(r'(Table|Section|Appendix|Figure)~\\ref\{[^}]*\}', r'the paper', frag)
    frag = re.sub(r'~?\\citep?\{[^}]*\}', '', frag)
    return subprocess.run(['pandoc', '-f', 'latex', '-t', 'gfm', '--wrap=none'], input=frag, cwd=BASE, capture_output=True, text=True, check=True).stdout.strip()


def table_md(name: str) -> str:
    return latex_to_md((TABLES / f'{name}.tex').read_text())


def caption_of(label: str) -> str:
    """The \\caption{...} immediately preceding \\label{label}, with balanced braces."""
    at = TEX.find('\\label{' + label + '}')
    start = TEX.rfind('\\caption{', 0, at)
    if at < 0 or start < 0:
        return ''
    i, depth = start + len('\\caption{'), 1
    while depth:
        depth += {'{': 1, '}': -1}.get(TEX[i], 0); i += 1
    return latex_to_md(TEX[start + len('\\caption{'):i - 1])


title = re.search(r'\\title\{(.*?)\}', TEX).group(1)
abstract = latex_to_md(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', TEX, flags=re.S).group(1).strip())
contrib = latex_to_md(re.search(r'\\paragraph\{Contributions\.\}\n(\\begin\{enumerate\}.*?\\end\{enumerate\})', TEX, flags=re.S).group(1))
today = time.strftime('%Y-%m-%d')

KEY = f"""## Key results

Every number below is produced by `scripts/make_tex_tables.py` from the result files under `outputs/` (the same tables the paper prints). Advantage is 2|AUC - 0.5| with its nested 95 percent interval; the audit is the likelihood-ratio AUC against a model retrained on exactly the retain split.

### PBMC-33k cluster 13 (30 megakaryocytes against 194 matched negatives)

{caption_of('tab:main')}

{table_md('main')}

### PBMC-68k, two rare clusters with within-type negatives

{caption_of('tab:pbmc68k')}

{table_md('pbmc68k')}

### The confound across cohorts

{caption_of('tab:confound')}

{table_md('confound')}

### Long retain-only fine-tuning

{caption_of('tab:longft')}

{table_md('longft')}

{caption_of('tab:longft68k')}

{table_md('longft68k')}
"""

index = f"""---
layout: default
title: Home
---

# {title}

**Status ({today}).** Every experiment was re-derived from one baseline with corrected code and clean retrain references (September 2026), the paper was rewritten around that evidence, and a second single-cell cohort (PBMC-68k, two rare clusters with within-type negatives) was added. This page is generated from the paper source by `scripts/make_index_md.py`.

David Benson, Columbia University

[View Code](https://github.com/db-d2/Machine_Unlearning) | [Writeup (Markdown)](./Writeup.md) | [Writeup (PDF)](./Writeup.pdf)

## Abstract

{abstract}

## Contributions

{contrib}

## Main figures

### Evaluation pipeline

![Evaluation pipeline](figures/eval_pipeline.svg)

### Where every PBMC-33k checkpoint lands relative to the retrained models

![Calibration axis](figures/calibration_axis.png)

### Long retain-only fine-tuning on PBMC-33k

![Long fine-tune](figures/pbmc_long_finetune.png)

### Erosion by support in the simulation

![Erosion by support](figures/erosion_by_support.png)

### Fisher overlap across datasets

![Fisher overlap](figures/fisher_overlap_cross_dataset.png)

{KEY}
## Reproducing

See `README.md` for the environment and the order of the notebooks (01 to 44) and scripts; `HANDOFF_PEER_REVIEW.md` records the state of the work and `CLAIMS_AUDIT.md` the verdict on every claim.
"""
(BASE / 'index.md').write_text(index)
print(f"wrote index.md ({len(index.splitlines())} lines)")

readme_path = BASE / 'README.md'
readme = readme_path.read_text()
start, end = '<!-- KEY RESULTS START -->', '<!-- KEY RESULTS END -->'
if start in readme and end in readme:
    pre, rest = readme.split(start, 1)
    _, post = rest.split(end, 1)
    readme_path.write_text(pre + start + '\n' + KEY + end + post)
    print('README.md key results replaced')
else:
    print('README.md has no KEY RESULTS markers; not touched')
