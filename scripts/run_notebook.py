#!/usr/bin/env python3
"""Execute one notebook in place with the stat4243 kernel (cwd = notebooks/).

Usage: python scripts/run_notebook.py 33_tcga_brca_unlearning [more names]
Exit code 1 if any cell errored. Never point this at a training notebook of the
canonical pipeline without a backup: it overwrites checkpoints (round-5 review, B2).
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent.parent / 'notebooks'
failed = False
for name in sys.argv[1:]:
    path = NB / f'{name}.ipynb'
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=7200, kernel_name='stat4243', resources={'metadata': {'path': str(NB)}})
    try:
        client.execute()
    except Exception as e:  # keep the partially executed notebook for inspection
        print(f"{name}: ERROR {type(e).__name__}", flush=True)
        failed = True
    finally:
        nbformat.write(nb, path)
    n_err = sum(1 for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error')
    print(f"{name}: executed in {time.time() - t0:.0f}s, error outputs={n_err}", flush=True)
    failed = failed or n_err > 0
sys.exit(1 if failed else 0)
