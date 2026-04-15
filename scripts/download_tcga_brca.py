#!/usr/bin/env python3
"""Download TCGA-BRCA data for the unlearning project.

Sources:
  - Expression: UCSC Xena GDC hub (HTSeq counts)
  - Clinical: UCSC Xena GDC hub (GDC phenotype)
  - L1000 genes: GEO GSE92742 gene_info file

Usage:
    python scripts/download_tcga_brca.py
"""

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tcga_brca"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = {
    # TCGA-BRCA HTSeq raw counts from GDC Xena hub
    "htseq_counts": {
        "url": "https://gdc.xenahubs.net/download/TCGA-BRCA.htseq_counts.tsv.gz",
        "filename": "TCGA-BRCA.htseq_counts.tsv.gz",
        "description": "TCGA-BRCA HTSeq counts (log2(count+1) transformed)",
    },
    # Clinical / phenotype data
    "phenotype": {
        "url": "https://gdc.xenahubs.net/download/TCGA-BRCA.GDC_phenotype.tsv.gz",
        "filename": "TCGA-BRCA.GDC_phenotype.tsv.gz",
        "description": "TCGA-BRCA clinical phenotype data",
    },
    # L1000 landmark gene info from GEO
    "l1000_genes": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz",
        "filename": "GSE92742_Broad_LINCS_gene_info.txt.gz",
        "description": "LINCS L1000 gene info (contains landmark flag)",
    },
}

# Alternative Xena URLs if the above don't work
XENA_ALTERNATIVES = {
    "htseq_counts": [
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.htseq_counts.tsv.gz",
        "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.htseq_counts.tsv.gz",
    ],
    "phenotype": [
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.GDC_phenotype.tsv.gz",
        "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.GDC_phenotype.tsv.gz",
    ],
}


def download_file(url, dest_path, description=""):
    """Download a file with progress reporting."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return True

    print(f"  Downloading: {description or url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(response, f)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {dest_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_l1000_genes():
    """Extract 978 landmark gene symbols from the GEO gene_info file."""
    gz_path = DATA_DIR / "GSE92742_Broad_LINCS_gene_info.txt.gz"
    out_path = DATA_DIR / "lincs_l1000_genes.txt"

    if out_path.exists():
        with open(out_path) as f:
            n = sum(1 for line in f if line.strip())
        print(f"  L1000 gene list already exists: {n} genes")
        return

    if not gz_path.exists():
        print("  ERROR: gene_info file not downloaded yet")
        return

    print("  Extracting landmark genes from gene_info file...")
    genes = []
    with gzip.open(gz_path, "rt") as f:
        header = f.readline().strip().split("\t")
        # Find column indices
        symbol_idx = header.index("pr_gene_symbol")
        lm_idx = header.index("pr_is_lmark")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > max(symbol_idx, lm_idx) and fields[lm_idx] == "1":
                genes.append(fields[symbol_idx])

    genes = sorted(set(genes))
    with open(out_path, "w") as f:
        for g in genes:
            f.write(g + "\n")

    print(f"  Extracted {len(genes)} landmark genes -> {out_path.name}")


def main():
    print(f"Download directory: {DATA_DIR}\n")

    for key, info in DOWNLOADS.items():
        dest = DATA_DIR / info["filename"]
        print(f"[{key}]")
        success = download_file(info["url"], dest, info["description"])

        # Try alternatives if primary URL fails
        if not success and key in XENA_ALTERNATIVES:
            for alt_url in XENA_ALTERNATIVES[key]:
                print(f"  Trying alternative: {alt_url}")
                success = download_file(alt_url, dest)
                if success:
                    break

        if not success:
            print(f"  WARNING: Could not download {key}. See manual instructions below.\n")

    # Extract L1000 gene list
    print("\n[l1000_extraction]")
    extract_l1000_genes()

    # Summary
    print("\n" + "=" * 60)
    print("Download summary:")
    for key, info in DOWNLOADS.items():
        dest = DATA_DIR / info["filename"]
        status = "OK" if dest.exists() else "MISSING"
        print(f"  [{status}] {info['filename']}")
    l1000_path = DATA_DIR / "lincs_l1000_genes.txt"
    status = "OK" if l1000_path.exists() else "MISSING"
    print(f"  [{status}] lincs_l1000_genes.txt")

    missing = [
        info["filename"]
        for info in DOWNLOADS.values()
        if not (DATA_DIR / info["filename"]).exists()
    ]
    if missing:
        print("\nManual download instructions for missing files:")
        print("  1. Go to https://xenabrowser.net/datapages/")
        print("  2. Search for 'TCGA-BRCA'")
        print("  3. Select the GDC hub dataset")
        print(f"  4. Download to {DATA_DIR}/")
        print("\n  For L1000 genes, go to:")
        print("  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742")
        print("  Download GSE92742_Broad_LINCS_gene_info.txt.gz")
    else:
        print("\nAll files downloaded successfully!")
        print(f"\nNext step: Run notebooks/31_tcga_brca_data_prep.ipynb")


if __name__ == "__main__":
    main()
