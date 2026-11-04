"""Data loaders shared by the tutorials.

Public data download on first use into ``cache_dir``. Local data (the Greenleaf reference
as h5ad, a large query) come from ``tutorials/data_paths.json``, which is not in git. Copy
``data_paths.example.json`` to ``data_paths.json`` and fill in your paths.

==================  ==========================================================  ==========
Dataset             Source                                                      Access
==================  ==========================================================  ==========
Greenleaf BM atlas  Granja et al. 2019 (GEO GSE139369), as an h5ad of counts    local path
BM multiome RNA     Persad et al. 2023 (SEACells), T-cell-depleted bone marrow,   download
                    8,627 cells, 18 cell types, with the authors' UMAP. The
                    file the Mellon tutorial downloads (Otto et al. 2024)
CD34 multiome RNA   Persad et al. 2023 (SEACells), Zenodo 6383269,              download
                    CD34+ enriched bone marrow
Corces bulk RNA     Corces et al. 2016 (GEO GSE74246), as shipped with the      download
                    viewmastR R package
Large query         any large h5ad of raw counts (cells x genes)                local path
==================  ==========================================================  ==========
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/api/records/6383269/files/{}/content"
MELLON = ("https://fh-pi-setty-m-eco-public.s3.amazonaws.com/mellon-tutorial/"
          "preprocessed_t-cell-depleted-bm-rna.h5ad")
MELLON_MD5 = "64909a0be6cd51fe79b7d70b5f19bf1c"
VIEWMASTR_RAW = "https://raw.githubusercontent.com/furlan-lab/viewmastR/main/inst/extdata/deconvolute/{}"


def config() -> dict:
    path = HERE / "data_paths.json"
    if not path.exists():
        raise FileNotFoundError(f"copy {HERE / 'data_paths.example.json'} to {path} and set your paths")
    cfg = json.loads(path.read_text())
    cfg.setdefault("cache_dir", str(HERE / "data"))
    return cfg


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, name: str, md5: str | None = None) -> Path:
    cache = Path(config()["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / name
    if not out.exists():
        print(f"downloading {name} ...")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if md5 and _md5(tmp) != md5:
            raise OSError(f"{name}: md5 mismatch, download again")
        tmp.rename(out)
    return out


def greenleaf_reference() -> ad.AnnData:
    """Healthy bone marrow atlas (Granja et al. 2019): counts in ``.X``, labels in ``.obs``."""
    return ad.read_h5ad(config()["greenleaf_reference"])


def multiome_rna(which: str) -> ad.AnnData:
    """SEACells multiome RNA, ``which`` = ``"bm"`` (T-cell depleted) or ``"cd34"`` (CD34+).

    Returns raw counts in ``.X`` (from ``.raw``), the authors' ``celltype`` labels and UMAP.
    Colours come from ``viewmastr.palette``, so the file's own colours are not kept.
    """
    a = ad.read_h5ad(_download(ZENODO.format(f"{which}_multiome_rna.h5ad"), f"{which}_multiome_rna.h5ad"))
    return ad.AnnData(a.raw.X.copy(), obs=a.obs[["celltype"]].copy(), var=pd.DataFrame(index=a.var_names),
                      obsm={"X_umap": a.obsm["X_umap"]})


def bm_tcelldep() -> ad.AnnData:
    """T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), loaded as is.

    The file is the one the Mellon tutorial downloads (github.com/settylab/Mellon,
    ``notebooks/basic_tutorial.ipynb``; 2.5 GB). Its counts are those of
    ``bm_multiome_rna.h5ad`` in Zenodo 6383269, plus 1,188 B lineage cells, finer labels and
    a UMAP. Returns raw counts in ``.X`` (from ``.raw``; its columns are ``var_names`` in
    order), ``sample`` and ``celltype`` in ``.obs``, and the stored UMAP. Nothing is
    re-normalised or re-embedded.
    """
    a = ad.read_h5ad(_download(MELLON, "preprocessed_t-cell-depleted-bm-rna.h5ad", MELLON_MD5))
    return ad.AnnData(a.raw.X.copy(), obs=a.obs[["sample", "celltype"]].copy(),
                      var=pd.DataFrame(index=a.var_names), obsm={"X_umap": a.obsm["X_umap"]})


def corces_bulk() -> pd.DataFrame:
    """Sorted-population bulk RNA-seq counts (genes x samples); columns are ``<donor>-<celltype>``."""
    path = _download(VIEWMASTR_RAW.format("GSE74246_RNAseq_All_Counts.txt"), "GSE74246_RNAseq_All_Counts.txt")
    return pd.read_csv(path, sep="\t", index_col=0)


def gene_weights() -> pd.DataFrame:
    """Gene lengths and deconvolution weights (``symbol``, ``length``, ``weight``)."""
    path = _download(VIEWMASTR_RAW.format("geneWeights_withSymbol.tsv"), "geneWeights_withSymbol.tsv")
    return pd.read_csv(path, sep="\t")


def large_query(backed: bool = True) -> ad.AnnData:
    """The large query of the big-query tutorial (raw counts), opened on disk by default."""
    return ad.read_h5ad(config()["large_query"], backed="r" if backed else None)
