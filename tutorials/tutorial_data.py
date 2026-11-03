"""Data loaders shared by the tutorials.

Public data download on first use into ``cache_dir``. Local data (the Greenleaf reference
as h5ad, a large query) come from ``tutorials/data_paths.json``, which is not in git. Copy
``data_paths.example.json`` to ``data_paths.json`` and fill in your paths.

==================  ==========================================================  ==========
Dataset             Source                                                      Access
==================  ==========================================================  ==========
Greenleaf BM atlas  Granja et al. 2019 (GEO GSE139369), as an h5ad of counts    local path
BM multiome RNA     Persad et al. 2023 (SEACells), Zenodo 6383269,              download
                    T-cell-depleted bone marrow, both replicates: every cell
                    Cell Ranger called, plus the authors' labels where present
CD34 multiome RNA   Persad et al. 2023 (SEACells), Zenodo 6383269,              download
                    CD34+ enriched bone marrow
Corces bulk RNA     Corces et al. 2016 (GEO GSE74246), as shipped with the      download
                    viewmastR R package
Large query         any large h5ad of raw counts (cells x genes)                local path
==================  ==========================================================  ==========
"""

from __future__ import annotations

import json
import urllib.request
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/api/records/6383269/files/{}/content"
VIEWMASTR_RAW = "https://raw.githubusercontent.com/furlan-lab/viewmastR/main/inst/extdata/deconvolute/{}"


def config() -> dict:
    path = HERE / "data_paths.json"
    if not path.exists():
        raise FileNotFoundError(f"copy {HERE / 'data_paths.example.json'} to {path} and set your paths")
    cfg = json.loads(path.read_text())
    cfg.setdefault("cache_dir", str(HERE / "data"))
    return cfg


def _download(url: str, name: str) -> Path:
    cache = Path(config()["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / name
    if not out.exists():
        print(f"downloading {name} ...")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
    return out


def greenleaf_reference() -> ad.AnnData:
    """Healthy bone marrow atlas (Granja et al. 2019): counts in ``.X``, labels in ``.obs``."""
    return ad.read_h5ad(config()["greenleaf_reference"])


def multiome_rna(which: str) -> ad.AnnData:
    """SEACells multiome RNA, ``which`` = ``"bm"`` (T-cell depleted) or ``"cd34"`` (CD34+).

    Returns raw counts in ``.X`` (from ``.raw``), the authors' ``celltype`` labels and UMAP.
    """
    a = ad.read_h5ad(_download(ZENODO.format(f"{which}_multiome_rna.h5ad"), f"{which}_multiome_rna.h5ad"))
    out = ad.AnnData(a.raw.X.copy(), obs=a.obs[["celltype"]].copy(), var=pd.DataFrame(index=a.var_names),
                     obsm={"X_umap": a.obsm["X_umap"]}, uns={"celltype_colors": a.uns["celltype_colors"]})
    return out


def bm_tcelldep_full() -> ad.AnnData:
    """All T-cell-depleted bone marrow cells (Persad et al. 2023), both replicates.

    Reads the Cell Ranger filtered matrices, so it keeps the cells the authors removed in
    their quality control. Returns raw RNA counts in ``.X``, ``sample`` in ``.obs``, and the
    authors' ``celltype``; cells without an author label get ``"unlabelled"``.
    """
    parts = []
    for rep in (1, 2):
        name = f"BM_Tcelldep_Rep{rep}_filtered_feature_bc_matrix.h5"
        with warnings.catch_warnings():  # duplicate gene names, made unique below
            warnings.simplefilter("ignore", UserWarning)
            a = sc.read_10x_h5(_download(ZENODO.format(name), name))
        a.var_names_make_unique()
        sample = f"IM-1393_BoneMarrow_TcellDep_{rep}_multiome"
        a.obs_names = [f"{sample}#{b}" for b in a.obs_names]
        a.obs["sample"] = f"rep{rep}"
        parts.append(a)
    out = ad.concat(parts, join="inner")
    out.var = pd.DataFrame(index=out.var_names)
    authors = multiome_rna("bm").obs["celltype"].astype(str)
    out.obs["celltype"] = pd.Categorical(authors.reindex(out.obs_names).fillna("unlabelled").to_numpy())
    return out


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
