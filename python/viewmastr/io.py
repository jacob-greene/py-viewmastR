"""The "3file" exchange format of R ``make3file()``: a Cell Ranger-style folder plus metadata.

``<dir>/3file/`` holds ``matrix.mtx.gz`` (genes x cells), ``features.tsv.gz``,
``barcodes.tsv.gz``, ``meta.csv``, one ``<name>_reduction.tsv.gz`` per embedding and
``variablefeatures.tsv.gz``. R writes it from a Seurat object; these functions read it into
AnnData and write AnnData back out for R.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def read_3file(path: str | Path) -> ad.AnnData:
    """Read a ``3file`` folder (or its parent) written by R ``make3file()``."""
    d = Path(path)
    d = d / "3file" if (d / "3file").is_dir() else d
    x = sp.csr_matrix(scipy.io.mmread(d / "matrix.mtx.gz").T)
    genes = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None)[0].astype(str)
    cells = pd.read_csv(d / "barcodes.tsv.gz", header=None)[0].astype(str)
    obs = pd.read_csv(d / "meta.csv", index_col=0) if (d / "meta.csv").exists() else pd.DataFrame(index=cells)
    adata = ad.AnnData(x, obs=obs.reindex(cells), var=pd.DataFrame(index=genes.to_numpy()))
    for f in sorted(d.glob("*_reduction.tsv.gz")):
        emb = pd.read_csv(f, sep="\t", index_col=0).reindex(cells)
        adata.obsm[f"X_{f.name[: -len('_reduction.tsv.gz')]}"] = emb.to_numpy()
    vf = d / "variablefeatures.tsv.gz"
    if vf.exists():
        with gzip.open(vf, "rt") as fh:
            adata.var["variable_feature"] = adata.var_names.isin([l.strip() for l in fh if l.strip()])
    return adata


def write_3file(adata: ad.AnnData, path: str | Path, layer: str | None = None,
                variable_features: list[str] | None = None) -> Path:
    """Write ``adata`` as ``<path>/3file`` so R can rebuild a Seurat object from it."""
    d = Path(path) / "3file"
    d.mkdir(parents=True, exist_ok=True)
    x = adata.X if layer is None else adata.layers[layer]
    with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, sp.csc_matrix(x).T)
    genes = list(adata.var_names)
    pd.DataFrame({0: genes, 1: genes, 2: "Gene Expression"}).to_csv(
        d / "features.tsv.gz", sep="\t", header=False, index=False)
    pd.Series(adata.obs_names).to_csv(d / "barcodes.tsv.gz", header=False, index=False)
    adata.obs.to_csv(d / "meta.csv")
    for key, emb in adata.obsm.items():
        name = key[2:] if key.startswith("X_") else key
        pd.DataFrame(np.asarray(emb), index=adata.obs_names).to_csv(d / f"{name}_reduction.tsv.gz", sep="\t")
    pd.Series(variable_features or [], dtype=str).to_csv(d / "variablefeatures.tsv.gz", header=False, index=False)
    return d
