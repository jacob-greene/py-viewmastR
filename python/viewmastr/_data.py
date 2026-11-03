"""Count access and the viewmastR normalisation.

R viewmastR normalises each object on its own (``get_norm_counts`` with ``norm_method =
"log"``):

1. Keep the genes the reference and the query share (``common_features``), in reference order.
2. Size factor of a cell = its total ROUNDED count over those genes, divided by the
   geometric mean of that total over all cells of the same object (monocle3
   ``estimate_size_factors``, method "mean-geometric-mean-total").
3. Value = ``log10(count / size_factor + 1)`` on the selected features.

The query is read in row chunks, so a backed (on-disk) AnnData of millions of cells works.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

NORM_METHODS = ("log", "binary", "size_only", "none")


def counts(adata: ad.AnnData, layer: str | None = None, rows: slice | None = None) -> sp.csr_matrix:
    """Raw counts (cells x genes, CSR float64) from ``.X`` or ``.layers[layer]``."""
    m = adata.X if layer is None else adata.layers[layer]
    if rows is not None:
        m = m[rows]
    return sp.csr_matrix(m, dtype=np.float64)


def chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def common_features(ref_genes: Sequence[str], query_genes: Sequence[str]) -> list[str]:
    """Genes found in both objects, in reference order (R ``common_features``)."""
    q = set(query_genes)
    return [g for g in dict.fromkeys(ref_genes) if g in q]


def library_sizes(adata: ad.AnnData, genes: Sequence[str], layer: str | None = None, chunk_size: int = 50_000) -> np.ndarray:
    """Total rounded count of each cell over ``genes``."""
    cols = pd.Index(adata.var_names).get_indexer(genes)
    if (cols < 0).any():
        raise KeyError("some genes are missing from the object")
    tot = []
    for rows in chunks(adata.n_obs, chunk_size):
        x = counts(adata, layer, rows)[:, cols]
        x.data = np.round(x.data)
        tot.append(np.asarray(x.sum(1)).ravel())
    return np.concatenate(tot)


def size_factors(library: np.ndarray) -> np.ndarray:
    """Library size over its geometric mean ("mean-geometric-mean-total")."""
    if (library <= 0).any():
        raise ValueError(
            f"{int((library <= 0).sum())} cells have no counts over the shared genes; remove them first"
        )
    return library / np.exp(np.mean(np.log(library)))


def normalise(x: sp.csr_matrix, sf: np.ndarray, method: str = "log") -> sp.csr_matrix:
    """Apply a viewmastR ``norm_method`` to raw counts ``x`` (cells x features)."""
    if method == "none":
        return x
    if method == "binary":
        return (x > 0).astype(np.float64)
    y = sp.csr_matrix(sp.diags(1.0 / sf) @ x)
    if method == "log":
        y.data = np.log10(y.data + 1.0)
    return y


def labels_of(adata: ad.AnnData, column: str) -> tuple[np.ndarray, list[str]]:
    """Integer codes and class names of an obs column, in the order R's ``as.factor`` gives.

    Classes are the labels present, sorted case-insensitively (R's ``sort()`` in an English
    locale), so class indices and output columns match an R model on the same labels.
    """
    col = adata.obs[column].astype(str)
    levels = sorted(set(col), key=lambda s: (s.lower(), s))
    return pd.Index(levels).get_indexer(col).astype(np.int32), levels
