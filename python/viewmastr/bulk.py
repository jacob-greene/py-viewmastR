"""Bulk RNA-seq as a reference, class balancing and deconvolution.

- :func:`splat_bulk_reference` simulates single cells from bulk profiles, so a bulk dataset
  can train ``viewmastR()``.
- :func:`augment_data` tops up rare cell types with simulated cells (same simulator).
- :func:`deconvolve_bulk` estimates cell-type exposures of bulk samples from single-cell
  signatures.

The simulator and both deconvolution solvers are the R package's Rust code (``rust/``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _core
from ._data import chunks, counts


def bw_nrd0(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb kernel bandwidth, R ``bw.nrd0`` (R's ``density()`` default)."""
    x = np.asarray(x, dtype=float)
    lo = min(np.std(x, ddof=1), np.subtract(*np.percentile(x, [75, 25])) / 1.34)
    if lo == 0:
        lo = np.std(x, ddof=1) or abs(x[0]) or 1.0
    return 0.9 * lo * len(x) ** -0.2


def _cell_totals(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    return np.concatenate([np.asarray(counts(adata, layer, r).sum(1)).ravel() for r in chunks(adata.n_obs, 50_000)])


def splat_bulk_reference(
    query: ad.AnnData | None,
    ref: ad.AnnData,
    N: int = 2,
    dist: Literal["sc-direct", "sc-mimic", "bulk"] = "sc-direct",
    bulk_feature_col: str | None = None,
    layer: str | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Simulate ``N`` single cells from every bulk sample of ``ref``.

    ``ref`` holds bulk counts as samples x genes (``.obs`` = sample annotation). Each cell
    gets a library size drawn from a kernel density of the query's cell totals
    (``dist="sc-direct"`` or ``"sc-mimic"``, molecules drawn without replacement) or of the
    bulk totals (``dist="bulk"``, with replacement), then molecules drawn from the bulk
    profile. Genes = those in both objects. Returns a cells x genes AnnData whose ``.obs``
    repeats the sample annotation; cells are named ``bulk_<sample>_cell_<k>`` (1-based).
    """
    if dist == "bulk":
        sizes, replace = _cell_totals(ref, layer), True
    else:
        if query is None:
            raise ValueError(f"dist={dist!r} needs a query to take cell sizes from")
        sizes, replace = _cell_totals(query), False
    genes = pd.Index(ref.var[bulk_feature_col] if bulk_feature_col else ref.var_names).astype(str)
    universe = list(genes) if query is None else [g for g in dict.fromkeys(genes) if g in set(query.var_names)]
    first = pd.Series(np.arange(len(genes)), index=genes)
    cols = first[~first.index.duplicated()][universe].to_numpy()  # first row of a repeated gene, as R match()
    bulk = np.rint(counts(ref, layer)[:, cols].toarray()).clip(min=0).astype(np.int64).T  # genes x samples
    r, c, v = _core.splat_bulk(np.ascontiguousarray(bulk), sizes.tolist(), bw_nrd0(sizes), N, replace, seed)
    x = sp.csr_matrix((v, (c, r)), shape=(ref.n_obs * N, len(universe)))
    obs = ref.obs.iloc[np.repeat(np.arange(ref.n_obs), N)].copy()
    obs.index = [f"bulk_{i + 1}_cell_{k + 1}" for i in range(ref.n_obs) for k in range(N)]
    return ad.AnnData(x, obs=obs, var=pd.DataFrame(index=universe))


def augment_data(adata: ad.AnnData, column: str, norm_number: int = 2000, prune: bool = False,
                 layer: str | None = None, seed: int | None = None) -> ad.AnnData:
    """Balance cell types to ``norm_number`` cells each.

    Types with fewer cells get simulated cells: library sizes from a kernel density of the
    type's own cell totals, molecules drawn without replacement from the type's pooled
    counts. With ``prune=True``, types with more cells are randomly down-sampled.
    Simulated cells are named ``stimcell_<type>`` (made unique), as in R.
    """
    rng = np.random.default_rng(seed)
    labels = adata.obs[column].astype(str).to_numpy()
    keep = np.ones(adata.n_obs, dtype=bool)
    types, n_per = np.unique(labels, return_counts=True)
    if prune:
        for t, n in zip(types, n_per):
            if n > norm_number:
                found = np.flatnonzero(labels == t)
                keep[rng.choice(found, n - norm_number, replace=False)] = False
    x = counts(adata, layer)
    new_x, new_labels = [], []
    for t, n in zip(types, n_per):
        if n >= norm_number:
            continue
        xt = x[labels == t]
        pool = np.rint(np.asarray(xt.sum(0)).ravel()).astype(np.int64)[:, None]
        sizes = np.asarray(xt.sum(1)).ravel()
        r, c, v = _core.splat_bulk(pool, sizes.tolist(), bw_nrd0(sizes) if n > 1 else 0.0,
                                   norm_number - n, False, int(rng.integers(2**31)))
        new_x.append(sp.csr_matrix((v, (c, r)), shape=(norm_number - n, adata.n_vars)))
        new_labels += [t] * (norm_number - n)
    names = ad.utils.make_index_unique(pd.Index([f"stimcell_{t}" for t in new_labels]), join=".")
    sim_obs = pd.DataFrame({column: new_labels}, index=names)
    with warnings.catch_warnings():  # input names may repeat; they are made unique below
        warnings.filterwarnings("ignore", "Observation names are not unique")
        parts = [ad.AnnData(x[keep], obs=adata.obs[keep], var=pd.DataFrame(index=adata.var_names)),
                 ad.AnnData(sp.vstack(new_x) if new_x else sp.csr_matrix((0, adata.n_vars)),
                            obs=sim_obs, var=pd.DataFrame(index=adata.var_names))]
        out = ad.concat(parts, join="outer", merge="same")
    out.obs_names_make_unique(join=".")
    return out


# --- deconvolution -------------------------------------------------------------------------

@dataclass
class Deconvolution:
    """Result of :func:`deconvolve_bulk` (R: the list ``deconvolve_bulk()`` returns)."""

    exposures: pd.DataFrame
    pred_counts: pd.DataFrame
    proportions: pd.DataFrame
    proportions_with_intercept: pd.DataFrame
    intercept_fraction: pd.Series
    metrics: dict = field(default_factory=dict)


def deconvolve_bulk(
    signatures: pd.DataFrame,
    bulk_counts: pd.DataFrame,
    gene_lengths: np.ndarray,
    gene_weights: np.ndarray | None = None,
    method: Literal["gd", "em"] = "gd",
    insert_size: float = 500,
    init_log_exp: float = -10,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    max_iter: int = 10_000,
    poll_interval: int = 100,
    ll_tolerance: float = 1e-6,
    sparsity_tolerance: float = 1e-4,
    seed: int | None = None,
    verbose: bool = True,
) -> Deconvolution:
    """Estimate how much of each cell type (columns of ``signatures``) is in each bulk sample.

    ``signatures`` is genes x cell types, ``bulk_counts`` genes x samples, on the same genes in
    the same order. Signature columns are rescaled to sum to 1. ``method="gd"`` fits a
    weighted Poisson model with Adam (a null fit of the intercept first); ``"em"`` uses
    expectation maximisation and ignores gene lengths, as in R. ``gene_weights`` in [0, 1]
    down-weight genes (default 1). Exposures include an ``Intercept`` row: the uniform
    "noise" source.
    """
    if not signatures.index.equals(bulk_counts.index):
        raise ValueError("signatures and bulk_counts must have the same genes in the same order")
    n_genes = len(signatures)
    gene_lengths = np.asarray(gene_lengths, dtype=np.float32)
    w = np.ones(n_genes, np.float32) if gene_weights is None else np.asarray(gene_weights, dtype=np.float32)
    if len(gene_lengths) != n_genes or len(w) != n_genes:
        raise ValueError("gene_lengths and gene_weights need one value per gene")
    if (w < 0).any() or (w > 1).any():
        raise ValueError("gene_weights must be in [0, 1]")
    s = signatures.to_numpy(dtype=np.float64)
    if np.any(np.abs(s.sum(0) - 1) > 1e-6):
        if verbose:
            print("deconvolve_bulk: signature columns do not sum to 1; normalising")
        s = s / s.sum(0)
    s, k = np.ascontiguousarray(s, np.float32), np.ascontiguousarray(bulk_counts.to_numpy(), np.float32)
    if method == "gd":
        e, p = _core.deconvolve_gd(s, k, gene_lengths.tolist(), w.tolist(), insert_size, init_log_exp,
                                   learning_rate, l1_lambda, l2_lambda, max_iter, poll_interval,
                                   ll_tolerance, sparsity_tolerance, seed, verbose)
    elif method == "em":
        e, p = _core.deconvolve_em(s, k, w.tolist(), max_iter, ll_tolerance, l1_lambda, verbose)
    else:
        raise ValueError("method must be 'gd' or 'em'")

    exposures = pd.DataFrame(e, index=[*signatures.columns, "Intercept"], columns=bulk_counts.columns)
    pred = pd.DataFrame(p, index=bulk_counts.index, columns=bulk_counts.columns)
    cells = exposures.iloc[:-1]
    with_int = exposures / exposures.sum(axis=0)
    return Deconvolution(exposures, pred, cells / cells.sum(axis=0), with_int, with_int.loc["Intercept"],
                         fit_metrics(bulk_counts.to_numpy(float), p.astype(float), w))


def fit_metrics(observed: np.ndarray, predicted: np.ndarray, gene_weights: np.ndarray) -> dict:
    """Goodness-of-fit summaries on genes with weight > 0 (R ``calculate_fit_metrics``)."""
    keep = gene_weights > 0
    o, p = observed[keep], predicted[keep]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    per_sample = np.array([corr(o[:, j], p[:, j]) for j in range(o.shape[1])])
    varying = o.std(1, ddof=1) >= 1e-10
    per_gene = np.array([corr(o[i], p[i]) for i in np.flatnonzero(varying)])
    rmse = np.sqrt(((o - p) ** 2).mean(0))

    def deviance(pred):
        return float(np.sum(2 * (o * np.log((o + 1e-10) / (pred + 1e-10)) - (o - pred))))

    null = deviance(np.broadcast_to(o.mean(0), o.shape))
    full = 2 * np.sum(o * np.log((o + 1e-10) / (p + 1e-10)) - (o - p))
    return {
        "n_genes_used": int(keep.sum()), "n_genes_excluded": int((~keep).sum()),
        "gene_correlation": per_sample, "rmse": rmse, "nrmse": rmse / o.mean(0),
        "count_ratio": p.sum(0) / o.sum(0), "global_correlation": corr(o.ravel(), p.ravel()),
        "r_squared": 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2),
        "pseudo_r2": 1 - full / null, "mape": float(np.mean(np.abs((o - p) / (o + 1))) * 100),
        "mean_gene_correlation": float(per_sample.mean()), "median_gene_correlation": float(np.median(per_sample)),
        "mean_sample_correlation": float(np.nanmean(per_gene)), "mean_rmse": float(rmse.mean()),
        "mean_nrmse": float((rmse / o.mean(0)).mean()),
    }
