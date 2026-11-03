"""Feature (gene) selection by dispersion, as in R viewmastR's ``select_features.R``.

For each gene: mean and coefficient of variation of size-factor-normalised counts, then a
cubic fit of log dispersion on log mean. Genes far above the fit are the most variable.
Results live in ``adata.var``:

========================  ==================================================
``log_mean``              natural log of the mean normalised count
``log_dispersion``        natural log of the coefficient of variation (in %)
``dispersion_fit``        cubic fit at ``log_mean`` (with ``_lci`` / ``_uci``)
``selected_feature``      set by :func:`select_features`
========================  ==================================================
"""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from ._data import chunks, counts


def calculate_feature_dispersion(adata: ad.AnnData, q: int = 3, layer: str | None = None,
                                 chunk_size: int = 10_000) -> ad.AnnData:
    """Per-gene log mean, log dispersion and the degree-``q`` polynomial fit between them.

    Size factor = total count / mean total count (all genes). Mean and variance (n - 1
    denominator) are accumulated over cell chunks, so large and backed objects work.
    """
    n, g = adata.n_obs, adata.n_vars
    lib = np.zeros(n)
    for rows in chunks(n, chunk_size):
        lib[rows] = np.asarray(counts(adata, layer, rows).sum(1)).ravel()
    sf = lib / lib.mean()
    s1, s2 = np.zeros(g), np.zeros(g)
    for rows in chunks(n, chunk_size):
        x = sp.diags(1.0 / sf[rows]) @ counts(adata, layer, rows)
        s1 += np.asarray(x.sum(0)).ravel()
        s2 += np.asarray(x.multiply(x).sum(0)).ravel()
    mean = s1 / n
    var = np.maximum(s2 - s1**2 / n, 0.0) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_disp = np.log(np.sqrt(var) / mean * 100.0)
        log_mean = np.log(mean)

    ok = np.isfinite(log_disp) & np.isfinite(log_mean)
    if ok.sum() < 10:
        raise ValueError("too few genes with a finite mean and dispersion to fit")
    fit, se = _poly_fit(log_mean[ok], log_disp[ok], log_mean, q)
    adata.var["log_mean"] = log_mean
    adata.var["log_dispersion"] = log_disp
    adata.var["dispersion_fit"] = fit
    adata.var["dispersion_lci"] = fit - 1.96 * se
    adata.var["dispersion_uci"] = fit + 1.96 * se
    return adata


def _poly_fit(x: np.ndarray, y: np.ndarray, at: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial of degree ``q`` and its standard error at ``at``.

    Same fitted values as R ``lm(y ~ x + poly(x, q))`` (the extra linear term is aliased).
    ``x`` is centred and scaled first for a well-conditioned design.
    """
    mu, sd = x.mean(), x.std()
    design = lambda v: np.vander((v - mu) / sd, q + 1)
    X = design(x)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma2 = np.sum((y - X @ beta) ** 2) / (len(y) - X.shape[1])
    xtx_inv = np.linalg.inv(X.T @ X)
    fit = np.full(at.shape, np.nan)
    se = np.full(at.shape, np.nan)
    good = np.isfinite(at)
    A = design(at[good])
    fit[good] = A @ beta
    se[good] = np.sqrt(np.einsum("ij,jk,ik->i", A, xtx_inv, A) * sigma2)
    return fit, se


def select_features(adata: ad.AnnData, fit_min: float = 1.0, fit_max: float = np.inf,
                    logmean_ul: float = np.inf, logmean_ll: float = -np.inf,
                    top_n: int | None = None) -> ad.AnnData:
    """Mark genes for training in ``adata.var["selected_feature"]``.

    ``ratio = log_dispersion / dispersion_fit``. With ``top_n``, take the ``top_n`` genes
    with the largest ratio among those with ``logmean_ll < log_mean < logmean_ul``;
    otherwise take every gene with ``fit_min < ratio < fit_max`` inside that window.
    """
    if "dispersion_fit" not in adata.var:
        raise ValueError("run calculate_feature_dispersion() first")
    v = adata.var
    ratio = (v["log_dispersion"] / v["dispersion_fit"]).to_numpy()
    in_range = ((v["log_mean"] > logmean_ll) & (v["log_mean"] < logmean_ul)).to_numpy()
    if top_n is not None:
        idx = np.flatnonzero(in_range)
        order = idx[np.argsort(-ratio[idx], kind="stable")]  # NaN ratios sort last, as in R order()
        selected = np.zeros(adata.n_vars, dtype=bool)
        selected[order[:top_n]] = True
    else:
        selected = in_range & (ratio > fit_min) & (ratio < fit_max)
    adata.var["selected_feature"] = selected
    return adata


def get_selected_features(adata: ad.AnnData) -> list[str]:
    """Genes marked by :func:`select_features` or :func:`set_selected_features`."""
    return list(adata.var_names[adata.var["selected_feature"].to_numpy(dtype=bool)])


def set_selected_features(adata: ad.AnnData, genes: Sequence[str]) -> ad.AnnData:
    """Mark exactly ``genes`` as selected."""
    adata.var["selected_feature"] = adata.var_names.isin(list(genes))
    if not adata.var["selected_feature"].any():
        print("set_selected_features: none of the genes are in the object")
    return adata


def plot_feature_dispersion(adata: ad.AnnData, size: float = 1, alpha: float = 0.4, ax=None):
    """Log dispersion against log mean, selected genes in red, with the fitted curve."""
    v = adata.var
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    sel = v["selected_feature"].to_numpy(dtype=bool) if "selected_feature" in v else np.zeros(len(v), bool)
    ax.scatter(v["log_mean"][~sel], v["log_dispersion"][~sel], s=size, alpha=alpha, c="lightgray",
               rasterized=True, label="other")
    if sel.any():
        ax.scatter(v["log_mean"][sel], v["log_dispersion"][sel], s=size, alpha=alpha, c="firebrick",
                   rasterized=True, label=f"selected ({sel.sum()})")
    o = np.argsort(v["log_mean"].to_numpy())
    ax.plot(v["log_mean"].to_numpy()[o], v["dispersion_fit"].to_numpy()[o], c="black", lw=1)
    ax.set(xlabel="log mean expression", ylabel="log dispersion")
    ax.legend(frameon=False, markerscale=5)
    return ax


def common_features(*adatas: ad.AnnData) -> list[str]:
    """Genes present in every object, in the order of the first."""
    shared = set.intersection(*(set(a.var_names) for a in adatas))
    return [g for g in adatas[0].var_names if g in shared]


__all__ = ["calculate_feature_dispersion", "select_features", "get_selected_features",
           "set_selected_features", "plot_feature_dispersion", "common_features"]
