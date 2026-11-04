"""Plots that mirror the R package's ``confusion_matrix()``, ``plot_training_data()`` and
``plot_deconvolution()`` (matplotlib instead of ComplexHeatmap, plotly and ggplot2)."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from . import palette


def confusion_matrix(pred: Sequence, gt: Sequence, cols: Mapping[str, str] | None = None,
                     ax=None, fontsize: float = 7):
    """Predicted (rows) against true (columns) labels, coloured by percent of each column.

    Cells show counts. When one label set contains the other (one vocabulary), the colour-bar
    label gives the accuracy: the share of cells whose predicted label equals the true label.
    Two different vocabularies that share a few names get no accuracy. ``cols`` maps
    labels to colours; each tick label then sits on a box of its colour.
    """
    pred, gt = pd.Series(np.asarray(pred, dtype=str)), pd.Series(np.asarray(gt, dtype=str))
    tab = pd.crosstab(pred, gt)
    pct = tab / tab.sum(0) * 100
    rows, columns = set(tab.index), set(tab.columns)
    one_vocabulary = rows <= columns or columns <= rows
    acc = 100 * float((pred.to_numpy() == gt.to_numpy()).mean()) if one_vocabulary else None

    nr, nc = tab.shape
    ax = ax or plt.subplots(figsize=(0.35 * nc + 2.5, 0.3 * nr + 2))[1]
    im = ax.imshow(pct.to_numpy(), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            ax.text(j, i, f"{tab.iat[i, j]:d}", ha="center", va="center", fontsize=fontsize,
                    color="black" if pct.iat[i, j] > 60 else "white")
    ax.set_xticks(range(nc), tab.columns, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(nr), tab.index, fontsize=fontsize)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set(xlabel="True label", ylabel="Predicted label")
    ax.grid(False)
    if cols:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            c = cols.get(t.get_text())
            if c is not None:
                r, g, b_ = to_rgb(c)
                t.set_color("black" if 0.299 * r + 0.587 * g + 0.114 * b_ > 0.5 else "white")
                t.set_bbox(dict(facecolor=c, edgecolor="none", pad=1.0))
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Percent of column" + (f" (accuracy {acc:.2f}%)" if acc is not None else ""))
    return ax


def plot_training_data(result, axes=None):
    """Accuracy and loss per epoch for training and validation cells.

    ``result`` is a :class:`viewmastr.ViewmastrResult` (or its ``history`` dict).
    """
    h = getattr(result, "history", result)
    epochs = np.arange(1, len(h["train_acc"]) + 1)
    axes = axes if axes is not None else plt.subplots(2, 1, figsize=(5, 5), sharex=True)[1]
    for key, name in (("train", "train"), ("test", "validation")):
        axes[0].plot(epochs, 100 * np.asarray(h[f"{key}_acc"]), "o-", label=f"{name} accuracy")
        axes[1].plot(epochs, h[f"{key}_loss"], "o-", label=f"{name} loss")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set(xlabel="Epoch", ylabel="Loss")
    for a in axes:
        a.legend(frameon=False)
        a.grid(alpha=0.3)
    return axes


def plot_deconvolution(result, observed_counts: pd.DataFrame | None = None,
                       kind: str = "all"):
    """Diagnostic panels for a :class:`viewmastr.Deconvolution`.

    ``kind``: ``"proportions"`` (stacked bars), ``"fit"`` (observed vs predicted counts),
    ``"correlation"`` (per-sample correlation), ``"residuals"``, ``"intercept"`` or ``"all"``.
    The fit and residual panels need ``observed_counts``.
    """
    panels = ["proportions", "correlation", "intercept"]
    if observed_counts is not None:
        panels[1:1] = ["fit", "residuals"]
    if kind != "all":
        panels = [kind]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.2 * len(panels)), squeeze=False)
    for ax, p in zip(axes[:, 0], panels):
        if p == "proportions":
            types = list(result.proportions.index)
            style = ({"color": [palette.color(t) for t in types]} if all(palette.known(t) for t in types)
                     else {"colormap": "tab20"})
            result.proportions.T.plot.bar(stacked=True, ax=ax, width=0.9, legend=False, **style)
            ax.set(title="Cell-type proportions (without the intercept)", ylabel="Proportion", xlabel="")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, frameon=False)
            ax.tick_params(axis="x", labelsize=5)
        elif p in ("fit", "residuals"):
            o, pr = observed_counts.to_numpy(float).ravel(), result.pred_counts.to_numpy(float).ravel()
            if p == "fit":
                k = (o > 0) & (pr > 0)
                ax.hexbin(np.log1p(o[k]), np.log1p(pr[k]), gridsize=100, bins="log", cmap="viridis")
                lim = [0, np.log1p(max(o.max(), pr.max()))]
                ax.plot(lim, lim, "r--", lw=1)
                ax.set(title=f"Fit (r = {np.corrcoef(o[k], pr[k])[0, 1]:.3f})",
                       xlabel="log1p(observed)", ylabel="log1p(predicted)")
            else:
                ax.hexbin(np.log1p(pr), (o - pr) / (o + 1), gridsize=100, bins="log", cmap="viridis",
                          extent=(0, np.log1p(pr.max()), -2, 2))
                ax.axhline(0, color="red", ls="--", lw=1)
                ax.set(title="Residuals", xlabel="log1p(predicted)", ylabel="(obs - pred) / (obs + 1)", ylim=(-2, 2))
        elif p == "correlation":
            r = pd.Series(result.metrics["gene_correlation"], index=result.proportions.columns)
            r.plot.bar(ax=ax, color="steelblue", width=0.9)
            ax.axhline(r.mean(), color="red", ls="--", lw=1)
            ax.set(title=f"Per-sample correlation (mean {r.mean():.3f})", ylabel="Pearson r", ylim=(0, 1))
            ax.tick_params(axis="x", labelsize=5)
        elif p == "intercept":
            result.intercept_fraction.plot.bar(ax=ax, color="coral", width=0.9)
            ax.axhline(0.5, color="red", ls="--", lw=1)
            ax.set(title="Intercept share of total exposure", ylabel="Intercept / total")
            ax.tick_params(axis="x", labelsize=5)
    fig.tight_layout()
    return fig
