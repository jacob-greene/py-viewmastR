"""Panels of the README figure, shared by ``scripts/readme_figures.py`` and tutorial 01.

Both draw the reference UMAP, the query UMAPs and the prediction heatmap with these
functions, so the tutorial plots match the README figure: same colours (``celltype_colors``),
dot sizes, legends below each UMAP, title and font sizes, and heatmap layout.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import viewmastr as vm

from celltype_colors import FINE_ORDER, color

# Editable text in the PDF (TrueType, not outlines); Helvetica or Arial if installed, else Liberation Sans
STYLE = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]}
# Font sizes (points). The README figure is 27 inches wide and shown about 900 pixels wide, so
# 1 point is about half a pixel there: panel titles need 24 points or more.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 26, 18, 13, 11
UMAP_W, UMAP_H = 6.4, 5.6  # inches, one UMAP panel
REF_SIZE, QSIZE = 1.5, 6  # dot area: the 118,509-cell reference, the 8,627-cell query
ORDER = list(vm.palette.COLORS)
# Mature reference types, by the palette name of the population. The prediction legend shows a
# mature type only if the query authors label that population too; developing types need no match.
MATURE = {
    "Mature B": "NaiveB", "Plasma Cell": "Plasma", "CD14 Mono": "Mono", "CD16 Mono": "CD16.Mono",
    "cDC1": "cDC", "cDC2": "cDC", "pDC": "pDC", "Megakaryocyte": "Mega",
    "Orthochromatic Erythroblast": "Ery", "CD4 Naive": "CD4.N", "CD4 Central Memory": "CD4.M",
    "CD4 Effector Memory": "CD4.M", "CD4 Regulatory": "T", "CD8 Naive": "CD8.N",
    "CD8 Central Memory": "CD8.CM", "CD8 Effector Memory 1": "CD8.EM",
    "CD8 Effector Memory 2": "CD8.EM", "CD8 Tissue Resident Memory": "CD8.EM",
    "T Proliferating": "T", "NK CD56high": "NK", "NK": "NK", "NK Proliferating": "NK",
    "Stromal": "Stromal",
}
MIN_CELLS = 10  # prediction legend: types predicted for at least this many query cells


def lineage_order(labels) -> list[str]:
    """Labels in palette order (roughly by lineage); unknown labels last."""
    if set(map(str, labels)) <= set(FINE_ORDER):
        return [k for k in FINE_ORDER if k in set(map(str, labels))]
    return vm.plotting.palette_order(labels)


def box(fig, left, bottom, width, height):
    """Axes at a position given in inches from the lower left corner of ``fig``."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def umap_panel(ax, xy, labels, title, legend=None, size=REF_SIZE):
    """UMAP coloured by label (dot area ``size``); the legend lists ``legend`` (default: every label)."""
    labels = np.asarray(labels, dtype=str)
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(xy[order, 0], xy[order, 1], s=size, linewidths=0, rasterized=True,
               c=[color(x) for x in labels[order]])
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=color(k), label=k)
               for k in (lineage_order(labels) if legend is None else legend)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              frameon=False, fontsize=FS_LEGEND, handletextpad=0.2, columnspacing=0.8)


def umap_figure(xy, labels, title, legend=None, size=REF_SIZE):
    """One UMAP panel on its own figure, at its size in the README figure."""
    fig = plt.figure(figsize=(UMAP_W, UMAP_H))
    umap_panel(fig.add_axes([0, 0, 1, 1]), xy, labels, title, legend=legend, size=size)
    return fig


def prediction_legend(pred, author) -> tuple[list[str], pd.DataFrame]:
    """Types the prediction legend lists, and a table of why each type is shown or hidden.

    A type is listed if it is called on at least ``MIN_CELLS`` query cells; a mature type also
    needs its population among the query authors' labels. The heatmap keeps all types.
    """
    n_called = pd.Series(np.asarray(pred, dtype=str)).value_counts()
    authors = set(np.asarray(author, dtype=str))
    rows = []
    for k in FINE_ORDER:
        n = int(n_called.get(k, 0))
        why = ("fewer than %d cells" % MIN_CELLS if n < MIN_CELLS else
               "mature, not in authors' labels" if k in MATURE and MATURE[k] not in authors else "")
        rows.append((k, n, "hidden" if why else "shown", why))
    tab = pd.DataFrame(rows, columns=["type", "cells", "legend", "reason"])
    return tab.loc[tab["legend"] == "shown", "type"].tolist(), tab


def label_fraction(pred, author) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell counts of predicted type (rows, all 55) by author label, and the fraction of each column.

    Columns follow the rows they mostly go to, so the heatmap runs along its diagonal.
    """
    tab = pd.crosstab(np.asarray(pred, dtype=str), np.asarray(author, dtype=str))
    tab = vm.plotting.order_diagonal(tab, rows=FINE_ORDER)
    return tab, tab.div(tab.sum(axis=0), axis=1)


def heatmap_panel(fig, frac, x0, x1, y0, y1, cbar, title="(d) Prediction vs\nauthors' label"):
    """Heatmap of ``frac`` in the box ``x0``-``x1``, ``y0``-``y1`` (inches), with colour strips.

    Rows are predicted types, columns author labels. ``cbar`` is the colour bar box
    (left, bottom, width, height) in inches.
    """
    ax = box(fig, x0, y0, x1 - x0, y1 - y0)
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(frac.shape[1]), frac.columns, rotation=90)
    ax.set_yticks(range(frac.shape[0]), frac.index)
    ax.tick_params(labelsize=FS_TICK - 1, length=0)
    ax.set_xlabel("Query authors' label", fontsize=FS_LABEL)
    rows = box(fig, x0 - 0.2, y0, 0.15, y1 - y0)
    rows.sharey(ax)
    rows.imshow(np.array([[matplotlib.colors.to_rgb(color(k))] for k in frac.index]),
                aspect="auto", interpolation="nearest")
    cols = box(fig, x0, y1 + 0.08, x1 - x0, 0.2)
    cols.sharex(ax)
    cols.imshow(np.array([[matplotlib.colors.to_rgb(color(k)) for k in frac.columns]]),
                aspect="auto", interpolation="nearest")
    cols.set_title(title, fontsize=FS_TITLE, pad=10)
    for a in (ax, rows, cols):
        a.grid(False)  # scanpy's figure style turns the grid on
    for a in (rows, cols):
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in a.spines.values():
            sp.set_visible(False)
    # Row names sit left of the row colour strip
    ax.tick_params(labelleft=False)
    rows.tick_params(labelleft=True, labelsize=FS_TICK - 1, length=0)
    rows.set_ylabel("viewmastR prediction (reference label)", fontsize=FS_LABEL)
    ax.xaxis.set_tick_params(labelbottom=True)
    cax = box(fig, *cbar)
    fig.colorbar(im, cax=cax).set_label("Fraction of column", fontsize=FS_LABEL)
    cax.tick_params(labelsize=FS_TICK)
    return ax
