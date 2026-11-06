"""Make the README figure from the tutorial data.

Reference: BoneMarrowMap healthy bone marrow atlas (Zeng et al. 2025) with its finest labels,
``CellType`` (55 types), balanced to 3,000 cells per type with ``augment_data``
(``tutorial_data.bonemarrowmap_augmented``). ``--no-augment`` drops the simulated cells and trains
on the real cells only. Accuracy is measured on held-out real reference cells.
Query: T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), 8,627 cells, loaded
as is with the authors' labels and UMAP (``tutorial_data.bm_tcelldep``).

Both datasets are shown on their own stored UMAP (real BoneMarrowMap cells only); nothing is
re-embedded. Colours come from ``tutorials/celltype_colors.py``, which the tutorials share.

Writes ``docs/img/transfer_umap.png`` and an editable vector copy, ``docs/img/transfer_umap.pdf``:
three UMAPs, a heatmap of the predicted label against the query authors' label, and under the
first two UMAPs the number of shared genes.
The prediction legend lists types called on at least 10 query cells; a mature type is
listed only if the query authors label that population too (``MATURE``). The heatmap keeps all 55
types, so uncalled types show as empty rows.

    PYTHONPATH=tutorials python scripts/readme_figures.py [--subsample N] [--replot]

``--subsample N`` keeps N cells of each dataset, for a quick test run. Each run caches what the
figure draws in ``cache_dir``; ``--replot`` redraws from that cache without training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import tutorial_data as td
from celltype_colors import FINE_ORDER, color
import viewmastr as vm

# Editable text in the PDF (TrueType, not outlines); Helvetica or Arial if installed, else Liberation Sans
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]})

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 20261103
TITLE = ("ViewmastR: de novo celltyping and refinement based on transfer learning from reference "
         "annotations")
QSIZE = 6  # dot area of the two query UMAPs (the 118,509-cell reference keeps 1.5)
ORDER = list(vm.palette.COLORS)
# Mature reference types, by the palette name of the population. The column 3 legend shows a mature
# type only if the query authors label that population too; developing types need no such match.
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
MIN_CELLS = 10  # column 3 legend: types predicted for at least this many query cells


def lineage_order(labels) -> list[str]:
    """Labels in palette order (roughly by lineage); unknown labels last."""
    if set(map(str, labels)) <= set(FINE_ORDER):
        return [k for k in FINE_ORDER if k in set(map(str, labels))]
    rank = {k: i for i, k in enumerate(ORDER)}
    key = lambda x: (rank.get(vm.palette.canonical(x), len(ORDER)), x)
    return sorted(pd.unique(pd.Series(list(labels), dtype=str)), key=key)


# Font sizes (points). The figure is 27 inches wide and shown about 900 pixels wide in the
# README, so 1 point is about half a pixel there: panel titles need 24 points or more.
FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 34, 26, 18, 13, 11
FIG_W, FIG_H = 27, 13.5  # inches


def box(fig, left, bottom, width, height):
    """Axes at a position given in inches from the lower left corner."""
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def umap_panel(ax, xy, labels, title, legend=None, size=1.5):
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


def compute(args) -> dict:
    """Train, predict and score; return what the figure draws."""
    label = "CellType"
    ref, held = td.bonemarrowmap_augmented(label, norm_number=3000)
    if args.no_augment:
        ref = ref[~ref.obs["simulated"].to_numpy()].copy()
    query = td.bm_tcelldep()
    if args.subsample:
        for a in (ref, held, query):
            sc.pp.subsample(a, n_obs=min(args.subsample, a.n_obs), random_state=0)
    print("reference", ref.shape, "held out", held.shape, "query", query.shape)

    # Shared variable genes, as in tutorial 01
    for a in (query, ref):
        vm.calculate_feature_dispersion(a)
        vm.select_features(a, top_n=10_000, logmean_ul=-1, logmean_ll=-8)
    vgr = set(vm.get_selected_features(ref))
    vg = [g for g in vm.get_selected_features(query) if g in vgr]
    print(len(vg), "shared variable genes")

    res = vm.viewmastR(query, ref, label, selected_features=vg, seed=SEED, workers=1,
                       return_type="list")
    query = res.adata
    vm.viewmastR_infer(held, res.model_dir, query_celldata_col="pred")
    acc = 100 * float((held.obs["pred"].astype(str) == held.obs[label].astype(str)).mean())
    print(f"accuracy on {held.n_obs:,} held-out real reference cells {acc:.1f}%")
    real = ref[~ref.obs["simulated"].to_numpy()]
    return {"ref_xy": np.asarray(real.obsm["X_umap"]), "ref_label": real.obs[label].astype(str).to_numpy(),
            "query_xy": np.asarray(query.obsm["X_umap"]),
            "author": query.obs["celltype"].astype(str).to_numpy(),
            "pred": query.obs["viewmastR_pred"].astype(str).to_numpy(),
            "n_genes": len(vg), "accuracy": acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--no-augment", action="store_true", help="train on real reference cells only")
    p.add_argument("--panel", action="store_true", help="test: draw only the prediction panel")
    p.add_argument("--replot", action="store_true",
                   help="redraw from the results the last run cached (no training)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path(td.config()["cache_dir"]) / (
        "readme_figure" + (f"_sub{args.subsample}" if args.subsample else "")
        + ("_noaug" if args.no_augment else "") + ".pkl")
    if args.replot:
        d = pd.read_pickle(cache)
    else:
        d = compute(args)
        pd.to_pickle(d, cache)
    print(f"{d['n_genes']} shared variable genes; accuracy {d['accuracy']:.1f}%")

    pred = pd.Series(d["pred"])
    # Column 3 legend: types called on at least MIN_CELLS query cells; a mature type also needs its
    # population among the query authors' labels. The heatmap keeps all types.
    n_called = pred.value_counts()
    authors = set(d["author"])
    legend_rows = []
    for k in FINE_ORDER:
        n = int(n_called.get(k, 0))
        why = ("fewer than %d cells" % MIN_CELLS if n < MIN_CELLS else
               "mature, not in authors' labels" if k in MATURE and MATURE[k] not in authors else "")
        legend_rows.append((k, n, "hidden" if why else "shown", why))
    legend_tab = pd.DataFrame(legend_rows, columns=["type", "cells", "legend", "reason"])
    shown = legend_tab.loc[legend_tab["legend"] == "shown", "type"].tolist()
    print(f"column 3 legend: {len(shown)} of {len(FINE_ORDER)} types shown")
    print(legend_tab.to_string(index=False))

    n_query = len(pred)
    if args.panel:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        umap_panel(box(fig, 0.3, 5.8, 6.4, 5.6), d["query_xy"], pred,
                   "(c) Query: viewmastR prediction", legend=shown, size=QSIZE)
        fig.savefig(OUT / "test_panel.png", dpi=150)
        return
    # Predicted reference label vs the query authors' label, fraction of each column
    tab = pd.crosstab(pred.to_numpy(), d["author"])
    tab = tab.reindex(index=FINE_ORDER, fill_value=0).loc[:, lineage_order(tab.columns)]
    frac = tab.div(tab.sum(axis=0), axis=1)
    print(tab.to_string())

    # Top row: reference labels, query author labels, query prediction (UMAPs with legends below).
    # Under the first two columns: the shared gene count.
    # Right: the heatmap.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cols_x = [0.3, 7.0, 13.7]
    umap_panel(box(fig, cols_x[0], 5.8, 6.4, 5.6), d["ref_xy"], d["ref_label"],
               f"(a) Reference: BoneMarrowMap\n({len(d['ref_label']):,} real cells)")
    umap_panel(box(fig, cols_x[1], 5.8, 6.4, 5.6), d["query_xy"], d["author"],
               f"(b) Query: authors' labels\n({n_query:,} cells)", size=QSIZE)
    umap_panel(box(fig, cols_x[2], 5.8, 6.4, 5.6), d["query_xy"], pred,
               "(c) Query: viewmastR prediction\n(reference labels)", legend=shown, size=QSIZE)
    fig.text((cols_x[0] + cols_x[1] + 6.4) / 2 / FIG_W, 0.5 / FIG_H,
             f"{d['n_genes']:,} common genes used for transfer", ha="center", va="center",
             fontsize=FS_TITLE, fontweight="bold")

    x0, x1, y0, y1 = 23.0, 25.6, 1.5, 11.1  # heatmap, inches
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
    cols.set_title("(d) Prediction vs\nauthors' label", fontsize=FS_TITLE, pad=10)
    for a in (rows, cols):
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in a.spines.values():
            sp.set_visible(False)
    # Row names sit left of the row colour strip
    ax.tick_params(labelleft=False)
    rows.tick_params(labelleft=True, labelsize=FS_TICK - 1, length=0)
    rows.set_ylabel("viewmastR prediction (reference label)", fontsize=FS_LABEL)
    ax.xaxis.set_tick_params(labelbottom=True)
    cax = box(fig, 25.85, 4.5, 0.18, 3.5)
    fig.colorbar(im, cax=cax).set_label("Fraction of column", fontsize=FS_LABEL)
    cax.tick_params(labelsize=FS_TICK)
    fig.suptitle(TITLE, fontsize=FS_SUPTITLE, y=1 - 0.35 / FIG_H, va="top")
    fig.savefig(OUT / "transfer_umap.png", dpi=150)
    # Vector copy for Illustrator: text and axes stay editable; the scatter layers are 300 dpi images
    fig.savefig(OUT / "transfer_umap.pdf", dpi=300)
    plt.close("all")
    print("wrote", *sorted(OUT.glob("transfer_umap.*")))


if __name__ == "__main__":
    main()
