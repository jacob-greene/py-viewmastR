"""Make the two README figures from the tutorial data.

Reference: BoneMarrowMap healthy bone marrow atlas (Zeng et al. 2025), 24 broad cell types,
balanced to 3,000 cells per type with ``augment_data`` (``tutorial_data.bonemarrowmap_augmented``,
as in tutorials 01 and 04). Accuracy is measured on held-out real reference cells.
Query: T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), 8,627 cells, loaded
as is with the authors' labels and UMAP (``tutorial_data.bm_tcelldep``).

Both datasets are shown on their own stored UMAP (real reference cells only); nothing is
re-embedded. Colours come
from ``viewmastr.palette``.

Writes ``docs/img/transfer_umap.png`` and ``docs/img/confusion_matrix.png``.

    PYTHONPATH=tutorials python scripts/readme_figures.py [--subsample N]

``--subsample N`` keeps N cells of each dataset, for a quick test run.
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
import seaborn as sns

import tutorial_data as td
import viewmastr as vm

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 20261103
ORDER = list(vm.palette.COLORS)


def lineage_order(labels) -> list[str]:
    """Labels in palette order (roughly by lineage); unknown labels last."""
    rank = {k: i for i, k in enumerate(ORDER)}
    key = lambda x: (rank.get(vm.palette.canonical(x), len(ORDER)), x)
    return sorted(pd.unique(pd.Series(list(labels), dtype=str)), key=key)


def umap_panel(ax, xy, labels, title):
    labels = np.asarray(labels, dtype=str)
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(xy[order, 0], xy[order, 1], s=1.5, linewidths=0, rasterized=True,
               c=[vm.palette.color(x) for x in labels[order]])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=vm.palette.color(k), label=k)
               for k in lineage_order(labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              frameon=False, fontsize=8, handletextpad=0.2, columnspacing=0.8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    label = "CellType_Broad"
    ref, held = td.bonemarrowmap_augmented(label, norm_number=3000)
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

    # Figure 1: reference labels, query author labels, query predictions; each on its own UMAP
    pred = query.obs["viewmastR_pred"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.2))
    real = ref[~ref.obs["simulated"].to_numpy()]
    umap_panel(axes[0], np.asarray(real.obsm["X_umap"]), real.obs[label],
               f"Reference: BoneMarrowMap labels ({real.n_obs:,} real cells)")
    umap_panel(axes[1], np.asarray(query.obsm["X_umap"]), query.obs["celltype"],
               f"Query: authors' labels ({query.n_obs:,} cells)")
    umap_panel(axes[2], np.asarray(query.obsm["X_umap"]), pred,
               "Query: viewmastR predictions")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.25, wspace=0.05)
    fig.savefig(OUT / "transfer_umap.png", dpi=150)
    plt.close(fig)

    # Figure 2: predicted reference label vs the query authors' label, fraction of each column
    author = query.obs["celltype"].astype(str)
    tab = pd.crosstab(pred, author)
    tab = tab.loc[lineage_order(tab.index), lineage_order(tab.columns)]
    frac = tab.div(tab.sum(axis=0), axis=1)
    print(tab.to_string())
    g = sns.clustermap(frac, row_cluster=False, col_cluster=False, cmap="viridis", vmin=0, vmax=1,
                       row_colors=pd.Series({k: vm.palette.color(k) for k in frac.index}, name=""),
                       col_colors=pd.Series({k: vm.palette.color(k) for k in frac.columns}, name=""),
                       figsize=(8, 7.5), dendrogram_ratio=0.04, colors_ratio=0.025, linewidths=0,
                       cbar_pos=(1.0, 0.35, 0.025, 0.2), cbar_kws={"label": "Fraction of column"})
    g.ax_heatmap.set(xlabel=f"Query authors' label ({query.n_obs:,} cells)",
                     ylabel="viewmastR prediction (reference label)")
    g.ax_heatmap.tick_params(labelsize=9, length=0)
    g.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("wrote", *sorted(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
