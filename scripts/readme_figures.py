"""Make the two README figures from the public tutorial data.

Reference: healthy bone marrow atlas (Granja et al. 2019, GEO GSE139369), 16 cell types.
Query: T-cell-depleted bone marrow multiome RNA, both replicates, every cell Cell Ranger
called (Persad et al. 2023, Zenodo 6383269), 11,744 cells.

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


def palette(labels) -> dict[str, str]:
    cols = sns.color_palette("tab20", 20).as_hex()
    return {lab: cols[i % 20] for i, lab in enumerate(sorted(labels))}


def shared_embedding(ref, query, genes):
    """Reference UMAP on the model genes; the query is projected onto it (scanpy ingest)."""
    r, q = ref[:, genes].copy(), query[:, genes].copy()
    for a in (r, q):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    sc.pp.pca(r, n_comps=30, random_state=0)
    sc.pp.neighbors(r, random_state=0)
    sc.tl.umap(r, random_state=0)
    sc.tl.ingest(q, r, obs=[], embedding_method="umap")
    return r.obsm["X_umap"], q.obsm["X_umap"]


def umap_panel(ax, xy, labels, cols, title):
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(xy[order, 0], xy[order, 1], s=1.5, linewidths=0, rasterized=True,
               c=[cols[x] for x in np.asarray(labels)[order]])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    label = td.config().get("greenleaf_label", "celltype")
    ref = td.greenleaf_reference()
    query = td.bm_tcelldep_full()
    if args.subsample:
        sc.pp.subsample(ref, n_obs=min(args.subsample, ref.n_obs), random_state=0)
        sc.pp.subsample(query, n_obs=min(args.subsample, query.n_obs), random_state=0)
    print("reference", ref.shape, "query", query.shape)

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
    val_acc = 100 * float(res.history["test_acc"][-1])
    print(f"held-out reference accuracy {val_acc:.1f}%")

    # Figure 1: transfer learning on a shared embedding
    cols = palette(ref.obs[label].astype(str).unique())
    ref_xy, q_xy = shared_embedding(ref, query, vg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    umap_panel(axes[0], ref_xy, ref.obs[label].astype(str), cols,
               f"Reference: known labels ({ref.n_obs:,} cells)")
    umap_panel(axes[1], q_xy, query.obs["viewmastR_pred"].astype(str), cols,
               f"Query: viewmastR predictions ({query.n_obs:,} cells)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=c, label=k) for k, c in cols.items()]
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=9)
    fig.text(0.43, 0.02, "Query cells are projected onto the reference UMAP (scanpy ingest).",
             ha="center", fontsize=8, color="0.35")
    fig.subplots_adjust(left=0.01, right=0.86, wspace=0.04, bottom=0.07)
    fig.savefig(OUT / "transfer_umap.png", dpi=150)
    plt.close(fig)

    # Figure 2: predicted reference label vs the query authors' label, percent of each column
    lab = query.obs["celltype"].astype(str)
    keep = (lab != "unlabelled").to_numpy()
    tab = pd.crosstab(query.obs["viewmastR_pred"].astype(str)[keep], lab[keep])
    frac = tab.div(tab.sum(axis=0), axis=1)
    print(f"{int(keep.sum())} query cells with an author label")
    print(tab.to_string())
    g = sns.clustermap(frac, cmap="viridis", vmin=0, vmax=1, linewidths=0,
                       row_colors=pd.Series({k: cols[k] for k in frac.index}, name="predicted"),
                       figsize=(7.5, 7.5), dendrogram_ratio=0.12, colors_ratio=0.03,
                       cbar_pos=(1.02, 0.35, 0.025, 0.2), cbar_kws={"label": "Fraction of column"})
    g.ax_heatmap.set(xlabel=f"Query author label ({int(keep.sum()):,} labelled cells)",
                     ylabel="viewmastR prediction (reference label)")
    g.ax_heatmap.tick_params(labelsize=9)
    g.ax_row_colors.set_xticks([])
    g.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("wrote", *sorted(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
