"""Make the README figure from the tutorial data.

Reference: BoneMarrowMap healthy bone marrow atlas (Zeng et al. 2025) with its finest labels,
``CellType`` (55 types), balanced to 3,000 cells per type with ``augment_data``
(``tutorial_data.bonemarrowmap_augmented``). ``--no-augment`` drops the simulated cells and trains
on the real cells only. Accuracy is measured on held-out real reference cells.
Query: T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), 8,627 cells, loaded
as is with the authors' labels and UMAP (``tutorial_data.bm_tcelldep``).

Both datasets are shown on their own stored UMAP (real BoneMarrowMap cells only); nothing is
re-embedded. Colours come from ``tutorials/celltype_colors.py``, which the tutorials share. The
panels are drawn by ``tutorials/readme_panels.py``, which tutorial 01 uses too.

Writes ``docs/img/transfer_umap.png`` and an editable vector copy, ``docs/img/transfer_umap.pdf``:
three UMAPs, a heatmap of the predicted label against the query authors' label, and under the
first two UMAPs the number of shared genes. On both query UMAPs a ring marks the authors'
``Mega`` cluster, which viewmastR labels EoBasoMast Precursor.
The prediction legend lists types called on at least 10 query cells; a mature type is
listed only if the query authors label that population too (``readme_panels.MATURE``). The
heatmap keeps all 55 types, so uncalled types show as empty rows.

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
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import scanpy as sc

import tutorial_data as td
from readme_panels import (FS_TITLE, QSIZE, STYLE, UMAP_H, UMAP_W, box, heatmap_panel,
                           label_fraction, prediction_legend, umap_panel)
import viewmastr as vm

plt.rcParams.update(STYLE)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 20261103
FIG_W, FIG_H = 27, 12.6  # inches
RING = "Mega"  # author label of the query cluster ringed on both query UMAPs


def ring(ax, xy, pad=0.7):
    """Draw an ellipse around the points ``xy``, ``pad`` UMAP units outside them."""
    lo, hi = xy.min(0), xy.max(0)
    ax.add_patch(Ellipse((lo + hi) / 2, *(hi - lo + 2 * pad), fill=False, edgecolor="black",
                         linewidth=2.5, zorder=3))


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
    shown, legend_tab = prediction_legend(pred, d["author"])
    print(f"column 3 legend: {len(shown)} of {len(legend_tab)} types shown")
    print(legend_tab.to_string(index=False))

    n_query = len(pred)
    if args.panel:
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        umap_panel(box(fig, 0.3, 5.8, UMAP_W, UMAP_H), d["query_xy"], pred,
                   "(c) Query: viewmastR prediction", legend=shown, size=QSIZE)
        fig.savefig(OUT / "test_panel.png", dpi=150)
        return
    # Predicted reference label vs the query authors' label, fraction of each column
    tab, frac = label_fraction(pred, d["author"])
    print(tab.to_string())

    # Top row: reference labels, query author labels, query prediction (UMAPs with legends below).
    # Under the first two columns: the shared gene count.
    # Right: the heatmap.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cols_x = [0.3, 7.0, 13.7]
    umap_panel(box(fig, cols_x[0], 5.8, UMAP_W, UMAP_H), d["ref_xy"], d["ref_label"],
               f"(a) Reference: BoneMarrowMap\n({len(d['ref_label']):,} real cells)")
    umap_panel(box(fig, cols_x[1], 5.8, UMAP_W, UMAP_H), d["query_xy"], d["author"],
               f"(b) Query: authors' labels\n({n_query:,} cells)", size=QSIZE)
    umap_panel(box(fig, cols_x[2], 5.8, UMAP_W, UMAP_H), d["query_xy"], pred,
               "(c) Query: viewmastR prediction\n(reference labels)", legend=shown, size=QSIZE)
    ringed = d["query_xy"][d["author"] == RING]
    print(f"ring: {len(ringed)} {RING} cells, predicted",
          pred[d["author"] == RING].value_counts().to_dict())
    for ax in fig.axes[1:3]:
        ring(ax, ringed)
    fig.text((cols_x[0] + cols_x[1] + UMAP_W) / 2 / FIG_W, 0.5 / FIG_H,
             f"{d['n_genes']:,} common genes used for transfer", ha="center", va="center",
             fontsize=FS_TITLE, fontweight="bold")

    heatmap_panel(fig, frac, 23.0, 25.6, 1.5, 11.1, cbar=(25.85, 4.5, 0.18, 3.5))
    fig.savefig(OUT / "transfer_umap.png", dpi=150)
    # Vector copy for Illustrator: text and axes stay editable; the scatter layers are 300 dpi images
    fig.savefig(OUT / "transfer_umap.pdf", dpi=300)
    plt.close("all")
    print("wrote", *sorted(OUT.glob("transfer_umap.*")))


if __name__ == "__main__":
    main()
