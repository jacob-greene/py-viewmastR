"""Make the README figure from the tutorial data.

Reference: BoneMarrowMap healthy bone marrow atlas (Zeng et al. 2025) with its finest labels,
``CellType`` (55 types), balanced to 3,000 cells per type with ``augment_data``
(``tutorial_data.bonemarrowmap_augmented``). ``--no-augment`` drops the simulated cells and trains
on the real cells only. Accuracy is measured on held-out real reference cells.
Query: T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), 8,627 cells, loaded
as is with the authors' labels and UMAP (``tutorial_data.bm_tcelldep``).

Both datasets are shown on their own stored UMAP (real BoneMarrowMap cells only); nothing is
re-embedded. Colours come from ``viewmastr.palette``; a fine type with no palette colour of its
own gets the next colour of ``EXTRA`` (31 colours far from the palette in CIELAB).

Writes ``docs/img/transfer_umap.png``: three UMAPs and a heatmap of the predicted label against the
query authors' label. The prediction legend lists types called on at least 10 query cells; a mature type is
listed only if the query authors label that population too (``MATURE``). The heatmap keeps all 55
types, so uncalled types show as empty rows.

    PYTHONPATH=tutorials python scripts/readme_figures.py [--subsample N]

``--subsample N`` keeps N cells of each dataset, for a quick test run.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import tutorial_data as td
import viewmastr as vm

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
SEED = 20261103
TITLE = ("ViewmastR: de novo celltyping and refinement based on transfer learning from reference "
         "annotations")
QSIZE = 6  # dot area of the two query UMAPs (the 118,509-cell reference keeps 1.5)
ORDER = list(vm.palette.COLORS)
# BoneMarrowMap ``CellType``, by lineage
FINE_ORDER = [
    "HSC", "MPP-MyLy", "MPP-MkEry", "LMPP", "Cycling Progenitor", "MLP", "MLP-II", "CLP", "Pre-ProB",
    "Pro-B VDJ", "Pro-B Cycling", "Large Pre-B", "Small Pre-B", "Immature B", "Mature B", "Plasma Cell",
    "Early GMP", "GMP-Cycle", "GMP-Neut", "GMP-Mono", "Early ProMono", "Late ProMono", "CD14 Mono",
    "CD16 Mono", "Pre-cDC", "cDC1", "cDC2", "ASDC", "Pre-pDC", "Pre-pDC Cycling", "pDC",
    "EoBasoMast Precursor", "MEP", "Megakaryocyte Precursor", "Megakaryocyte", "BFU-E", "CFU-E",
    "Pro-Erythroblast", "Basophilic Erythroblast", "Polychromatic Erythroblast",
    "Orthochromatic Erythroblast", "CD4 Naive", "CD4 Central Memory", "CD4 Effector Memory",
    "CD4 Regulatory", "CD8 Naive", "CD8 Central Memory", "CD8 Effector Memory 1",
    "CD8 Effector Memory 2", "CD8 Tissue Resident Memory", "T Proliferating", "NK CD56high", "NK",
    "NK Proliferating", "Stromal",
]
# Fine types that take the palette colour of the same population (one fine type per colour)
FINE_ALIASES = {
    "MPP-MyLy": "HMP", "Pre-ProB": "preproB", "Pro-B VDJ": "proB", "Large Pre-B": "preB",
    "Mature B": "NaiveB", "Plasma Cell": "Plasma", "Early ProMono": "MonoPre", "CD14 Mono": "Mono",
    "CD16 Mono": "CD16.Mono", "cDC2": "cDC", "GMP-Neut": "GMP.Neut", "CD4 Naive": "CD4.N",
    "CD8 Naive": "CD8.N",
}
# Colours for fine types with no palette colour, in order of use. Picked by greedy farthest-point
# sampling in CIELAB (lightness 30-85, chroma < 65), each at least 20 (Delta E 1976) from every
# palette colour and from each other.
EXTRA = [
    "#b496ff", "#a54b00", "#a5005a", "#783ca5", "#00f0b4", "#1e69d2", "#ff6996", "#ff875a",
    "#c3961e", "#3cc3ff", "#69a5ff", "#69874b", "#96e1c3", "#b47896", "#f0b4ff", "#2d4b69",
    "#5a96b4", "#f0d25a", "#870f78", "#ff6969", "#961e0f", "#783c0f", "#878700", "#f0963c",
    "#00872d", "#c3785a", "#4b5a00", "#a569b4", "#4ba596", "#e1c396", "#872d4b",
]
# Fine types grouped under one query author label, so that like types in the reference and the
# query share a hue: the label of the same population by name if there is one, else the author
# label that most predicted cells of the type carry (the panel d cross-tab), else by lineage
# (Megakaryocyte, Basophilic Erythroblast: never called). The first type keeps the author label's
# palette colour; each other type gets the shade of it (hue within 0.04, HLS lightness within 0.3)
# farthest in CIELAB from every colour already used. Types in no group keep the colour set above.
FAMILY = {
    "HSC": ["HSC", "MPP-MkEry"],
    "HMP": ["MPP-MyLy", "LMPP", "Cycling Progenitor", "MLP"],
    "CLP": ["CLP", "MLP-II"],
    "preproB": ["Pre-ProB", "Pro-B VDJ"],
    "proB": ["Large Pre-B", "Pro-B Cycling"],
    "preB": ["Small Pre-B"],
    "NaiveB": ["Mature B", "Immature B"],
    "MyeloidPre": ["Early GMP", "GMP-Cycle"],
    "MonoPre": ["Early ProMono", "GMP-Mono"],
    "Mono": ["CD14 Mono", "Late ProMono", "CD16 Mono"],
    "cDC": ["cDC2", "Pre-cDC", "cDC1"],
    "pDC": ["pDC", "Pre-pDC", "Pre-pDC Cycling", "ASDC"],
    "MEP": ["MEP", "BFU-E"],
    "Mega": ["Megakaryocyte Precursor", "Megakaryocyte"],
    "EryPre1": ["Pro-Erythroblast", "CFU-E", "Basophilic Erythroblast"],
    "EryPre2": ["Polychromatic Erythroblast"],
    "Ery": ["Orthochromatic Erythroblast"],
}


def lab(hex_) -> np.ndarray:
    """CIELAB (D65) of a hex colour."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_))
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]]) @ rgb / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def shade(hex_, taken) -> str:
    """A shade of ``hex_`` (hue within 0.04, lightness within 0.3) farthest (CIELAB) from ``taken``."""
    h, l, s = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(hex_))
    cand = [matplotlib.colors.to_hex(colorsys.hls_to_rgb((h + dh) % 1, q, s * ds))
            for dh in (-0.04, -0.02, 0, 0.02, 0.04)
            for q in np.linspace(0.2, 0.88, 18) if abs(q - l) <= 0.3
            for ds in (0.6, 1.0)]
    pts = np.array([lab(c) for c in taken])
    return max(cand, key=lambda c: np.linalg.norm(pts - lab(c), axis=1).min())


_extra = iter(EXTRA)
COLOR = {}
for _k in FINE_ORDER:
    _a = FINE_ALIASES.get(_k, _k)
    COLOR[_k] = vm.palette.color(_a) if vm.palette.known(_a) else next(_extra)
_shaded = {t for ts in FAMILY.values() for t in ts[1:]}
COLOR.update({ts[0]: vm.palette.color(f) for f, ts in FAMILY.items()})
_taken = list(vm.palette.COLORS.values()) + [COLOR[k] for k in FINE_ORDER if k not in _shaded]
for _types in FAMILY.values():
    for _k in _types[1:]:
        COLOR[_k] = shade(COLOR[_types[0]], _taken)
        _taken.append(COLOR[_k])
# Types that must not share a hue with the author label of their group (Mature B against the query
# authors' NaiveB): each gets the unused ``EXTRA`` colour farthest in CIELAB from every colour in use.
DISTINCT = ["Mature B"]
for _k in DISTINCT:
    _free = [c for c in EXTRA if c not in COLOR.values()]
    _pts = np.array([lab(c) for c in _taken])
    COLOR[_k] = max(_free, key=lambda c: np.linalg.norm(_pts - lab(c), axis=1).min())
    _taken.append(COLOR[_k])
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


def color(label) -> str:
    """Fine BoneMarrowMap types first, then the palette (query author labels)."""
    return COLOR.get(str(label)) or vm.palette.color(label)


def lineage_order(labels) -> list[str]:
    """Labels in palette order (roughly by lineage); unknown labels last."""
    if set(map(str, labels)) <= set(FINE_ORDER):
        return [k for k in FINE_ORDER if k in set(map(str, labels))]
    rank = {k: i for i, k in enumerate(ORDER)}
    key = lambda x: (rank.get(vm.palette.canonical(x), len(ORDER)), x)
    return sorted(pd.unique(pd.Series(list(labels), dtype=str)), key=key)


def umap_panel(ax, xy, labels, title, legend=None, size=1.5):
    """UMAP coloured by label (dot area ``size``); the legend lists ``legend`` (default: every label)."""
    labels = np.asarray(labels, dtype=str)
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(xy[order, 0], xy[order, 1], s=size, linewidths=0, rasterized=True,
               c=[color(x) for x in labels[order]])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=color(k), label=k)
               for k in (lineage_order(labels) if legend is None else legend)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              frameon=False, fontsize=7, handletextpad=0.2, columnspacing=0.8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--no-augment", action="store_true", help="train on real reference cells only")
    p.add_argument("--panel", action="store_true", help="test: draw only the prediction panel")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

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

    # Each dataset on its own stored UMAP
    pred = query.obs["viewmastR_pred"].astype(str)
    # Column 3 legend: types called on at least MIN_CELLS query cells; a mature type also needs its
    # population among the query authors' labels. The heatmap keeps all types.
    n_called = pred.value_counts()
    authors = set(query.obs["celltype"].astype(str))
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

    if args.panel:
        fig, ax = plt.subplots(figsize=(6, 9))
        umap_panel(ax, np.asarray(query.obsm["X_umap"]), pred, "Query: viewmastR predictions",
                   legend=shown, size=QSIZE)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.36)
        fig.savefig(OUT / "test_panel.png", dpi=150)
        return
    # Predicted reference label vs the query authors' label, fraction of each column
    author = query.obs["celltype"].astype(str)
    tab = pd.crosstab(pred, author)
    tab = tab.reindex(index=FINE_ORDER, fill_value=0).loc[:, lineage_order(tab.columns)]
    frac = tab.div(tab.sum(axis=0), axis=1)
    print(tab.to_string())

    # One figure: reference labels, query author labels, query predictions, heatmap
    real = ref[~ref.obs["simulated"].to_numpy()]
    fig = plt.figure(figsize=(26, 10))
    gs = fig.add_gridspec(1, 3, left=0.005, right=0.70, top=0.90, bottom=0.36, wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    umap_panel(axes[0], np.asarray(real.obsm["X_umap"]), real.obs[label],
               f"(a) Reference: BoneMarrowMap CellType ({real.n_obs:,} real cells)")
    umap_panel(axes[1], np.asarray(query.obsm["X_umap"]), query.obs["celltype"],
               f"(b) Query: authors' labels ({query.n_obs:,} cells)", size=QSIZE)
    umap_panel(axes[2], np.asarray(query.obsm["X_umap"]), pred,
               "(c) Query: viewmastR predictions", legend=shown, size=QSIZE)
    x0, x1, y0, y1 = 0.800, 0.955, 0.16, 0.86
    ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(frac.shape[1]), frac.columns, rotation=90)
    ax.set_yticks(range(frac.shape[0]), frac.index)
    ax.tick_params(labelsize=7, length=0)
    ax.set_xlabel(f"Query authors' label ({query.n_obs:,} cells)")
    rows = fig.add_axes([x0 - 0.008, y0, 0.006, y1 - y0], sharey=ax)
    rows.imshow(np.array([[matplotlib.colors.to_rgb(color(k))] for k in frac.index]),
                aspect="auto", interpolation="nearest")
    cols = fig.add_axes([x0, y1 + 0.006, x1 - x0, 0.015], sharex=ax)
    cols.imshow(np.array([[matplotlib.colors.to_rgb(color(k)) for k in frac.columns]]),
                aspect="auto", interpolation="nearest")
    cols.set_title("(d) Prediction vs authors' label", fontsize=11, pad=8)
    for a in (rows, cols):
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in a.spines.values():
            sp.set_visible(False)
    # Row names sit left of the row colour strip
    ax.tick_params(labelleft=False)
    rows.tick_params(labelleft=True, labelsize=7, length=0)
    rows.set_ylabel("viewmastR prediction (reference label)")
    ax.xaxis.set_tick_params(labelbottom=True)
    cax = fig.add_axes([0.972, 0.55, 0.008, 0.2])
    fig.suptitle(TITLE, fontsize=16, y=0.985)
    fig.colorbar(im, cax=cax, label="Fraction of column")
    fig.savefig(OUT / "transfer_umap.png", dpi=150)
    plt.close("all")
    print("wrote", *sorted(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
