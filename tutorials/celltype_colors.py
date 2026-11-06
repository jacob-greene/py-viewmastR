"""Colours of the BoneMarrowMap ``CellType`` labels (55 fine types), as in the README figure.

Fine types take the ``viewmastr.palette`` colour of the same population where there is one;
like types share a hue. ``set_colors(adata, key)`` stores them for scanpy plots, so every
tutorial and the README figure colour a type the same way.
"""

from __future__ import annotations

import colorsys

import matplotlib.colors
import numpy as np
import viewmastr as vm

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
# authors' NaiveB). Mature B: of the sRGB colours (step 5, CIELAB lightness 30-80) at least 25
# (Delta E 1976) from every other colour in the figure, the one farthest (CIEDE2000) from them all.
DISTINCT = {"Mature B": "#0000ff"}
COLOR.update(DISTINCT)
_taken.extend(DISTINCT.values())


def color(label) -> str:
    """Fine BoneMarrowMap types first, then the palette (query author labels)."""
    return COLOR.get(str(label)) or vm.palette.color(label)


def set_colors(adata, key: str) -> None:
    """Make ``adata.obs[key]`` categorical, in lineage order, and store its colours for scanpy."""
    col = adata.obs[key].astype(str)
    cats = [k for k in FINE_ORDER if k in set(col)] + sorted(set(col) - set(FINE_ORDER))
    adata.obs[key] = col.astype("category").cat.set_categories(cats)
    adata.uns[f"{key}_colors"] = [color(c) for c in cats]
