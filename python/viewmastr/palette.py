"""One colour per bone marrow cell type, shared by every tutorial and README figure.

The colours of the T-cell-depleted bone marrow multiome object (Persad et al. 2023,
``uns["celltype_colors"]``) are the base. Labels of the other datasets use the colour of
the same population under its name there (``ALIASES``). Labels with no counterpart got new
colours, chosen to be far (CIEDE2000) from every other colour here.

    import viewmastr as vm
    vm.palette.set_colors(adata, "viewmastR_pred")   # scanpy then uses these colours
    cols = vm.palette.colors(adata.obs["celltype"])  # label -> hex, for matplotlib
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

#: Base colours, keyed by the multiome object's labels, then the other atlases' own labels.
COLORS: dict[str, str] = {
    # T-cell-depleted bone marrow multiome object
    "HSC": "#e377c2",
    "HMP": "#8c564b",
    "CLP": "#ff7f0e",
    "preproB": "#c5b0d5",
    "proB": "#1f77b4",
    "preB": "#ad494a",
    "NaiveB": "#dbdb8d",
    "MyeloidPre": "#98df8a",
    "MonoPre": "#ffbb78",
    "Mono": "#aec7e8",
    "cDC": "#c49c94",
    "pDC": "#f7b6d2",
    "MEP": "#b5bd61",
    "Mega": "#17becf",
    "MylEry": "#ff9896",
    "EryPre1": "#d62728",
    "EryPre2": "#aa40fc",
    "Ery": "#279e68",
    # Granja et al. 2019 labels with no counterpart above
    "GMP.Neut": "#00ced1",
    "Early.Baso": "#000000",
    "CD16.Mono": "#9acd32",
    "Plasma": "#7b4173",
    "T": "#7f7f7f",
    "NK": "#393b79",
    # New colours
    "CD4.N": "#7b73a5",
    "CD4.M": "#4a4242",
    "CD8.N": "#107373",
    "CD8.CM": "#103a3a",
    "CD8.EM": "#004a10",
    "CMP": "#6b0019",
    "LMPP": "#7b6308",
    "Blast": "#3a3108",
    "LSC": "#4a7b00",
    "rHSC": "#2100ff",
    "Other": "#d9d9d9",
}

#: Other names for the same population (Granja et al. 2019; Corces et al. 2016 bulk).
ALIASES: dict[str, str] = {
    "CMP.LMPP": "HMP",
    "MPP": "HMP",
    "GMP": "MyeloidPre",
    "CD14.Mono": "Mono",
    "Early.Eryth": "EryPre1",
    "Late.Eryth": "Ery",
    "B": "NaiveB",
    "Bcell": "NaiveB",
    "Pre.B": "preB",
    "NKcell": "NK",
    "CD4Tcell": "CD4.N",
    "CD8Tcell": "CD8.N",
    "Blasts": "Blast",
}

_PREFIX = re.compile(r"^\d+_")  # "11_CD14.Mono" -> "CD14.Mono"


def canonical(label) -> str:
    """The palette name of ``label``: no numeric prefix, aliases resolved."""
    name = _PREFIX.sub("", str(label))
    return ALIASES.get(name, name)


def known(label) -> bool:
    """Whether ``label`` (or its alias) has its own colour."""
    return canonical(label) in COLORS


def color(label: str) -> str:
    """Colour of one label; unknown labels get the ``Other`` colour."""
    return COLORS.get(canonical(label), COLORS["Other"])


def colors(labels: Iterable) -> dict[str, str]:
    """Map each distinct label to its colour."""
    return {lab: color(lab) for lab in pd.unique(pd.Series(list(labels), dtype=str))}


def set_colors(adata, key: str) -> None:
    """Make ``adata.obs[key]`` categorical and store its colours in ``adata.uns[key + "_colors"]``."""
    col = adata.obs[key]
    if not isinstance(col.dtype, pd.CategoricalDtype):
        col = col.astype(str).astype("category")
    col = col.cat.remove_unused_categories()
    adata.obs[key] = col
    adata.uns[f"{key}_colors"] = [color(c) for c in col.cat.categories]
