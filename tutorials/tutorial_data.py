"""Data loaders shared by the tutorials.

Public data download on first use into ``cache_dir``. Local data (the Greenleaf reference
as h5ad, a large query) come from ``tutorials/data_paths.json``, which is not in git. Copy
``data_paths.example.json`` to ``data_paths.json`` and fill in your paths.

==================  ==========================================================  ==========
Dataset             Source                                                      Access
==================  ==========================================================  ==========
Greenleaf BM atlas  Granja et al. 2019 (GEO GSE139369), as an h5ad of counts    local path
BoneMarrowMap       Zeng et al. 2025, bone marrow atlas of 6 studies, 263,159     download
                    cells, 55 fine / 24 broad cell types, counts and UMAP
BM multiome RNA     Persad et al. 2023 (SEACells), T-cell-depleted bone marrow,   download
                    8,627 cells, 18 cell types, with the authors' UMAP. The
                    file the Mellon tutorial downloads (Otto et al. 2024)
CD34 multiome RNA   Persad et al. 2023 (SEACells), Zenodo 6383269,              download
                    CD34+ enriched bone marrow
Corces bulk RNA     Corces et al. 2016 (GEO GSE74246), as shipped with the      download
                    viewmastR R package
Large query         any large h5ad of raw counts (cells x genes)                local path
==================  ==========================================================  ==========
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/api/records/6383269/files/{}/content"
MELLON = ("https://fh-pi-setty-m-eco-public.s3.amazonaws.com/mellon-tutorial/"
          "preprocessed_t-cell-depleted-bm-rna.h5ad")
MELLON_MD5 = "64909a0be6cd51fe79b7d70b5f19bf1c"
BONEMARROWMAP = ("https://bonemarrowmap.s3.us-east-2.amazonaws.com/"
                 "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad")
BONEMARROWMAP_MD5 = "9cd413810545bb1bed79ad4634f3985c"
VIEWMASTR_RAW = "https://raw.githubusercontent.com/furlan-lab/viewmastR/main/inst/extdata/deconvolute/{}"


def config() -> dict:
    path = HERE / "data_paths.json"
    if not path.exists():
        raise FileNotFoundError(f"copy {HERE / 'data_paths.example.json'} to {path} and set your paths")
    cfg = json.loads(path.read_text())
    cfg.setdefault("cache_dir", str(HERE / "data"))
    return cfg


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, name: str, md5: str | None = None) -> Path:
    cache = Path(config()["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / name
    if not out.exists():
        print(f"downloading {name} ...")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if md5 and _md5(tmp) != md5:
            raise OSError(f"{name}: md5 mismatch, download again")
        tmp.rename(out)
    return out


def greenleaf_reference() -> ad.AnnData:
    """Healthy bone marrow atlas (Granja et al. 2019): counts in ``.X``, labels in ``.obs``."""
    return ad.read_h5ad(config()["greenleaf_reference"])


def multiome_rna(which: str) -> ad.AnnData:
    """SEACells multiome RNA, ``which`` = ``"bm"`` (T-cell depleted) or ``"cd34"`` (CD34+).

    Returns raw counts in ``.X`` (from ``.raw``), the authors' ``celltype`` labels and UMAP.
    Colours come from ``viewmastr.palette``, so the file's own colours are not kept.
    """
    a = ad.read_h5ad(_download(ZENODO.format(f"{which}_multiome_rna.h5ad"), f"{which}_multiome_rna.h5ad"))
    return ad.AnnData(a.raw.X.copy(), obs=a.obs[["celltype"]].copy(), var=pd.DataFrame(index=a.var_names),
                      obsm={"X_umap": a.obsm["X_umap"]})


def bm_tcelldep() -> ad.AnnData:
    """T-cell-depleted bone marrow multiome RNA (Persad et al. 2023), loaded as is.

    The file is the one the Mellon tutorial downloads (github.com/settylab/Mellon,
    ``notebooks/basic_tutorial.ipynb``; 2.5 GB). Its counts are those of
    ``bm_multiome_rna.h5ad`` in Zenodo 6383269, plus 1,188 B lineage cells, finer labels and
    a UMAP. Returns raw counts in ``.X`` (from ``.raw``; its columns are ``var_names`` in
    order), ``sample`` and ``celltype`` in ``.obs``, and the stored UMAP. Nothing is
    re-normalised or re-embedded.
    """
    a = ad.read_h5ad(_download(MELLON, "preprocessed_t-cell-depleted-bm-rna.h5ad", MELLON_MD5))
    return ad.AnnData(a.raw.X.copy(), obs=a.obs[["sample", "celltype"]].copy(),
                      var=pd.DataFrame(index=a.var_names), obsm={"X_umap": a.obsm["X_umap"]})


def bonemarrowmap_reference() -> ad.AnnData:
    """BoneMarrowMap healthy bone marrow atlas (Zeng et al. 2025), loaded as is.

    The annotated h5ad from github.com/andygxzeng/BoneMarrowMap (3.7 GB): 263,159 cells from
    six studies, 34,009 genes. Returns raw counts in ``.X``, ``CellType`` (55 types),
    ``CellType_Broad`` (24 types) and ``Study`` in ``.obs``, and the stored UMAP. Needs about
    12 GB of memory.
    """
    a = ad.read_h5ad(_download(BONEMARROWMAP, "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad",
                               BONEMARROWMAP_MD5))
    return ad.AnnData(a.X, obs=a.obs[["CellType", "CellType_Broad", "Study"]].copy(),
                      var=pd.DataFrame(index=a.var_names), obsm={"X_umap": a.obsm["X_umap"]})


def bonemarrowmap_split(ref: ad.AnnData, label: str = "CellType_Broad", holdout: float = 0.2,
                        max_held: int = 1000, seed: int = 0) -> tuple[ad.AnnData, ad.AnnData]:
    """Split real reference cells into (train, held out) per type.

    Of each type, ``holdout`` of the cells (at most ``max_held``) are held out. Held-out cells
    are never augmented, so they give an honest accuracy for every model trained on ``train``.
    """
    rng = np.random.default_rng(seed)
    held = np.zeros(ref.n_obs, dtype=bool)
    for _, idx in ref.obs.groupby(label, observed=True).indices.items():
        held[rng.choice(idx, min(int(len(idx) * holdout), max_held), replace=False)] = True
    return ref[~held].copy(), ref[held].copy()


def bonemarrowmap_augmented(label: str = "CellType_Broad", norm_number: int = 3000,
                            seed: int = 0) -> tuple[ad.AnnData, ad.AnnData]:
    """BoneMarrowMap reference balanced with ``viewmastr.augment_data``, and its held-out cells.

    Splits the atlas with :func:`bonemarrowmap_split`, then balances the training cells to
    ``norm_number`` per type (``prune=True``): larger types are down-sampled, smaller types
    get simulated cells. ``obs["simulated"]`` marks simulated cells; real cells keep the
    stored UMAP (simulated cells have NaN). Both parts are cached in ``cache_dir``.
    Returns ``(augmented, held_out)``.
    """
    import viewmastr as vm

    cache = Path(config()["cache_dir"])
    stem = f"bonemarrowmap_{label}_n{norm_number}_s{seed}"
    paths = cache / f"{stem}_augmented.h5ad", cache / f"{stem}_heldout.h5ad"
    if not all(p.exists() for p in paths):
        ref = bonemarrowmap_reference()
        train, held = bonemarrowmap_split(ref, label, seed=seed)
        del ref
        aug = vm.augment_data(train, label, norm_number=norm_number, prune=True, seed=seed)
        aug.obs["simulated"] = aug.obs_names.str.startswith("stimcell_")
        umap = np.full((aug.n_obs, 2), np.nan, dtype=np.float32)
        real = ~aug.obs["simulated"].to_numpy()
        umap[real] = train[aug.obs_names[real]].obsm["X_umap"]
        aug.obsm["X_umap"] = umap
        aug.obs = aug.obs[[label, "simulated"]]
        for a, p in zip((aug, held), paths):
            a.write_h5ad(p.with_suffix(".part"), compression="gzip")
            p.with_suffix(".part").rename(p)
    return ad.read_h5ad(paths[0]), ad.read_h5ad(paths[1])


def corces_bulk() -> pd.DataFrame:
    """Sorted-population bulk RNA-seq counts (genes x samples); columns are ``<donor>-<celltype>``."""
    path = _download(VIEWMASTR_RAW.format("GSE74246_RNAseq_All_Counts.txt"), "GSE74246_RNAseq_All_Counts.txt")
    return pd.read_csv(path, sep="\t", index_col=0)


def gene_weights() -> pd.DataFrame:
    """Gene lengths and deconvolution weights (``symbol``, ``length``, ``weight``)."""
    path = _download(VIEWMASTR_RAW.format("geneWeights_withSymbol.tsv"), "geneWeights_withSymbol.tsv")
    return pd.read_csv(path, sep="\t")


def large_query(backed: bool = True) -> ad.AnnData:
    """The large query of the big-query tutorial (raw counts), opened on disk by default."""
    return ad.read_h5ad(config()["large_query"], backed="r" if backed else None)
