"""Data loaders shared by the tutorials.

Public data download on first use into ``cache_dir``. The two local files (the Greenleaf
reference as h5ad, and the ScPCA project of tutorial 07) come from ``tutorials/data_paths.json``, which is not in git. Copy
``data_paths.example.json`` to ``data_paths.json`` and fill in your paths.

==================  ==========================================================  ==========
Dataset             Source                                                      Access
==================  ==========================================================  ==========
Greenleaf BM atlas  Granja et al. 2019 (GEO GSE139369), as an h5ad of counts    local path
BoneMarrowMap       Zeng et al. 2025, bone marrow atlas of 6 studies, 263,159     download
                    cells, 55 fine / 24 broad cell types, counts and UMAP
B-cell dev. atlas   Iacobucci et al. 2025 (PMC12377259) fig 3b, healthy B-cell   download
                    development, 130,085 cells; labels from its Symphony
                    reference, counts from the source studies (GEO GSE155259,
                    GSE166895; ArrayExpress E-MTAB-9389, E-MTAB-7407)
BM multiome RNA     Persad et al. 2023 (SEACells), T-cell-depleted bone marrow,   download
                    8,627 cells, 18 cell types, with the authors' UMAP. The
                    file the Mellon tutorial downloads (Otto et al. 2024)
CD34 multiome RNA   Persad et al. 2023 (SEACells), Zenodo 6383269,              download
                    CD34+ enriched bone marrow
Corces bulk RNA     Corces et al. 2016 (GEO GSE74246), counts from the          download
                    viewmastR R package repository
AML scRNA-seq       van Galen et al. 2019 (GEO GSE116256), Seq-Well, the 16     download
                    diagnosis (day 0) AML bone marrow samples
ScPCA SCPCP000003   pediatric T-ALL, AML and T/myeloid MPAL, 511,195 cells,     local path
                    the merged AnnData of ScPCA Portal project SCPCP000003
==================  ==========================================================  ==========
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/api/records/6383269/files/{}/content"
MELLON = ("https://fh-pi-setty-m-eco-public.s3.amazonaws.com/mellon-tutorial/"
          "preprocessed_t-cell-depleted-bm-rna.h5ad")
MELLON_MD5 = "64909a0be6cd51fe79b7d70b5f19bf1c"
BONEMARROWMAP = ("https://bonemarrowmap.s3.us-east-2.amazonaws.com/"
                 "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad")
BONEMARROWMAP_MD5 = "9cd413810545bb1bed79ad4634f3985c"
BDEV_SYMPHONY = "https://bdevelopmentmap.s3.us-east-2.amazonaws.com/BDevelopmentMap_SymphonyReference.rds"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
EBI = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-"
ENSEMBL98_GTF = "https://ftp.ensembl.org/pub/release-98/gtf/homo_sapiens/Homo_sapiens.GRCh38.98.gtf.gz"
# fig 3b state -> BoneMarrowMap CellType_Broad, by name through BoneMarrowMap's own fine -> broad map
BDEV_TO_BROAD = {
    "HSC/MPP": "HSC MPP", "MPP-MyLy": "HSC MPP", "LMPP": "LMPP", "Early GMP": "Early GMP",
    "MLP": "Early Lymphoid", "CLP": "Early Lymphoid", "Pre-Pro-B": "Early Lymphoid",
    "Pre-Pro-B Cycling": "Early Lymphoid", "Pro-B VDJ": "Pro-B", "Pro-B Cycling": "Pro-B",
    "Large Pre-B": "Pre-B", "Small Pre-B": "Pre-B", "Immature B": "B", "Mature B": "B",
    "Pre-pDC": "pDC", "Pre-pDC Cycling": "pDC", "pDC": "pDC",
}
VIEWMASTR_RAW = "https://raw.githubusercontent.com/furlan-lab/viewmastR/main/inst/extdata/deconvolute/{}"
VANGALEN = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116256/suppl/GSE116256_RAW.tar"


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
    are never augmented, so they give an unbiased accuracy for every model trained on ``train``.
    """
    rng = np.random.default_rng(seed)
    held = np.zeros(ref.n_obs, dtype=bool)
    for _, idx in ref.obs.groupby(label, observed=True).indices.items():
        held[rng.choice(idx, min(int(len(idx) * holdout), max_held), replace=False)] = True
    return ref[~held].copy(), ref[held].copy()


def _bdev_meta() -> pd.DataFrame:
    """Cell IDs, study, donor, tissue and labels of the fig 3b atlas (its Symphony reference).

    Reading the ``.rds`` file needs R (``Rscript`` on ``PATH``); the result is cached as CSV.
    """
    cache = Path(config()["cache_dir"])
    csv = cache / "bdev_meta.csv"
    if not csv.exists():
        rds = _download(BDEV_SYMPHONY, "BDevelopmentMap_SymphonyReference.rds")
        code = (f'm <- readRDS("{rds}")$meta_data; m$BDevelopment_CellType_Formatted <- NULL; '
                f'write.csv(m, "{csv}.part")')
        subprocess.run(["Rscript", "-e", code], check=True)
        Path(f"{csv}.part").rename(csv)
    return pd.read_csv(csv, index_col=0).set_index("Cell", drop=False)


def _old_to_new_symbols(genes_tsv: pd.DataFrame) -> dict:
    """Old Sanger symbols (Cell Ranger GRCh38-1.2.0) -> Ensembl 98 symbols (Cell Ranger 2020-A).

    Maps through the Ensembl IDs in ``genes_tsv`` (columns id, symbol). Only old symbols that
    are unique, and Ensembl 98 symbols that are unique, are kept.
    """
    rows = []
    with gzip.open(_download(ENSEMBL98_GTF, "Homo_sapiens.GRCh38.98.gtf.gz"), "rt") as fh:
        for line in fh:
            if "\tgene\t" in line:
                at = line.split("\t")[8]
                gid = re.search(r'gene_id "([^"]+)"', at).group(1)
                nm = re.search(r'gene_name "([^"]+)"', at)
                rows.append((gid, nm.group(1) if nm else gid))
    new = pd.DataFrame(rows, columns=["id", "new"]).drop_duplicates("id")
    old = genes_tsv.set_axis(["id", "old"], axis=1)
    old = old[~old.old.duplicated(keep=False)]
    mp = old.merge(new, on="id")
    mp = mp[~mp.new.duplicated(keep=False)]
    return dict(zip(mp.old, mp.new))


def _bdev_part(x, cells, genes, meta) -> ad.AnnData:
    obs = meta.loc[cells, ["Study", "Donor", "Tissue", "Technology", "Ontogeny", "BDevelopment_CellType"]].copy()
    obs["CellType_Broad"] = obs["BDevelopment_CellType"].map(BDEV_TO_BROAD)
    a = ad.AnnData(sp.csr_matrix(x, dtype=np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    a.var_names_make_unique()
    return a


def bdevelopment_added_cells() -> ad.AnnData:
    """Healthy fig 3b cells (Iacobucci et al. 2025, PMC12377259) that BoneMarrowMap does not contain.

    The fig 3b atlas has 130,085 cells from 10 public datasets. The paper deposits labels and
    cell IDs only (a Symphony reference), so counts come from each study's own deposit,
    matched by exact cell ID. 55,450 cells (Ainciburu, HCA bone marrow, Setty 2019, Oetjen)
    are already in BoneMarrowMap and are not added again. Not added: 1,181 HCA cord blood
    cells (hashed cell IDs) and 1,098 Popescu cells of donor F45 (not in E-MTAB-7407).

    ===============================  ======  ==================================
    Study                            Cells   Counts
    ===============================  ======  ==================================
    Jardine 2021, fetal bone marrow  26,407  E-MTAB-9389 ``fbm_tot_gex`` (14 GB)
    Jardine 2021, CITE-seq (2 sets)  16,470  GEO GSE166895 postQC mRNA raw
    Roy 2021, CD34+                  25,106  GEO GSE155259 h5ad
    Popescu 2019, fetal liver         4,373  E-MTAB-7407 per-sample matrices
    ===============================  ======  ==================================

    Old gene symbols (Jardine, Popescu) are renamed to Ensembl 98 through their Ensembl IDs.
    Returns raw counts on the genes every source shares, the fig 3b label
    (``BDevelopment_CellType``), its BoneMarrowMap broad label (``CellType_Broad``, by
    name), ``Study`` and ``Donor``. Cached as ``bdevelopment_added_cells.h5ad``.
    """
    cache = Path(config()["cache_dir"])
    out = cache / "bdevelopment_added_cells.h5ad"
    if out.exists():
        return ad.read_h5ad(out)
    import scipy.io
    meta = _bdev_meta()
    parts = {}
    part = cache / "bdevelopment"
    part.mkdir(exist_ok=True)

    def cached(name, build):
        f = part / name
        if not f.exists():
            build().write_h5ad(f, compression="gzip")
        return ad.read_h5ad(f)

    def roy():
        h5 = cache / "GSE155259_Roy_et_al.h5ad"
        if not h5.exists():
            gz = _download(f"{GEO}/GSE155nnn/GSE155259/suppl/GSE155259_Roy_et_al.h5ad.gz", "GSE155259_Roy_et_al.h5ad.gz")
            with gzip.open(gz, "rb") as fi, open(h5.with_suffix(".part"), "wb") as fo:
                while block := fi.read(1 << 24):
                    fo.write(block)
            h5.with_suffix(".part").rename(h5)
        a = ad.read_h5ad(h5)
        cells = meta.index[meta.Study == "RoyHSC_CD34p"]
        return _bdev_part(a[cells].X, cells, a.var["name"].astype(str), meta)

    def cite(fn, study):
        def build():
            df = pd.read_csv(_download(f"{GEO}/GSE166nnn/GSE166895/suppl/{fn}", fn), index_col=0)
            cells = meta.index[meta.Study == study]
            return _bdev_part(df.loc[cells].to_numpy(), cells, df.columns, meta)
        return build

    genes_tsv = None

    def popescu():
        nonlocal genes_tsv
        cells = meta.index[meta.Study == "Popescu_FetalLiver"]
        key = cells.to_series().str.extract(r"_((?:FCAImmP|4834STDY)\d+)_([ACGT]+)$")
        listing = urllib.request.urlopen(f"{EBI}/407/E-MTAB-7407/Files/").read().decode()
        avail = set(re.findall(r'href="([^"/]+)\.tar\.gz"', listing))
        key = key[key[0].isin(avail)]  # donor F45 is not in E-MTAB-7407
        xs, names = [], []
        for s, grp in key.groupby(0):
            with tarfile.open(_download(f"{EBI}/407/E-MTAB-7407/Files/{s}.tar.gz", f"bdevelopment/{s}.tar.gz")) as tf:
                mem = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
                mtx = scipy.io.mmread(tf.extractfile(mem["matrix.mtx"])).tocsc()
                g = pd.read_csv(tf.extractfile(mem["genes.tsv"]), sep="\t", header=None)
                bc = pd.read_csv(tf.extractfile(mem["barcodes.tsv"]), header=None)[0].str.replace("-1$", "", regex=True)
            genes_tsv = g if genes_tsv is None else genes_tsv
            idx = pd.Index(bc).get_indexer(grp[1])
            assert (idx >= 0).all(), f"{s}: barcodes missing"
            xs.append(mtx[:, idx].T.tocsr())
            names.extend(grp.index)
        return _bdev_part(sp.vstack(xs), pd.Index(names), genes_tsv[1].astype(str), meta)

    def jardine():
        f = _download(f"{EBI}/389/E-MTAB-9389/Files/fbm_tot_gex_20201104.csv", "fbm_tot_gex_20201104.csv")
        with open(f) as fh:
            col = pd.Index(fh.readline().rstrip("\n").split("\t")[1:])
        idx = np.sort(col.get_indexer(meta.index[meta.Study == "Jardine_FetalBoneMarrow"]))
        assert (idx >= 0).all()
        order = col[idx]
        blocks, genes = [], []
        for chunk in pd.read_csv(f, sep="\t", usecols=[0, *(idx + 1)], index_col=0, chunksize=1000,
                                 dtype={c: np.float32 for c in order}):
            genes.extend(chunk.index.astype(str))
            blocks.append(sp.csr_matrix(chunk[order].to_numpy()))
        return _bdev_part(sp.vstack(blocks).T, order, genes, meta)

    parts["roy"] = cached("new_roy.h5ad", roy)
    parts["cite_mnc"] = cached("new_Jardine_FetalBoneMarrow_CITEseq.h5ad",
                               cite("GSE166895_postQC_mRNAraw_FBM-MNCs.csv.gz", "Jardine_FetalBoneMarrow_CITEseq"))
    parts["cite_cd34"] = cached("new_Jardine_FetalCD34p_CITEseq.h5ad",
                                cite("GSE166895_postQC_mRNAraw_FL-FBM-CB.csv.gz", "Jardine_FetalCD34p_CITEseq"))
    parts["popescu"] = cached("new_popescu.h5ad", popescu)
    parts["jardine"] = cached("new_jardine.h5ad", jardine)
    if genes_tsv is None:
        t = next((cache / "bdevelopment").glob("*.tar.gz"))
        with tarfile.open(t) as tf:
            m = next(x for x in tf.getmembers() if x.name.endswith("genes.tsv"))
            genes_tsv = pd.read_csv(tf.extractfile(m), sep="\t", header=None)
    mp = _old_to_new_symbols(genes_tsv)
    for k in ("popescu", "jardine"):
        a = parts[k]
        keep = [g for g in a.var_names if g in mp]
        parts[k] = a[:, keep].copy()
        parts[k].var_names = [mp[g] for g in keep]
    shared = set.intersection(*(set(a.var_names) for a in parts.values()))
    genes = [g for g in parts["roy"].var_names if g in shared]
    a = ad.concat([p[:, genes] for p in parts.values()], merge="same")
    a.write_h5ad(out.with_suffix(".part"), compression="gzip")
    out.with_suffix(".part").rename(out)
    return a


def bonemarrowmap_bdev_reference() -> ad.AnnData:
    """BoneMarrowMap plus the fig 3b cells it lacks (:func:`bdevelopment_added_cells`).

    Genes = those both parts share. ``obs["source"]`` is ``"BoneMarrowMap"`` or
    ``"fig 3b added"``. Added cells have no position on the BoneMarrowMap UMAP (NaN).
    """
    bmm = bonemarrowmap_reference()
    add = bdevelopment_added_cells()
    genes = [g for g in bmm.var_names if g in set(add.var_names)]
    bmm = bmm[:, genes].copy()
    bmm.obs["source"] = "BoneMarrowMap"
    add = add[:, genes].copy()
    add.obs = add.obs[["CellType_Broad", "Study", "BDevelopment_CellType"]].assign(source="fig 3b added")
    add.obsm["X_umap"] = np.full((add.n_obs, 2), np.nan, dtype=np.float32)
    ref = ad.concat([bmm, add], merge="same")
    ref.obs["Study"] = ref.obs["Study"].astype(str)
    return ref


def bonemarrowmap_augmented(label: str = "CellType_Broad", norm_number: int = 3000,
                            seed: int = 0, bdevelopment: bool = False) -> tuple[ad.AnnData, ad.AnnData]:
    """BoneMarrowMap reference balanced with ``viewmastr.augment_data``, and its held-out cells.

    Splits the atlas with :func:`bonemarrowmap_split`, then balances the training cells to
    ``norm_number`` per type (``prune=True``): larger types are down-sampled, smaller types
    get simulated cells. ``obs["simulated"]`` marks simulated cells; real cells keep the
    stored UMAP (simulated cells have NaN). Both parts are cached in ``cache_dir``.
    ``bdevelopment=True`` starts from :func:`bonemarrowmap_bdev_reference` (BoneMarrowMap plus
    the healthy fig 3b cells it lacks); ``obs["source"]`` then tells the two parts apart, and
    held-out cells come from both. Returns ``(augmented, held_out)``.
    """
    import viewmastr as vm

    cache = Path(config()["cache_dir"])
    stem = f"bonemarrowmap{'_bdev' if bdevelopment else ''}_{label}_n{norm_number}_s{seed}"
    paths = cache / f"{stem}_augmented.h5ad", cache / f"{stem}_heldout.h5ad"
    if not all(p.exists() for p in paths):
        ref = bonemarrowmap_bdev_reference() if bdevelopment else bonemarrowmap_reference()
        train, held = bonemarrowmap_split(ref, label, seed=seed)
        del ref
        aug = vm.augment_data(train, label, norm_number=norm_number, prune=True, seed=seed)
        aug.obs["simulated"] = aug.obs_names.str.startswith("stimcell_")
        umap = np.full((aug.n_obs, 2), np.nan, dtype=np.float32)
        real = ~aug.obs["simulated"].to_numpy()
        umap[real] = train[aug.obs_names[real]].obsm["X_umap"]
        aug.obsm["X_umap"] = umap
        keep = [label, "simulated"] + (["source"] if bdevelopment else [])
        aug.obs = aug.obs[[label, "simulated"]].join(train.obs[keep[2:]])[keep]
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


def vangalen_aml() -> ad.AnnData:
    """AML bone marrow at diagnosis (van Galen et al. 2019, GEO GSE116256), raw counts.

    The 16 day-0 AML samples of the Seq-Well data set (the healthy donors, later time points
    and cell lines are left out). ``.obs`` keeps ``sample`` and the authors' ``CellType``
    and ``PredictionRefined`` (``malignant`` or ``normal``). Cached as
    ``vangalen_aml_d0.h5ad``.
    """
    cache = Path(config()["cache_dir"])
    out = cache / "vangalen_aml_d0.h5ad"
    if out.exists():
        return ad.read_h5ad(out)
    parts = []
    with tarfile.open(_download(VANGALEN, "GSE116256_RAW.tar")) as tf:
        files = {}  # sample -> {"dem": member, "anno": member}; the two have different GSM numbers
        for m in tf.getmembers():
            hit = re.fullmatch(r"GSM\d+_(AML[^_]*-D0)\.(dem|anno)\.txt\.gz", m.name)
            if hit:
                files.setdefault(hit[1], {})[hit[2]] = m
        for sample, f in sorted(files.items()):
            x = pd.read_csv(gzip.open(tf.extractfile(f["dem"])), sep="\t", index_col=0)
            anno = pd.read_csv(gzip.open(tf.extractfile(f["anno"])), sep="\t", index_col=0)
            obs = anno.loc[x.columns, ["CellType", "PredictionRefined"]].assign(sample=sample)
            parts.append(ad.AnnData(sp.csr_matrix(x.T.to_numpy(dtype=np.float32)), obs=obs,
                                    var=pd.DataFrame(index=x.index.astype(str))))
    a = ad.concat(parts, join="inner")
    a.write_h5ad(out.with_suffix(".part"), compression="gzip")
    out.with_suffix(".part").rename(out)
    return a


def large_query() -> ad.AnnData:
    """The large query of the big-query tutorial: ScPCA project SCPCP000003, opened on disk.

    Pediatric leukaemia (T-ALL, AML and T/myeloid mixed phenotype acute leukaemia),
    511,195 cells. Download the project as merged AnnData from the ScPCA Portal
    (https://scpca.alexslemonade.org/projects/SCPCP000003; free account) and set
    ``scpca_SCPCP000003`` in ``data_paths.json`` to ``SCPCP000003_merged_rna.h5ad``.
    The file is read with ``backed="r"``, so the counts stay on disk. Gene symbols replace
    the Ensembl IDs as gene names, so the query matches the reference genes; a symbol
    shared by two or more genes keeps the Ensembl IDs.
    """
    query = ad.read_h5ad(config()["scpca_SCPCP000003"], backed="r")
    ids = query.var_names.to_series()
    symbol = query.var["gene_symbol"].astype(str).where(query.var["gene_symbol"].notna(), ids)
    query.var_names = symbol.where(~symbol.duplicated(keep=False), ids).to_numpy()
    return query
