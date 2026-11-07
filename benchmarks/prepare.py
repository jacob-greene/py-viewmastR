"""Build the benchmark inputs: the same cells, genes and labels for every tool.

Two tasks, each a reference split into training and held-out cells plus an external query:

| Task | Training cells | Held-out cells | External query |
|---|---|---|---|
| ``greenleaf`` | Granja et al. 2019 atlas: 80 % of the real cells of each type, plus simulated cells made from them | the other 20 % of the real cells | CD34+ bone marrow multiome RNA (Persad et al. 2023) |
| ``bmm`` | BoneMarrowMap ``CellType``, the real training cells of the README model | the README held-out cells | T-cell-depleted bone marrow multiome RNA (Persad et al. 2023) |

Genes: the variable genes shared by the training cells and the query, picked as in tutorial 01.
Every tool gets raw counts on these genes and normalises them its own way.

Writes ``$BENCH_DIR/<task>[_sub<N>]/``: ``{train,held,query}.h5ad`` (raw counts; ``label`` in
``.obs``, and ``author`` for the query), the same as Matrix Market files with ``*_obs.csv`` for
R, and ``genes.txt``. For ``bmm`` it also writes ``train_aug.h5ad``, the README's augmented
training set (real plus simulated cells), for the "viewmastR + augment" row.

    PYTHONPATH=tutorials python benchmarks/prepare.py greenleaf [--subsample 5000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.io
import scipy.sparse as sp

import tutorial_data as td
import viewmastr as vm


def counts(a):
    """Raw counts as float32 CSR, with ``label`` (and ``author``) the only ``.obs`` columns."""
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    return a


def write(a, out: Path, name: str):
    a.write_h5ad(out / f"{name}.h5ad")
    scipy.io.mmwrite(out / f"{name}.mtx", a.X.T.tocsc())  # genes x cells, as R expects
    a.obs.to_csv(out / f"{name}_obs.csv")


# Granja et al. 2019 ``BioClassification`` (25 clusters) -> the 16 benchmark types.
# The ``NN_`` prefix is dropped, then sub-clusters are merged; the two ``Unk`` clusters are left out.
GREENLEAF_MERGE = {"CLP.1": "CLP", "CLP.2": "CLP", "CD14.Mono.1": "CD14.Mono", "CD14.Mono.2": "CD14.Mono",
                   "Pre.B": "B", "CD8.N": "T", "CD4.N1": "T", "CD4.N2": "T", "CD4.M": "T",
                   "CD8.EM": "T", "CD8.CM": "T"}


def greenleaf_split(ref):
    """Real Greenleaf cells with 16 labels, split 80/20 per type; then simulated training cells.

    Labels come from the published ``BioClassification`` column. Simulated cells already in
    the file (``stimcell_*``) are dropped: they were made from all real cells, the held-out
    ones too. New simulated cells are made from the training cells only, after the split.
    """
    bio = ref.obs["BioClassification"].astype("string")
    real = bio.notna().to_numpy() & ~ref.obs_names.str.startswith("stimcell")
    real &= ~bio.fillna("").str.endswith("_Unk").to_numpy()
    ref = ref[real].copy()
    base = ref.obs["BioClassification"].astype(str).str.split("_", n=1).str[1]
    ref.obs["label"] = base.replace(GREENLEAF_MERGE).to_numpy()
    if "celltype" in ref.obs:  # the 21,854-cell copy carries the same 16 labels; check they agree
        assert (ref.obs["celltype"].astype(str) == ref.obs["label"]).all()
    assert ref.obs["label"].nunique() == 16
    ref.obs = ref.obs[["label"]]
    train, held = td.bonemarrowmap_split(ref, "label", holdout=0.2, max_held=10**9, seed=0)
    assert not set(train.obs_names) & set(held.obs_names)
    # The generator sees only the training cells, so no held-out molecule reaches training
    train = vm.augment_data(train, "label", norm_number=2000, seed=0)
    sim = train.obs_names.str.startswith("stimcell_")
    n_real = train.obs.loc[~sim, "label"].value_counts()
    n_sim = train.obs.loc[sim, "label"].value_counts().reindex(n_real.index, fill_value=0)
    assert ((n_sim == (2000 - n_real).clip(lower=0)) | (n_real >= 2000)).all()
    assert not set(train.obs_names[~sim]) & set(held.obs_names)
    return train, held


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["greenleaf", "bmm"])
    p.add_argument("--subsample", type=int, default=0, help="test run: N training cells")
    args = p.parse_args()
    out = Path(os.environ["BENCH_DIR"]) / (args.task + (f"_sub{args.subsample}" if args.subsample else ""))
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "greenleaf":
        ref = td.greenleaf_reference()
        train, held = greenleaf_split(ref)
        query = td.multiome_rna("cd34")
        aug = None
    else:
        aug, held = td.bonemarrowmap_augmented("CellType", norm_number=3000)
        train = aug[~aug.obs["simulated"].to_numpy()].copy()
        for a in (aug, train, held):
            a.obs["label"] = a.obs["CellType"].astype(str)
        query = td.bm_tcelldep()
    query.obs["author"] = query.obs["celltype"].astype(str)

    if args.subsample:
        rng = np.random.default_rng(0)
        pick = lambda a, n: a[np.sort(rng.choice(a.n_obs, min(n, a.n_obs), replace=False))].copy()
        train, held, query = pick(train, args.subsample), pick(held, 2000), pick(query, 2000)
        aug = None

    # Shared variable genes, as in tutorial 01 and the README figure
    for a in (query, train):
        vm.calculate_feature_dispersion(a)
        vm.select_features(a, top_n=10_000, logmean_ul=-1, logmean_ll=-8)
    in_train = set(vm.get_selected_features(train))
    genes = [g for g in vm.get_selected_features(query) if g in in_train and g in set(held.var_names)]
    (out / "genes.txt").write_text("\n".join(genes) + "\n")

    for name, a, cols in (("train", train, ["label"]), ("held", held, ["label"]),
                          ("query", query, ["author"]), ("train_aug", aug, ["label", "simulated"])):
        if a is None:
            continue
        b = counts(a[:, genes].copy())
        b.obs = a.obs[cols].copy()
        b.var = b.var.iloc[:, :0]
        b.obsm.clear() if hasattr(b.obsm, "clear") else None
        if name == "train_aug":
            b.write_h5ad(out / f"{name}.h5ad")  # Python only
        else:
            write(b, out, name)
        print(name, b.shape, "types:", b.obs[cols[0]].nunique())
    print(len(genes), "genes ->", out)


if __name__ == "__main__":
    main()
