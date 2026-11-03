"""Measure label agreement between py-viewmastR and R viewmastR on the same query.

Agreement = share of query cells that get the same label. The script trains ``--models``
Python models on the R train split of ``--seed`` (R: ``set.seed(seed)`` before
``viewmastR()``), labels the query, and compares every Python model with every R run given.
It also reports R against R and Python against Python, the noise floor of each.

R predictions: one TSV per R run (``--r-pred``), first column = cell name, column
``prediction`` = R label. Write them with ``return_type = "list"`` and
``data.table::fwrite(data.frame(cell = colnames(obj), prediction = obj$viewmastR), ...)``.

Example::

    python scripts/parity_r.py --ref ref.h5ad --ref-col celltype --query query.h5ad \\
        --features features.txt --r-pred r_runs/pred_*.tsv.gz --seed 20261029 --models 5 \\
        --epochs 10 --out parity/
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import viewmastr as vm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-col", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--query-genes", help="query .var column with gene symbols (made unique as in R)")
    p.add_argument("--features", required=True, help="text file, one gene per line (R selected_features)")
    p.add_argument("--r-pred", required=True, nargs="+", help="R prediction TSVs (globs allowed)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--models", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = ad.read_h5ad(a.ref)
    query = ad.read_h5ad(a.query)
    if a.query_genes:
        query.var_names = pd.Index(query.var[a.query_genes].astype(str))
        query.var_names_make_unique(join=".")  # R make.unique(): GENE, GENE.1, GENE.2
    features = [l.strip() for l in open(a.features) if l.strip()]

    py = {}
    for m in range(a.models):
        q = query.copy()
        vm.viewmastR(q, ref, a.ref_col, selected_features=features, max_epochs=a.epochs, seed=a.seed,
                     workers=a.workers, model_dir=out / f"model_{m}", verbose=m == 0)
        py[f"py{m}"] = q.obs["viewmastR_pred"].astype(str)
    labels = pd.DataFrame(py)
    labels.to_csv(out / "python_labels.tsv.gz", sep="\t")

    files = sorted(f for g in a.r_pred for f in glob.glob(g))
    r = {Path(f).name.split(".")[0]: pd.read_csv(f, sep="\t", index_col=0, usecols=[0, 1]).iloc[:, 0].astype(str)
         for f in files}
    cells = labels.index
    for k, v in r.items():
        if not v.index.isin(cells).all() or len(v) != len(cells):
            raise SystemExit(f"R run {k} does not cover the same cells as the query")
    r = {k: v.reindex(cells).to_numpy() for k, v in r.items()}
    py = {k: v.to_numpy() for k, v in labels.items()}

    rows = [("R vs R", x, y, (r[x] == r[y]).mean()) for x, y in itertools.combinations(r, 2)]
    rows += [("Python vs Python", x, y, (py[x] == py[y]).mean()) for x, y in itertools.combinations(py, 2)]
    rows += [("Python vs R", x, y, (py[x] == r[y]).mean()) for x in py for y in r]
    pairs = pd.DataFrame(rows, columns=["comparison", "a", "b", "agreement"])
    pairs.to_csv(out / "agreement_pairs.tsv", sep="\t", index=False)
    summary = pairs.groupby("comparison").agreement.agg(["count", "mean", "min", "max"]).round(4)
    summary.to_csv(out / "agreement_summary.tsv", sep="\t")
    print(summary.to_string())


if __name__ == "__main__":
    main()
