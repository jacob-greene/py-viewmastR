"""Run one Python label-transfer tool on one benchmark task.

Each tool trains on ``train.h5ad`` and predicts ``held.h5ad`` and ``query.h5ad`` (raw counts on
the shared genes; see ``prepare.py``). Settings are each tool's documented defaults, except
where noted.

| Tool | Normalisation | Model |
|---|---|---|
| ``viewmastr`` | its own (log) | multinomial logistic regression, 5 epochs (the default) |
| ``viewmastr_aug`` | its own (log) | the same, trained on ``train_aug.h5ad`` (``bmm`` only) |
| ``celltypist`` | counts per 10,000, log1p | logistic regression, ``feature_selection=False`` (genes are fixed) |
| ``scanvi`` | none (raw counts) | scVI, then scANVI from it, on the CPU; predicts new cells without retraining |

Writes ``<task dir>/pred_<tool>.csv`` (cell, set, label) and ``time_<tool>.json`` (seconds to
train and to predict, data loading excluded).

    python benchmarks/run_python.py celltypist $BENCH_DIR/greenleaf --threads 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SEED = 0


def lognorm(a):
    import scanpy as sc
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    return b


def run_viewmastr(train, sets, threads, d):
    import viewmastr as vm
    genes = list(train.var_names)
    t0 = time.perf_counter()
    res = vm.viewmastR(None, train, "label", selected_features=genes, seed=SEED, workers=threads,
                       train_only=True, model_dir=str(d / "model_viewmastr"), return_type="list",
                       verbose=False)
    t1 = time.perf_counter()
    out = {}
    for name, a in sets.items():
        vm.viewmastR_infer(a, res.model_dir, selected_features=genes, query_celldata_col="pred")
        out[name] = a.obs["pred"].astype(str).to_numpy()
    return out, t1 - t0, time.perf_counter() - t1, "cpu"


def run_celltypist(train, sets, threads, d):
    import celltypist
    t0 = time.perf_counter()
    model = celltypist.train(lognorm(train), labels="label", feature_selection=False, n_jobs=threads,
                             check_expression=False, use_SGD=False)
    t1 = time.perf_counter()
    out = {n: celltypist.annotate(lognorm(a), model=model, majority_voting=False)
           .predicted_labels["predicted_labels"].astype(str).to_numpy() for n, a in sets.items()}
    return out, t1 - t0, time.perf_counter() - t1, "cpu"


def run_scanvi(train, sets, threads, d):
    import scvi
    import torch
    scvi.settings.seed = SEED
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    scvi.model.SCVI.setup_anndata(train, labels_key="label")
    vae = scvi.model.SCVI(train)
    vae.train(accelerator="cpu")
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown", labels_key="label")
    lvae.train(accelerator="cpu")
    t1 = time.perf_counter()
    out = {}
    for n, a in sets.items():
        a.obs["label"] = "Unknown"
        out[n] = np.asarray(lvae.predict(a)).astype(str)
    return out, t1 - t0, time.perf_counter() - t1, "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tool", choices=["viewmastr", "viewmastr_aug", "celltypist", "scanvi"])
    p.add_argument("task_dir", type=Path)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    d = args.task_dir
    train = ad.read_h5ad(d / ("train_aug.h5ad" if args.tool == "viewmastr_aug" else "train.h5ad"))
    sets = {n: ad.read_h5ad(d / f"{n}.h5ad") for n in ("held", "query")}
    for a in sets.values():
        a.obs = a.obs.iloc[:, :0].copy()  # no labels reach the predictor
    run = {"viewmastr": run_viewmastr, "viewmastr_aug": run_viewmastr,
           "celltypist": run_celltypist, "scanvi": run_scanvi}[args.tool]
    out, t_train, t_pred, device = run(train, sets, args.threads, d)
    pd.concat([pd.DataFrame({"cell": sets[n].obs_names, "set": n, "pred": out[n]}) for n in sets]) \
        .to_csv(d / f"pred_{args.tool}.csv", index=False)
    info = {"tool": args.tool, "train_s": t_train, "predict_s": t_pred, "device": device,
            "threads": args.threads, "n_train": train.n_obs}
    (d / f"time_{args.tool}.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
