"""Score every tool's predictions and draw the benchmark figure.

| Metric | Cells | Meaning |
|---|---|---|
| Accuracy | held-out reference cells | fraction of cells given their reference label |
| Macro-F1 | held-out reference cells | F1 score per type, averaged over types with equal weight |
| Rare-type F1 | held-out reference cells | F1 of each of the 5 types with the fewest training cells |
| ARI, NMI | external query | agreement of the predicted labels with the query authors' labels. The two label sets differ, so these need no label mapping; 1 = the same partition |
| Train, predict | all | seconds, data loading excluded |
| Peak memory | all | maximum resident set size of the whole process (``/usr/bin/time -v``), loading included |

Writes ``$BENCH_DIR/results.csv``, ``$BENCH_DIR/rare_f1.csv`` and ``docs/img/benchmark.{png,pdf}`` (or ``$BENCH_FIG_DIR``).

    python benchmarks/summarize.py greenleaf bmm
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]})
TOOLS = {"viewmastr": "py-viewmastR", "viewmastr_aug": "py-viewmastR + augment",
         "viewmastr_r": "viewmastR (R)", "celltypist": "CellTypist", "scanvi": "scANVI", "scanvi_cpu": "scANVI (CPU)",
         "singler": "SingleR", "seurat": "Seurat"}
TASKS = {"greenleaf_sub5000": "Subset test (5,000 training cells)", "greenleaf": "Granja 2019 atlas (16 types) -> CD34+ BM",
         "bmm": "BoneMarrowMap CellType (55 types) -> T-cell-depleted BM"}
OUT = Path(os.environ.get("BENCH_FIG_DIR", Path(__file__).resolve().parents[1] / "docs" / "img"))


def peak_gb(path: Path) -> float:
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(m.group(1)) / 2**20 if m else np.nan


def score(task: str, d: Path):
    held = pd.read_csv(d / "held_obs.csv", index_col=0)["label"].astype(str)
    query = pd.read_csv(d / "query_obs.csv", index_col=0)["author"].astype(str)
    ntrain = pd.read_csv(d / "train_obs.csv", index_col=0)["label"].value_counts()
    rare = ntrain.sort_values(kind="stable").index[:5].tolist()
    rows, rare_rows = [], []
    for tool, name in TOOLS.items():
        f = d / f"pred_{tool}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f, dtype=str).set_index("cell")
        ph = p.loc[p["set"] == "held", "pred"].reindex(held.index.astype(str))
        pq = p.loc[p["set"] == "query", "pred"].reindex(query.index.astype(str))
        assert ph.notna().all() and pq.notna().all(), f"{tool}: missing predictions"
        t = json.loads((d / f"time_{tool}.json").read_text())
        f1 = f1_score(held, ph, labels=ntrain.index, average=None, zero_division=0)
        f1 = pd.Series(f1, index=ntrain.index)
        rows.append({"task": task, "tool": name, "device": t["device"].upper(),
                     "accuracy": (ph.to_numpy() == held.to_numpy()).mean(), "macro_f1": f1.mean(),
                     "query_ari": adjusted_rand_score(query, pq),
                     "query_nmi": normalized_mutual_info_score(query, pq),
                     "train_s": t["train_s"], "predict_s": t["predict_s"],
                     "peak_gb": peak_gb(d / f"rusage_{tool}.txt"), "n_train": t["n_train"]})
        rare_rows += [{"task": task, "tool": name, "type": k, "n_train": int(ntrain[k]), "f1": f1[k]}
                      for k in rare]
    return pd.DataFrame(rows), pd.DataFrame(rare_rows)


def main():
    bench = Path(os.environ["BENCH_DIR"])
    tasks = sys.argv[1:] or ["greenleaf", "bmm"]
    res, rare = map(pd.concat, zip(*[score(t, bench / t) for t in tasks]))
    res.to_csv(bench / "results.csv", index=False)
    rare.to_csv(bench / "rare_f1.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(3).to_string(index=False))
        print(rare.round(3).to_string(index=False))

    fig, axes = plt.subplots(len(tasks), 3, figsize=(15, 4.2 * len(tasks)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1.2, 1]})
    for i, task in enumerate(tasks):
        r = res[res["task"] == task].reset_index(drop=True)
        names = r["tool"].tolist()
        y = np.arange(len(names))
        # (a) accuracy, macro-F1, query ARI
        ax = axes[i, 0]
        for j, (col, lab, c) in enumerate([("accuracy", "Accuracy (held-out)", "#1f78b4"),
                                           ("macro_f1", "Macro-F1 (held-out)", "#a6cee3"),
                                           ("query_ari", "ARI vs query authors", "#33a02c")]):
            ax.barh(y + (j - 1) * 0.27, r[col], height=0.27, color=c, label=lab)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(f"({'abcdef'[3 * i]}) {TASKS[task]}", loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        # (b) F1 of the 5 rarest training types
        ax = axes[i, 1]
        m = rare[rare["task"] == task].pivot(index="tool", columns="type", values="f1").loc[names]
        nt = rare[rare["task"] == task].drop_duplicates("type").set_index("type")["n_train"]
        m = m[nt.sort_values().index]
        im = ax.imshow(m.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for (a, b), v in np.ndenumerate(m.to_numpy()):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.6 else "black")
        ax.set_xticks(range(m.shape[1]), [f"{k}\n(n={nt[k]:,})" for k in m.columns], fontsize=7)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(f"({'abcdef'[3 * i + 1]}) F1 of the 5 rarest training types", loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, label="F1")
        # (c) time and memory
        ax = axes[i, 2]
        tot = r["train_s"] + r["predict_s"]
        ax.scatter(tot, r["peak_gb"], s=40, c=["#e31a1c" if d == "GPU" else "#333333" for d in r["device"]])
        for k, n in enumerate(names):
            ax.annotate(n, (tot[k], r["peak_gb"][k]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.margins(x=0.25, y=0.15)
        ax.set_xlabel("Train + predict time (s, log scale)")
        ax.set_ylabel("Peak memory (GB)")
        ax.set_title(f"({'abcdef'[3 * i + 2]}) Cost (red = GPU, black = 8 CPU cores)", loc="left", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "benchmark.png", dpi=150)
    fig.savefig(OUT / "benchmark.pdf")
    print("wrote", OUT / "benchmark.png")


if __name__ == "__main__":
    main()
