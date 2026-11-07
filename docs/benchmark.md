# Benchmark against other label-transfer tools

py-viewmastR was compared with five other label-transfer tools on two tasks. Every tool got the same cells, the same genes and the same labels. The code is in `benchmarks/`.

## Terms

| Term | Meaning |
|---|---|
| Held-out cells | Reference cells kept out of training and used to score accuracy |
| Accuracy | Share of held-out cells that get their true label |
| Macro-F1 | F1 score per cell type, averaged over types with equal weight. Rare types count as much as common types |
| ARI, NMI | Adjusted Rand index and normalised mutual information. Each one scores how well the predicted query labels agree with the authors' query labels, with no label mapping (1 = the same partition). They are not accuracy, because the query authors used different type names |
| Simulated cells | Cells named `stimcell_*`, made by `augment_data()` from the training cells of one type. They top each `greenleaf` type up to 2,000 training cells |

## Tasks

| Task | Reference (training) | Held-out cells | Query | Genes |
|---|---|---|---|---|
| `greenleaf` | Granja et al. 2019 atlas, 16 types (see [Greenleaf source](#greenleaf-source)): 80 % of the real cells of each type (17,490 cells) plus 17,327 simulated cells made from them (34,817 cells) | the other 20 % of the real cells (4,364 cells) | CD34+ bone marrow multiome RNA, Persad et al. 2023 (Zenodo 6383269), 6,881 cells | 6,042 |
| `bmm` | BoneMarrowMap (Zeng et al. 2025), `CellType` with 55 types: the cells that are not held out (118,509 cells) | 20 % of each type, at most 1,000 per type (35,575 cells); the README uses the same split | T-cell-depleted bone marrow, Persad et al. 2023 (Zenodo 6383269), 8,627 cells | 3,867 |

The genes are the variable genes shared by the reference and the query, picked as in [How to use viewmastR](tutorials/01_how_to.ipynb). Every tool got raw counts on these genes and normalised them its own way.

On `greenleaf`, only real cells are held out. The simulated cells are made after the split, from the training cells only, so no molecule of a held-out cell reaches training. `prepare.py` asserts this.

### Greenleaf source

| Item | Detail |
|---|---|
| Public file | `scRNA-Healthy-Hematopoiesis-191120.rds` (Granja et al. 2019), linked from the GreenleafLab/MPAL-Single-Cell-2019 repository: `https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Data/scRNA-Healthy-Hematopoiesis-191120.rds` (md5 `8a98a0690009e0cd5c965baac7ca92d1`). It is a `SummarizedExperiment` with 35,582 cells, raw counts and the authors' `BioClassification` (25 clusters) |
| Labels | `prepare.py` makes the 16 labels from `BioClassification`. It drops the `NN_` prefix, merges CLP.1 and CLP.2, CD14.Mono.1 and CD14.Mono.2, and Pre.B into B, merges the six CD4 and CD8 clusters into T, and leaves out the two `Unk` clusters (681 cells) |
| Cells used here | A copy of that file with 21,854 of the 35,582 cells (some types down-sampled; the rule is not recorded). Every one of its cells is in the public file with the same counts and the same `BioClassification` |
| From the public file | Convert the `.rds` to an h5ad of counts with a `BioClassification` column. For example, write `assay(se, "counts")` with `Matrix::writeMM` in R, then build the AnnData in Python. Point `greenleaf_reference` in `tutorials/data_paths.json` at it. `prepare.py` then runs as written. It uses all 34,901 labelled cells, so the numbers differ from this page |

## Tools

| Tool | Version | Normalisation | Method |
|---|---|---|---|
| py-viewmastR | 0.1.0 | its own (log) | multinomial logistic regression, 5 epochs (the default) |
| py-viewmastR + augment | 0.1.0 | its own (log) | the same, trained on a reference balanced with `augment_data()` (`bmm` only) |
| viewmastR (R) | 0.5.0 | its own (log) | multinomial logistic regression, 5 epochs |
| CellTypist | 1.7.1 | counts per 10,000, log1p | logistic regression, `feature_selection=False` (the genes are fixed), no majority voting |
| scANVI (scvi-tools) | 1.5.2 | none (raw counts) | scVI, then scANVI from it, default epochs, on the CPU; predicts new cells without retraining |
| SingleR | 2.6.0 | counts per 10,000, log1p | `de.method = "wilcox"`, as advised for single-cell references |
| Seurat | 5.1.0 | `LogNormalize` | PCA (30 dimensions), `FindTransferAnchors` and `TransferData` |

All settings are each tool's documented defaults, except where the table says otherwise. Every run used 8 CPU cores on one node.

## Results

![Benchmark](img/benchmark.png)

*py-viewmastR is in colour and bold; the other tools are grey. (a, d) Accuracy and macro-F1 on held-out reference cells, and ARI against the query authors' labels. (b, e) F1 of the 5 types with the fewest real training cells; under each type, the number of real training cells and held-out cells. (c, f) Train plus predict time against peak memory, on 8 CPU cores.*

### Metrics

#### `greenleaf`

| Tool | Accuracy | Macro-F1 | ARI | NMI | Train (s) | Predict (s) | Peak memory (GB) |
|---|---|---|---|---|---|---|---|
| py-viewmastR | 0.918 | 0.882 | 0.585 | 0.619 | 21.5 | 0.5 | 2.4 |
| viewmastR (R) | 0.919 | 0.882 | 0.569 | 0.613 | 21.0 | 2.2 | 3.9 |
| CellTypist | 0.888 | 0.847 | 0.391 | 0.449 | 37.3 | 1.9 | 3.2 |
| scANVI | 0.913 | 0.884 | 0.485 | 0.542 | 3,966.7 | 0.3 | 1.9 |
| SingleR | 0.814 | 0.753 | 0.232 | 0.326 | 11.6 | 76.9 | 2.1 |
| Seurat | 0.926 | 0.901 | 0.576 | 0.621 | 40.6 | 60.6 | 7.8 |

#### `bmm`

| Tool | Accuracy | Macro-F1 | ARI | NMI | Train (s) | Predict (s) | Peak memory (GB) |
|---|---|---|---|---|---|---|---|
| py-viewmastR | 0.732 | 0.732 | 0.433 | 0.693 | 44.1 | 0.5 | 4.6 |
| py-viewmastR + augment | 0.732 | 0.738 | 0.470 | 0.699 | 57.1 | 0.5 | 6.3 |
| viewmastR (R) | 0.734 | 0.734 | 0.434 | 0.696 | 74.8 | 6.0 | 5.9 |
| CellTypist | 0.662 | 0.657 | 0.038 | 0.218 | 18,649.2 | 3.1 | 6.2 |
| scANVI | 0.729 | 0.726 | 0.492 | 0.660 | 1,580.7 | 0.6 | 3.2 |
| SingleR | 0.611 | 0.591 | 0.260 | 0.515 | 46.6 | 1,119.8 | 3.5 |
| Seurat | 0.683 | 0.678 | 0.622 | 0.710 | 87.6 | 204.8 | 17.2 |

### Rare types

Each column header gives the number of real training cells / held-out cells.

#### `greenleaf`: F1 of the 5 types with fewest real training cells

| Tool | Plasma (50 / 12) | Early.Baso (83 / 20) | CD16.Mono (234 / 58) | cDC (260 / 65) | Late.Eryth (354 / 88) |
|---|---|---|---|---|---|
| py-viewmastR | 0.96 | 0.63 | 0.90 | 0.82 | 0.82 |
| viewmastR (R) | 0.96 | 0.63 | 0.89 | 0.83 | 0.83 |
| CellTypist | 0.83 | 0.67 | 0.88 | 0.73 | 0.87 |
| scANVI | 0.96 | 0.68 | 0.91 | 0.84 | 0.84 |
| SingleR | 0.75 | 0.55 | 0.51 | 0.64 | 0.75 |
| Seurat | 0.92 | 0.77 | 0.91 | 0.88 | 0.90 |

#### `bmm`: F1 of the 5 types with fewest real training cells

| Tool | Megakaryocyte (44 / 10) | Stromal (171 / 42) | NK Proliferating (172 / 43) | cDC1 (296 / 74) | MLP-II (405 / 101) |
|---|---|---|---|---|---|
| py-viewmastR | 0.87 | 0.99 | 0.44 | 0.87 | 0.63 |
| py-viewmastR + augment | 0.77 | 0.99 | 0.65 | 0.88 | 0.67 |
| viewmastR (R) | 0.83 | 0.99 | 0.44 | 0.88 | 0.63 |
| CellTypist | 0.80 | 1.00 | 0.20 | 0.88 | 0.44 |
| scANVI | 0.74 | 0.99 | 0.58 | 0.85 | 0.55 |
| SingleR | 0.40 | 0.91 | 0.48 | 0.63 | 0.43 |
| Seurat | 0.87 | 1.00 | 0.56 | 0.61 | 0.54 |

## Conclusions

| Reading | `greenleaf` | `bmm` |
|---|---|---|
| py-viewmastR accuracy | 0.918 | 0.732 |
| Highest accuracy, any tool | 0.926 (Seurat) | 0.734 (viewmastR (R)) |
| py-viewmastR train + predict (s) | 22 | 45 |
| Next shortest train + predict (s) | 23 (viewmastR (R)) | 58 (py-viewmastR + augment), 81 (viewmastR (R)) |

py-viewmastR and R viewmastR gave the same accuracy and macro-F1 to within 0.002 on both tasks. On both tasks, py-viewmastR accuracy was within 0.01 of the best tool. py-viewmastR had the shortest train plus predict time on both tasks; on `greenleaf`, R viewmastR was as fast. On `bmm`, `augment_data()` raised macro-F1 from 0.732 to 0.738 and F1 of NK Proliferating from 0.44 to 0.65.

CellTypist on `bmm` took 5.2 hours to train with its default settings (no stochastic gradient descent). Its query ARI was 0.038, and it labelled 19 % of the query cells Stromal; the other tools labelled at most 0.3 % Stromal. Its `use_SGD=True` and `mini_batch=True` options for large references were not tested.

### Precision of the readings

| Reading | Limit |
|---|---|
| Accuracy, macro-F1 | One random 80/20 split per task; differences under about 0.01 are not resolved |
| Rare-type F1 | 10 to 101 held-out cells per type; with 10 cells, one cell moves F1 by up to about 0.1 |
| ARI, NMI | Agreement with the query authors' partition, not label correctness |
| Time | One run on a shared node; three py-viewmastR `greenleaf` trainings took 15.0, 15.6 and 21.5 s, so differences under about 1.5-fold are not resolved. Data loading is excluded |
| Peak memory | Includes data loading. SingleR runs in forked processes; its value is the largest single process |
| Threads | R viewmastR used its default data-loader threads, not 8 |

Seurat and SingleR do most of their work at prediction time, so their cost grows with each new query.

## Reproduce

```bash
export BENCH_DIR=/path/to/bench
PYTHONPATH=tutorials python benchmarks/prepare.py greenleaf      # and again with bmm
python benchmarks/run_python.py viewmastr $BENCH_DIR/greenleaf --threads 8
Rscript benchmarks/run_r.R seurat $BENCH_DIR/greenleaf 8
python benchmarks/summarize.py greenleaf bmm                    # scores, tables and the figure
```

`prepare.py` lists the source file of each dataset. The `bmm` data and both queries download on first use. For `greenleaf`, set `greenleaf_reference` in `tutorials/data_paths.json` first (see [Greenleaf source](#greenleaf-source)). `summarize.py` skips any tool with no prediction file.
