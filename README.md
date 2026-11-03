# py-viewmastR

**Cell-type label transfer for single-cell RNA-seq, in Python, with the viewmastR burn backend.**

py-viewmastR is a Python interface to [viewmastR](https://github.com/furlan-lab/viewmastR),
the Furlan lab's R package that learns cell types from a labelled reference and predicts
them for a query. Models train in viewmastR's own Rust code (the
[burn](https://burn.dev) deep-learning framework), compiled as a Python extension. A
Python model and an R model trained on the same cells agree as closely as two R runs do.

- Works on [AnnData](https://anndata.readthedocs.io), including on-disk (backed) objects of
  millions of cells.
- Mirrors the R API: `viewmastR()`, `viewmastR_infer()`, feature selection, bulk references,
  class augmentation and bulk deconvolution.
- Model files use R viewmastR's format: Python labels data with R-trained models exactly as R did.
- `seed=` reproduces R's `set.seed()` train split exactly.

## Installation

py-viewmastR builds a small Rust extension, so it needs a Rust toolchain (1.86 or newer,
from [rustup](https://rustup.rs) or your cluster's module system).

```bash
uv pip install "git+https://github.com/jacob-greene/py-viewmastR"
# or, from a clone:
uv pip install .            # add ".[tutorials]" for scanpy and Jupyter
```

## Quick start

```python
import anndata as ad
import viewmastr as vm

ref = ad.read_h5ad("reference.h5ad")   # raw counts in .X, labels in .obs["celltype"]
query = ad.read_h5ad("query.h5ad")     # raw counts in .X

# Optional: train on genes that vary in both datasets
for a in (ref, query):
    vm.calculate_feature_dispersion(a)
    vm.select_features(a, top_n=10_000, logmean_ul=-1, logmean_ll=-8)
genes = set(vm.get_selected_features(ref))
vg = [g for g in vm.get_selected_features(query) if g in genes]

vm.viewmastR(query, ref, "celltype", selected_features=vg, return_probs=True)
query.obs["viewmastR_pred"]            # predicted cell type
```

## Functions

| R function | Python | Notes |
|---|---|---|
| `viewmastR()` | `viewmastR()` | `func="mlr"` (default), `"nn"` or `"nb"`; same defaults as R |
| `viewmastR_infer()` | `viewmastR_infer()` | reads Python and R model directories; works in row chunks |
| `setup_training()` | `setup_training()` | normalised train and validation matrices |
| `get_weights()` | `get_weights()` | genes x classes weights of an `mlr` model |
| `calculate_feature_dispersion()`, `select_features()`, `get_selected_features()`, `set_selected_features()`, `plot_feature_dispersion()` | same names | results in `adata.var` |
| `splat_bulk_reference()` | `splat_bulk_reference()` | simulates single cells from bulk profiles |
| `augment_data()` | `augment_data()` | balances rare cell types with simulated cells |
| `deconvolve_bulk()`, `plot_deconvolution()` | same names | gradient descent or EM |
| `confusion_matrix()`, `plot_training_data()` | same names | matplotlib |
| `make3file()` | `read_3file()`, `write_3file()` | exchange objects with R |

## Agreement with R viewmastR

Agreement is the share of query cells that get the same label. The test query is 6,942
healthy bone marrow cells (Caron et al. 2020, GEO GSE132509). The reference is the Greenleaf
atlas (16 types). All runs use the same features, 10 epochs, and the train split of
`set.seed(20261029)`. R is viewmastR 0.5.0 at commit b8acbff (the `main` branch on
2026-11-03).

| Comparison | Pairs | Mean | Min | Max |
|---|---|---|---|---|
| R run vs R run | 28 | 0.9915 | 0.9859 | 0.9952 |
| Python model vs R run | 40 | 0.9929 | 0.9863 | 0.9961 |
| Python model vs Python model | 10 | 0.9961 | 0.9942 | 0.9983 |

R runs differ from each other because burn's 16 data-loader threads deliver batches in a
slightly different order each time. Python inherits the same behaviour (`workers=16`).
`workers=1` with a `seed` makes a Python model fully reproducible.

Model files are interchangeable. Python labels the same query with an R-trained model
(R run `a` above) exactly as R did:

| Check | Result |
|---|---|
| Label agreement, Python inference vs R's own labels | 1.0000 (6,942 of 6,942 cells) |
| Largest logit difference | 2.3e-5 |

This uses `viewmastR_infer(..., sf_genes=<genes shared by reference and query>)`, the
size-factor genes R used at training time. Python writes the same files, and R's
`viewmastR_infer()` and `get_weights()` open them. R's `viewmastR_infer()` itself is not
reliable at b8acbff, though (see below), so use Python for saved-model inference.
[`scripts/parity_r.py`](scripts/parity_r.py) repeats this measurement on any data.

## Differences from the R package

| Topic | R viewmastR 0.5.0 | py-viewmastR |
|---|---|---|
| Backend | `nd`, `wgpu` or `candle` | `nd` (CPU) only |
| Saved-model inference | `viewmastR_infer()` takes size factors over the model features only, then divides the log values again by their column sums in Rust. On the Caron test, R's inference with R's own model (run `a`) matched R's training-time labels for 1.6 % of cells | the training-time normalisation; saved-model labels equal training-time labels |
| `augment_data()` | every simulated cell of a type gets the same library size (`newsizes[1]`) | each simulated cell gets its own drawn library size |
| `tf_idf`, `scale`, `LSImethod` | available | not ported (they change the feature set between reference and query) |
| Naive Bayes | linfa-bayes | the same model (multinomial, alpha = 1) in numpy |

## Tutorials

One notebook per article on the [viewmastR website](https://furlan-lab.github.io/viewmastR/).
They run on public data where possible; set local paths in `tutorials/data_paths.json`
(copy `data_paths.example.json`).

| R article | Python notebook | Data |
|---|---|---|
| [How to use viewmastR](https://furlan-lab.github.io/viewmastR/articles/HowTo.html) | [01_how_to](tutorials/01_how_to.ipynb) | Greenleaf reference; T-cell-depleted bone marrow multiome RNA |
| [A deeper look at viewmastR](https://furlan-lab.github.io/viewmastR/articles/InDepth.html) | [02_in_depth](tutorials/02_in_depth.ipynb) | Greenleaf reference; CD34+ bone marrow multiome RNA |
| [Bulk dataset as a reference](https://furlan-lab.github.io/viewmastR/articles/BulkReference.html) | [03_bulk_reference](tutorials/03_bulk_reference.ipynb) | Corces bulk RNA-seq; Greenleaf atlas; a large query of your own (h5ad) |
| [Augment data](https://furlan-lab.github.io/viewmastR/articles/Augment.html) | [04_augment](tutorials/04_augment.ipynb) | Greenleaf reference (fine labels); T-cell-depleted bone marrow |
| [Saved models](https://furlan-lab.github.io/viewmastR/articles/SavedModel.html) | [05_saved_model](tutorials/05_saved_model.ipynb) | Greenleaf reference; CD34+ bone marrow |
| [Extra functions](https://furlan-lab.github.io/viewmastR/articles/Extras.html) | [06_extras](tutorials/06_extras.ipynb) | CD34+ bone marrow |
| [Large query objects](https://furlan-lab.github.io/viewmastR/articles/BigQuery.html) | [07_big_query](tutorials/07_big_query.ipynb) | Greenleaf reference; a large query of your own (h5ad) |
| [Deconvolute bulk RNA-seq](https://furlan-lab.github.io/viewmastR/articles/Deconvolute.html) | [08_deconvolute](tutorials/08_deconvolute.ipynb) | Greenleaf signatures; Corces bulk RNA-seq |

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop --release
pytest
```

| Path | Content |
|---|---|
| `python/viewmastr/` | the Python package |
| `rust/src/` | the burn backend: `classifier.rs` (training), `splat.rs` (bulk simulation), `deconv.rs` (deconvolution), `lib.rs` (Python binding) |
| `tests/` | known-answer tests on synthetic data |
| `scripts/parity_r.py` | agreement with R viewmastR on real data |
| `tutorials/` | the notebooks and their data loaders |

## Citation

If you use py-viewmastR, please cite the viewmastR R package and the datasets you use:

- Furlan SN. viewmastR: cell-type classification for single-cell genomics.
  https://github.com/furlan-lab/viewmastR
- Greene J. py-viewmastR. https://github.com/jacob-greene/py-viewmastR
- Greenleaf reference: Granja JM, Klemm S, McGinnis LM, et al. Single-cell multiomic
  analysis identifies regulatory programs in mixed-phenotype acute leukemia. *Nat
  Biotechnol* 37, 1458-1465 (2019).
- Multiome bone marrow queries: Persad S, Choo ZN, Dien C, et al. SEACells infers
  transcriptional and epigenomic cellular states from single-cell genomics data. *Nat
  Biotechnol* 41, 1746-1757 (2023). Data: Zenodo record 6383269.
- Bulk haematopoiesis data: Corces MR, Buenrostro JD, Wu B, et al. Lineage-specific and
  single-cell chromatin accessibility charts human hematopoiesis and leukemia evolution.
  *Nat Genet* 48, 1193-1203 (2016).

## License

MIT. The Rust code in `rust/src/` is adapted from viewmastR (MIT, Copyright 2026 Scott N
Furlan); see [LICENSE](LICENSE).

## Author

Jacob Greene
