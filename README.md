# py-viewmastR

**Cell-type label transfer for single-cell RNA-seq, in Python.**

py-viewmastR is a Python port of [viewmastR](https://github.com/furlan-lab/viewmastR), the
Furlan lab's R package. It learns cell types from a labelled reference dataset and predicts
them for a query dataset. Models train in viewmastR's own Rust code (the
[burn](https://burn.dev) deep-learning framework), compiled as a Python extension.

- Works on [AnnData](https://anndata.readthedocs.io), including on-disk (backed) objects of
  millions of cells.
- Mirrors the R API: `viewmastR()`, `viewmastR_infer()`, feature selection, bulk references,
  class augmentation and bulk deconvolution.
- Reads and writes R viewmastR model files. A Python model and an R model trained on the
  same cells agree as closely as two R runs do ([details](docs/r_comparison.md)).

## Installation

py-viewmastR builds a small Rust extension, so it needs a Rust toolchain (1.86 or newer,
from [rustup](https://rustup.rs)).

```bash
pip install "git+https://github.com/jacob-greene/py-viewmastR"
# or, from a clone:
pip install .               # add ".[tutorials]" for scanpy and Jupyter
```

## Quick start

```python
import anndata as ad
import viewmastr as vm

ref = ad.read_h5ad("reference.h5ad")   # raw counts in .X, labels in .obs["celltype"]
query = ad.read_h5ad("query.h5ad")     # raw counts in .X

# Train on genes that vary in both datasets
for a in (ref, query):
    vm.calculate_feature_dispersion(a)
    vm.select_features(a, top_n=10_000, logmean_ul=-1, logmean_ll=-8)
genes = set(vm.get_selected_features(ref))
vg = [g for g in vm.get_selected_features(query) if g in genes]

vm.viewmastR(query, ref, "celltype", selected_features=vg, return_probs=True)
query.obs["viewmastR_pred"]            # predicted cell type
```

## How it works

| Step | Function | What it does |
|---|---|---|
| Feature selection | `calculate_feature_dispersion()`, `select_features()` | Picks variable genes in each dataset; training uses the genes both share |
| Class balancing (optional) | `augment_data()` | Simulates cells for small types (and, with `prune=True`, down-samples large ones) to a set number per type |
| Training | `viewmastR()` | Trains a classifier on the reference (in Rust, with burn) and predicts the query; `return_probs=True` adds class probabilities |
| Inference | `viewmastR_infer()` | Labels new cells with a saved model |

## Example: refining the cell types of a bone marrow dataset

![ViewmastR: de novo celltyping and refinement based on transfer learning from reference annotations](docs/img/transfer_umap.png)

**ViewmastR: de novo celltyping and refinement based on transfer learning from reference
annotations.** We trained viewmastR on a healthy bone marrow atlas (BoneMarrowMap; Zeng et al.
2025) with its finest labels, `CellType` (55 types). We then used the model to label a
T-cell-depleted bone marrow multiome dataset (RNA only; Persad et al. 2023). The query authors
gave its 8,627 cells 18 labels. viewmastR gives each cell one of the 55 reference types, so it
splits several author labels into finer stages (table below).
(**a**) Reference cells on the atlas UMAP, coloured by `CellType`; only real cells are shown.
(**b**) Query cells on their published UMAP, coloured by the authors' labels.
(**c**) The same UMAP, coloured by the viewmastR prediction. The legend lists a type if at least
10 query cells get it. A mature type must also be one the authors label in the query. Together,
the two conditions hide types the query should not contain, such as T cells, NK cells, plasma
cells and CD16 monocytes.
(**d**) For each author label (column), the fraction of its cells that get each predicted type
(row). All 55 reference types are rows, so types the model never calls are empty rows. The two
label sets were made independently, so read this panel as a map between them, not as an accuracy.
Like cell types share a hue across panels. Mature B cells have their own colour, so that they
stand apart from the authors' naive B cells.

| Item | Value |
|---|---|
| Reference | BoneMarrowMap, `CellType`, 55 types; balanced to 3,000 training cells per type with `augment_data()` (165,000 cells: 118,509 real, 46,491 simulated) |
| Query | Persad et al. 2023 T-cell-depleted bone marrow multiome RNA, 8,627 cells, 18 author labels, published UMAP |
| Genes | 3,878 variable genes shared by reference and query |
| Model | multinomial logistic regression (the default), 5 epochs |
| Accuracy on held-out real reference cells | 72.9 % over 55 types (35,575 cells, never trained on or simulated) |
| Types predicted for at least one query cell | 43 of 55 |

Balancing helps rare types; it does not change overall accuracy. In a control (2 seeds, one
split), copies of real cells in place of simulated cells gave the same overall accuracy (73.0 %).
[`scripts/readme_figures.py`](scripts/readme_figures.py) makes the figure and documents its
colour rules.

## Documentation

| Topic | Where |
|---|---|
| Tutorials, one per article on the [viewmastR website](https://furlan-lab.github.io/viewmastR/) | [`tutorials/`](tutorials/README.md) |
| Function list, agreement with R, and differences from R | [`docs/r_comparison.md`](docs/r_comparison.md) |
| Building from source and running the tests | [`docs/development.md`](docs/development.md) |

## Citation

If you use py-viewmastR, please cite the viewmastR R package and the datasets you use:

- Furlan SN. viewmastR: cell-type classification for single-cell genomics.
  https://github.com/furlan-lab/viewmastR
- Greene J, Furlan SN. py-viewmastR. https://github.com/jacob-greene/py-viewmastR

The example above uses these public datasets:

- Zeng AGX, Iacobucci I, Shah S, et al. Single-cell transcriptional atlas of human
  hematopoiesis reveals genetic and hierarchy-based determinants of aberrant AML
  differentiation. *Blood Cancer Discov* 6, 307-324 (2025). doi:10.1158/2643-3230.BCD-24-0342.
  The annotated h5ad linked from https://github.com/andygxzeng/BoneMarrowMap.
- Persad S, Choo ZN, Dien C, et al. SEACells infers transcriptional and epigenomic cellular
  states from single-cell genomics data. *Nat Biotechnol* 41, 1746-1757 (2023). Zenodo
  record 6383269.
  The processed object with UMAP is the file the [Mellon](https://github.com/settylab/Mellon)
  tutorial downloads (Otto DJ, et al. *Nat Methods* 2024).

The tutorials also use these datasets:

- Iacobucci I, Zeng AGX, Gao Q, et al. Multipotent lineage potential in B cell acute
  lymphoblastic leukemia is associated with distinct cellular origins and clinical features.
  *Nat Cancer* 6, 1242-1262 (2025). doi:10.1038/s43018-025-00987-2. Fig 3b atlas cell IDs and
  labels from `BDevelopmentMap_SymphonyReference.rds`.
- Jardine L, Webb S, Goh I, et al. Blood and immune development in human fetal bone marrow
  and Down syndrome. *Nature* 598, 327-331 (2021). doi:10.1038/s41586-021-03929-x.
  ArrayExpress E-MTAB-9389; GEO GSE166895.
- Roy A, Wang G, Iskander D, et al. Transitions in lineage specification and gene regulatory
  networks in hematopoietic stem/progenitor cells over human development. *Cell Rep* 36,
  109698 (2021). doi:10.1016/j.celrep.2021.109698. GEO GSE155259.
- Popescu DM, Botting RA, Stephenson E, et al. Decoding human fetal liver haematopoiesis.
  *Nature* 574, 365-371 (2019). doi:10.1038/s41586-019-1652-y. ArrayExpress E-MTAB-7407.
- Granja JM, Klemm S, McGinnis LM, et al. Single-cell multiomic analysis identifies
  regulatory programs in mixed-phenotype acute leukemia. *Nat Biotechnol* 37, 1458-1465
  (2019). GEO GSE139369.
- van Galen P, Hovestadt V, Wadsworth MH 2nd, et al. Single-cell RNA-seq reveals AML
  hierarchies relevant to disease progression and immunity. *Cell* 176, 1265-1281.e24
  (2019). doi:10.1016/j.cell.2019.01.031. GEO GSE116256.
- Corces MR, Buenrostro JD, Wu B, et al. Lineage-specific and single-cell chromatin
  accessibility charts human hematopoiesis and leukemia evolution. *Nat Genet* 48,
  1193-1203 (2016). GEO GSE74246.

## Authors

Jacob Greene and Scott Furlan.

## License

MIT. The Rust code in `rust/src/` is adapted from viewmastR (MIT, Copyright 2026 Scott N
Furlan); see [LICENSE](LICENSE).
