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

## Example

A model trained on a healthy bone marrow atlas (BoneMarrowMap, plus the healthy fetal and
CD34+ cells of the Iacobucci et al. 2025 B-cell development atlas) labels a separate bone
marrow dataset. The left panel shows the BoneMarrowMap cells on their stored UMAP.
The query is shown on its own published UMAP, coloured by its authors' labels (middle) and by
the viewmastR prediction (right). Colours are those of `viewmastr.palette`.

![Reference labels, query author labels and query predictions](docs/img/transfer_umap.png)

The query authors labelled these cells independently, with their own names. Each column is
one author label; colour is the fraction of its cells that get each predicted label. The two
vocabularies differ, so read it as a mapping between them, not as an accuracy.

![Predicted label against the query authors' label](docs/img/confusion_matrix.png)

| Item | Value |
|---|---|
| Reference | BoneMarrowMap atlas (Zeng et al. 2025), 24 broad cell types, plus 72,356 healthy cells that the Iacobucci et al. 2025 fig 3b atlas adds (counts from the source deposits); 3,000 cells per type after `augment_data()` (72,000 cells, 6,567 simulated) |
| Query | Persad et al. 2023 T-cell-depleted bone marrow multiome RNA, 8,627 cells, stored UMAP |
| Model | default multinomial logistic regression, 5 epochs |
| Accuracy on held-out real reference cells | 85.1 % (21,356 cells, never simulated) |

[`scripts/readme_figures.py`](scripts/readme_figures.py) makes both figures.

## Installation

py-viewmastR builds a small Rust extension, so it needs a Rust toolchain (1.86 or newer,
from [rustup](https://rustup.rs)).

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

# Train on genes that vary in both datasets
for a in (ref, query):
    vm.calculate_feature_dispersion(a)
    vm.select_features(a, top_n=10_000, logmean_ul=-1, logmean_ll=-8)
genes = set(vm.get_selected_features(ref))
vg = [g for g in vm.get_selected_features(query) if g in genes]

vm.viewmastR(query, ref, "celltype", selected_features=vg, return_probs=True)
query.obs["viewmastR_pred"]            # predicted cell type
```

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
- Persad S, Choo ZN, Dien C, et al. SEACells infers transcriptional and epigenomic cellular
  states from single-cell genomics data. *Nat Biotechnol* 41, 1746-1757 (2023). Zenodo
  record 6383269.
  The processed object with UMAP is the file the [Mellon](https://github.com/settylab/Mellon)
  tutorial downloads (Otto DJ, et al. *Nat Methods* 2024).

## Authors

Jacob Greene and Scott Furlan.

## License

MIT. The Rust code in `rust/src/` is adapted from viewmastR (MIT, Copyright 2026 Scott N
Furlan); see [LICENSE](LICENSE).
