# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `viewmastr.palette`: one colour per bone marrow cell type, used by every tutorial and
  README figure. Base colours are those of the T-cell-depleted bone marrow multiome object;
  the Granja et al. 2019, Corces et al. 2016 and BoneMarrowMap broad labels map onto them.
  `plot_deconvolution()` uses it when every cell type has a colour.
- README figure: reference labels, query author labels and query predictions on their
  stored UMAPs, and the fraction of each author label given each predicted type
  (`scripts/readme_figures.py`; PNG and editable PDF in `docs/img/`).
- A documentation site (MkDocs), built from the README, the docs pages, the tutorials and
  the docstrings, and deployed to GitHub Pages on every push to `main`.
- A benchmark against five other label-transfer tools (`benchmarks/`, `docs/benchmark.md`).
- Tutorial loaders for the BoneMarrowMap atlas (`bonemarrowmap_reference()`,
  `bonemarrowmap_split()`, `bonemarrowmap_augmented()`), downloaded from its public URL and
  md5-checked.

### Changed

- Tutorials 01 and 04 and the README figure use BoneMarrowMap (Zeng et al. 2025) with its
  55 fine `CellType` labels as the reference, balanced with `augment_data()` to 3,000 cells
  per type. The query is the T-cell-depleted bone marrow multiome object as published for the
  Mellon tutorial (8,627 cells), with its stored UMAP and labels. Tutorial 01 trains once
  with a fixed seed and reproduces the README figure. Other tutorials keep the Granja et al.
  2019 reference.
- `confusion_matrix()` reports an accuracy only when one label set contains the other. Two
  vocabularies that share a few names (for example `cDC`) no longer get a meaningless accuracy.
- `viewmastR_infer()` builds the query gene set once, not once per gene.
- Shorter README with a workflow flowchart (`scripts/workflow_svg.py`) and a `uv pip`
  install. The function table, the R agreement measurements and the R differences are in
  `docs/r_comparison.md`; the tutorial table and dataset citations in `tutorials/README.md`;
  build notes in `docs/development.md`.
- Scott Furlan is a package author (`pyproject.toml`, README, LICENSE).
- The `tutorials` extra pins `h5py<3.16`; newer h5py does not build on older glibc.
- Tutorial 04 checks the simulated cells on the atlas's stored, study-corrected UMAP. A new
  UMAP of the raw counts split the six studies of the atlas apart. Its query plots and
  heatmap now use the README figure code.

## [0.1.0] - 2026-11-03

### Added

- `viewmastR()`, `viewmastR_infer()`, `setup_training()` and `get_weights()` on AnnData.
- Training through the viewmastR burn backend (burn 0.18, CPU), compiled as a Python
  extension with PyO3. Models: multinomial logistic regression (`mlr`), perceptrons with one
  or two hidden layers (`nn`), and multinomial naive Bayes (`nb`).
- `seed=` reproduces the R train split of `set.seed(seed)` exactly.
- Model directories (`model.mpk`, `meta.mpk`) that R and Python both read.
- Feature selection: `calculate_feature_dispersion()`, `select_features()`,
  `get_selected_features()`, `set_selected_features()`, `plot_feature_dispersion()`.
- Bulk tools: `splat_bulk_reference()`, `augment_data()` and `deconvolve_bulk()`
  (gradient descent and EM), with the R package's Rust simulator and solvers.
- Plots: `confusion_matrix()`, `plot_training_data()`, `plot_deconvolution()`.
- `read_3file()` / `write_3file()` for the R `make3file()` exchange format.
- Eight tutorials, one per article on the viewmastR website.
- Known-answer tests and an R parity script.
