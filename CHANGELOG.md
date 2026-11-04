# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `viewmastr.palette`: one colour per bone marrow cell type, used by every tutorial and
  README figure. Base colours are those of the T-cell-depleted bone marrow multiome object;
  the Granja et al. 2019 and Corces et al. 2016 labels map onto them. `plot_deconvolution()`
  uses it when every cell type has a colour.
- README figures: reference labels and query predictions on a shared UMAP, and a
  confusion matrix of predicted against author labels (`scripts/readme_figures.py`,
  `docs/img/`).

### Changed

- Shorter README. The function table, the R agreement measurements and the R differences
  moved to `docs/r_comparison.md`; the tutorial table to `tutorials/README.md`; build notes
  to `docs/development.md`.
- Scott Furlan is a package author (`pyproject.toml`, README, LICENSE).
- Tutorials 01 and 04 and the README figures use the T-cell-depleted bone marrow multiome
  object as published for the Mellon tutorial (8,627 cells), with its stored UMAP and labels.
  Nothing is re-normalised or re-embedded. `tutorial_data.bm_tcelldep()` replaces
  `bm_tcelldep_full()`.
- The README figures show the reference and the query on their own stored UMAPs, not on a
  new UMAP of the model genes.
- README: the R saved-model inference difference is the gene order of Seurat queries, not
  the normalisation. The normalisation differences are listed separately and are small.
- The `tutorials` extra pins `h5py<3.16`; newer h5py does not build on older glibc.

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
