# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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
