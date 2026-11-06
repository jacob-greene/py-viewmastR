# py-viewmastR and R viewmastR

This page lists the Python functions with their R counterparts, measures how closely the two packages agree, and lists where they differ.

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
| Python model vs R run | 40 | 0.9928 | 0.9866 | 0.9960 |
| Python model vs Python model | 10 | 0.9954 | 0.9934 | 0.9980 |

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
`viewmastR_infer()` and `get_weights()` open them. With a Seurat query, R's
`viewmastR_infer()` at b8acbff needs the query genes in the order of `selected_features`
(see below). With the query rows in that order, R labels the Caron query with each of 5
Python models; agreement with Python's own labels is 99.5 % to 99.6 %. Python matches
genes by name, so it does not need the rows in any order.
[`scripts/parity_r.py`](../scripts/parity_r.py) repeats this measurement on any data.

## Differences from the R package

| Topic | R viewmastR 0.5.0 | py-viewmastR |
|---|---|---|
| Backend | `nd`, `wgpu` or `candle` | `nd` (CPU) only |
| Saved-model inference, gene order | For a Seurat query, `viewmastR_infer()` subsets with `query[selected_features, ]` (`R/inference.R` line 53 at b8acbff). That keeps the object's row order, and the matrix goes to Rust by position. So when `selected_features` is not in query row order, each model weight meets the wrong gene. On the Caron test with R model `a`, agreement with R's training-time labels was 1.6 % with the query rows as loaded, and 99.6 % with the rows pre-sorted into `selected_features` order. Workaround in R: reorder the query rows before the call | matches model features to query genes by name; row order does not matter |
| Saved-model inference, normalisation | size factors over the model features only, then a second division by column sums in Rust. Each changes few labels (0.4 % and 0.1 % of Caron cells) | the training-time normalisation; saved-model labels equal training-time labels |
| `augment_data()` | every simulated cell of a type gets the same library size (`newsizes[1]`) | each simulated cell gets its own drawn library size |
| `tf_idf`, `scale`, `LSImethod` | available | not ported (they change the feature set between reference and query) |
| Naive Bayes | linfa-bayes | the same model (multinomial, alpha = 1) in numpy |

