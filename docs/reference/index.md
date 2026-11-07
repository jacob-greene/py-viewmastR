# Reference

All functions are available from the top-level package: `import viewmastr as vm`.

## Training and inference

| Function | Description |
|---|---|
| [`viewmastR()`](viewmastR.md) | Learn cell types from ``ref`` and predict them for ``query``. |
| [`viewmastR_infer()`](viewmastR_infer.md) | Label ``query`` with a saved model (from Python or R ``viewmastR``). |
| [`setup_training()`](setup_training.md) | Select features, normalise the reference and split it into train and test cells. |
| [`get_weights()`](get_weights.md) | Weights of a saved mlr model: one row per feature, one column per class. |
| [`TrainingSet`](TrainingSet.md) | Normalised reference cells split for training, as R ``setup_training()`` returns them. |
| [`ViewmastrResult`](ViewmastrResult.md) | What ``viewmastR(..., return_type="list")`` returns (R: ``object``, ``training_output``). |

## Feature selection

| Function | Description |
|---|---|
| [`calculate_feature_dispersion()`](calculate_feature_dispersion.md) | Per-gene log mean, log dispersion and the degree-``q`` polynomial fit between them. |
| [`select_features()`](select_features.md) | Mark genes for training in ``adata.var["selected_feature"]``. |
| [`get_selected_features()`](get_selected_features.md) | Genes marked by `select_features` or `set_selected_features`. |
| [`set_selected_features()`](set_selected_features.md) | Mark exactly ``genes`` as selected. |
| [`common_features()`](common_features.md) | Genes present in every object, in the order of the first. |
| [`plot_feature_dispersion()`](plot_feature_dispersion.md) | Log dispersion against log mean, selected genes in red, with the fitted curve. |

## Bulk references, augmentation and deconvolution

| Function | Description |
|---|---|
| [`splat_bulk_reference()`](splat_bulk_reference.md) | Simulate ``N`` single cells from every bulk sample of ``ref``. |
| [`augment_data()`](augment_data.md) | Balance cell types to ``norm_number`` cells each. |
| [`deconvolve_bulk()`](deconvolve_bulk.md) | Estimate how much of each cell type (columns of ``signatures``) is in each bulk sample. |
| [`Deconvolution`](Deconvolution.md) | Result of `deconvolve_bulk`. |

## Plotting

| Function | Description |
|---|---|
| [`confusion_matrix()`](confusion_matrix.md) | Predicted (rows) against true (columns) labels, coloured by percent of each column. |
| [`plot_training_data()`](plot_training_data.md) | Accuracy and loss per epoch for training and validation cells. |
| [`plot_deconvolution()`](plot_deconvolution.md) | Diagnostic panels for a `Deconvolution`. |

## Colours

| Function | Description |
|---|---|
| [`palette`](palette.md) | One colour per bone marrow cell type, shared by every tutorial and README figure. |

## Model files

| Function | Description |
|---|---|
| [`read_3file()`](read_3file.md) | Read a ``3file`` folder (or its parent) written by R ``make3file()``. |
| [`write_3file()`](write_3file.md) | Write ``adata`` as ``<path>/3file`` so R can rebuild a Seurat object from it. |
