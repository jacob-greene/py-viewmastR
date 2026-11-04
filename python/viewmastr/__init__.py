"""py-viewmastR: cell-type label transfer for single-cell RNA-seq, in Python.

A Python interface to viewmastR (Furlan lab, https://github.com/furlan-lab/viewmastR). Models
train in the R package's own Rust/burn code, so Python and R results are interchangeable.
"""

from . import palette
from .bulk import Deconvolution, augment_data, deconvolve_bulk, splat_bulk_reference
from .core import TrainingSet, ViewmastrResult, get_weights, setup_training, viewmastR, viewmastR_infer
from .features import (
    calculate_feature_dispersion,
    common_features,
    get_selected_features,
    plot_feature_dispersion,
    select_features,
    set_selected_features,
)
from .io import read_3file, write_3file
from .plotting import confusion_matrix, plot_deconvolution, plot_training_data

__version__ = "0.1.0"

__all__ = [
    "viewmastR", "viewmastR_infer", "setup_training", "get_weights", "TrainingSet", "ViewmastrResult",
    "calculate_feature_dispersion", "select_features", "get_selected_features", "set_selected_features",
    "plot_feature_dispersion", "common_features",
    "splat_bulk_reference", "augment_data", "deconvolve_bulk", "Deconvolution",
    "confusion_matrix", "plot_training_data", "plot_deconvolution",
    "read_3file", "write_3file", "palette",
]
