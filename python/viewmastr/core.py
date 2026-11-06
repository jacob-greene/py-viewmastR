"""Train a classifier on a labelled reference and label a query: ``viewmastR()`` and friends.

The public functions mirror the R package (furlan-lab/viewmastR 0.5.0) on AnnData objects.
Training runs in the R package's own burn code (see ``rust/``), so a Python model and an R
model trained on the same split differ only as two R runs differ.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _core
from ._data import NORM_METHODS, chunks, common_features, counts, labels_of, library_sizes, normalise, size_factors
from ._model import Model, softmax
from ._rrandom import r_sample


@dataclass
class TrainingSet:
    """Normalised reference cells split for training, as R ``setup_training()`` returns them.

    ``train_x`` / ``test_x`` are cells x features (CSR). ``sf_genes`` are the genes whose
    totals set the size factors; ``query_library`` holds the query totals over them.
    """

    train_x: sp.csr_matrix
    train_y: np.ndarray
    test_x: sp.csr_matrix
    test_y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    labels: list[str]
    features: list[str]
    sf_genes: list[str]
    query_library: np.ndarray | None = None


@dataclass
class ViewmastrResult:
    """What ``viewmastR(..., return_type="list")`` returns (R: ``object``, ``training_output``)."""

    adata: ad.AnnData | None
    history: dict = field(default_factory=dict)
    logits: pd.DataFrame | None = None
    probs: pd.DataFrame | None = None
    model_dir: str | None = None
    seconds: float = 0.0


def setup_training(
    query: ad.AnnData | None,
    ref: ad.AnnData,
    ref_celldata_col: str,
    selected_features: Sequence[str] | None = None,
    norm_method: str = "log",
    train_frac: float = 0.8,
    seed: int | None = None,
    layer: str | None = None,
    ref_layer: str | None = None,
    chunk_size: int = 50_000,
) -> TrainingSet:
    """Select features, normalise the reference and split it into train and test cells.

    Parameters
    ----------
    query, ref
        Query and reference AnnData with raw counts in ``.X`` (or ``layer`` / ``ref_layer``).
        ``query`` may be ``None`` (train only); it may be backed (``backed="r"``).
    ref_celldata_col
        Reference ``.obs`` column with the cell-type labels.
    selected_features
        Genes to train on. Only those present in both objects are used, in this order.
        ``None`` uses every shared gene.
    norm_method
        ``"log"`` (default), ``"size_only"``, ``"binary"`` or ``"none"``.
    train_frac
        Share of reference cells used for training; the rest validate.
    seed
        An integer draws exactly the split of R ``set.seed(seed)`` before ``viewmastR()``.
        ``None`` draws a random split, as R does without ``set.seed``.
    """
    if norm_method not in NORM_METHODS:
        raise ValueError(f"norm_method must be one of {NORM_METHODS}")
    shared = list(ref.var_names) if query is None else common_features(ref.var_names, query.var_names)
    if selected_features is None:
        features = shared
    else:
        keep = set(shared)
        features = [g for g in dict.fromkeys(selected_features) if g in keep]
    if not features:
        raise ValueError("no selected features are shared by the reference and the query")

    y, labels = labels_of(ref, ref_celldata_col)
    if (y < 0).any():
        raise ValueError(f"missing labels in ref.obs[{ref_celldata_col!r}]")
    sf = size_factors(library_sizes(ref, shared, ref_layer, chunk_size))
    x = counts(ref, ref_layer)[:, pd.Index(ref.var_names).get_indexer(features)]
    x = normalise(x, sf, norm_method)

    n = ref.n_obs
    size = int(round(train_frac * n))
    train_idx = r_sample(n, size, seed) if seed is not None else np.random.default_rng().choice(n, size, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)  # ascending, like R's which()
    q_lib = None if query is None else library_sizes(query, shared, layer, chunk_size)
    return TrainingSet(x[train_idx], y[train_idx], x[test_idx], y[test_idx], train_idx, test_idx,
                       labels, features, shared, q_lib)


def viewmastR(
    query: ad.AnnData | None,
    ref: ad.AnnData,
    ref_celldata_col: str,
    query_celldata_col: str = "viewmastR_pred",
    func: Literal["mlr", "nn", "nb"] = "mlr",
    norm_method: str = "log",
    selected_features: Sequence[str] | None = None,
    train_frac: float = 0.8,
    hidden_layers: Sequence[int] = (500, 100),
    learning_rate: float = 1e-3,
    max_epochs: int = 5,
    model_dir: str | Path | None = None,
    return_probs: bool = False,
    return_logits: bool = False,
    return_type: Literal["object", "list"] = "object",
    train_only: bool = False,
    seed: int | None = None,
    workers: int = 16,
    batch_size: int = 64,
    layer: str | None = None,
    ref_layer: str | None = None,
    chunk_size: int = 50_000,
    verbose: bool = True,
) -> ad.AnnData | ViewmastrResult:
    """Learn cell types from ``ref`` and predict them for ``query``.

    The defaults are those of R ``viewmastR()``. ``func`` selects the model: ``"mlr"``
    (multinomial logistic regression), ``"nn"`` (perceptron with one or two hidden layers,
    ``hidden_layers``) or ``"nb"`` (multinomial naive Bayes). Predictions go to
    ``query.obs[query_celldata_col]``; ``return_probs`` / ``return_logits`` add one
    ``prob_<label>`` / ``logit_<label>`` column per class.

    Python-only arguments: ``seed`` (see [`setup_training`][viewmastr.setup_training]),
    ``workers`` (burn data loader threads; R uses 16, and ``workers=1`` with a ``seed`` makes
    training fully reproducible), ``layer`` / ``ref_layer`` (count layers) and ``chunk_size``
    (query rows read at a time).

    Returns the query (``return_type="object"``) or a [`ViewmastrResult`][viewmastr.ViewmastrResult].
    """
    if train_only:
        query = None
    elif query is None:
        raise ValueError("query is required unless train_only=True")
    ts = setup_training(query, ref, ref_celldata_col, selected_features, norm_method, train_frac,
                        seed, layer, ref_layer, chunk_size)
    if verbose:
        print(f"viewmastR: {len(ts.features)} features, {len(ts.labels)} classes, "
              f"{len(ts.train_y)} train / {len(ts.test_y)} test cells")

    if func == "nb":
        if return_probs or return_logits:
            print("viewmastR: naive Bayes returns labels only, as in R")
        model = _naive_bayes(ts)
        result = ViewmastrResult(query, model_dir=None)
    else:
        if func not in ("mlr", "nn"):
            raise ValueError("func must be 'mlr', 'nn' or 'nb'")
        hidden = [] if func == "mlr" else [int(h) for h in hidden_layers]
        model_dir = Path(model_dir or tempfile.mkdtemp(prefix="viewmastr_"))
        model_dir.mkdir(parents=True, exist_ok=True)
        fit = _core.train_classifier(
            _dense(ts.train_x), ts.train_y, _dense(ts.test_x), ts.test_y, len(ts.labels), hidden,
            learning_rate, max_epochs, batch_size, workers, None if seed is None else int(seed),
            str(model_dir / "model"), verbose,
        )
        model = Model([(w, b) for w, b in fit["weights"]], ts.features, ts.labels, ts.sf_genes, fit["history"])
        model.save_meta(model_dir)
        result = ViewmastrResult(query, fit["history"], model_dir=str(model_dir), seconds=fit["seconds"])
    if query is None:
        return result

    sf = size_factors(ts.query_library)
    logits = _predict(query, model, ts.features, sf, norm_method, layer, chunk_size)
    _annotate(query, logits, model.labels, query_celldata_col, return_probs, return_logits)
    if return_type == "object":
        return query
    result.logits = pd.DataFrame(logits, index=query.obs_names, columns=model.labels)
    result.probs = pd.DataFrame(softmax(logits), index=query.obs_names, columns=model.labels)
    return result


def viewmastR_infer(
    query: ad.AnnData,
    model_dir: str | Path,
    selected_features: Sequence[str] | None = None,
    query_celldata_col: str = "viewmastR_inferred",
    labels: Sequence[str] | None = None,
    return_probs: bool = False,
    return_type: Literal["object", "list"] = "object",
    layer: str | None = None,
    chunk_size: int = 50_000,
    sf_genes: Sequence[str] | None = None,
) -> ad.AnnData | ViewmastrResult:
    """Label ``query`` with a saved model (from Python or R ``viewmastR``).

    The query is normalised exactly as during training, so inference with a saved model
    reproduces the labels ``viewmastR()`` gave the same query. Size factors use
    ``sf_genes`` if given, else the genes stored at training time (``viewmastr.json``),
    else (an R model) the model features. To match an R model's training-time labels,
    pass the genes the reference and query share as ``sf_genes``. ``chunk_size`` bounds
    memory on large or backed queries.
    """
    model = Model.load(model_dir)
    if selected_features is not None and list(selected_features) != model.features:
        raise ValueError("selected_features differ from the features the model was trained on")
    query_genes = set(query.var_names)
    missing = [g for g in model.features if g not in query_genes]
    if missing:
        raise ValueError(f"{len(missing)} model features are missing from the query, e.g. {missing[:5]}")
    if labels is not None:
        if len(labels) != len(model.labels):
            raise ValueError("labels has a different length from the model's classes")
        model.labels = list(labels)
    sf_genes = [g for g in (sf_genes or model.sf_genes or model.features) if g in query_genes]
    sf = size_factors(library_sizes(query, sf_genes, layer, chunk_size))
    logits = _predict(query, model, model.features, sf, "log", layer, chunk_size)
    _annotate(query, logits, model.labels, query_celldata_col, return_probs, False)
    if return_type == "object":
        return query
    return ViewmastrResult(query, model.history,
                           pd.DataFrame(logits, index=query.obs_names, columns=model.labels),
                           pd.DataFrame(softmax(logits), index=query.obs_names, columns=model.labels),
                           str(model_dir))


def get_weights(model_dir: str | Path) -> pd.DataFrame:
    """Weights of a saved mlr model: one row per feature, one column per class."""
    model = Model.load(model_dir)
    if model.kind != "mlr":
        raise ValueError("get_weights() reads mlr models only")
    return pd.DataFrame(model.layers[0][0], index=model.features, columns=model.labels)


# --- helpers ------------------------------------------------------------------------------

def _dense(x: sp.spmatrix) -> np.ndarray:
    return np.ascontiguousarray(x.toarray(), dtype=np.float32)


def _predict(query, model, features, sf, norm_method, layer, chunk_size) -> np.ndarray:
    """Logits for every query cell, normalising one row chunk at a time."""
    cols = pd.Index(query.var_names).get_indexer(features)
    out = []
    for rows in chunks(query.n_obs, chunk_size):
        x = normalise(counts(query, layer, rows)[:, cols], sf[rows], norm_method)
        out.append(model.logits(x.astype(np.float32)))
    return np.vstack(out)


def _annotate(query, logits, labels, column, return_probs, return_logits) -> None:
    query.obs[column] = pd.Categorical(np.asarray(labels)[logits.argmax(1)], categories=labels)
    extra = {}
    if return_logits:
        extra.update({f"logit_{l}": logits[:, k] for k, l in enumerate(labels)})
    if return_probs:
        p = softmax(logits)
        extra.update({f"prob_{l}": p[:, k] for k, l in enumerate(labels)})
    if extra:
        query.obs = query.obs.drop(columns=[c for c in extra if c in query.obs]).assign(**extra)


def _naive_bayes(ts: TrainingSet) -> Model:
    """Multinomial naive Bayes with Laplace smoothing (alpha = 1), as R's linfa-bayes call.

    Written as a linear model: logit = x . log P(feature | class) + log P(class).
    """
    k = len(ts.labels)
    y = sp.csr_matrix((np.ones_like(ts.train_y, dtype=np.float64), (ts.train_y, np.arange(len(ts.train_y)))),
                      shape=(k, len(ts.train_y)))
    feature_counts = np.asarray((y @ ts.train_x).todense()) + 1.0
    log_lik = np.log(feature_counts / feature_counts.sum(1, keepdims=True))
    prior = np.log(np.maximum(np.asarray(y.sum(1)).ravel(), 1e-300) / len(ts.train_y))
    return Model([(log_lik.T.astype(np.float32), prior.astype(np.float32))], ts.features, ts.labels, ts.sf_genes)
