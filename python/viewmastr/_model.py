"""Saved models: read burn ``model.mpk`` files and run them in numpy.

A model directory holds the files R viewmastR writes, so models move between R and Python:

- ``model.mpk``: burn weights (MessagePack, ``NamedMpkFileRecorder``, full precision);
- ``meta.mpk``: ``feature_names`` and ``class_labels`` (MessagePack);
- ``viewmastr.json``: Python only; the genes that set the size factors at training time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import msgpack
import numpy as np


@dataclass
class Model:
    """Layer weights (W [in, out], b [out]) with ReLU between layers, plus their metadata."""

    layers: list[tuple[np.ndarray, np.ndarray]]
    features: list[str]
    labels: list[str]
    sf_genes: list[str] | None = None
    history: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return {1: "mlr", 2: "ann1", 3: "ann2"}[len(self.layers)]

    def logits(self, x) -> np.ndarray:
        """Class logits for normalised cells ``x`` (cells x features, dense or sparse)."""
        h = x
        for i, (w, b) in enumerate(self.layers):
            h = np.asarray(h @ w, dtype=np.float32) + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def save_meta(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        meta = {"feature_names": list(self.features), "class_labels": list(self.labels)}
        (d / "meta.mpk").write_bytes(msgpack.packb(meta))
        (d / "viewmastr.json").write_text(json.dumps({"sf_genes": self.sf_genes, "history": self.history}))

    @classmethod
    def load(cls, model_dir: str | Path) -> "Model":
        """Load a model written by Python or by R viewmastR."""
        d = Path(model_dir)
        item = msgpack.unpackb((d / "model.mpk").read_bytes(), raw=False)["item"]
        layers = []
        for name in ("linear1", "linear2", "linear3"):
            if name in item:
                layers.append((_tensor(item[name]["weight"]), _tensor(item[name]["bias"])))
        meta = msgpack.unpackb((d / "meta.mpk").read_bytes(), raw=False)
        side = json.loads((d / "viewmastr.json").read_text()) if (d / "viewmastr.json").exists() else {}
        return cls(layers, list(meta["feature_names"]), list(meta["class_labels"]),
                   side.get("sf_genes"), side.get("history", {}))


def _tensor(rec: dict) -> np.ndarray:
    """Decode one burn parameter record (raw little-endian bytes + shape + dtype)."""
    p = rec["param"]
    dtype = {"F32": np.float32, "F64": np.float64, "F16": np.float16}[str(p["dtype"]).upper()]
    return np.frombuffer(bytes(p["bytes"]), dtype=dtype).reshape(p["shape"]).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
