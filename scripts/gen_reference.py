"""Write the API reference pages in docs/reference/, laid out like the R package index.

docs/reference/index.md lists every function by group, with the first sentence of its
docstring. Each function gets its own page, docs/reference/<name>.md. Run this after adding,
removing or renaming a public function, or after editing the first line of a docstring:

    python scripts/gen_reference.py

It reads the source with griffe (installed with mkdocstrings) and does not import viewmastr,
so the compiled extension is not needed. Add new functions to GROUPS and to the Reference
section of ``nav`` in mkdocs.yml.
"""

import re
from pathlib import Path

import griffe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "reference"

GROUPS = {
    "Training and inference": [
        "viewmastR", "viewmastR_infer", "setup_training", "get_weights", "TrainingSet",
        "ViewmastrResult",
    ],
    "Feature selection": [
        "calculate_feature_dispersion", "select_features", "get_selected_features",
        "set_selected_features", "common_features", "plot_feature_dispersion",
    ],
    "Bulk references, augmentation and deconvolution": [
        "splat_bulk_reference", "augment_data", "deconvolve_bulk", "Deconvolution",
    ],
    "Plotting": ["confusion_matrix", "plot_training_data", "plot_deconvolution"],
    "Colours": ["palette"],
    "Model files": ["read_3file", "write_3file"],
}


def summary(obj) -> str:
    """First paragraph of the docstring on one line, with mkdocstrings cross-references made
    into plain code (the index table has its own links)."""
    text = obj.docstring.value.strip().split("\n\n")[0] if obj.docstring else ""
    text = " ".join(text.split())
    return re.sub(r"\[(`[^`]+`)\]\[[\w.]+\]", r"\1", text)


def main():
    pkg = griffe.load("viewmastr", search_paths=[ROOT / "python"], allow_inspection=False)
    OUT.mkdir(exist_ok=True)
    index = ["# Reference", "", "All functions are available from the top-level package: "
             "`import viewmastr as vm`.", ""]
    for group, names in GROUPS.items():
        index += [f"## {group}", "", "| Function | Description |", "|---|---|"]
        for name in names:
            obj = pkg[name]
            call = f"{name}()" if obj.is_function else name
            index.append(f"| [`{call}`]({name}.md) | {summary(obj)} |")
            (OUT / f"{name}.md").write_text(f"::: viewmastr.{name}\n")
        index.append("")
    (OUT / "index.md").write_text("\n".join(index))


if __name__ == "__main__":
    main()
