# Development

```bash
uv venv && source .venv/bin/activate
uv pip install maturin pytest
maturin develop --release
pytest
```

| Path | Content |
|---|---|
| `python/viewmastr/` | the Python package |
| `rust/src/` | the burn backend: `classifier.rs` (training), `splat.rs` (bulk simulation), `deconv.rs` (deconvolution), `lib.rs` (Python binding) |
| `tests/` | known-answer tests on synthetic data |
| `scripts/parity_r.py` | agreement with R viewmastR on real data |
| `scripts/readme_figures.py` | the README figure, as a PNG and as an editable PDF (`docs/img/transfer_umap.pdf`) |
| `docs/` | longer documentation and the README images |
| `tutorials/` | the notebooks and their data loaders; `readme_panels.py` draws the README figure panels for both the script and tutorial 01 |


## Documentation site

The site is built with [MkDocs](https://www.mkdocs.org) and the Material theme. Notebooks are
shown with their stored outputs (mkdocs-jupyter, `execute: false`); the build never runs them.
The API reference is read from the docstrings (mkdocstrings), so it needs no compiled extension.

```bash
uv pip install mkdocs mkdocs-material mkdocs-jupyter "mkdocstrings[python]" ruff
mkdocs serve            # preview at http://127.0.0.1:8000
mkdocs build --strict   # what the workflow runs
```

`.github/workflows/docs.yml` builds the site and deploys it to GitHub Pages on every push to
`main`. In the repository settings, set Pages > Source to "GitHub Actions".
