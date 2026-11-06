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
| `tutorials/` | the notebooks and their data loaders |

