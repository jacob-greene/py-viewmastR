# Tutorials

One notebook per article on the [viewmastR website](https://furlan-lab.github.io/viewmastR/).
The table in the main [README](../README.md#tutorials) maps each R article to its notebook.

## Run them

```bash
uv pip install ".[tutorials]"
cp tutorials/data_paths.example.json tutorials/data_paths.json   # then edit the paths
cd tutorials && jupyter lab
```

| Key in `data_paths.json` | What it points to |
|---|---|
| `greenleaf_reference` | the Granja et al. 2019 bone marrow atlas as an h5ad of raw counts (cells x genes) |
| `greenleaf_label` | the `.obs` column with its cell types (default `celltype`) |
| `large_query` | any large h5ad of raw counts, for tutorials 03 and 07 |
| `large_query_gene_column` | `.var` column with gene symbols, if `var_names` are not symbols |
| `cache_dir` | where public data are downloaded on first use |

The public data (SEACells multiome RNA from Zenodo 6383269, and the Corces et al. 2016 bulk
counts shipped with viewmastR) download automatically. The rendered notebooks in this folder
were executed on a CPU node with 16 cores; the slowest (07)
took about 12 minutes.
