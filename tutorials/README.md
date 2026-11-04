# Tutorials

One notebook per article on the [viewmastR website](https://furlan-lab.github.io/viewmastR/).
Each notebook follows one R article; the table maps them.

| R article | Python notebook | Data |
|---|---|---|
| [How to use viewmastR](https://furlan-lab.github.io/viewmastR/articles/HowTo.html) | [01_how_to](01_how_to.ipynb) | BoneMarrowMap atlas, augmented; T-cell-depleted bone marrow multiome RNA, 8,627 cells, stored UMAP |
| [A deeper look at viewmastR](https://furlan-lab.github.io/viewmastR/articles/InDepth.html) | [02_in_depth](02_in_depth.ipynb) | Greenleaf reference; CD34+ bone marrow multiome RNA |
| [Bulk dataset as a reference](https://furlan-lab.github.io/viewmastR/articles/BulkReference.html) | [03_bulk_reference](03_bulk_reference.ipynb) | Corces bulk RNA-seq; Greenleaf atlas; a large query of your own (h5ad) |
| [Augment data](https://furlan-lab.github.io/viewmastR/articles/Augment.html) | [04_augment](04_augment.ipynb) | BoneMarrowMap atlas (broad labels); T-cell-depleted bone marrow, 8,627 cells, stored UMAP |
| [Saved models](https://furlan-lab.github.io/viewmastR/articles/SavedModel.html) | [05_saved_model](05_saved_model.ipynb) | Greenleaf reference; CD34+ bone marrow |
| [Extra functions](https://furlan-lab.github.io/viewmastR/articles/Extras.html) | [06_extras](06_extras.ipynb) | CD34+ bone marrow |
| [Large query objects](https://furlan-lab.github.io/viewmastR/articles/BigQuery.html) | [07_big_query](07_big_query.ipynb) | Greenleaf reference; a large query of your own (h5ad) |
| [Deconvolute bulk RNA-seq](https://furlan-lab.github.io/viewmastR/articles/Deconvolute.html) | [08_deconvolute](08_deconvolute.ipynb) | Greenleaf signatures; Corces bulk RNA-seq |

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

## Data sources

| Dataset | Reference |
|---|---|
| Greenleaf bone marrow atlas | Granja JM, et al. *Nat Biotechnol* 37, 1458-1465 (2019). GEO GSE139369 |
| BoneMarrowMap atlas | Zeng AGX, et al. *Blood Cancer Discov* 6, 307-324 (2025). doi:10.1158/2643-3230.BCD-24-0342. The annotated h5ad linked from github.com/andygxzeng/BoneMarrowMap |
| Multiome bone marrow (T-cell-depleted and CD34+) | Persad S, et al. *Nat Biotechnol* 41, 1746-1757 (2023). Zenodo 6383269. The T-cell-depleted object with UMAP is the file the [Mellon](https://github.com/settylab/Mellon) tutorial downloads (Otto DJ, et al. *Nat Methods* 2024) |
| Bulk haematopoiesis RNA-seq | Corces MR, et al. *Nat Genet* 48, 1193-1203 (2016). GEO GSE74246 |
