# Run one R label-transfer tool on one benchmark task.
#
# Each tool trains on train.mtx and predicts held.mtx and query.mtx (raw counts on the shared
# genes; see prepare.py). Settings are each tool's documented defaults, except where noted.
#
# | Tool        | Normalisation                         | Method                                                   |
# |-------------|---------------------------------------|----------------------------------------------------------|
# | singler     | counts per 10,000, log1p              | SingleR, de.method = "wilcox" (advised for single-cell references) |
# | seurat      | Seurat LogNormalize                   | PCA (30), FindTransferAnchors + TransferData             |
# | viewmastr_r | its own (log)                         | R viewmastR, multinomial logistic regression, 5 epochs   |
#
# Writes <task dir>/pred_<tool>.csv and time_<tool>.json, as run_python.py does.
#
#   Rscript benchmarks/run_r.R seurat $BENCH_DIR/greenleaf 8

args <- commandArgs(trailingOnly = TRUE)
tool <- args[1]; d <- args[2]; threads <- as.integer(args[3])
suppressPackageStartupMessages({library(Matrix); library(Seurat)})
set.seed(0)

read_set <- function(name) {
  m <- as(readMM(file.path(d, paste0(name, ".mtx"))), "CsparseMatrix")
  obs <- read.csv(file.path(d, paste0(name, "_obs.csv")), row.names = 1, check.names = FALSE)
  rownames(m) <- readLines(file.path(d, "genes.txt")); colnames(m) <- rownames(obs)
  list(m = m, obs = obs)
}
lognorm <- function(m) log1p(t(t(m) / pmax(colSums(m), 1)) * 1e4)

tr <- read_set("train"); sets <- list(held = read_set("held"), query = read_set("query"))
genes <- rownames(tr$m); labels <- tr$obs$label
out <- list(); device <- "cpu"

if (tool == "singler") {
  suppressPackageStartupMessages({library(SingleR); library(BiocParallel)})
  bp <- MulticoreParam(threads)
  t0 <- proc.time()[["elapsed"]]
  trained <- trainSingleR(lognorm(tr$m), labels = labels, de.method = "wilcox", BPPARAM = bp)
  t1 <- proc.time()[["elapsed"]]
  for (n in names(sets)) out[[n]] <- classifySingleR(lognorm(sets[[n]]$m), trained, BPPARAM = bp)$labels
} else if (tool == "seurat") {
  t0 <- proc.time()[["elapsed"]]
  ref <- CreateSeuratObject(tr$m, meta.data = tr$obs, min.cells = 0, min.features = 0)
  ref <- NormalizeData(ref, verbose = FALSE)
  VariableFeatures(ref) <- genes
  ref <- ScaleData(ref, verbose = FALSE)
  ref <- RunPCA(ref, npcs = 30, verbose = FALSE)
  t1 <- proc.time()[["elapsed"]]
  for (n in names(sets)) {
    q <- NormalizeData(CreateSeuratObject(sets[[n]]$m, min.cells = 0, min.features = 0), verbose = FALSE)
    anc <- FindTransferAnchors(reference = ref, query = q, features = genes, dims = 1:30,
                               reference.reduction = "pca", verbose = FALSE)
    out[[n]] <- as.character(TransferData(anc, refdata = ref$label, dims = 1:30, verbose = FALSE)$predicted.id)
  }
} else if (tool == "viewmastr_r") {
  suppressPackageStartupMessages(library(viewmastR))
  md <- file.path(d, "model_viewmastr_r")
  ref <- CreateSeuratObject(tr$m, meta.data = tr$obs, min.cells = 0, min.features = 0)
  t0 <- proc.time()[["elapsed"]]
  viewmastR(NULL, ref, "label", selected_features = genes, model_dir = md, train_only = TRUE,
            verbose = FALSE)
  t1 <- proc.time()[["elapsed"]]
  labs <- colnames(get_weights(md))
  for (n in names(sets)) {
    q <- CreateSeuratObject(sets[[n]]$m, min.cells = 0, min.features = 0)
    out[[n]] <- as.character(viewmastR_infer(q, model_dir = md, selected_features = genes, labels = labs,
                                             query_celldata_col = "pred", verbose = FALSE, threads = threads)$pred)
  }
} else stop("unknown tool ", tool)
t2 <- proc.time()[["elapsed"]]

pred <- do.call(rbind, lapply(names(sets), function(n)
  data.frame(cell = rownames(sets[[n]]$obs), set = n, pred = out[[n]])))
write.csv(pred, file.path(d, paste0("pred_", tool, ".csv")), row.names = FALSE, quote = TRUE)
writeLines(sprintf('{"tool": "%s", "train_s": %.2f, "predict_s": %.2f, "device": "%s", "threads": %d, "n_train": %d}',
                   tool, t1 - t0, t2 - t1, device, threads, ncol(tr$m)),
           file.path(d, paste0("time_", tool, ".json")))
cat(tool, "train", t1 - t0, "s, predict", t2 - t1, "s\n")
