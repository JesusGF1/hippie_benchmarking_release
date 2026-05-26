#!/usr/bin/env Rscript

# --- Progress logging helpers + perf CSV -------------------------------------
assign(".start_time", proc.time(), envir = .GlobalEnv)

# Event counter
assign(".__event_idx", 0L, envir = .GlobalEnv)

# In-memory performance log (data.frame)
assign(".__perf_log", data.frame(
  event_id      = integer(),
  kind          = character(),
  label         = character(),
  start_clock   = as.POSIXct(character()),
  end_clock     = as.POSIXct(character()),
  elapsed_s     = numeric(),
  rss_before_mb = numeric(),
  rss_after_mb  = numeric(),
  rss_delta_mb  = numeric(),
  status        = character(),
  notes         = character(),
  stringsAsFactors = FALSE
), envir = .GlobalEnv)

# Elapsed wall time since script start (s)
.time_since <- function() as.numeric((proc.time() - get(".start_time", envir = .GlobalEnv))[3])

# Best-effort resident memory (MB)
.mem_mb <- local({
  have_proc <- file.exists("/proc/self/status")
  function() {
    if (have_proc) {
      x <- tryCatch(readLines("/proc/self/status", warn = FALSE), error = function(e) character())
      line <- x[startsWith(x, "VmRSS:")]
      if (length(line)) {
        kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", line)))
        if (!is.na(kb)) return(kb / 1024)
      }
    }
    kb <- suppressWarnings(as.numeric(system("ps -o rss= -p $$", intern = TRUE)))
    if (!is.na(kb)) return(kb / 1024)
    NA_real_
  }
})

# Append a row to perf log
.__record_perf <- function(kind, label, t0, t1, rss0, rss1, status="ok", notes="") {
  .__event_idx <<- .__event_idx + 1L
  elapsed <- if (!is.null(t0) && !is.null(t1)) as.numeric(difftime(t1, t0, units = "secs")) else NA_real_

  safe_time <- function(x) {
    if (is.null(x) || length(x) == 0) return(as.POSIXct(NA))
    return(as.POSIXct(x))
  }
  safe_num <- function(x) {
    if (is.null(x) || length(x) == 0) return(NA_real_)
    return(as.numeric(x)[1])
  }

  row <- data.frame(
    event_id      = .__event_idx,
    kind          = as.character(kind)[1],
    label         = as.character(label)[1],
    start_clock   = safe_time(t0),
    end_clock     = safe_time(t1),
    elapsed_s     = safe_num(elapsed),
    rss_before_mb = safe_num(rss0),
    rss_after_mb  = safe_num(rss1),
    rss_delta_mb  = if (is.na(safe_num(rss0)) || is.na(safe_num(rss1))) NA_real_ else (safe_num(rss1) - safe_num(rss0)),
    status        = as.character(status)[1],
    notes         = as.character(notes)[1],
    stringsAsFactors = FALSE
  )

  assign(".__perf_log",
         rbind(get(".__perf_log", envir = .GlobalEnv), row),
         envir = .GlobalEnv)
  invisible(NULL)
}
# Pretty step printer (also logs a "marker" row)
.step <- local({
  i <- 0
  function(msg) {
    i <<- i + 1
    stamp <- format(Sys.time(), "%H:%M:%S")
    cat(sprintf("[%-2d][%s | +%6.1fs] %s\n", i, stamp, .time_since(), msg))
    flush.console()
    now <- Sys.time()
    rss <- .mem_mb()
    .__record_perf(kind="marker", label=msg, t0=now, t1=now, rss0=rss, rss1=rss,
                   status="ok", notes="step")
  }
})

# Timed evaluator
.timeit <- function(label, expr) {
  .step(paste0(label, " ..."))
  t_clock0 <- Sys.time()
  rss0 <- .mem_mb()
  t_ptm0 <- proc.time()
  res <- NULL
  status <- "ok"
  note <- ""
  tryCatch(
    {
      res <<- eval.parent(substitute(expr))
    },
    error = function(e) {
      status <<- "error"
      note <<- paste0("error: ", conditionMessage(e))
      stop(e)
    },
    finally = {
      t_ptm1 <- proc.time()
      t_clock1 <- Sys.time()
      rss1 <- .mem_mb()
      elapsed <- as.numeric((t_ptm1 - t_ptm0)["elapsed"])
      .__record_perf(kind="timeit", label=label, t0=t_clock0, t1=t_clock1,
                     rss0=rss0, rss1=rss1, status=status, notes=note)
      .step(sprintf("%s ✓ (%.2fs)", label, elapsed))
    }
  )
  invisible(res)
}

# --- CLI parser --------------------------------------------------------------
.parse_cli <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  for (a in args) {
    if (startsWith(a, "--")) {
      a2 <- sub("^--", "", a)
      if (grepl("=", a2, fixed = TRUE)) {
        parts <- strsplit(a2, "=", fixed = TRUE)[[1]]
        kv[[parts[1]]] <- parts[2]
      } else {
        kv[[a2]] <- TRUE
      }
    }
  }
  kv
}

cli <- .parse_cli()

# Defaults
# Shell runners cd to the script directory before sourcing, so getwd() is reliable
project_root <- normalizePath(file.path(getwd(), "..", ".."), mustWork = FALSE)
default_out_dir   <- file.path(project_root, "results", "physmap", "holdout")
default_root      <- file.path(project_root, "datasets")
default_dataset   <- "allen_scope_neuropixel_area_subset"

out_dir <- if (!is.null(cli[["out-dir"]])) cli[["out-dir"]] else
           if (nzchar(Sys.getenv("OUT_DIR"))) Sys.getenv("OUT_DIR") else
           default_out_dir

dataset_root <- if (!is.null(cli[["dataset-root"]])) cli[["dataset-root"]] else
                if (nzchar(Sys.getenv("DATASET_ROOT"))) Sys.getenv("DATASET_ROOT") else
                default_root

dataset_folder <- if (!is.null(cli[["dataset-folder"]])) cli[["dataset-folder"]] else
                  if (nzchar(Sys.getenv("DATASET_FOLDER"))) Sys.getenv("DATASET_FOLDER") else
                  default_dataset

# Holdout-specific parameters
n_holdout_folds <- if (!is.null(cli[["n-holdout-folds"]])) as.numeric(cli[["n-holdout-folds"]]) else 5
holdout_fold <- if (!is.null(cli[["holdout-fold"]])) as.numeric(cli[["holdout-fold"]]) else 0
min_cells_per_animal <- if (!is.null(cli[["min-cells-per-animal"]])) as.numeric(cli[["min-cells-per-animal"]]) else 50

# Grouping column for animal/session splits.
# Both Allen and IBL metadata files have a standardized 'animal_id' column
# (added 2026-04-12).  Fall back to dataset-specific names if missing.
grouping_col <- if (!is.null(cli[["grouping-col"]])) {
  cli[["grouping-col"]]
} else {
  "animal_id"
}

dataset_folder <- sub("^/+","", dataset_folder)
dataset_path <- file.path(dataset_root, dataset_folder)

# --- Libraries ----------------------------------------------------------------
.step("Loading libraries")
library(tidyr)
library(dplyr)
library(caret)
library(nnet)
library(reshape2)
library(Seurat)
library(aricode)
library(ggplot2)
library(patchwork)

# Balanced accuracy = mean per-class recall (matches sklearn.metrics.balanced_accuracy_score).
# Defined locally because the holdout script doesn't share helpers with physmap_script.r.
balanced_accuracy <- function(truth, pred) {
  truth <- as.character(truth)
  pred  <- as.character(pred)
  classes <- sort(unique(truth))
  recalls <- vapply(classes, function(cls) {
    in_cls <- truth == cls
    if (sum(in_cls) == 0) return(NA_real_)
    sum(pred[in_cls] == cls) / sum(in_cls)
  }, numeric(1))
  mean(recalls, na.rm = TRUE)
}

# --- Config -------------------------------------------------------------------
UMAP.SEED <- 3
ALGORITHM <- 3
dimV <- 1:5
n.comp <- 5
numPCs <- 5

.step(sprintf("Config: UMAP.SEED=%d | numPCs=%d | dims=%s | n.comp=%d",
              UMAP.SEED, numPCs, paste(dimV, collapse=","), n.comp))
.step(sprintf("Holdout config: n_folds=%d | current_fold=%d | min_cells_per_animal=%d",
              n_holdout_folds, holdout_fold, min_cells_per_animal))
.step(sprintf("Paths: dataset_root=%s | dataset_folder=%s | dataset_path=%s | out_dir=%s",
              dataset_root, dataset_folder, dataset_path, out_dir))

print("Dataset path")
print(dataset_path)

# Ensure output dirs exist (including the parent out_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold)),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)

.perf_csv_path <- file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold), "perf_log.csv")

# Always write perf log on exit
.on_exit_write_perf <- function() {
  .__record_perf(
    kind="marker",
    label=sprintf("Session end: R %s | Seurat %s | caret %s",
                  getRversion(), as.character(packageVersion("Seurat")),
                  as.character(packageVersion("caret"))),
    t0=Sys.time(), t1=Sys.time(),
    rss0=.mem_mb(), rss1=.mem_mb(),
    status="ok", notes="session"
  )
  try({ write.csv(get(".__perf_log", envir = .GlobalEnv), .perf_csv_path, row.names = FALSE) }, silent = TRUE)
}
reg.finalizer(environment(), function(e) .on_exit_write_perf(), onexit = TRUE)

# --- Working directory --------------------------------------------------------
.step(sprintf("Setting working directory: %s", out_dir))
setwd(out_dir)

# --- Load data ----------------------------------------------------------------
.step("Loading CSVs")
waveform <- read.csv(file.path(dataset_path, "waveforms.csv"))
isi <- read.csv(file.path(dataset_path, "isi_dist.csv"))
autocorr <- read.csv(file.path(dataset_path, "acg.csv"))
celltypes <- read.csv(file.path(dataset_path, "labels.csv"))
metadata <- read.csv(file.path(dataset_path, "metadata.csv"))

cType <- celltypes$label

print(ls())              # list objects available
print(dim(waveform))     # dimensions of waveform
print(dim(isi))          # dimensions of isi
print(dim(autocorr))    # dimensions of autocorr
print(dim(cType))       # dimensions of cType
print(dim(metadata))    # dimensions of metadata

# Coerce all columns to numeric (avoids accidental character columns)
numify <- function(df) {
  as.data.frame(lapply(df, function(x) as.numeric(as.character(x))), stringsAsFactors = FALSE)
}
waveform <- numify(waveform)
isi      <- numify(isi)
autocorr <- numify(autocorr)

rows_ok <- complete.cases(waveform) & complete.cases(isi) & complete.cases(autocorr)
if (!all(rows_ok)) {
  dropped <- sum(!rows_ok)
  .step(sprintf("Dropping %d cells with NA/NaN/Inf across modalities", dropped))
  waveform  <- waveform[rows_ok, , drop = FALSE]
  isi       <- isi[rows_ok, , drop = FALSE]
  autocorr  <- autocorr[rows_ok, , drop = FALSE]
  cType     <- cType[rows_ok]
  metadata  <- metadata[rows_ok, , drop = FALSE]
}
cType[is.na(cType) | cType == ""] <- "unknown"

# Drop zero-waveform units. These are units where the source NWB had no valid
# waveform data (e.g., ~7% of IBL brainwide_good). PCA on all-zero rows
# produces identical projections → KNN "too many ties" error.  HIPPIE handles
# these via its learned encoder; PhysMAP's PCA+UMAP pipeline cannot.
# The dropped unit IDs are logged so downstream analysis can report the gap.
if (!is.null(waveform)) {
  wf_rowsums <- rowSums(abs(as.matrix(waveform)))
  zero_wf <- wf_rowsums == 0
  n_zero_wf <- sum(zero_wf)
  if (n_zero_wf > 0) {
    .step(sprintf("Dropping %d zero-waveform units (%.1f%%) — PhysMAP requires non-zero WF for PCA",
                  n_zero_wf, 100 * n_zero_wf / length(zero_wf)))
    keep <- !zero_wf
    waveform <- waveform[keep, , drop = FALSE]
    isi      <- isi[keep, , drop = FALSE]
    autocorr <- autocorr[keep, , drop = FALSE]
    cType    <- cType[keep]
    metadata <- metadata[keep, , drop = FALSE]
  }
}

# Optional: fail fast if anything non-finite remains
stopifnot(all(is.finite(as.matrix(waveform))),
          all(is.finite(as.matrix(isi))),
          all(is.finite(as.matrix(autocorr))))

# --- Animal-based holdout split -----------------------------------------------
.step("Performing animal-based holdout split")

# Filter animals/sessions with sufficient cells
# Fall back to dataset-specific column if animal_id is missing
if (!grouping_col %in% colnames(metadata)) {
  fallback <- if (grepl("ibl", tolower(dataset_folder))) "subject" else "specimen_id"
  if (fallback %in% colnames(metadata)) {
    .step(sprintf("'%s' not in metadata; falling back to '%s'", grouping_col, fallback))
    grouping_col <- fallback
  } else {
    stop(sprintf(
      "Grouping column '%s' (or '%s') not found in metadata.csv. Available: %s",
      grouping_col, fallback, paste(colnames(metadata), collapse = ", ")
    ))
  }
}
.step(sprintf("Using grouping column: '%s'", grouping_col))

animal_counts <- table(metadata[[grouping_col]])
valid_animals <- names(animal_counts)[animal_counts >= min_cells_per_animal]
valid_indices <- which(metadata[[grouping_col]] %in% valid_animals)

# Update all data to only include cells from valid animals/sessions
waveform <- waveform[valid_indices, , drop = FALSE]
isi <- isi[valid_indices, , drop = FALSE]
autocorr <- autocorr[valid_indices, , drop = FALSE]
cType <- cType[valid_indices]
metadata <- metadata[valid_indices, , drop = FALSE]

.step(sprintf("Using %d groups with >= %d cells each", length(valid_animals), min_cells_per_animal))
.step(sprintf("Total cells after filtering: %d", nrow(metadata)))

# Create animal/session-based splits (reproducible)
set.seed(42)
shuffled_animals <- sample(valid_animals)

# Split into folds
fold_size <- length(valid_animals) %/% n_holdout_folds
fold_start <- holdout_fold * fold_size + 1
fold_end <- ifelse(holdout_fold < n_holdout_folds - 1,
                   fold_start + fold_size - 1,
                   length(valid_animals))

test_animals <- shuffled_animals[fold_start:fold_end]
train_animals <- shuffled_animals[-c(fold_start:fold_end)]

.step(sprintf("Train groups: %d, Test groups: %d", length(train_animals), length(test_animals)))

# Get indices for train/test split
train_indices <- which(metadata[[grouping_col]] %in% train_animals)
test_indices <- which(metadata[[grouping_col]] %in% test_animals)

.step(sprintf("Train cells: %d, Test cells: %d", length(train_indices), length(test_indices)))

# --- Prepare matrices for training animals only ------------------------------
.step("Preparing matrices and cell IDs for training set")
X_waveform_train <- as.matrix(waveform[train_indices, , drop = FALSE])
X_isi_train <- as.matrix(isi[train_indices, , drop = FALSE])
X_autocorr_train <- as.matrix(autocorr[train_indices, , drop = FALSE])
cType_train <- cType[train_indices]

cellIds_train <- seq_len(nrow(X_waveform_train))
rownames(X_waveform_train) <- cellIds_train
colnames(X_waveform_train) <- seq_len(ncol(X_waveform_train))

rownames(X_isi_train) <- cellIds_train
colnames(X_isi_train) <- seq_len(ncol(X_isi_train))

rownames(X_autocorr_train) <- cellIds_train
colnames(X_autocorr_train) <- seq_len(ncol(X_autocorr_train))

stopifnot(nrow(X_isi_train) == length(cellIds_train),
          nrow(X_autocorr_train) == length(cellIds_train),
          length(cType_train) == length(cellIds_train))

.step("Training matrices prepared and validated")

# --- Seurat object for ALL data (train + test together) ---------------------
.step("Creating Seurat object for ALL data (train + test together)")

# Combine all data for joint embedding
X_waveform_all <- as.matrix(waveform)
X_isi_all <- as.matrix(isi)
X_autocorr_all <- as.matrix(autocorr)
cType_all <- cType

cellIds_all <- seq_len(nrow(X_waveform_all))
rownames(X_waveform_all) <- cellIds_all
colnames(X_waveform_all) <- seq_len(ncol(X_waveform_all))
rownames(X_isi_all) <- cellIds_all
colnames(X_isi_all) <- seq_len(ncol(X_isi_all))
rownames(X_autocorr_all) <- cellIds_all
colnames(X_autocorr_all) <- seq_len(ncol(X_autocorr_all))

refdata <- CreateSeuratObject(counts = t(X_waveform_all), assay = "WF")
refdata[["ISI"]] <- CreateAssayObject(counts = t(X_isi_all), assay = "ISI")
refdata[["AUTOCORR"]] <- CreateAssayObject(counts = t(X_autocorr_all), assay = "AUTOCORR")

refdata@meta.data <- cbind(refdata@meta.data, cType = cType_all)
.step("Seurat object ready for all data")

# --- WF pipeline --------------------------------------------------------------
refdata <- ScaleData(refdata, layer = "counts")
refdata <- FindVariableFeatures(refdata, assay = "WF", layer = "counts",
                                selection.method = "vst", nfeatures = 20)
refdata <- RunPCA(refdata, verbose = FALSE, reduction.name = "WFpca",
                  reduction.key = "WFpca_", npcs = numPCs)
.step("Available reductions after WF PCA:")
print(names(refdata@reductions))

refdata <- FindNeighbors(refdata, reduction = "WFpca", dims = dimV, verbose = FALSE)
refdata <- RunUMAP(refdata, reduction = "WFpca", dims = dimV,
                   reduction.name = "WF_umap", metric = "euclidean",
                   n.components = n.comp, seed.use = UMAP.SEED)

# --- ISI pipeline -------------------------------------------------------------
refdata <- ScaleData(refdata, assay = "ISI", layer = "counts")
refdata <- FindVariableFeatures(refdata, assay = "ISI", layer = "counts",
                                selection.method = "vst", nfeatures = 20)
refdata <- RunPCA(refdata, assay = "ISI", verbose = FALSE, reduction.name = "ISIpca",
                  reduction.key = "ISIpca_", npcs = numPCs)
.step("Available reductions after ISI PCA:")
print(names(refdata@reductions))

refdata <- FindNeighbors(refdata, reduction = "ISIpca", dims = dimV, verbose = FALSE)
refdata <- RunUMAP(refdata, reduction = "ISIpca", dims = dimV,
                   reduction.name = "ISI_umap", metric = "euclidean",
                   n.components = n.comp, seed.use = UMAP.SEED)

# --- AUTOCORR pipeline --------------------------------------------------------
refdata <- ScaleData(refdata, assay = "AUTOCORR", layer = "counts")
refdata <- FindVariableFeatures(refdata, assay = "AUTOCORR", layer = "counts",
                                selection.method = "vst", nfeatures = 20)
refdata <- RunPCA(refdata, assay = "AUTOCORR", verbose = FALSE, reduction.name = "AUTOCORRpca",
                  reduction.key = "AUTOCORRpca_", npcs = numPCs)
.step("Available reductions after AUTOCORR PCA:")
print(names(refdata@reductions))

refdata <- FindNeighbors(refdata, reduction = "AUTOCORRpca", dims = dimV, verbose = FALSE)
refdata <- RunUMAP(refdata, reduction = "AUTOCORRpca", dims = dimV,
                   reduction.name = "AUTOCORR_umap", metric = "euclidean",
                   n.components = n.comp, seed.use = UMAP.SEED)

# --- WNN pipeline -------------------------------------------------------------
# Check what reductions actually exist
.step("Checking available reductions before WNN:")
print(names(refdata@reductions))

# Get the actual reduction names (in case they were renamed due to conflicts)
available_reductions <- names(refdata@reductions)
wf_reduction <- available_reductions[grepl("WFpca", available_reductions)][1]
isi_reduction <- available_reductions[grepl("ISIpca", available_reductions)][1]
autocorr_reduction <- available_reductions[grepl("AUTOCORRpca", available_reductions)][1]

cat(sprintf("Using reductions: WF=%s, ISI=%s, AUTOCORR=%s\n",
            wf_reduction, isi_reduction, autocorr_reduction))

# Check for non-finite values before WNN
.step("Checking PCA embeddings for non-finite values")
wf_pca_train <- refdata@reductions[[wf_reduction]]@cell.embeddings
isi_pca_train <- refdata@reductions[[isi_reduction]]@cell.embeddings
autocorr_pca_train <- refdata@reductions[[autocorr_reduction]]@cell.embeddings

cat(sprintf("WF PCA non-finite values: %d\n", sum(!is.finite(wf_pca_train))))
cat(sprintf("ISI PCA non-finite values: %d\n", sum(!is.finite(isi_pca_train))))
cat(sprintf("AUTOCORR PCA non-finite values: %d\n", sum(!is.finite(autocorr_pca_train))))

# Replace any non-finite values with 0
if (any(!is.finite(wf_pca_train))) {
  .step("Found non-finite values in WF PCA, replacing with 0")
  wf_pca_train[!is.finite(wf_pca_train)] <- 0
  refdata@reductions[[wf_reduction]]@cell.embeddings <- wf_pca_train
}
if (any(!is.finite(isi_pca_train))) {
  .step("Found non-finite values in ISI PCA, replacing with 0")
  isi_pca_train[!is.finite(isi_pca_train)] <- 0
  refdata@reductions[[isi_reduction]]@cell.embeddings <- isi_pca_train
}
if (any(!is.finite(autocorr_pca_train))) {
  .step("Found non-finite values in AUTOCORR PCA, replacing with 0")
  autocorr_pca_train[!is.finite(autocorr_pca_train)] <- 0
  refdata@reductions[[autocorr_reduction]]@cell.embeddings <- autocorr_pca_train
}

.step("Running WNN pipeline")
refdata <- FindMultiModalNeighbors(refdata,
  reduction.list = list(wf_reduction, isi_reduction, autocorr_reduction),
  dims.list      = list(dimV, dimV, dimV)
)

.step("Checking WNN neighbor graph for non-finite values")
# Check the WNN neighbor graph
wnn_graph <- refdata@graphs$wsnn
if (!is.null(wnn_graph)) {
  cat(sprintf("WNN SNN graph non-finite values: %d\n", sum(!is.finite(wnn_graph@x))))
  if (any(!is.finite(wnn_graph@x))) {
    .step("Found non-finite values in WNN graph, replacing with 0")
    wnn_graph@x[!is.finite(wnn_graph@x)] <- 0
    refdata@graphs$wsnn <- wnn_graph
  }
}

# Also check WNN distance matrix if it exists
wnn_nn <- refdata@neighbors$weighted.nn
if (!is.null(wnn_nn)) {
  if (!is.null(wnn_nn@nn.dist)) {
    cat(sprintf("WNN distance matrix non-finite values: %d\n", sum(!is.finite(wnn_nn@nn.dist))))
    if (any(!is.finite(wnn_nn@nn.dist))) {
      .step("Found non-finite values in WNN distances, replacing with small value")
      wnn_nn@nn.dist[!is.finite(wnn_nn@nn.dist)] <- 1e-6
      refdata@neighbors$weighted.nn <- wnn_nn
    }
  }
}

.step("Running WNN UMAP")
refdata <- RunUMAP(refdata, nn.name = "weighted.nn",
                   reduction.name = "wnn.umap", reduction.key = "wnnUMAP_",
                   seed.use = 3)

# All data has been embedded together above, so we just split the results

# --- Split embeddings based on animal holdout ------------------------------------
.step("Splitting embeddings based on animal holdout")

# Extract all embeddings (all data embedded together)
reducedDimDataset_all   <- as.data.frame(refdata@reductions$wnn.umap@cell.embeddings);    reducedDimDataset_all$CellType   <- cType_all
wfPCADataset_all        <- as.data.frame(refdata@reductions[[wf_reduction]]@cell.embeddings);       wfPCADataset_all$CellType        <- cType_all
isiPCADataset_all       <- as.data.frame(refdata@reductions[[isi_reduction]]@cell.embeddings);      isiPCADataset_all$CellType       <- cType_all
autocorrPCADataset_all  <- as.data.frame(refdata@reductions[[autocorr_reduction]]@cell.embeddings); autocorrPCADataset_all$CellType  <- cType_all

# Split embeddings based on train/test indices
reducedDimDataset_train <- reducedDimDataset_all[train_indices, ]
reducedDimDataset_test  <- reducedDimDataset_all[test_indices, ]

wfPCADataset_train <- wfPCADataset_all[train_indices, ]
wfPCADataset_test  <- wfPCADataset_all[test_indices, ]

isiPCADataset_train <- isiPCADataset_all[train_indices, ]
isiPCADataset_test  <- isiPCADataset_all[test_indices, ]

autocorrPCADataset_train <- autocorrPCADataset_all[train_indices, ]
autocorrPCADataset_test  <- autocorrPCADataset_all[test_indices, ]

# --- Save embeddings as csv --------------------------------------------------
fold_embeddings_dir <- file.path(out_dir, "embeddings", paste0("fold_", holdout_fold))
dir.create(fold_embeddings_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(reducedDimDataset_train, file.path(fold_embeddings_dir, "physmap_wnn_train.csv"), row.names = FALSE)
write.csv(wfPCADataset_train, file.path(fold_embeddings_dir, "physmap_wf_train.csv"), row.names = FALSE)
write.csv(isiPCADataset_train, file.path(fold_embeddings_dir, "physmap_isi_train.csv"), row.names = FALSE)
write.csv(autocorrPCADataset_train, file.path(fold_embeddings_dir, "physmap_autocorr_train.csv"), row.names = FALSE)

write.csv(reducedDimDataset_test, file.path(fold_embeddings_dir, "physmap_wnn_test.csv"), row.names = FALSE)
write.csv(wfPCADataset_test, file.path(fold_embeddings_dir, "physmap_wf_test.csv"), row.names = FALSE)
write.csv(isiPCADataset_test, file.path(fold_embeddings_dir, "physmap_isi_test.csv"), row.names = FALSE)
write.csv(autocorrPCADataset_test, file.path(fold_embeddings_dir, "physmap_autocorr_test.csv"), row.names = FALSE)

# --- Helper to run holdout evaluation ----------------------------------------
run_holdout_evaluation <- function(train_df, test_df, label, outname) {
  .step(sprintf("%s: training KNN on train animals", label))

  # Train KNN on training animals.
  # Sweep k = 1..20 (matching HIPPIE / NEMO benchmark protocol) and pick the
  # best k by balanced accuracy on the holdout fold. We sweep manually rather
  # than letting caret tune internally because trainControl("none") disables
  # tuning, and we want exactly the same k grid as the rest of the benchmark.
  k_grid <- seq_len(20)
  per_k_results <- lapply(k_grid, function(k) {
    set.seed(123)
    knn_model <- train(CellType ~ ., data = train_df, method = "knn",
                       trControl = trainControl(method = "none"),
                       tuneGrid = data.frame(k = k))
    preds <- predict(knn_model, newdata = test_df[, !names(test_df) %in% "CellType", drop = FALSE])
    list(k = k, preds = preds, bal_acc = balanced_accuracy(test_df$CellType, preds))
  })
  best_idx <- which.max(vapply(per_k_results, function(r) r$bal_acc, numeric(1)))
  best <- per_k_results[[best_idx]]
  test_predictions <- best$preds
  test_accuracy <- mean(test_predictions == test_df$CellType)

  .step(sprintf("%s: Holdout best k=%d, balanced_accuracy=%.3f, raw_accuracy=%.3f",
                label, best$k, best$bal_acc, test_accuracy))

  # Create results dataframe
  results_df <- data.frame(
    label = test_df$CellType,
    prediction = test_predictions,
    fold = paste0("holdout_", holdout_fold),
    stringsAsFactors = FALSE
  )

  # Save results
  out_path <- file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold), outname)
  write.csv(results_df, out_path, row.names = FALSE)
  .step(sprintf("%s: wrote holdout predictions → %s", label, out_path))

  return(test_accuracy)
}

# WNN embeddings for test data are created above by splitting the joint embeddings

# --- Run holdout evaluation for each embedding --------------------------------
wf_accuracy <- run_holdout_evaluation(wfPCADataset_train, wfPCADataset_test, "WF PCA", "PCA_WF_holdout_results.csv")
isi_accuracy <- run_holdout_evaluation(isiPCADataset_train, isiPCADataset_test, "ISI PCA", "PCA_ISI_holdout_results.csv")
autocorr_accuracy <- run_holdout_evaluation(autocorrPCADataset_train, autocorrPCADataset_test, "Autocorr PCA", "PCA_AUTOCORR_holdout_results.csv")
wnn_accuracy <- run_holdout_evaluation(reducedDimDataset_train, reducedDimDataset_test, "Physmap (WNN)", "physmap_holdout_results.csv")

# --- Save holdout summary ----------------------------------------------------
summary_results <- data.frame(
  fold = holdout_fold,
  n_train_animals = length(train_animals),
  n_test_animals = length(test_animals),
  n_train_cells = length(train_indices),
  n_test_cells = length(test_indices),
  wf_accuracy = wf_accuracy,
  isi_accuracy = isi_accuracy,
  autocorr_accuracy = autocorr_accuracy,
  wnn_accuracy = wnn_accuracy,
  stringsAsFactors = FALSE
)

summary_path <- file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold), "holdout_summary.csv")
write.csv(summary_results, summary_path, row.names = FALSE)

# Save animal split information
animal_split <- data.frame(
  animal_id = c(train_animals, test_animals),
  split = c(rep("train", length(train_animals)), rep("test", length(test_animals))),
  fold = holdout_fold,
  stringsAsFactors = FALSE
)

animal_split_path <- file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold), "animal_split.csv")
write.csv(animal_split, animal_split_path, row.names = FALSE)

# --- Done --------------------------------------------------------------------
.step("All holdout steps complete 🎉")
.step(sprintf("Results saved to: %s", file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold))))

try(write.csv(get(".__perf_log", envir = .GlobalEnv), .perf_csv_path, row.names = FALSE), silent = TRUE)
.step(sprintf("Perf log written → %s", .perf_csv_path))