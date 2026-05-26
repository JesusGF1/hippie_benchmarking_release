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
default_out_dir   <- file.path(project_root, "results", "physmap", "transductive")
default_root      <- file.path(project_root, "datasets")
default_dataset   <- "cellexplorer_cell_type"

out_dir <- if (!is.null(cli[["out-dir"]])) cli[["out-dir"]] else
           if (nzchar(Sys.getenv("OUT_DIR"))) Sys.getenv("OUT_DIR") else
           default_out_dir

dataset_root <- if (!is.null(cli[["dataset-root"]])) cli[["dataset-root"]] else
                if (nzchar(Sys.getenv("DATASET_ROOT"))) Sys.getenv("DATASET_ROOT") else
                default_root

dataset_folder <- if (!is.null(cli[["dataset-folder"]])) cli[["dataset-folder"]] else
                  if (nzchar(Sys.getenv("DATASET_FOLDER"))) Sys.getenv("DATASET_FOLDER") else
                  default_dataset

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

# --- Config -------------------------------------------------------------------
# Hyperparameters: can be overridden via CLI --num-pcs, --nfeatures, --metric
# or environment variables PHYSMAP_NUMPCS, PHYSMAP_NFEATURES, PHYSMAP_METRIC
UMAP.SEED <- 3
ALGORITHM <- 3

# numPCs / dimV: number of principal components for PCA and UMAP dims
numPCs <- if (!is.null(cli[["num-pcs"]])) as.integer(cli[["num-pcs"]]) else
          if (nzchar(Sys.getenv("PHYSMAP_NUMPCS"))) as.integer(Sys.getenv("PHYSMAP_NUMPCS")) else 5
dimV <- 1:numPCs
n.comp <- numPCs

# nfeatures: number of variable features for Seurat's FindVariableFeatures
.nfeatures <- if (!is.null(cli[["nfeatures"]])) as.integer(cli[["nfeatures"]]) else
              if (nzchar(Sys.getenv("PHYSMAP_NFEATURES"))) as.integer(Sys.getenv("PHYSMAP_NFEATURES")) else 20

# metric: distance metric for neighbor finding (cosine, euclidean, correlation)
# Note: This is used in FindNeighbors/UMAP; not all Seurat functions support all metrics
.metric <- if (!is.null(cli[["metric"]])) cli[["metric"]] else
           if (nzchar(Sys.getenv("PHYSMAP_METRIC"))) Sys.getenv("PHYSMAP_METRIC") else "euclidean"

.step(sprintf("Config: UMAP.SEED=%d | numPCs=%d | dims=%s | n.comp=%d | nfeatures=%d | metric=%s",
              UMAP.SEED, numPCs, paste(dimV, collapse=","), n.comp, .nfeatures, .metric))
.step(sprintf("Paths: dataset_root=%s | dataset_folder=%s | dataset_path=%s | out_dir=%s",
              dataset_root, dataset_folder, dataset_path, out_dir))

print("Dataset path")
print(dataset_path)

# Ensure output dirs exist
dir.create(file.path(out_dir, "benchmark_results", dataset_folder),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)

.perf_csv_path <- file.path(out_dir, "benchmark_results", dataset_folder, "perf_log.csv")

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

# Modality files are optional individually. We require labels.csv plus at
# least one of waveforms / isi_dist / acg.
modality_files <- list(
  WF       = "waveforms.csv",
  ISI      = "isi_dist.csv",
  AUTOCORR = "acg.csv"
)

available_modalities <- c()
modality_data <- list()
for (mod in names(modality_files)) {
  fpath <- file.path(dataset_path, modality_files[[mod]])
  if (file.exists(fpath)) {
    .step(sprintf("Loading modality %s from %s", mod, modality_files[[mod]]))
    modality_data[[mod]] <- read.csv(fpath)
    available_modalities <- c(available_modalities, mod)
  } else {
    .step(sprintf("Modality %s missing (%s) - skipping", mod, modality_files[[mod]]))
  }
}

if (length(available_modalities) == 0) {
  stop(sprintf("No modality CSVs found under %s. Need at least one of: %s",
               dataset_path, paste(unlist(modality_files), collapse = ", ")))
}

labels_path <- file.path(dataset_path, "labels.csv")
if (!file.exists(labels_path)) {
  stop(sprintf("labels.csv not found in %s", dataset_path))
}
celltypes <- read.csv(labels_path)
cType <- celltypes$label

# Backwards-compatible names for the rest of the script. Modalities that are
# missing remain NULL and are guarded against below.
waveform <- modality_data[["WF"]]
isi      <- modality_data[["ISI"]]
autocorr <- modality_data[["AUTOCORR"]]
# .step(sprintf("Loaded: waveform=%dx%d | isi=%dx%d | autocorr=%dx%d | labels=%d",
#               nrow(waveform), ncol(waveform),
#               nrow(isi), ncol(isi),
#               nrow(autocorr), ncol(autocorr),
#               length(cType)))

.step(sprintf("Available modalities: %s", paste(available_modalities, collapse = ", ")))
for (mod in available_modalities) {
  .step(sprintf("  %s dims: %s", mod, paste(dim(modality_data[[mod]]), collapse = " x ")))
}
.step(sprintf("Labels: %d", length(cType)))

# Coerce all columns to numeric (avoids accidental character columns)
numify <- function(df) {
  as.data.frame(lapply(df, function(x) as.numeric(as.character(x))), stringsAsFactors = FALSE)
}
for (mod in available_modalities) {
  modality_data[[mod]] <- numify(modality_data[[mod]])
}

# Drop rows with NA/NaN/Inf in ANY available modality
rows_ok <- rep(TRUE, length(cType))
for (mod in available_modalities) {
  rows_ok <- rows_ok & complete.cases(modality_data[[mod]])
}
if (!all(rows_ok)) {
  dropped <- sum(!rows_ok)
  .step(sprintf("Dropping %d cells with NA/NaN/Inf across modalities", dropped))
  for (mod in available_modalities) {
    modality_data[[mod]] <- modality_data[[mod]][rows_ok, , drop = FALSE]
  }
  cType <- cType[rows_ok]
}
# --- Universal label filtering (MUST match HIPPIE / NEMO behaviour) -----------
# Task doc "Dataset Filtering Rules (MANDATORY for all benchmarks)":
#   1. Exclude the "unlabeled" class entirely (kept for pretraining but not
#      benchmarking). This also catches b'' / bytes-literal artifacts the
#      hausser unified-layer CSV contains for 94% of its rows.
#   2. Exclude classes with fewer than MIN_CELLS_PER_CLASS=10 samples.
# Both rules mirror `_is_unlabeled()` + the `count < 10` filter in
# scripts/train_multimodal_transductive.py and
# comparison_methods/nemo/scripts/nemo_transductive_test.py.
MIN_CELLS_PER_CLASS <- 10L

# Normalise labels: decode b'...' / b"..." bytes-literal strings, trim
# whitespace, lowercase check for filter words. Preserves original casing
# otherwise so downstream class labels stay human-readable.
normalize_label <- function(lbl) {
  s <- as.character(lbl)
  if (is.na(s) || length(s) == 0L) return("")
  # Strip bytes-literal wrapper: b'foo' or b"foo" -> foo
  if (nchar(s) >= 3L && substr(s, 1L, 2L) %in% c("b'", "b\"")) {
    last <- substr(s, nchar(s), nchar(s))
    if (last %in% c("'", "\"")) {
      s <- substr(s, 3L, nchar(s) - 1L)
    }
  }
  trimws(s)
}
is_unlabeled <- function(lbl) {
  s <- normalize_label(lbl)
  if (nchar(s) == 0L) return(TRUE)
  s_low <- tolower(s)
  any(vapply(c("unlabeled", "unlabelled", "unknown", "juxt", "axo", "vgat"),
             function(w) grepl(w, s_low, fixed = TRUE),
             logical(1)))
}

# Apply normalisation first so filter word-matching works consistently.
cType <- vapply(cType, normalize_label, character(1))
cType[cType == ""] <- "unknown"  # residual blanks become "unknown" then get dropped

unlabeled_mask <- vapply(cType, is_unlabeled, logical(1))
if (any(unlabeled_mask)) {
  n_unlabeled <- sum(unlabeled_mask)
  .step(sprintf("Dropping %d unlabeled/b''/unknown/juxt/axo/vgat rows from %s",
                n_unlabeled, dataset_folder))
  for (mod in available_modalities) {
    modality_data[[mod]] <- modality_data[[mod]][!unlabeled_mask, , drop = FALSE]
  }
  cType <- cType[!unlabeled_mask]
}

# Drop classes below the min-samples floor. Done BEFORE CV/train so caret
# never sees rare classes that would break its stratified folds.
class_counts <- table(cType)
keep_classes <- names(class_counts[class_counts >= MIN_CELLS_PER_CLASS])
rare_classes <- names(class_counts[class_counts < MIN_CELLS_PER_CLASS])
if (length(rare_classes) > 0L) {
  keep_mask <- cType %in% keep_classes
  n_rare <- sum(!keep_mask)
  .step(sprintf("Dropping %d cells from %d rare classes (< %d samples): %s",
                n_rare, length(rare_classes), MIN_CELLS_PER_CLASS,
                paste(rare_classes, collapse = ", ")))
  for (mod in available_modalities) {
    modality_data[[mod]] <- modality_data[[mod]][keep_mask, , drop = FALSE]
  }
  cType <- cType[keep_mask]
}

.step(sprintf("Post-filter: %d cells, %d classes (%s)",
              length(cType), length(unique(cType)),
              paste(sort(unique(cType)), collapse = ", ")))

# Drop zero-waveform units. These are units where the source NWB had no valid
# waveform data (e.g., ~7% of IBL brainwide_good). PCA on all-zero rows
# produces identical projections → KNN "too many ties" error.  HIPPIE handles
# these via its learned encoder; PhysMAP's PCA+UMAP pipeline cannot.
# The dropped unit count is logged for the run summary.
if ("WF" %in% available_modalities) {
  wf_rowsums <- rowSums(abs(as.matrix(modality_data[["WF"]])))
  zero_wf <- wf_rowsums == 0
  n_zero_wf <- sum(zero_wf)
  if (n_zero_wf > 0) {
    .step(sprintf("Dropping %d zero-waveform units (%.1f%%) — PhysMAP requires non-zero WF for PCA",
                  n_zero_wf, 100 * n_zero_wf / length(zero_wf)))
    keep <- !zero_wf
    for (mod in available_modalities) {
      modality_data[[mod]] <- modality_data[[mod]][keep, , drop = FALSE]
    }
    cType <- cType[keep]
  }
}

# Optional: fail fast if anything non-finite remains
for (mod in available_modalities) {
  stopifnot(all(is.finite(as.matrix(modality_data[[mod]]))))
}

# Remove duplicate rows (across all available modalities concatenated)
combined_data <- do.call(cbind, modality_data[available_modalities])
dup_rows <- duplicated(combined_data)
if (any(dup_rows)) {
  n_dup <- sum(dup_rows)
  .step(sprintf("Removing %d duplicate cells", n_dup))
  for (mod in available_modalities) {
    modality_data[[mod]] <- modality_data[[mod]][!dup_rows, , drop = FALSE]
  }
  cType <- cType[!dup_rows]
}

# Refresh aliases
waveform <- modality_data[["WF"]]
isi      <- modality_data[["ISI"]]
autocorr <- modality_data[["AUTOCORR"]]


# --- Prepare matrices ---------------------------------------------------------
.step("Preparing matrices and cell IDs")
modality_matrices <- list()
n_rows <- length(cType)
cellIds <- seq_len(n_rows)

for (mod in available_modalities) {
  M <- as.matrix(modality_data[[mod]])
  stopifnot(nrow(M) == n_rows)
  rownames(M) <- cellIds
  colnames(M) <- seq_len(ncol(M))
  modality_matrices[[mod]] <- M
}

.step("Matrices prepared and validated")

# --- Seurat object ------------------------------------------------------------
# The first available modality becomes the "default" assay; the others are
# attached as additional assays.
.step("Creating Seurat object")
default_mod <- available_modalities[1]
refdata <- CreateSeuratObject(counts = t(modality_matrices[[default_mod]]), assay = default_mod)
for (mod in available_modalities) {
  if (mod == default_mod) next
  refdata[[mod]] <- CreateAssayObject(counts = t(modality_matrices[[mod]]), assay = mod)
}
refdata@meta.data <- cbind(refdata@meta.data, cType = cType)
.step("Seurat object ready")

# --- Per-modality PCA + UMAP --------------------------------------------------
modality_reduction_names <- c(WF = "WFpca", ISI = "ISIpca", AUTOCORR = "AUTOCORRpca")
modality_umap_names      <- c(WF = "WF_umap", ISI = "ISI_umap", AUTOCORR = "AUTOCORR_umap")

for (mod in available_modalities) {
  red_name <- modality_reduction_names[[mod]]
  umap_name <- modality_umap_names[[mod]]
  .step(sprintf("Modality %s: ScaleData -> FindVariableFeatures -> PCA -> UMAP", mod))
  refdata <- ScaleData(refdata, assay = mod, layer = "counts")
  refdata <- FindVariableFeatures(refdata, assay = mod, layer = "counts",
                                  selection.method = "vst", nfeatures = .nfeatures)
  refdata <- RunPCA(refdata, assay = mod, verbose = FALSE,
                    reduction.name = red_name,
                    reduction.key = paste0(red_name, "_"),
                    npcs = numPCs)
  stopifnot(red_name %in% names(refdata@reductions))
  refdata <- RunUMAP(refdata, reduction = red_name, dims = dimV,
                     reduction.name = umap_name, metric = .metric,
                     n.components = n.comp, seed.use = UMAP.SEED)
}

# --- Multi-modal integration --------------------------------------------------
cat("Checking PCA embeddings for NA values:\n")
for (mod in available_modalities) {
  red <- refdata@reductions[[modality_reduction_names[[mod]]]]
  cat(sprintf("  %s NAs: %d, dims: %s\n", mod,
              sum(is.na(red@cell.embeddings)),
              paste(dim(red@cell.embeddings), collapse = " x ")))
}

n_cells <- ncol(refdata)
have_multimodal <- length(available_modalities) >= 2

if (have_multimodal) {
  if (n_cells < 150 || length(available_modalities) == 2) {
    # Small datasets (or only 2 modalities) -> concatenated PCA fallback. WNN
    # is unstable below ~150 cells and a 2-modality WNN is barely meaningful.
    cat(sprintf("Using concatenated PCA for integration (n_cells=%d, modalities=%d)\n",
                n_cells, length(available_modalities)))
    pca_blocks <- lapply(available_modalities, function(m) {
      refdata@reductions[[modality_reduction_names[[m]]]]@cell.embeddings
    })
    concat_pca <- do.call(cbind, pca_blocks)
    rownames(concat_pca) <- colnames(refdata)
    colnames(concat_pca) <- paste0("wnnUMAP_", seq_len(ncol(concat_pca)))

    refdata[["concat"]] <- CreateDimReducObject(
      embeddings = concat_pca,
      key = "concat_",
      assay = DefaultAssay(refdata)
    )
    refdata <- RunUMAP(refdata, reduction = "concat",
                       dims = seq_len(ncol(concat_pca)),
                       reduction.name = "wnn.umap", reduction.key = "wnnUMAP_",
                       metric = .metric, n.components = n.comp, seed.use = UMAP.SEED)
  } else {
    cat("Using FindMultiModalNeighbors (WNN)\n")
    reduction_list <- as.list(modality_reduction_names[available_modalities])
    dims_list <- replicate(length(available_modalities), dimV, simplify = FALSE)
    refdata <- FindMultiModalNeighbors(refdata,
      reduction.list = reduction_list,
      dims.list      = dims_list
    )
    refdata <- RunUMAP(refdata, nn.name = "weighted.nn",
                       reduction.name = "wnn.umap", reduction.key = "wnnUMAP_",
                       seed.use = 3)
  }
} else {
  cat(sprintf("Only one modality (%s) - skipping multimodal integration. PhysMAP/WNN result will be the single-modality embedding.\n",
              available_modalities[1]))
}

# --- Build data.frames for classification ------------------------------------
modality_to_outname <- c(WF = "physmap_wf.csv", ISI = "physmap_isi.csv", AUTOCORR = "physmap_autocorr.csv")
modality_to_label   <- c(WF = "WF PCA",         ISI = "ISI PCA",         AUTOCORR = "Autocorr PCA")
modality_to_results <- c(WF = "PCA_WF_CV_results.csv",
                         ISI = "PCA_ISI_CV_results.csv",
                         AUTOCORR = "PCA_AUTOCORR_CV_results.csv")

per_modality_dfs <- list()
for (mod in available_modalities) {
  red_name <- modality_reduction_names[[mod]]
  df <- as.data.frame(refdata@reductions[[red_name]]@cell.embeddings)
  df$CellType <- cType
  per_modality_dfs[[mod]] <- df
  write.csv(df, file.path(out_dir, "embeddings", modality_to_outname[[mod]]),
            row.names = FALSE)
}

if (have_multimodal) {
  reducedDimDataset <- as.data.frame(refdata@reductions$wnn.umap@cell.embeddings)
  reducedDimDataset$CellType <- cType
  write.csv(reducedDimDataset,
            file.path(out_dir, "embeddings", "physmap_wnn.csv"),
            row.names = FALSE)
}

# --- Helper to run KNN CV -----------------------------------------------------
#
# Mirror the HIPPIE evaluation: try k in {1, 3, 5, 10}, pick the k that
# maximises balanced accuracy across the 10-fold CV, and write ONE prediction
# per cell (the prediction made when that cell was in its held-out fold) using
# the winning k. This matches train_multimodal_transductive.py which writes
# `transductive_predictions.csv` with just the best-k (pred, true) pairs.

balanced_accuracy <- function(truth, pred) {
  # mean per-class recall, ignoring classes with no support
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

run_knn_cv <- function(df, label, outname){
  .step(sprintf("%s: training KNN (5-fold CV, k in 1..20)", label))

  # L2-normalize feature columns to match HIPPIE/NEMO KNN protocol
  # (train_multimodal_transductive.py ~line 1240). On unit-normalized vectors,
  # Euclidean KNN produces the same nearest-neighbor ordering as cosine KNN,
  # so we get cosine-similarity geometry without switching caret's KNN engine.
  feature_cols <- setdiff(names(df), "CellType")
  feature_mat <- as.matrix(df[, feature_cols])
  row_norms <- sqrt(rowSums(feature_mat^2))
  row_norms[row_norms < 1e-12] <- 1e-12
  df[, feature_cols] <- feature_mat / row_norms

  k_grid <- data.frame(k = seq_len(20))
  train_control <- trainControl(method = "cv", number = 5,
                                savePredictions = "all")
  set.seed(42)  # Match HIPPIE/NEMO seed convention for consistency
  knn_model <- train(CellType ~ ., data = df, method = "knn",
                     trControl = train_control, tuneGrid = k_grid)

  # Per-fold best-k selection to match the HIPPIE/NEMO evaluation protocol:
  # each method picks the k that maximizes balanced accuracy on the held-out
  # test fold of its cross-validation iteration. The equivalent for PhysMAP's
  # single-job setup is to pick a k per caret internal fold on that fold's
  # held-out cells, and write per-cell predictions using the fold-local best k.
  # Each cell is in exactly one fold, so this still yields one prediction per
  # cell, keeping the k-selection rule identical across the three methods.
  per_k <- knn_model$pred

  fold_best_k <- per_k %>%
    dplyr::group_by(Resample, k) %>%
    dplyr::summarise(
      bal_acc = balanced_accuracy(obs, pred),
      .groups = "drop"
    ) %>%
    dplyr::group_by(Resample) %>%
    dplyr::slice_max(bal_acc, n = 1, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    dplyr::select(Resample, k, fold_bal_acc = bal_acc)

  for (i in seq_len(nrow(fold_best_k))) {
    .step(sprintf("  %s %s: best k=%d (balanced_acc=%.3f)",
                  label, fold_best_k$Resample[i],
                  fold_best_k$k[i], fold_best_k$fold_bal_acc[i]))
  }

  # Pick each fold's best-k predictions, then concatenate.
  best_preds <- per_k %>%
    dplyr::inner_join(fold_best_k %>% dplyr::select(Resample, k),
                      by = c("Resample", "k")) %>%
    dplyr::arrange(rowIndex) %>%
    dplyr::transmute(true = obs, pred = pred, fold = Resample)

  # Reported accuracy is the balanced_accuracy of the pooled per-fold-best-k
  # predictions — the honest metric given the optimistic k-selection rule.
  best_bal_acc <- balanced_accuracy(best_preds$true, best_preds$pred)

  # For logging: report the mode of the per-fold best k's as a scalar.
  best_k_mode <- as.integer(
    names(sort(table(fold_best_k$k), decreasing = TRUE))[1]
  )
  .step(sprintf("%s: per-fold best k mode=%d, pooled balanced_acc=%.3f",
                label, best_k_mode, best_bal_acc))

  # Also compute the legacy pooled-k table for the perf log, so downstream
  # scripts that expect a per-k summary still see one.
  k_scores <- per_k %>%
    dplyr::group_by(k) %>%
    dplyr::summarise(
      bal_acc = balanced_accuracy(obs, pred),
      raw_acc = mean(obs == pred),
      .groups = "drop"
    )

  out_path <- file.path(out_dir, "benchmark_results", dataset_folder, outname)
  write.csv(best_preds, out_path, row.names = FALSE)
  .step(sprintf("%s: wrote %d predictions -> %s",
                label, nrow(best_preds), out_path))

  # Return the per-k summary so callers can stash it in the perf log later.
  invisible(list(
    label = label,
    best_k = best_k_mode,
    best_balanced_accuracy = best_bal_acc,
    per_k = k_scores,
    fold_best_k = fold_best_k
  ))
}

# --- Run KNN CV for each available modality + multimodal --------------------
knn_summaries <- list()
for (mod in available_modalities) {
  knn_summaries[[mod]] <- run_knn_cv(
    per_modality_dfs[[mod]],
    modality_to_label[[mod]],
    modality_to_results[[mod]]
  )
}
if (have_multimodal) {
  knn_summaries[["WNN"]] <- run_knn_cv(reducedDimDataset, "Physmap (WNN)",
                                       "physmap_CV_results.csv")
} else {
  .step("Skipping physmap_CV_results.csv (need >=2 modalities)")
}

# Per-modality best-k summary
summary_rows <- do.call(rbind, lapply(names(knn_summaries), function(mod) {
  s <- knn_summaries[[mod]]
  data.frame(
    modality = mod,
    label = s$label,
    best_k = s$best_k,
    best_balanced_accuracy = s$best_balanced_accuracy,
    stringsAsFactors = FALSE
  )
}))
summary_path <- file.path(out_dir, "benchmark_results", dataset_folder,
                          "knn_summary.csv")
write.csv(summary_rows, summary_path, row.names = FALSE)
.step(sprintf("Wrote KNN summary -> %s", summary_path))

# --- Compute parity: append one unified-schema row to timings.csv -----------
# Mirrors the schema written by hippie/compute_parity.py so the aggregator can
# concatenate HIPPIE / NEMO / PhysMAP rows directly. PhysMAP runs are CPU-only
# (peak_gpu_mb = NA) and don't have HIPPIE-style phases, so we emit a single
# `phase = "total"` row per dataset launch.
.write_compute_parity_row <- function() {
  csv_path <- Sys.getenv("COMPUTE_PARITY_CSV")
  if (!nzchar(csv_path)) {
    csv_path <- file.path(project_root, "results", "compute_parity", "timings.csv")
  }
  dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)

  # Run id: stable per launch. Avoids extra package deps by using base R only.
  run_id <- sprintf(
    "physmap-%d-%s",
    as.integer(Sys.time()),
    paste(sample(c(letters, 0:9), 8, replace = TRUE), collapse = "")
  )

  # `.start_time` is from proc.time() (CPU clock since R started), not wall
  # clock. Reconstruct the wall-clock start by subtracting elapsed seconds
  # from "now".
  elapsed_s <- .time_since()
  started_at <- format(Sys.time() - elapsed_s, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

  # Best-k per modality, JSON-encoded for the `extra` column
  per_modality_best_k <- list()
  for (mod in names(knn_summaries)) {
    per_modality_best_k[[mod]] <- list(
      best_k = knn_summaries[[mod]]$best_k,
      best_balanced_accuracy = knn_summaries[[mod]]$best_balanced_accuracy
    )
  }
  extra_list <- list(
    numPCs = numPCs,
    n_modalities = length(available_modalities),
    available_modalities = available_modalities,
    seurat_version = as.character(packageVersion("Seurat")),
    caret_version  = as.character(packageVersion("caret")),
    per_modality_best_k = per_modality_best_k,
    script = "physmap_script.r"
  )
  extra_json <- tryCatch(
    as.character(jsonlite::toJSON(extra_list, auto_unbox = TRUE, null = "null")),
    error = function(e) "{}"
  )

  row <- data.frame(
    method       = "physmap",
    dataset      = dataset_folder,
    fold         = -1L,
    phase        = "total",
    wall_seconds = .time_since(),
    peak_gpu_mb  = NA_real_,
    peak_cpu_mb  = .mem_mb(),
    n_cells      = length(cType),
    n_classes    = length(unique(cType)),
    control      = "",
    config       = "",
    run_id       = run_id,
    started_at   = started_at,
    extra        = as.character(extra_json),
    stringsAsFactors = FALSE
  )

  # Append (writing header if the file doesn't exist yet). No file lock - if
  # multiple PhysMAP datasets launch in parallel on the same node and all
  # finish in the same instant they could interleave; in practice the windows
  # are wide enough that this is fine for the supplement table.
  write_header <- !file.exists(csv_path)
  # quote = column-indices-of-character-cols only (NOT TRUE which would also
  # quote the header row). The `extra` column needs quoting because its JSON
  # body contains commas; the other character columns get quoted defensively.
  # Headers stay unquoted so pandas concat with HIPPIE/NEMO timings.csv
  # (which use python csv.DictWriter, no header quoting) doesn't mismatch
  # column names across method rows.
  char_col_idx <- which(sapply(row, is.character))
  suppressWarnings(
    write.table(row, file = csv_path, sep = ",", row.names = FALSE,
                col.names = write_header, append = !write_header,
                quote = char_col_idx, qmethod = "double")
  )
  .step(sprintf("[compute_parity] wrote total row to %s (wall_seconds=%.1f)",
                csv_path, .time_since()))
}
# jsonlite is normally pulled in transitively by Seurat. If it's missing the
# helper still writes a row, just with extra="{}".
try(suppressPackageStartupMessages(library(jsonlite)), silent = TRUE)
try(.write_compute_parity_row(), silent = FALSE)

# --- Done --------------------------------------------------------------------
.step("All steps complete 🎉")

try(write.csv(get(".__perf_log", envir = .GlobalEnv), .perf_csv_path, row.names = FALSE), silent = TRUE)
.step(sprintf("Perf log written → %s", .perf_csv_path))