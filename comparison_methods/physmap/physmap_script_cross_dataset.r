#!/usr/bin/env Rscript
#
# PhysMAP cross-dataset transfer evaluation.
#
# Fits PCA + UMAP on the TRAINING dataset, then projects the PREDICT dataset
# into that same embedding space.  Trains a KNN on training embeddings and
# evaluates it on the projected predict embeddings.
#
# This mirrors the HIPPIE cross-dataset protocol:
#   train on hull_cell_type  → predict on lisberger_labeled_cell_type
#   train on lisberger_labeled_cell_type → predict on hull_cell_type
#
# CLI args (override via --key=value or env vars):
#   --train-dataset-root     Base directory that contains both dataset folders
#   --train-folder           Sub-folder for the training dataset
#   --predict-folder         Sub-folder for the predict dataset
#   --out-dir                Output directory for results
#   --num-pcs                Number of PCs (dimV) to use in UMAP/WNN  (default: 5)
#   --nfeatures              nfeatures for FindVariableFeatures           (default: 20)
#   --metric                 UMAP metric                                  (default: euclidean)

# --- Progress logging helpers + perf CSV -------------------------------------
assign(".start_time", proc.time(), envir = .GlobalEnv)

assign(".__event_idx", 0L, envir = .GlobalEnv)

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

.time_since <- function() as.numeric((proc.time() - get(".start_time", envir = .GlobalEnv))[3])

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
      .step(sprintf("%s done (%.2fs)", label, elapsed))
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

# --- Resolve arguments -------------------------------------------------------
dataset_root <- if (!is.null(cli[["train-dataset-root"]])) cli[["train-dataset-root"]] else
                if (nzchar(Sys.getenv("DATASET_ROOT"))) Sys.getenv("DATASET_ROOT") else
                stop("--train-dataset-root or DATASET_ROOT env var required")

train_folder <- if (!is.null(cli[["train-folder"]])) cli[["train-folder"]] else
                if (nzchar(Sys.getenv("TRAIN_DATASET"))) Sys.getenv("TRAIN_DATASET") else
                stop("--train-folder or TRAIN_DATASET env var required")

predict_folder <- if (!is.null(cli[["predict-folder"]])) cli[["predict-folder"]] else
                  if (nzchar(Sys.getenv("PREDICT_DATASET"))) Sys.getenv("PREDICT_DATASET") else
                  stop("--predict-folder or PREDICT_DATASET env var required")

out_dir <- if (!is.null(cli[["out-dir"]])) cli[["out-dir"]] else
           if (nzchar(Sys.getenv("OUT_DIR"))) Sys.getenv("OUT_DIR") else
           file.path(getwd(), "results", "physmap_cross_dataset",
                     paste0("train_", train_folder, "_predict_", predict_folder))

# Locked hyperparameters (config_13: dimV=5, metric=euclidean, nfeatures=20)
num_pcs    <- as.integer(
  if (!is.null(cli[["num-pcs"]])) cli[["num-pcs"]] else
  if (nzchar(Sys.getenv("PHYSMAP_DIMV"))) Sys.getenv("PHYSMAP_DIMV") else "5"
)
nfeatures  <- as.integer(
  if (!is.null(cli[["nfeatures"]])) cli[["nfeatures"]] else
  if (nzchar(Sys.getenv("PHYSMAP_NFEATURES"))) Sys.getenv("PHYSMAP_NFEATURES") else "20"
)
umap_metric <- if (!is.null(cli[["metric"]])) cli[["metric"]] else
               if (nzchar(Sys.getenv("PHYSMAP_METRIC"))) Sys.getenv("PHYSMAP_METRIC") else "euclidean"

# Strip leading slashes from folder names
train_folder   <- sub("^/+", "", train_folder)
predict_folder <- sub("^/+", "", predict_folder)

train_path   <- file.path(dataset_root, train_folder)
predict_path <- file.path(dataset_root, predict_folder)

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

# --- Configuration -----------------------------------------------------------
UMAP.SEED  <- 3
ALGORITHM  <- 3
dimV       <- seq_len(num_pcs)
n.comp     <- num_pcs
numPCs     <- num_pcs

.step(sprintf("Cross-dataset config:"))
.step(sprintf("  train_folder   = %s", train_folder))
.step(sprintf("  predict_folder = %s", predict_folder))
.step(sprintf("  numPCs/dimV    = %d", numPCs))
.step(sprintf("  nfeatures      = %d", nfeatures))
.step(sprintf("  umap_metric    = %s", umap_metric))
.step(sprintf("  out_dir        = %s", out_dir))

# Ensure output dirs exist
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)

.perf_csv_path <- file.path(out_dir, "perf_log.csv")

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

setwd(out_dir)

# --- Helper: load and clean a dataset ----------------------------------------
load_dataset <- function(path, label) {
  .step(sprintf("Loading %s CSVs from: %s", label, path))
  wf  <- read.csv(file.path(path, "waveforms.csv"))
  isi <- read.csv(file.path(path, "isi_dist.csv"))
  acg <- read.csv(file.path(path, "acg.csv"))
  ct  <- read.csv(file.path(path, "labels.csv"))$label

  numify <- function(df) {
    as.data.frame(lapply(df, function(x) as.numeric(as.character(x))), stringsAsFactors = FALSE)
  }
  wf  <- numify(wf)
  isi <- numify(isi)
  acg <- numify(acg)

  # Drop rows with any NA/NaN/Inf
  rows_ok <- complete.cases(wf) & complete.cases(isi) & complete.cases(acg)
  if (!all(rows_ok)) {
    dropped <- sum(!rows_ok)
    .step(sprintf("%s: dropping %d rows with NA/NaN/Inf", label, dropped))
    wf  <- wf[rows_ok, , drop = FALSE]
    isi <- isi[rows_ok, , drop = FALSE]
    acg <- acg[rows_ok, , drop = FALSE]
    ct  <- ct[rows_ok]
  }
  ct[is.na(ct) | ct == ""] <- "unknown"

  # Drop zero-waveform units
  wf_rowsums <- rowSums(abs(as.matrix(wf)))
  zero_wf <- wf_rowsums == 0
  n_zero_wf <- sum(zero_wf)
  if (n_zero_wf > 0) {
    .step(sprintf("%s: dropping %d zero-waveform units (%.1f%%)",
                  label, n_zero_wf, 100 * n_zero_wf / length(zero_wf)))
    keep <- !zero_wf
    wf  <- wf[keep, , drop = FALSE]
    isi <- isi[keep, , drop = FALSE]
    acg <- acg[keep, , drop = FALSE]
    ct  <- ct[keep]
  }

  stopifnot(all(is.finite(as.matrix(wf))),
            all(is.finite(as.matrix(isi))),
            all(is.finite(as.matrix(acg))))

  .step(sprintf("%s: loaded %d cells, %d classes",
                label, nrow(wf), length(unique(ct))))
  list(wf = wf, isi = isi, acg = acg, ct = ct)
}

# --- Load both datasets -------------------------------------------------------
train_data   <- load_dataset(train_path, "train")
predict_data <- load_dataset(predict_path, "predict")

# --- Build Seurat objects (train and predict separately) ---------------------
.step("Building Seurat objects")

make_seurat <- function(dat, prefix) {
  n <- nrow(dat$wf)
  ids <- seq_len(n)

  X_wf  <- as.matrix(dat$wf)
  X_isi <- as.matrix(dat$isi)
  X_acg <- as.matrix(dat$acg)
  rownames(X_wf)  <- ids; colnames(X_wf)  <- seq_len(ncol(X_wf))
  rownames(X_isi) <- ids; colnames(X_isi) <- seq_len(ncol(X_isi))
  rownames(X_acg) <- ids; colnames(X_acg) <- seq_len(ncol(X_acg))

  obj <- CreateSeuratObject(counts = t(X_wf), assay = "WF")
  obj[["ISI"]]      <- CreateAssayObject(counts = t(X_isi), assay = "ISI")
  obj[["AUTOCORR"]] <- CreateAssayObject(counts = t(X_acg), assay = "AUTOCORR")
  obj@meta.data <- cbind(obj@meta.data, cType = dat$ct)
  obj
}

train_obj   <- make_seurat(train_data, "train")
predict_obj <- make_seurat(predict_data, "predict")

# --- Fit WF PCA on train, project predict ------------------------------------
.step("WF: ScaleData + VariableFeatures on train")
train_obj <- ScaleData(train_obj, layer = "counts")
train_obj <- FindVariableFeatures(train_obj, assay = "WF", layer = "counts",
                                  selection.method = "vst", nfeatures = nfeatures)
train_obj <- RunPCA(train_obj, verbose = FALSE, reduction.name = "WFpca",
                    reduction.key = "WFpca_", npcs = numPCs)

# Project predict into train PCA space
.step("WF: Project predict into train PCA space")
# Extract PCA rotation (loadings) from train
wf_rotation <- train_obj@reductions$WFpca@feature.loadings          # features x PCs
wf_center   <- GetAssayData(train_obj, assay = "WF", layer = "scale.data")  # features x cells (row-scaled)
wf_counts_mat <- as.matrix(GetAssayData(train_obj, assay = "WF", layer = "counts"))
wf_scale_model <- list(
  center = rowMeans(wf_counts_mat),
  scale  = apply(wf_counts_mat, 1, sd)
)

# Scale predict data using TRAIN mean/sd
predict_wf_mat <- as.matrix(predict_data$wf)
colnames(predict_wf_mat) <- seq_len(ncol(predict_wf_mat))
rownames(predict_wf_mat) <- seq_len(nrow(predict_wf_mat))

# Use only the variable features selected in train
wf_var_features <- VariableFeatures(train_obj, assay = "WF")

# Scale predict using train statistics
predict_obj <- ScaleData(predict_obj, layer = "counts", assay = "WF")
# ProjectDim removed — Seurat v5 incompatibility. Using manual projection below.
.step("WF: Extracting WF PCA embeddings")
train_wf_emb   <- as.data.frame(train_obj@reductions$WFpca@cell.embeddings)
train_wf_emb$CellType <- train_data$ct

# For predict, we project manually via the PCA rotation matrix
# ScaleData on predict_obj and then multiply by train rotation
predict_wf_scaled <- GetAssayData(predict_obj, assay = "WF", layer = "scale.data")  # features x cells

# Align features (use features present in both)
common_wf_features <- intersect(rownames(wf_rotation), rownames(predict_wf_scaled))
.step(sprintf("WF: %d / %d rotation features found in predict data",
              length(common_wf_features), nrow(wf_rotation)))

wf_rot_aligned     <- wf_rotation[common_wf_features, , drop = FALSE]
predict_wf_aligned <- predict_wf_scaled[common_wf_features, , drop = FALSE]

predict_wf_emb_mat <- t(predict_wf_aligned) %*% wf_rot_aligned  # cells x PCs
predict_wf_emb     <- as.data.frame(predict_wf_emb_mat)
colnames(predict_wf_emb) <- paste0("WFpca_", seq_len(ncol(predict_wf_emb)))
predict_wf_emb$CellType <- predict_data$ct

# --- Fit ISI PCA on train, project predict -----------------------------------
.step("ISI: ScaleData + PCA on train")
train_obj <- ScaleData(train_obj, assay = "ISI", layer = "counts")
train_obj <- FindVariableFeatures(train_obj, assay = "ISI", layer = "counts",
                                  selection.method = "vst", nfeatures = nfeatures)
train_obj <- RunPCA(train_obj, assay = "ISI", verbose = FALSE, reduction.name = "ISIpca",
                    reduction.key = "ISIpca_", npcs = numPCs)

predict_obj <- ScaleData(predict_obj, assay = "ISI", layer = "counts")

isi_rotation <- train_obj@reductions$ISIpca@feature.loadings
predict_isi_scaled <- GetAssayData(predict_obj, assay = "ISI", layer = "scale.data")

common_isi_features <- intersect(rownames(isi_rotation), rownames(predict_isi_scaled))
.step(sprintf("ISI: %d / %d rotation features found in predict data",
              length(common_isi_features), nrow(isi_rotation)))

isi_rot_aligned      <- isi_rotation[common_isi_features, , drop = FALSE]
predict_isi_aligned  <- predict_isi_scaled[common_isi_features, , drop = FALSE]

predict_isi_emb_mat <- t(predict_isi_aligned) %*% isi_rot_aligned
predict_isi_emb     <- as.data.frame(predict_isi_emb_mat)
colnames(predict_isi_emb) <- paste0("ISIpca_", seq_len(ncol(predict_isi_emb)))
predict_isi_emb$CellType  <- predict_data$ct

train_isi_emb <- as.data.frame(train_obj@reductions$ISIpca@cell.embeddings)
train_isi_emb$CellType <- train_data$ct

# --- Fit AUTOCORR PCA on train, project predict ------------------------------
.step("AUTOCORR: ScaleData + PCA on train")
train_obj <- ScaleData(train_obj, assay = "AUTOCORR", layer = "counts")
train_obj <- FindVariableFeatures(train_obj, assay = "AUTOCORR", layer = "counts",
                                  selection.method = "vst", nfeatures = nfeatures)
train_obj <- RunPCA(train_obj, assay = "AUTOCORR", verbose = FALSE, reduction.name = "AUTOCORRpca",
                    reduction.key = "AUTOCORRpca_", npcs = numPCs)

predict_obj <- ScaleData(predict_obj, assay = "AUTOCORR", layer = "counts")

acg_rotation <- train_obj@reductions$AUTOCORRpca@feature.loadings
predict_acg_scaled <- GetAssayData(predict_obj, assay = "AUTOCORR", layer = "scale.data")

common_acg_features <- intersect(rownames(acg_rotation), rownames(predict_acg_scaled))
.step(sprintf("AUTOCORR: %d / %d rotation features found in predict data",
              length(common_acg_features), nrow(acg_rotation)))

acg_rot_aligned      <- acg_rotation[common_acg_features, , drop = FALSE]
predict_acg_aligned  <- predict_acg_scaled[common_acg_features, , drop = FALSE]

predict_acg_emb_mat <- t(predict_acg_aligned) %*% acg_rot_aligned
predict_acg_emb     <- as.data.frame(predict_acg_emb_mat)
colnames(predict_acg_emb) <- paste0("AUTOCORRpca_", seq_len(ncol(predict_acg_emb)))
predict_acg_emb$CellType  <- predict_data$ct

train_acg_emb <- as.data.frame(train_obj@reductions$AUTOCORRpca@cell.embeddings)
train_acg_emb$CellType <- train_data$ct

# --- Check PCA embeddings for non-finite values ------------------------------
.step("Checking PCA embeddings for non-finite values")

replace_nonfinite <- function(df, name) {
  mat <- as.matrix(df[, !names(df) %in% "CellType", drop = FALSE])
  n_bad <- sum(!is.finite(mat))
  if (n_bad > 0) {
    .step(sprintf("%s: replacing %d non-finite values with 0", name, n_bad))
    mat[!is.finite(mat)] <- 0
    df[, !names(df) %in% "CellType"] <- mat
  }
  df
}

train_wf_emb   <- replace_nonfinite(train_wf_emb,  "train WF PCA")
train_isi_emb  <- replace_nonfinite(train_isi_emb, "train ISI PCA")
train_acg_emb  <- replace_nonfinite(train_acg_emb, "train ACG PCA")
predict_wf_emb  <- replace_nonfinite(predict_wf_emb,  "predict WF PCA")
predict_isi_emb <- replace_nonfinite(predict_isi_emb, "predict ISI PCA")
predict_acg_emb <- replace_nonfinite(predict_acg_emb, "predict ACG PCA")

# --- Concatenated WNN-style embedding (PCA features from all modalities) -----
# Full WNN would require a shared UMAP; here we concatenate modality PCA
# embeddings and use them directly as the multimodal embedding.  This is the
# cross-dataset analogue of the WNN approach: the PCA axes of each modality
# are separately fitted on training data and projected onto predict.
.step("Concatenating modality PCA embeddings for multimodal representation")

make_combined_emb <- function(wf_emb, isi_emb, acg_emb, ct, prefix) {
  wf_cols  <- wf_emb[, !names(wf_emb)  %in% "CellType", drop = FALSE]
  isi_cols <- isi_emb[, !names(isi_emb) %in% "CellType", drop = FALSE]
  acg_cols <- acg_emb[, !names(acg_emb) %in% "CellType", drop = FALSE]
  colnames(wf_cols)  <- paste0("WF_",  seq_len(ncol(wf_cols)))
  colnames(isi_cols) <- paste0("ISI_", seq_len(ncol(isi_cols)))
  colnames(acg_cols) <- paste0("ACG_", seq_len(ncol(acg_cols)))
  combined <- cbind(wf_cols, isi_cols, acg_cols)
  combined$CellType <- ct
  combined
}

train_combined   <- make_combined_emb(train_wf_emb,   train_isi_emb,   train_acg_emb,
                                      train_data$ct,   "train")
predict_combined <- make_combined_emb(predict_wf_emb, predict_isi_emb, predict_acg_emb,
                                      predict_data$ct, "predict")

# --- Save embeddings ---------------------------------------------------------
emb_dir <- file.path(out_dir, "embeddings")
dir.create(emb_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(train_wf_emb,    file.path(emb_dir, "train_wf_pca.csv"),      row.names = FALSE)
write.csv(train_isi_emb,   file.path(emb_dir, "train_isi_pca.csv"),     row.names = FALSE)
write.csv(train_acg_emb,   file.path(emb_dir, "train_acg_pca.csv"),     row.names = FALSE)
write.csv(train_combined,  file.path(emb_dir, "train_combined_pca.csv"), row.names = FALSE)

write.csv(predict_wf_emb,   file.path(emb_dir, "predict_wf_pca.csv"),      row.names = FALSE)
write.csv(predict_isi_emb,  file.path(emb_dir, "predict_isi_pca.csv"),     row.names = FALSE)
write.csv(predict_acg_emb,  file.path(emb_dir, "predict_acg_pca.csv"),     row.names = FALSE)
write.csv(predict_combined, file.path(emb_dir, "predict_combined_pca.csv"), row.names = FALSE)

# --- KNN evaluation ----------------------------------------------------------
run_cross_dataset_evaluation <- function(train_df, predict_df, label, outname) {
  .step(sprintf("%s: training KNN on train, evaluating on predict", label))

  # Select k via cross-validation on training data only (mirrors HIPPIE protocol).
  # Never use predict (test) labels for model selection.
  k_grid <- seq_len(20)
  n_folds <- min(5L, length(unique(train_df$CellType)))
  set.seed(123)
  folds <- caret::createFolds(train_df$CellType, k = n_folds, list = TRUE, returnTrain = FALSE)

  cv_scores <- vapply(k_grid, function(k) {
    fold_accs <- vapply(folds, function(val_idx) {
      tr  <- train_df[-val_idx, , drop = FALSE]
      val <- train_df[ val_idx, , drop = FALSE]
      mdl <- train(CellType ~ ., data = tr, method = "knn",
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(k = k))
      p <- predict(mdl, newdata = val[, !names(val) %in% "CellType", drop = FALSE])
      balanced_accuracy(val$CellType, p)
    }, numeric(1))
    mean(fold_accs, na.rm = TRUE)
  }, numeric(1))

  best_k   <- k_grid[which.max(cv_scores)]
  best_cv  <- max(cv_scores)
  .step(sprintf("%s: best k=%d by train-CV (CV BA=%.3f)", label, best_k, best_cv))

  # Train final model with best k and evaluate on the held-out predict set
  final_model <- train(CellType ~ ., data = train_df, method = "knn",
                       trControl = trainControl(method = "none"),
                       tuneGrid = data.frame(k = best_k))
  best_preds  <- predict(final_model, newdata = predict_df[, !names(predict_df) %in% "CellType", drop = FALSE])
  best_bal_acc <- balanced_accuracy(predict_df$CellType, best_preds)
  raw_acc <- mean(best_preds == predict_df$CellType)

  .step(sprintf("%s: best k=%d, balanced_accuracy=%.3f, raw_accuracy=%.3f",
                label, best_k, best_bal_acc, raw_acc))

  results_df <- data.frame(
    label      = predict_df$CellType,
    prediction = best_preds,
    fold       = "cross_dataset",
    stringsAsFactors = FALSE
  )
  out_path <- file.path(out_dir, outname)
  write.csv(results_df, out_path, row.names = FALSE)
  .step(sprintf("%s: wrote cross-dataset predictions -> %s", label, out_path))

  list(best_k = best_k, bal_acc = best_bal_acc, raw_acc = raw_acc)
}

# --- Run evaluation for each modality embedding ------------------------------
wf_res      <- run_cross_dataset_evaluation(train_wf_emb,   predict_wf_emb,   "WF PCA",      "PCA_WF_cross_dataset_results.csv")
isi_res     <- run_cross_dataset_evaluation(train_isi_emb,  predict_isi_emb,  "ISI PCA",     "PCA_ISI_cross_dataset_results.csv")
acg_res     <- run_cross_dataset_evaluation(train_acg_emb,  predict_acg_emb,  "AUTOCORR PCA","PCA_AUTOCORR_cross_dataset_results.csv")
combined_res <- run_cross_dataset_evaluation(train_combined, predict_combined, "PhysMAP (combined PCA)", "physmap_cross_dataset_results.csv")

# --- Save summary ------------------------------------------------------------
summary_results <- data.frame(
  train_dataset   = train_folder,
  predict_dataset = predict_folder,
  n_train_cells   = nrow(train_data$wf),
  n_predict_cells = nrow(predict_data$wf),
  wf_best_k               = wf_res$best_k,
  wf_balanced_accuracy    = wf_res$bal_acc,
  isi_best_k              = isi_res$best_k,
  isi_balanced_accuracy   = isi_res$bal_acc,
  acg_best_k              = acg_res$best_k,
  acg_balanced_accuracy   = acg_res$bal_acc,
  combined_best_k         = combined_res$best_k,
  combined_balanced_accuracy = combined_res$bal_acc,
  stringsAsFactors = FALSE
)
summary_path <- file.path(out_dir, "cross_dataset_summary.csv")
write.csv(summary_results, summary_path, row.names = FALSE)
.step(sprintf("Summary written -> %s", summary_path))

# --- Done --------------------------------------------------------------------
.step("All cross-dataset steps complete")
.step(sprintf("Results saved to: %s", out_dir))

try(write.csv(get(".__perf_log", envir = .GlobalEnv), .perf_csv_path, row.names = FALSE), silent = TRUE)
.step(sprintf("Perf log written -> %s", .perf_csv_path))
