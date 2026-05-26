#!/usr/bin/env Rscript
#
# PhysMAP-PCA: induction-safe multi-modal PCA + KNN benchmark.
#
# Parallel to physmap_script.r / physmap_script_holdout.r but replaces
# Seurat WNN (which requires all cells in one object and cannot project
# held-out animals inductively) with plain prcomp() + predict().
#
# Modes
# -----
#   Transductive (default): PCA fit on ALL cells; 5-fold CV KNN.
#   Holdout (--holdout-fold N): PCA fit on TRAIN animals only;
#     predict() projects test animals; KNN trained on train, evaluated on test.
#
# Multi-modal combination: concatenated PCA across available modalities
#   (WF, ISI, AUTOCORR) — inductive equivalent of WNN.
#
# Output files match physmap_script.r (transductive) and
#   physmap_script_holdout.r (holdout) so existing analysis scripts work.
#
# Usage
# -----
#   # Transductive
#   R -q -e 'source("physmap_script_pca.r")' --args \
#     --dataset-root=/data/datasets --dataset-folder=hull_cell_type \
#     --out-dir=/data/results/physmap_pca/hull_cell_type/fold_0 \
#     --num-pcs=5 --nfeatures=20
#
#   # Holdout
#   R -q -e 'source("physmap_script_pca.r")' --args \
#     --dataset-root=/data/datasets --dataset-folder=allen_scope_neuropixel_area_subset \
#     --out-dir=/data/results/physmap_pca_holdout \
#     --holdout-fold=0 --n-holdout-folds=5 \
#     --num-pcs=5 --nfeatures=20

# --- Progress logging helpers ------------------------------------------------
assign(".start_time", proc.time(), envir = .GlobalEnv)
assign(".__event_idx", 0L, envir = .GlobalEnv)
assign(".__perf_log", data.frame(
  event_id=integer(), kind=character(), label=character(),
  start_clock=as.POSIXct(character()), end_clock=as.POSIXct(character()),
  elapsed_s=numeric(), rss_before_mb=numeric(), rss_after_mb=numeric(),
  rss_delta_mb=numeric(), status=character(), notes=character(),
  stringsAsFactors=FALSE
), envir=.GlobalEnv)

.time_since <- function() as.numeric((proc.time() - get(".start_time", envir=.GlobalEnv))[3])
.mem_mb <- local({
  have_proc <- file.exists("/proc/self/status")
  function() {
    if (have_proc) {
      x <- tryCatch(readLines("/proc/self/status", warn=FALSE), error=function(e) character())
      line <- x[startsWith(x, "VmRSS:")]
      if (length(line)) {
        kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", line)))
        if (!is.na(kb)) return(kb / 1024)
      }
    }
    kb <- suppressWarnings(as.numeric(system("ps -o rss= -p $$", intern=TRUE)))
    if (!is.na(kb)) return(kb / 1024)
    NA_real_
  }
})
.__record_perf <- function(kind, label, t0, t1, rss0, rss1, status="ok", notes="") {
  .__event_idx <<- .__event_idx + 1L
  safe_time <- function(x) { if (is.null(x) || length(x)==0) as.POSIXct(NA) else as.POSIXct(x) }
  safe_num  <- function(x) { if (is.null(x) || length(x)==0) NA_real_ else as.numeric(x)[1] }
  row <- data.frame(
    event_id=.__event_idx, kind=as.character(kind)[1], label=as.character(label)[1],
    start_clock=safe_time(t0), end_clock=safe_time(t1),
    elapsed_s=if(!is.null(t0)&&!is.null(t1)) as.numeric(difftime(t1,t0,units="secs")) else NA_real_,
    rss_before_mb=safe_num(rss0), rss_after_mb=safe_num(rss1),
    rss_delta_mb=if(is.na(safe_num(rss0))||is.na(safe_num(rss1))) NA_real_ else safe_num(rss1)-safe_num(rss0),
    status=as.character(status)[1], notes=as.character(notes)[1],
    stringsAsFactors=FALSE
  )
  assign(".__perf_log", rbind(get(".__perf_log", envir=.GlobalEnv), row), envir=.GlobalEnv)
  invisible(NULL)
}
.step <- local({
  i <- 0
  function(msg) {
    i <<- i + 1
    stamp <- format(Sys.time(), "%H:%M:%S")
    cat(sprintf("[%-2d][%s | +%6.1fs] %s\n", i, stamp, .time_since(), msg))
    flush.console()
    now <- Sys.time(); rss <- .mem_mb()
    .__record_perf(kind="marker", label=msg, t0=now, t1=now, rss0=rss, rss1=rss, status="ok", notes="step")
  }
})

# --- CLI parser --------------------------------------------------------------
.parse_cli <- function() {
  args <- commandArgs(trailingOnly=TRUE)
  kv <- list()
  for (a in args) {
    if (startsWith(a, "--")) {
      a2 <- sub("^--", "", a)
      if (grepl("=", a2, fixed=TRUE)) {
        parts <- strsplit(a2, "=", fixed=TRUE)[[1]]
        kv[[parts[1]]] <- parts[2]
      } else {
        kv[[a2]] <- TRUE
      }
    }
  }
  kv
}
cli <- .parse_cli()

# --- Paths and mode ----------------------------------------------------------
project_root  <- normalizePath(file.path(getwd(), "..", ".."), mustWork=FALSE)

dataset_root  <- if (!is.null(cli[["dataset-root"]]))  cli[["dataset-root"]]  else
                 if (nzchar(Sys.getenv("DATASET_ROOT"))) Sys.getenv("DATASET_ROOT") else
                 file.path(project_root, "datasets")

dataset_folder <- if (!is.null(cli[["dataset-folder"]])) cli[["dataset-folder"]] else
                  if (nzchar(Sys.getenv("DATASET_FOLDER"))) Sys.getenv("DATASET_FOLDER") else
                  "cellexplorer_cell_type"
dataset_folder <- sub("^/+", "", dataset_folder)
dataset_path   <- file.path(dataset_root, dataset_folder)

out_dir <- if (!is.null(cli[["out-dir"]])) cli[["out-dir"]] else
           if (nzchar(Sys.getenv("OUT_DIR"))) Sys.getenv("OUT_DIR") else
           file.path(project_root, "results", "physmap_pca", dataset_folder)

# Holdout mode when --holdout-fold is provided
holdout_mode     <- !is.null(cli[["holdout-fold"]])
holdout_fold     <- if (holdout_mode) as.integer(cli[["holdout-fold"]]) else NA_integer_
n_holdout_folds  <- if (!is.null(cli[["n-holdout-folds"]])) as.integer(cli[["n-holdout-folds"]]) else 5L
min_cells_per_animal <- if (!is.null(cli[["min-cells-per-animal"]])) as.integer(cli[["min-cells-per-animal"]]) else 50L
grouping_col     <- if (!is.null(cli[["grouping-col"]])) cli[["grouping-col"]] else "animal_id"

# --- Hyperparameters ---------------------------------------------------------
numPCs     <- if (!is.null(cli[["num-pcs"]]))    as.integer(cli[["num-pcs"]])    else
              if (nzchar(Sys.getenv("PHYSMAP_NUMPCS"))) as.integer(Sys.getenv("PHYSMAP_NUMPCS")) else 5L
.nfeatures <- if (!is.null(cli[["nfeatures"]]))  as.integer(cli[["nfeatures"]]) else
              if (nzchar(Sys.getenv("PHYSMAP_NFEATURES"))) as.integer(Sys.getenv("PHYSMAP_NFEATURES")) else 20L

# --- Libraries ---------------------------------------------------------------
.step("Loading libraries")
library(dplyr)
library(caret)

# --- Output dirs -------------------------------------------------------------
if (holdout_mode) {
  fold_out <- file.path(out_dir, dataset_folder, paste0("fold_", holdout_fold))
  dir.create(fold_out, recursive=TRUE, showWarnings=FALSE)
  .perf_csv_path <- file.path(fold_out, "perf_log.csv")
} else {
  bm_dir <- file.path(out_dir, "benchmark_results", dataset_folder)
  dir.create(bm_dir, recursive=TRUE, showWarnings=FALSE)
  .perf_csv_path <- file.path(bm_dir, "perf_log.csv")
}

.on_exit_write_perf <- function() {
  .__record_perf(kind="marker", label=sprintf("Session end: R %s", getRversion()),
    t0=Sys.time(), t1=Sys.time(), rss0=.mem_mb(), rss1=.mem_mb(), status="ok", notes="session")
  try({ write.csv(get(".__perf_log", envir=.GlobalEnv), .perf_csv_path, row.names=FALSE) }, silent=TRUE)
}
reg.finalizer(environment(), function(e) .on_exit_write_perf(), onexit=TRUE)

.step(sprintf("Mode: %s | dataset: %s | numPCs: %d | nfeatures: %d",
              if (holdout_mode) sprintf("holdout fold %d/%d", holdout_fold, n_holdout_folds) else "transductive",
              dataset_folder, numPCs, .nfeatures))
.step(sprintf("Paths: root=%s | out=%s", dataset_root, out_dir))

# --- Load data ---------------------------------------------------------------
.step("Loading CSVs")
modality_files <- list(WF="waveforms.csv", ISI="isi_dist.csv", AUTOCORR="acg.csv")
available_modalities <- c()
modality_data <- list()
for (mod in names(modality_files)) {
  fpath <- file.path(dataset_path, modality_files[[mod]])
  if (file.exists(fpath)) {
    .step(sprintf("Loading modality %s", mod))
    modality_data[[mod]] <- read.csv(fpath)
    available_modalities <- c(available_modalities, mod)
  }
}
if (length(available_modalities) == 0)
  stop(sprintf("No modality CSVs found under %s", dataset_path))

labels_path <- file.path(dataset_path, "labels.csv")
if (!file.exists(labels_path)) stop(sprintf("labels.csv not found in %s", dataset_path))
cType <- read.csv(labels_path)$label

# Coerce to numeric
numify <- function(df)
  as.data.frame(lapply(df, function(x) as.numeric(as.character(x))), stringsAsFactors=FALSE)
for (mod in available_modalities) modality_data[[mod]] <- numify(modality_data[[mod]])

# Drop rows with NA/NaN/Inf
rows_ok <- rep(TRUE, length(cType))
for (mod in available_modalities) rows_ok <- rows_ok & complete.cases(modality_data[[mod]])
if (!all(rows_ok)) {
  .step(sprintf("Dropping %d cells with NA/NaN/Inf", sum(!rows_ok)))
  for (mod in available_modalities) modality_data[[mod]] <- modality_data[[mod]][rows_ok, , drop=FALSE]
  cType <- cType[rows_ok]
}

# Label filtering (mirrors HIPPIE protocol)
MIN_CELLS_PER_CLASS <- 10L
normalize_label <- function(lbl) {
  s <- as.character(lbl)
  if (is.na(s) || length(s) == 0L) return("")
  if (nchar(s) >= 3L && substr(s,1L,2L) %in% c("b'","b\"")) {
    last <- substr(s, nchar(s), nchar(s))
    if (last %in% c("'","\"")) s <- substr(s, 3L, nchar(s)-1L)
  }
  trimws(s)
}
is_unlabeled <- function(lbl) {
  s <- normalize_label(lbl)
  if (nchar(s) == 0L) return(TRUE)
  s_low <- tolower(s)
  any(vapply(c("unlabeled","unlabelled","unknown","juxt","axo","vgat"),
             function(w) grepl(w, s_low, fixed=TRUE), logical(1)))
}
cType <- vapply(cType, normalize_label, character(1))
cType[cType == ""] <- "unknown"
unlabeled_mask <- vapply(cType, is_unlabeled, logical(1))
if (any(unlabeled_mask)) {
  .step(sprintf("Dropping %d unlabeled rows", sum(unlabeled_mask)))
  for (mod in available_modalities) modality_data[[mod]] <- modality_data[[mod]][!unlabeled_mask, , drop=FALSE]
  cType <- cType[!unlabeled_mask]
}
class_counts <- table(cType)
keep_classes  <- names(class_counts[class_counts >= MIN_CELLS_PER_CLASS])
rare_mask     <- !(cType %in% keep_classes)
if (any(rare_mask)) {
  .step(sprintf("Dropping %d cells from rare classes (< %d)", sum(rare_mask), MIN_CELLS_PER_CLASS))
  for (mod in available_modalities) modality_data[[mod]] <- modality_data[[mod]][!rare_mask, , drop=FALSE]
  cType <- cType[!rare_mask]
}

# Drop zero-waveform units (all-zero WF rows → degenerate PCA)
if ("WF" %in% available_modalities) {
  wf_mat <- as.matrix(modality_data[["WF"]])
  nonzero <- rowSums(abs(wf_mat)) > 0
  if (!all(nonzero)) {
    .step(sprintf("Dropping %d zero-waveform units", sum(!nonzero)))
    for (mod in available_modalities) modality_data[[mod]] <- modality_data[[mod]][nonzero, , drop=FALSE]
    cType <- cType[nonzero]
  }
}

N <- length(cType)
.step(sprintf("After filtering: %d cells, %d classes, modalities: %s",
              N, length(unique(cType)), paste(available_modalities, collapse=", ")))

# --- Holdout: load metadata and build animal split ---------------------------
if (holdout_mode) {
  meta_path <- file.path(dataset_path, "metadata.csv")
  if (!file.exists(meta_path)) stop(sprintf("metadata.csv not found in %s (required for holdout)", dataset_path))
  metadata <- read.csv(meta_path, stringsAsFactors=FALSE)

  # Apply same row filtering as data (rows_ok + unlabeled_mask + rare_mask + nonzero)
  # We need to track the surviving row indices relative to the original metadata.
  # Re-apply filters in order on the metadata.
  meta_ok <- rows_ok
  meta_ok[meta_ok][unlabeled_mask] <- FALSE
  # rare_mask and nonzero_mask are relative to the already-filtered vector:
  # rebuild a full-length surviving mask
  # Simpler: just use nrow(modality_data[[ available_modalities[1] ]]) == N
  # and trust metadata rows align with data rows after the same filtering.
  # We apply the filters sequentially to metadata just as we did to modality_data.
  meta_filtered <- metadata[rows_ok, , drop=FALSE]
  meta_filtered <- meta_filtered[!unlabeled_mask[rows_ok], , drop=FALSE]
  meta_filtered <- meta_filtered[!rare_mask, , drop=FALSE]
  if ("WF" %in% available_modalities) {
    # nonzero mask was computed after rare filtering, so apply it here
    wf_after_rare <- as.matrix(modality_data[["WF"]])  # already at final N rows
    # We already filtered modality_data above; meta_filtered should also be length N
    # Apply nonzero filter if it changed things
    if (nrow(meta_filtered) > N) {
      # nonzero filter removed some rows — recompute on meta_filtered
      wf_pre_nonzero <- as.matrix(read.csv(file.path(dataset_path, "waveforms.csv")))
      wf_pre_nonzero <- numify(as.data.frame(wf_pre_nonzero[rows_ok,,drop=FALSE]))
      wf_pre_nonzero <- wf_pre_nonzero[!unlabeled_mask[rows_ok],,drop=FALSE]
      wf_pre_nonzero <- wf_pre_nonzero[!rare_mask,,drop=FALSE]
      nz2 <- rowSums(abs(as.matrix(wf_pre_nonzero))) > 0
      meta_filtered <- meta_filtered[nz2, , drop=FALSE]
    }
  }
  stopifnot(nrow(meta_filtered) == N)

  if (!(grouping_col %in% names(meta_filtered)))
    stop(sprintf("grouping column '%s' not found in metadata.csv. Columns: %s",
                 grouping_col, paste(names(meta_filtered), collapse=", ")))

  animal_ids <- meta_filtered[[grouping_col]]
  counts     <- table(animal_ids)
  valid_animals <- sort(names(counts[counts >= min_cells_per_animal]))
  if (length(valid_animals) < n_holdout_folds)
    stop(sprintf("Only %d valid animals (need >= %d folds)", length(valid_animals), n_holdout_folds))

  set.seed(42)
  shuffled <- sample(valid_animals)
  fold_size  <- length(valid_animals) %/% n_holdout_folds
  fold_start <- holdout_fold * fold_size + 1
  fold_end   <- if (holdout_fold < n_holdout_folds - 1) fold_start + fold_size - 1 else length(valid_animals)

  test_animals  <- shuffled[fold_start:fold_end]
  train_animals <- shuffled[-c(fold_start:fold_end)]

  train_idx <- which(animal_ids %in% train_animals)
  test_idx  <- which(animal_ids %in% test_animals)
  .step(sprintf("Holdout fold %d: train %d animals / %d cells, test %d animals / %d cells",
                holdout_fold, length(train_animals), length(train_idx),
                length(test_animals), length(test_idx)))
}

# --- Core PCA helper ---------------------------------------------------------
# fit_pca_inductive: select top-nfeat variable columns from X_tr, fit prcomp(),
#   project X_tr (train) and X_te (test) through the same rotation.
# For transductive mode pass X_tr == X_te == all data.
fit_pca_inductive <- function(X_tr, X_te, nfeat, n_pcs, label="") {
  col_vars  <- apply(X_tr, 2, var)
  keep_cols <- which(is.finite(col_vars) & col_vars > 0)
  if (length(keep_cols) == 0) stop(sprintf("[%s] no variable features in training data", label))
  if (length(keep_cols) > nfeat)
    keep_cols <- keep_cols[order(col_vars[keep_cols], decreasing=TRUE)[seq_len(nfeat)]]

  X_tr_sub <- X_tr[, keep_cols, drop=FALSE]
  X_te_sub <- X_te[, keep_cols, drop=FALSE]

  n_pcs_eff <- min(n_pcs, nrow(X_tr_sub) - 1L, ncol(X_tr_sub))
  pca <- prcomp(X_tr_sub, center=TRUE, scale.=TRUE, rank.=n_pcs_eff)

  tr_emb <- pca$x[, seq_len(n_pcs_eff), drop=FALSE]
  te_emb <- predict(pca, newdata=X_te_sub)[, seq_len(n_pcs_eff), drop=FALSE]

  tr_emb[!is.finite(tr_emb)] <- 0
  te_emb[!is.finite(te_emb)] <- 0

  .step(sprintf("[%s] prcomp: %d features -> %d PCs (train %d, test %d cells)",
                label, length(keep_cols), n_pcs_eff, nrow(tr_emb), nrow(te_emb)))
  list(train=tr_emb, test=te_emb, n_pcs=n_pcs_eff)
}

# --- Balanced accuracy -------------------------------------------------------
balanced_accuracy <- function(truth, pred) {
  truth <- as.character(truth); pred <- as.character(pred)
  classes <- sort(unique(truth))
  recalls <- vapply(classes, function(cls) {
    in_cls <- truth == cls
    if (sum(in_cls) == 0) return(NA_real_)
    sum(pred[in_cls] == cls) / sum(in_cls)
  }, numeric(1))
  mean(recalls, na.rm=TRUE)
}

# --- L2-normalise rows of a data frame (feature cols only) ------------------
l2_norm_df <- function(df) {
  feat_cols <- setdiff(names(df), "CellType")
  mat <- as.matrix(df[, feat_cols])
  nrms <- sqrt(rowSums(mat^2)); nrms[nrms < 1e-12] <- 1e-12
  df[, feat_cols] <- mat / nrms
  df
}

# =============================================================================
# TRANSDUCTIVE MODE
# =============================================================================
if (!holdout_mode) {
  .step("=== TRANSDUCTIVE MODE ===")

  # --- Fit PCA on ALL data (both train and test are the full dataset) --------
  per_mod_pca <- list()
  for (mod in available_modalities) {
    X_all <- as.matrix(modality_data[[mod]])
    pca_res <- fit_pca_inductive(X_all, X_all, .nfeatures, numPCs, label=mod)
    per_mod_pca[[mod]] <- pca_res$train  # same as test in transductive
  }

  # Combined (concatenated PCA)
  combined_all <- do.call(cbind, per_mod_pca)
  colnames(combined_all) <- paste0("PC", seq_len(ncol(combined_all)))

  # Build data frames
  modality_labels <- c(WF="WF PCA", ISI="ISI PCA", AUTOCORR="Autocorr PCA")
  modality_outnames <- c(WF="PCA_WF_CV_results.csv",
                         ISI="PCA_ISI_CV_results.csv",
                         AUTOCORR="PCA_AUTOCORR_CV_results.csv")

  per_mod_dfs <- list()
  for (mod in available_modalities) {
    df <- as.data.frame(per_mod_pca[[mod]])
    colnames(df) <- paste0("PC", seq_len(ncol(df)))
    df$CellType <- cType
    per_mod_dfs[[mod]] <- df
  }
  combined_df <- as.data.frame(combined_all)
  combined_df$CellType <- cType

  # --- KNN 5-fold CV ----------------------------------------------------------
  run_knn_cv <- function(df, label, outname) {
    .step(sprintf("%s: KNN 5-fold CV, k in 1..20", label))
    df <- l2_norm_df(df)

    k_grid <- data.frame(k=seq_len(20))
    set.seed(42)
    knn_model <- train(CellType ~ ., data=df, method="knn",
                       trControl=trainControl(method="cv", number=5, savePredictions="all"),
                       tuneGrid=k_grid)

    per_k <- knn_model$pred
    fold_best_k <- per_k %>%
      dplyr::group_by(Resample, k) %>%
      dplyr::summarise(bal_acc=balanced_accuracy(obs, pred), .groups="drop") %>%
      dplyr::group_by(Resample) %>%
      dplyr::slice_max(bal_acc, n=1, with_ties=FALSE) %>%
      dplyr::ungroup() %>%
      dplyr::select(Resample, k, fold_bal_acc=bal_acc)

    best_preds <- per_k %>%
      dplyr::inner_join(fold_best_k %>% dplyr::select(Resample, k), by=c("Resample","k")) %>%
      dplyr::arrange(rowIndex) %>%
      dplyr::transmute(true=obs, pred=pred, fold=Resample)

    best_bal_acc <- balanced_accuracy(best_preds$true, best_preds$pred)
    best_k_mode  <- as.integer(names(sort(table(fold_best_k$k), decreasing=TRUE))[1])
    .step(sprintf("%s: best k (mode)=%d, pooled balanced_acc=%.3f", label, best_k_mode, best_bal_acc))

    out_path <- file.path(out_dir, "benchmark_results", dataset_folder, outname)
    write.csv(best_preds, out_path, row.names=FALSE)
    invisible(list(label=label, best_k=best_k_mode, best_balanced_accuracy=best_bal_acc,
                   fold_best_k=fold_best_k))
  }

  knn_summaries <- list()
  for (mod in available_modalities)
    knn_summaries[[mod]] <- run_knn_cv(per_mod_dfs[[mod]], modality_labels[[mod]], modality_outnames[[mod]])

  knn_summaries[["CONCAT"]] <- run_knn_cv(combined_df, "PhysMAP-PCA (concat)", "physmap_CV_results.csv")

  # Summary CSV
  summary_rows <- do.call(rbind, lapply(names(knn_summaries), function(m) {
    s <- knn_summaries[[m]]
    data.frame(modality=m, label=s$label, best_k=s$best_k,
               best_balanced_accuracy=s$best_balanced_accuracy, stringsAsFactors=FALSE)
  }))
  summary_path <- file.path(out_dir, "benchmark_results", dataset_folder, "knn_summary.csv")
  write.csv(summary_rows, summary_path, row.names=FALSE)
  .step(sprintf("Wrote KNN summary -> %s", summary_path))

} else {

# =============================================================================
# HOLDOUT MODE
# =============================================================================
  .step("=== HOLDOUT MODE ===")

  modality_labels   <- c(WF="WF PCA", ISI="ISI PCA", AUTOCORR="Autocorr PCA")
  modality_outnames <- c(WF="PCA_WF_holdout_results.csv",
                         ISI="PCA_ISI_holdout_results.csv",
                         AUTOCORR="PCA_AUTOCORR_holdout_results.csv")

  # --- Fit PCA on TRAIN animals only; project test via predict() -------------
  per_mod_pca <- list()
  for (mod in available_modalities) {
    X_all   <- as.matrix(modality_data[[mod]])
    X_train <- X_all[train_idx, , drop=FALSE]
    X_test  <- X_all[test_idx,  , drop=FALSE]
    per_mod_pca[[mod]] <- fit_pca_inductive(X_train, X_test, .nfeatures, numPCs, label=mod)
  }

  cType_train <- cType[train_idx]
  cType_test  <- cType[test_idx]

  # Per-modality data frames
  per_mod_dfs_train <- list(); per_mod_dfs_test <- list()
  for (mod in available_modalities) {
    tr_df <- as.data.frame(per_mod_pca[[mod]]$train)
    colnames(tr_df) <- paste0("PC", seq_len(ncol(tr_df)))
    tr_df$CellType  <- cType_train
    per_mod_dfs_train[[mod]] <- tr_df

    te_df <- as.data.frame(per_mod_pca[[mod]]$test)
    colnames(te_df) <- paste0("PC", seq_len(ncol(te_df)))
    te_df$CellType  <- cType_test
    per_mod_dfs_test[[mod]] <- te_df
  }

  # Combined (concatenated PCA)
  combined_train_mat <- do.call(cbind, lapply(per_mod_pca, function(p) p$train))
  combined_test_mat  <- do.call(cbind, lapply(per_mod_pca, function(p) p$test))
  colnames(combined_train_mat) <- paste0("PC", seq_len(ncol(combined_train_mat)))
  colnames(combined_test_mat)  <- paste0("PC", seq_len(ncol(combined_test_mat)))

  combined_train_df <- as.data.frame(combined_train_mat); combined_train_df$CellType <- cType_train
  combined_test_df  <- as.data.frame(combined_test_mat);  combined_test_df$CellType  <- cType_test

  # --- KNN: train on train animals, evaluate on test animals -----------------
  run_holdout_eval <- function(train_df, test_df, label, outname) {
    .step(sprintf("%s: KNN holdout, k sweep 1..20", label))
    train_df <- l2_norm_df(train_df)
    test_df  <- l2_norm_df(test_df)

    feat_cols <- setdiff(names(train_df), "CellType")
    test_feats <- test_df[, feat_cols, drop=FALSE]

    per_k_results <- lapply(seq_len(20), function(k) {
      set.seed(123)
      mdl   <- train(CellType ~ ., data=train_df, method="knn",
                     trControl=trainControl(method="none"),
                     tuneGrid=data.frame(k=k))
      preds <- predict(mdl, newdata=test_feats)
      list(k=k, preds=preds, bal_acc=balanced_accuracy(test_df$CellType, preds))
    })

    best_idx   <- which.max(vapply(per_k_results, function(r) r$bal_acc, numeric(1)))
    best       <- per_k_results[[best_idx]]
    raw_acc    <- mean(best$preds == test_df$CellType)

    .step(sprintf("%s: best k=%d, balanced_acc=%.3f, raw_acc=%.3f",
                  label, best$k, best$bal_acc, raw_acc))

    results_df <- data.frame(
      label=test_df$CellType, prediction=best$preds,
      fold=paste0("holdout_", holdout_fold), stringsAsFactors=FALSE
    )
    out_path <- file.path(fold_out, outname)
    write.csv(results_df, out_path, row.names=FALSE)
    invisible(best$bal_acc)
  }

  acc <- list()
  for (mod in available_modalities)
    acc[[mod]] <- run_holdout_eval(per_mod_dfs_train[[mod]], per_mod_dfs_test[[mod]],
                                  modality_labels[[mod]], modality_outnames[[mod]])
  acc[["CONCAT"]] <- run_holdout_eval(combined_train_df, combined_test_df,
                                      "PhysMAP-PCA (concat)", "physmap_holdout_results.csv")

  # Save animal split
  animal_split_df <- data.frame(
    animal_id=c(train_animals, test_animals),
    split=c(rep("train", length(train_animals)), rep("test", length(test_animals))),
    fold=holdout_fold, stringsAsFactors=FALSE
  )
  write.csv(animal_split_df, file.path(fold_out, "animal_split.csv"), row.names=FALSE)

  # Holdout summary
  summary_df <- data.frame(
    fold=holdout_fold,
    n_train_animals=length(train_animals), n_test_animals=length(test_animals),
    n_train_cells=length(train_idx),       n_test_cells=length(test_idx),
    wf_accuracy=if("WF" %in% available_modalities) acc[["WF"]] else NA,
    isi_accuracy=if("ISI" %in% available_modalities) acc[["ISI"]] else NA,
    autocorr_accuracy=if("AUTOCORR" %in% available_modalities) acc[["AUTOCORR"]] else NA,
    wnn_accuracy=acc[["CONCAT"]],
    stringsAsFactors=FALSE
  )
  write.csv(summary_df, file.path(fold_out, "holdout_summary.csv"), row.names=FALSE)
  .step(sprintf("Wrote holdout summary -> %s", file.path(fold_out, "holdout_summary.csv")))

} # end holdout mode

# --- Compute parity CSV ------------------------------------------------------
.write_compute_parity_row <- function() {
  csv_path <- Sys.getenv("COMPUTE_PARITY_CSV")
  if (!nzchar(csv_path)) csv_path <- file.path(project_root, "results", "compute_parity", "timings.csv")
  dir.create(dirname(csv_path), recursive=TRUE, showWarnings=FALSE)
  run_id  <- sprintf("physmap-pca-%d-%s", as.integer(Sys.time()),
                     paste(sample(c(letters,0:9),8,replace=TRUE), collapse=""))
  elapsed <- as.numeric((proc.time() - get(".start_time", envir=.GlobalEnv))[3])
  row <- data.frame(
    run_id=run_id, method="physmap-pca", dataset=dataset_folder,
    phase="total", elapsed_s=elapsed, peak_gpu_mb=NA,
    n_cells=N, n_classes=length(unique(cType)),
    holdout_fold=if(holdout_mode) holdout_fold else NA,
    stringsAsFactors=FALSE
  )
  if (file.exists(csv_path)) {
    existing <- tryCatch(read.csv(csv_path), error=function(e) NULL)
    if (!is.null(existing)) { row <- dplyr::bind_rows(existing, row) }
  }
  write.csv(row, csv_path, row.names=FALSE)
  .step(sprintf("Wrote compute parity row -> %s", csv_path))
}
.write_compute_parity_row()

.step("physmap_script_pca.r complete")
try(write.csv(get(".__perf_log", envir=.GlobalEnv), .perf_csv_path, row.names=FALSE), silent=TRUE)
