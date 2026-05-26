#!/usr/bin/env Rscript

# =================== CLI ===================
.parse_cli <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- list()
  for (a in args) {
    if (startsWith(a, "--")) {
      a2 <- sub("^--", "", a)
      if (grepl("=", a2, fixed = TRUE)) {
        parts <- strsplit(a2, "=", fixed = TRUE)[[1]]
        if (length(parts) == 2) kv[[parts[1]]] <- parts[2]
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
default_out_dir <- file.path(project_root, "results", "physmap", "cross_dataset")
default_root    <- file.path(project_root, "datasets")
default_train   <- "cellexplorer_cell_type"
default_predict <- "hull_cell_type"

out_dir <- if (!is.null(cli[["out-dir"]])) cli[["out-dir"]] else
  if (nzchar(Sys.getenv("OUT_DIR"))) Sys.getenv("OUT_DIR") else default_out_dir

dataset_root <- if (!is.null(cli[["dataset-root"]])) cli[["dataset-root"]] else
  if (nzchar(Sys.getenv("DATASET_ROOT"))) Sys.getenv("DATASET_ROOT") else default_root

train_folder <- if (!is.null(cli[["train-folder"]])) cli[["train-folder"]] else
  if (nzchar(Sys.getenv("TRAIN_FOLDER"))) Sys.getenv("TRAIN_FOLDER") else default_train

predict_folder <- if (!is.null(cli[["predict-folder"]])) cli[["predict-folder"]] else
  if (nzchar(Sys.getenv("PREDICT_FOLDER"))) Sys.getenv("PREDICT_FOLDER") else default_predict

train_folder   <- sub("^/+","", train_folder)
predict_folder <- sub("^/+","", predict_folder)
train_path     <- file.path(dataset_root, train_folder)
predict_path   <- file.path(dataset_root, predict_folder)

# =================== Libs & Config ===================
suppressPackageStartupMessages({
  library(tidyr); library(dplyr); library(caret); library(ggplot2)
})

UMAP.SEED <- 3
numPCs    <- 10
nfeatures <- 40
set.seed(123)

dir.create(file.path(out_dir, "benchmark_results"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)

# =================== Helpers ===================
.sanitize_colnames <- function(df, prefix="V") {
  cn <- colnames(df); if (is.null(cn)) cn <- rep("", NCOL(df))
  cn[is.na(cn) | cn == ""] <- paste0(prefix, which(is.na(cn) | cn == ""))
  cn <- make.unique(cn, sep = "_"); colnames(df) <- cn; df
}

force_frame <- function(x, prefix="V") {
  if (is.null(x)) return(data.frame())
  if (is.data.frame(x)) {
    if (NCOL(x) == 0 && NROW(x) > 0) {
      x <- data.frame(V1 = as.numeric(x), check.names = FALSE)
    }
  } else if (is.matrix(x)) {
    x <- as.data.frame(x, check.names = FALSE)
  } else if (is.vector(x)) {
    x <- data.frame(V1 = as.numeric(x), check.names = FALSE)
  } else {
    x <- as.data.frame(x, check.names = FALSE)
  }
  x <- .sanitize_colnames(x, prefix)
  x
}

fallback_if_empty <- function(df, who) {
  if (NCOL(df) == 0) {
    rn <- NROW(df)
    if (rn == 0) return(data.frame(FALLBACK = numeric()))
    set.seed(42)
    df <- data.frame(FALLBACK = seq_len(rn) + rnorm(rn, 0, 0.1))
  }
  df
}

impute_numeric_df <- function(df, who, prefix="V") {
  df <- force_frame(df, prefix)
  if (NCOL(df) == 0) {
    return(fallback_if_empty(df, who))
  }
  df[] <- lapply(df, function(x) { 
    if (is.factor(x)) x <- as.character(x)
    suppressWarnings(as.numeric(x)) 
  })
  valid_cols <- vapply(df, function(x) {
    numeric_x <- suppressWarnings(as.numeric(x))
    sum(is.finite(numeric_x)) > 1
  }, logical(1))
  if (!any(valid_cols)) {
    return(fallback_if_empty(data.frame(), who))
  }
  if (!all(valid_cols)) {
    df <- df[, valid_cols, drop = FALSE]
  }
  all_na_cols <- vapply(df, function(x) all(is.na(x)), logical(1))
  if (any(all_na_cols)) {
    df <- df[, !all_na_cols, drop = FALSE]
  }
  df <- fallback_if_empty(df, who)
  if (NCOL(df) == 0) stop(sprintf("[%s] No usable numeric columns after cleaning.", who))
  nf_mask <- !is.finite(as.matrix(df))
  df[nf_mask] <- NA_real_
  for (j in seq_len(NCOL(df))) {
    col <- df[[j]]
    if (anyNA(col)) {
      med <- suppressWarnings(stats::median(col, na.rm = TRUE)); 
      if (!is.finite(med)) med <- 0
      col[is.na(col)] <- med; df[[j]] <- col
    }
  }
  stopifnot(all(is.finite(as.matrix(df))))
  df
}

load_dataset <- function(path) {
  read_mod <- function(fname, who, prefix) {
    p <- file.path(path, fname)
    if (!file.exists(p)) return(data.frame())
    df <- tryCatch({
      read.csv(p, check.names = FALSE, stringsAsFactors = FALSE)
    }, error = function(e) { data.frame() })
    if (NROW(df) == 0) return(data.frame())
    if (NCOL(df) > 0) {
      first <- colnames(df)[1]
      looks_index <- first %in% c("X","V1","","index","Id","ID","Unnamed: 0","row","row_id")
      if (!looks_index && is.numeric(suppressWarnings(as.numeric(df[[1]])))) {
        looks_index <- tryCatch({
          identical(suppressWarnings(as.numeric(df[[1]])), seq_len(NROW(df)))
        }, error = function(e) FALSE)
      }
      if (looks_index) df <- df[, -1, drop = FALSE]
    }
    impute_numeric_df(df, sprintf("%s/%s", who, fname), prefix = prefix)
  }
  who <- basename(path)
  wf  <- read_mod("waveforms.csv", who, "WF")
  isi <- read_mod("isi_dist.csv",  who, "ISI")
  acg <- read_mod("acg.csv",       who, "ACG")
  labels_path <- file.path(path, "labels.csv")
  if (!file.exists(labels_path)) stop(sprintf("labels.csv not found in %s", path))
  labels <- read.csv(labels_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (is.null(labels$label)) stop("labels.csv must contain a 'label' column")
  cType <- labels$label
  lengths <- c(NROW(wf), NROW(isi), NROW(acg), length(cType))
  lengths <- lengths[lengths > 0]
  if (length(lengths) == 0) stop("All input files are empty")
  n <- min(lengths)
  if (NROW(wf) == 0) wf <- data.frame(matrix(0, nrow = n, ncol = 1, dimnames = list(NULL, "WF_dummy")))
  if (NROW(isi) == 0) isi <- data.frame(matrix(0, nrow = n, ncol = 1, dimnames = list(NULL, "ISI_dummy")))
  if (NROW(acg) == 0) acg <- data.frame(matrix(0, nrow = n, ncol = 1, dimnames = list(NULL, "ACG_dummy")))
  if (NROW(wf) != n || NROW(isi) != n || NROW(acg) != n || length(cType) != n) {
    wf  <- wf [seq_len(min(n, NROW(wf))), , drop = FALSE]
    isi <- isi[seq_len(min(n, NROW(isi))), , drop = FALSE]
    acg <- acg[seq_len(min(n, NROW(acg))), , drop = FALSE]
    cType <- cType[seq_len(min(n, length(cType)))]

    if (NROW(wf) < n) {
      padding <- data.frame(matrix(0, nrow = n - NROW(wf), ncol = max(1, NCOL(wf))))
      if (NCOL(wf) > 0) colnames(padding) <- colnames(wf)
      wf <- rbind(wf, padding)
    }
    if (NROW(isi) < n) {
      padding <- data.frame(matrix(0, nrow = n - NROW(isi), ncol = max(1, NCOL(isi))))
      if (NCOL(isi) > 0) colnames(padding) <- colnames(isi)
      isi <- rbind(isi, padding)
    }
    if (NROW(acg) < n) {
      padding <- data.frame(matrix(0, nrow = n - NROW(acg), ncol = max(1, NCOL(acg))))
      if (NCOL(acg) > 0) colnames(padding) <- colnames(acg)
      acg <- rbind(acg, padding)
    }
    if (length(cType) < n) {
      cType <- c(cType, rep("unknown", n - length(cType)))
    }
  }
  wf  <- .sanitize_colnames(force_frame(wf,  "WF"),  "WF")
  isi <- .sanitize_colnames(force_frame(isi, "ISI"), "ISI")
  acg <- .sanitize_colnames(force_frame(acg, "ACG"), "ACG")
  list(wf = wf, isi = isi, acg = acg, label = factor(replace(cType, is.na(cType) | cType == "", "unknown")))
}

top_var_cols <- function(df, k) {
  df <- force_frame(df)
  if (NCOL(df) == 0) stop("top_var_cols: data frame has 0 columns.")
  v <- vapply(df, function(col) {
    if (length(unique(col)) <= 1) return(0)
    var_val <- stats::var(col, na.rm = TRUE)
    if (is.na(var_val) || !is.finite(var_val)) return(0)
    return(var_val)
  }, numeric(1))
  v <- v[is.finite(v) & v > 0]
  if (!length(v)) return(colnames(df)[1])
  o <- order(v, decreasing = TRUE)
  head(names(v)[o], min(k, length(o)))
}

fit_pca <- function(train_df, feature_names, npcs) {
  train_df <- force_frame(train_df)
  use <- intersect(feature_names, colnames(train_df))
  if (!length(use)) stop("fit_pca: no overlap between requested features and training columns.")
  X <- train_df[, use, drop = FALSE]
  if (NCOL(X) == 1) {
    center <- mean(X[[1]], na.rm = TRUE)
    scale_val <- sd(X[[1]], na.rm = TRUE)
    if (scale_val == 0 || !is.finite(scale_val)) scale_val <- 1
    pr <- list(
      rotation = matrix(1, nrow = 1, ncol = 1, dimnames = list(colnames(X), "PC1")),
      center = center,
      scale = scale_val,
      sdev = scale_val
    )
    class(pr) <- "prcomp"
    return(list(pr = pr, center = setNames(center, colnames(X)), 
                scale = setNames(scale_val, colnames(X)), features = colnames(X)))
  }
  X_scaled <- scale(X, center = TRUE, scale = TRUE)
  if (any(!is.finite(X_scaled))) X_scaled[!is.finite(X_scaled)] <- 0
  pr <- prcomp(X_scaled, center = FALSE, scale. = FALSE, rank. = min(npcs, NCOL(X_scaled), NROW(X_scaled) - 1))
  center <- attr(X_scaled, "scaled:center"); names(center) <- colnames(X_scaled)
  scalev <- attr(X_scaled, "scaled:scale");  names(scalev) <- colnames(X_scaled)
  if (any(!is.finite(center))) center[!is.finite(center)] <- 0
  if (any(!is.finite(scalev))) scalev[!is.finite(scalev)] <- 1
  list(pr = pr, center = center, scale = scalev, features = colnames(X_scaled))
}

transform_pca <- function(model, new_df) {
  new_df <- force_frame(new_df)
  feats <- model$features
  missing <- setdiff(feats, colnames(new_df))
  if (length(missing)) {
    add <- as.data.frame(matrix(rep(model$center[missing], each = NROW(new_df)),
                                nrow = NROW(new_df),
                                dimnames = list(NULL, missing)))
    new_df <- cbind(new_df, add)
  }
  X <- as.matrix(new_df[, feats, drop = FALSE])
  X <- sweep(X, 2, model$center[feats], "-")
  X <- sweep(X, 2, model$scale [feats], "/")
  X[!is.finite(X)] <- 0
  result <- X %*% model$pr$rotation
  as.data.frame(result)
}

make_embeddings <- function(wf_df, isi_df, acg_df, models=NULL) {
  wf_df  <- force_frame(wf_df,  "WF")
  isi_df <- force_frame(isi_df, "ISI")
  acg_df <- force_frame(acg_df, "ACG")
  wf_df  <- fallback_if_empty(wf_df,  "WF")
  isi_df <- fallback_if_empty(isi_df, "ISI")
  acg_df <- fallback_if_empty(acg_df, "ACG")
  if (is.null(models)) {
    wf_cols  <- top_var_cols(wf_df,  min(nfeatures, NCOL(wf_df)))
    isi_cols <- top_var_cols(isi_df, min(nfeatures, NCOL(isi_df)))
    acg_cols <- top_var_cols(acg_df, min(nfeatures, NCOL(acg_df)))
    wf_m  <- fit_pca(wf_df,  wf_cols,  min(numPCs, length(wf_cols)))
    isi_m <- fit_pca(isi_df, isi_cols, min(numPCs, length(isi_cols)))
    acg_m <- fit_pca(acg_df, acg_cols, min(numPCs, length(acg_cols)))
    wf_emb  <- transform_pca(wf_m,  wf_df)
    isi_emb <- transform_pca(isi_m, isi_df)
    acg_emb <- transform_pca(acg_m, acg_df)
  } else {
    wf_emb  <- transform_pca(models$wf,  wf_df)
    isi_emb <- transform_pca(models$isi, isi_df)
    acg_emb <- transform_pca(models$acg, acg_df)
  }
  emb_concat <- cbind(wf_emb, isi_emb, acg_emb)
  colnames(emb_concat) <- c(
    sprintf("WF_PC%02d",  seq_len(NCOL(wf_emb))),
    sprintf("ISI_PC%02d", seq_len(NCOL(isi_emb))),
    sprintf("ACG_PC%02d", seq_len(NCOL(acg_emb)))
  )
  if (is.null(models)) {
    list(emb=list(wf=wf_emb, isi=isi_emb, acg=acg_emb, physmap=emb_concat),
         models=list(wf=wf_m, isi=isi_m, acg=acg_m))
  } else {
    list(emb=list(wf=wf_emb, isi=isi_emb, acg=acg_emb, physmap=emb_concat), models=models)
  }
}

save_embeddings <- function(emb_list, split, tag) {
  dir.create(file.path(out_dir, "embeddings"), recursive = TRUE, showWarnings = FALSE)
  write.csv(emb_list$wf,      file.path(out_dir, "embeddings", sprintf("%s_%s_wf_pca.csv",      tag, split)), row.names = FALSE)
  write.csv(emb_list$isi,     file.path(out_dir, "embeddings", sprintf("%s_%s_isi_pca.csv",     tag, split)), row.names = FALSE)
  write.csv(emb_list$acg,     file.path(out_dir, "embeddings", sprintf("%s_%s_acg_pca.csv",     tag, split)), row.names = FALSE)
  write.csv(emb_list$physmap, file.path(out_dir, "embeddings", sprintf("%s_%s_physmap.csv",     tag, split)), row.names = FALSE)
}

# --- NEW: helpers to append truth/preds to embedding CSVs ----------------------
append_labels_to_embeddings <- function(tag, split, truth = NULL, preds = list()) {
  # preds: named list with optional elements: physmap, wf, isi, acg (character vectors)
  base_dir <- file.path(out_dir, "embeddings")
  # files to update
  files <- list(
    wf      = file.path(base_dir, sprintf("%s_%s_wf_pca.csv",      tag, split)),
    isi     = file.path(base_dir, sprintf("%s_%s_isi_pca.csv",     tag, split)),
    acg     = file.path(base_dir, sprintf("%s_%s_acg_pca.csv",     tag, split)),
    physmap = file.path(base_dir, sprintf("%s_%s_physmap.csv",     tag, split))
  )
  for (nm in names(files)) {
    fp <- files[[nm]]
    if (!file.exists(fp)) next
    df <- read.csv(fp, check.names = FALSE, stringsAsFactors = FALSE)
    # add true label if available
    if (!is.null(truth)) {
      df$true_label <- as.character(truth)[seq_len(NROW(df))]
    }
    # add predicted columns if provided (all four as available)
    if (!is.null(preds$physmap)) df$pred_physmap <- as.character(preds$physmap)[seq_len(NROW(df))]
    if (!is.null(preds$wf))      df$pred_wf      <- as.character(preds$wf)[seq_len(NROW(df))]
    if (!is.null(preds$isi))     df$pred_isi     <- as.character(preds$isi)[seq_len(NROW(df))]
    if (!is.null(preds$acg))     df$pred_acg     <- as.character(preds$acg)[seq_len(NROW(df))]
    write.csv(df, fp, row.names = FALSE)
  }
}

# =================== Pipeline ===================
cat("\n=== TIMING: Starting PhysMap pipeline ===\n")
time_total_start <- Sys.time()

cat("Loading datasets...\n")
time_load_start <- Sys.time()
train <- load_dataset(train_path)
pred  <- load_dataset(predict_path)
time_load_end <- Sys.time()
time_load <- as.numeric(difftime(time_load_end, time_load_start, units = "secs"))
cat(sprintf("  Data loading time: %.2f seconds\n", time_load))

cat("Generating embeddings (PCA)...\n")
time_embed_start <- Sys.time()
train_emb <- make_embeddings(train$wf, train$isi, train$acg, models = NULL)
pred_emb  <- make_embeddings(pred$wf, pred$isi, pred$acg, models = train_emb$models)
time_embed_end <- Sys.time()
time_embed <- as.numeric(difftime(time_embed_end, time_embed_start, units = "secs"))
cat(sprintf("  Embedding generation time: %.2f seconds\n", time_embed))

tag_train   <- sprintf("train_%s",   train_folder)
tag_predict <- sprintf("predict_%s", predict_folder)

# Save initial embeddings
save_embeddings(train_emb$emb, "train",   tag_train)
save_embeddings(pred_emb$emb,  "predict", tag_predict)

# Append true labels to TRAIN embeddings (no predictions on train in this flow)
append_labels_to_embeddings(tag_train, "train", truth = train$label)

# --- KNN with return of predictions + truth column in outputs ------------------
run_knn_train_predict <- function(train_df, train_y, test_df, test_y = NULL, label, outname) {
  train_df <- as.data.frame(train_df)
  test_df  <- as.data.frame(test_df)
  if (NROW(train_df) != length(train_y)) {
    stop(sprintf("Training data rows (%d) don't match labels (%d)", NROW(train_df), length(train_y)))
  }
  if (NCOL(train_df) == 0) stop("Training data has no features")

  df_train <- train_df
  df_train$CellType <- as.factor(train_y)

  tr_ctrl <- trainControl(method = "cv", number = 10, savePredictions = "final")
  set.seed(123)
  # Sweep k = 1..20 explicitly (matches HIPPIE / NEMO / physmap_script.r protocol).
  # tuneLength = 20 lets caret choose 20 candidate k values automatically, which
  # are not necessarily {1..20}, so we use an explicit tuneGrid instead.
  k_grid <- data.frame(k = seq_len(20))
  knn_model <- train(CellType ~ ., data = df_train, method = "knn",
                     trControl = tr_ctrl, tuneGrid = k_grid)

  preds <- predict(knn_model, newdata = test_df)
  out <- data.frame(
    sample_id = seq_len(NROW(test_df)),
    prediction = as.character(preds),
    stringsAsFactors = FALSE
  )
  # add truth if provided
  if (!is.null(test_y)) {
    out$truth <- as.character(test_y)[seq_len(NROW(out))]
  }

  out_path <- file.path(out_dir, "benchmark_results",
                        sprintf("%s__train_%s__predict_%s.csv", outname, train_folder, predict_folder))
  write.csv(out, out_path, row.names = FALSE)

  return(as.character(preds))
}

# KNN runs (collect predictions for predict split)
cat("Running KNN models...\n")
time_knn_start <- Sys.time()

cat("  Training PHYSmap KNN...\n")
preds_physmap <- run_knn_train_predict(train_emb$emb$physmap, train$label,
                                       pred_emb$emb$physmap, pred$label,
                                       "PHYSmap (WF+ISI+ACG PCs)", "knn_predictions_physmap")

cat("  Training WF KNN...\n")
preds_wf <- run_knn_train_predict(train_emb$emb$wf, train$label,
                                  pred_emb$emb$wf, pred$label,
                                  "WF PCA", "knn_predictions_wf")

cat("  Training ISI KNN...\n")
preds_isi <- run_knn_train_predict(train_emb$emb$isi, train$label,
                                   pred_emb$emb$isi, pred$label,
                                   "ISI PCA", "knn_predictions_isi")

cat("  Training ACG KNN...\n")
preds_acg <- run_knn_train_predict(train_emb$emb$acg, train$label,
                                   pred_emb$emb$acg, pred$label,
                                   "ACG PCA", "knn_predictions_acg")

time_knn_end <- Sys.time()
time_knn <- as.numeric(difftime(time_knn_end, time_knn_start, units = "secs"))
cat(sprintf("  KNN training and prediction time: %.2f seconds\n", time_knn))

# Append true/predicted labels to PREDICT embeddings (all modalities)
append_labels_to_embeddings(
  tag_predict, "predict",
  truth = pred$label,
  preds = list(
    physmap = preds_physmap,
    wf      = preds_wf,
    isi     = preds_isi,
    acg     = preds_acg
  )
)

time_total_end <- Sys.time()
time_total <- as.numeric(difftime(time_total_end, time_total_start, units = "secs"))

cat("\n=== TIMING SUMMARY ===\n")
cat(sprintf("Data loading/processing:  %.2f seconds (%.1f%%)\n", time_load, 100*time_load/time_total))
cat(sprintf("PCA embedding generation: %.2f seconds (%.1f%%)\n", time_embed, 100*time_embed/time_total))
cat(sprintf("KNN training/prediction:  %.2f seconds (%.1f%%)\n", time_knn, 100*time_knn/time_total))
cat(sprintf("TOTAL TIME:               %.2f seconds (%.2f minutes)\n", time_total, time_total/60))
cat("======================\n\n")

# Save timing results to file
timing_results <- data.frame(
  stage = c("data_loading", "pca_embedding", "knn_training", "total"),
  time_seconds = c(time_load, time_embed, time_knn, time_total),
  percentage = c(100*time_load/time_total, 100*time_embed/time_total, 100*time_knn/time_total, 100)
)
write.csv(timing_results, file.path(out_dir, "timing_results.csv"), row.names = FALSE)
cat(sprintf("Timing results saved to: %s/timing_results.csv\n", out_dir))
