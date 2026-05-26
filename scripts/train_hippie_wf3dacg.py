#!/usr/bin/env python3
"""
hippie-wf3dacg transductive training script.

Protocol (mirrors the HIPPIE training script exactly):
  1. Load all neurons from a C4 H5 file.
  2. Precompute 3D ACGs from spike_indices (expensive, done once).
  3. Train the hippie-wf3dacg CVAE transductively on ALL neurons for --epochs.
  4. CV split on labeled neurons only (stratified k-fold, seed 42).
  5. Evaluate with KNN (cosine k=1..20), sklearn MLP, LogisticRegression.
  6. Write transductive_predictions.csv, mlp_predictions.csv,
     linear_probe_predictions.csv to --output-dir.

Region conditioning
-------------------
  --use-region-conditioning  Enable super_region and layer embeddings.
                             Technology conditioning is always on.

Usage
-----
    python scripts/train_hippie_wf3dacg.py \\
        --dataset hull_cell_type \\
        --cv-fold 0 \\
        --z-dim 100 \\
        --beta 0.5 \\
        --epochs 100 \\
        --data-root /data/datasets \\
        --output-dir /data/results/hippie-wf3dacg/hull_cell_type/config_1/fold_0

    # With region conditioning:
    python scripts/train_hippie_wf3dacg.py ... --use-region-conditioning
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as SKLearnMLP
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── repo path setup ───────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
for _p in [str(_REPO), str(_REPO / "hippie_wf3dacg")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hippie_wf3dacg import (
    HippieWF3DACGConfig,
    HippieWF3DACGDataset,
    HippieWF3DACGLightning,
    LAYER_IDS,
    SUPER_REGION_IDS,
    TECHNOLOGY_IDS,
    load_c4_dataset,
    make_dataloader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_h5(data_root: str, dataset: str) -> str:
    """Return the single .h5 file under <data_root>/<dataset>/."""
    pattern = os.path.join(data_root, dataset, "*.h5")
    hits    = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No .h5 file found at {pattern}. "
            "Run download_unified_dataset.py --subdir hippie first."
        )
    if len(hits) > 1:
        print(f"WARNING: multiple H5 files found; using {hits[0]}", file=sys.stderr)
    return hits[0]


def _make_subset(ds: HippieWF3DACGDataset, indices: np.ndarray) -> HippieWF3DACGDataset:
    """Slice a HippieWF3DACGDataset to the given neuron indices."""
    return HippieWF3DACGDataset(
        waveforms=ds.waveforms[indices],
        spike_times=[ds.spike_times[i] for i in indices],
        labels=ds.labels[indices],
        source_ids=ds.source_ids[indices],
        wave_len=ds.wave_len,
        acg_win_ms=ds.acg_win_ms,
        acg_bin_ms=ds.acg_bin_ms,
        acg_n_deciles=ds.acg_n_deciles,
        acg_scale_factor=ds.acg_scale_factor,
        precomputed_acgs=ds.precomputed_acgs[indices] if ds.precomputed_acgs is not None else None,
        super_region_ids=ds.super_region_ids[indices] if ds.super_region_ids is not None else None,
        technology_ids=ds.technology_ids[indices]     if ds.technology_ids   is not None else None,
        layer_ids=ds.layer_ids[indices]               if ds.layer_ids        is not None else None,
    )


def _extract_embeddings(
    model: HippieWF3DACGLightning,
    ds: HippieWF3DACGDataset,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract z-mean embeddings for every neuron in ds."""
    loader = make_dataloader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    device = next(model.parameters()).device
    model.eval()
    all_embs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            bdata, _ = batch
            bdata    = {k: v.to(device) for k, v in bdata.items()}
            emb = model.get_embeddings(
                bdata["waveform"],
                bdata["acg"],
                bdata["source_id"],
                super_region_ids=bdata.get("super_region_id"),
                technology_ids=bdata.get("technology_id"),
                layer_ids=bdata.get("layer_id"),
            )
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def _run_probes(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    knn_random_state: int = 42,
):
    """KNN (cosine, best k in 1..20 by inner train-CV) + MLP + LogisticRegression."""
    def _l2norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    Xtr_l2, Xte_l2 = _l2norm(train_emb), _l2norm(test_emb)

    # k is selected by stratified inner CV on the TRAINING fold only, then
    # evaluated once on the held-out test fold. See
    # hippie/utils.py:select_knn_k_by_train_cv for rationale.
    from hippie.utils import select_knn_k_by_train_cv
    knn_result = select_knn_k_by_train_cv(
        Xtr_l2,
        train_labels,
        Xte_l2,
        test_labels,
        k_grid=range(1, 21),
        inner_cv_splits=5,
        metric="cosine",
        random_state=knn_random_state,
    )
    best_k = knn_result["best_k"]
    best_ba = knn_result["test_balanced_accuracy"]
    best_preds = knn_result["test_predictions"]
    print(f"  KNN  best k={best_k} (selected by train-CV)  BA={best_ba:.4f}")

    sc  = StandardScaler()
    Xtr = sc.fit_transform(train_emb)
    Xte = sc.transform(test_emb)

    mlp = SKLearnMLP(
        hidden_layer_sizes=(256, 128), max_iter=500,
        random_state=42, early_stopping=True, validation_fraction=0.1,
    )
    mlp.fit(Xtr, train_labels)
    mlp_preds = mlp.predict(Xte)
    print(f"  MLP  BA={balanced_accuracy_score(test_labels, mlp_preds):.4f}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr, train_labels)
    lr_preds = lr.predict(Xte)
    print(f"  LR   BA={balanced_accuracy_score(test_labels, lr_preds):.4f}")

    return best_preds, mlp_preds, lr_preds


def _write_timings(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="hippie-wf3dacg transductive sweep: CVAE on all neurons, probes on labeled subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name, e.g. hull_cell_type")
    parser.add_argument("--cv-fold",    type=int, default=0,
                        help="CV fold index (0 to n-cv-folds-1)")
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--z-dim",      type=int, default=100,
                        help="CVAE latent dimension (matches the cross-species "
                             "Fig 5 protocol)")
    parser.add_argument("--beta",       type=float, default=0.5,
                        help="KL regularisation weight (beta_kl)")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    # ACG parameters
    parser.add_argument("--acg-win-ms",  type=float, default=100.0,
                        help="ACG window in ms (default 100 → 101 bins at 1 ms)")
    parser.add_argument("--acg-bin-ms",  type=float, default=1.0,
                        help="ACG bin size in ms")
    parser.add_argument("--acg-n-deciles", type=int, default=10,
                        help="Number of IFR deciles for 3D ACG")
    parser.add_argument("--acg-scale-factor", type=float, default=10.0,
                        help="Multiply ACG values before loss (NEMO convention)")
    parser.add_argument("--beta-acg", type=float, default=1.0,
                        help="Weight for ACG reconstruction loss relative to wave. "
                             "Both losses are computed in unscaled space so 1.0 "
                             "means equal weight regardless of acg_scale_factor.")
    parser.add_argument("--cross-modal-weight", type=float, default=0.1,
                        help="Weight for cross-modal contrastive alignment loss "
                             "(wave_feats ↔ acg_feats). Restores NEMO's bimodal "
                             "alignment signal. Set 0 to disable.")
    parser.add_argument("--cross-modal-temperature", type=float, default=0.5,
                        help="Temperature for cross-modal contrastive loss.")
    parser.add_argument("--acg-cache-dir", default=None,
                        help="Directory to cache precomputed ACGs as .npy. "
                             "Defaults to the same directory as the H5 file. "
                             "Pass 'none' to disable caching.")
    parser.add_argument("--data-root",  default="./datasets",
                        help="Root dir containing downloaded C4 H5 files")
    parser.add_argument("--output-dir", required=True,
                        help="Write prediction CSVs and cv_split.json here")
    parser.add_argument("--timings-csv",
                        default="results/compute_parity/timings.csv",
                        help="Path for the per-run timings CSV (relative to "
                             "CWD). Default writes to the repo's "
                             "results/compute_parity/ tree.")
    parser.add_argument("--unlabeled-strings", nargs="+", default=["", "unlabeled"])
    # Region conditioning
    parser.add_argument("--use-region-conditioning", action="store_true",
                        help="Enable super_region and layer embeddings. "
                             "Technology conditioning is always active.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("hippie-wf3dacg transductive job")
    print(f"  dataset             : {args.dataset}")
    print(f"  cv_fold             : {args.cv_fold}/{args.n_cv_folds}")
    print(f"  z_dim               : {args.z_dim}")
    print(f"  beta_kl             : {args.beta}")
    print(f"  epochs              : {args.epochs}")
    print(f"  acg_win_ms          : {args.acg_win_ms}")
    print(f"  acg_bin_ms          : {args.acg_bin_ms}")
    print(f"  acg_n_deciles       : {args.acg_n_deciles}")
    print(f"  region_conditioning : {args.use_region_conditioning}")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    t_load  = time.time()
    h5_path = _find_h5(args.data_root, args.dataset)
    print(f"\nLoading {h5_path} ...")
    data    = load_c4_dataset(h5_path, args.dataset, source_id=0)
    N       = len(data["labels"])
    load_time = time.time() - t_load
    print(f"Loaded {N} neurons in {load_time:.1f}s")
    lbl_counts: dict = {}
    for s in data["str_labels"]:
        lbl_counts[s] = lbl_counts.get(s, 0) + 1
    print(f"Label distribution: {lbl_counts}")

    # ── 2. Identify labeled neurons ──────────────────────────────────────────
    unlabeled  = set(args.unlabeled_strings)
    is_labeled = np.array([s not in unlabeled for s in data["str_labels"]])
    labeled_idx = np.where(is_labeled)[0]
    print(f"\nLabeled neurons: {len(labeled_idx)} / {N}")

    if len(labeled_idx) < 2 * args.n_cv_folds:
        print(
            f"ERROR: only {len(labeled_idx)} labeled neurons — too few for "
            f"{args.n_cv_folds}-fold CV",
            file=sys.stderr,
        )
        return 1

    # ── 3. Label encoder + CV split ──────────────────────────────────────────
    labeled_str = [data["str_labels"][i] for i in labeled_idx]
    le          = LabelEncoder()
    le.fit(labeled_str)
    labeled_int = le.transform(labeled_str)

    skf    = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=42)
    splits = list(skf.split(labeled_idx, labeled_int))
    train_pos, test_pos = splits[args.cv_fold]
    print(f"CV fold {args.cv_fold}: train={len(train_pos)}  test={len(test_pos)}")
    print(f"Classes: {list(le.classes_)}")

    # ── 4. Build config and full dataset ─────────────────────────────────────
    config = HippieWF3DACGConfig(
        z_dim=args.z_dim,
        beta_kl=args.beta,
        beta_acg=args.beta_acg,
        use_augmentations=True,
        reconstruction_consistency_weight=0.15,
        warmup_epochs=args.warmup_epochs,
        acg_win_ms=args.acg_win_ms,
        acg_bin_ms=args.acg_bin_ms,
        acg_n_deciles=args.acg_n_deciles,
        acg_scale_factor=args.acg_scale_factor,
        cross_modal_contrastive_weight=args.cross_modal_weight,
        cross_modal_temperature=args.cross_modal_temperature,
    )

    print(f"\nBuilding dataset ({N} neurons) ...")
    full_ds = HippieWF3DACGDataset(
        waveforms=data["waveforms"],
        spike_times=data["spike_times"],
        labels=data["labels"],
        source_ids=data["source_ids"],
        wave_len=config.wave_len,
        acg_win_ms=config.acg_win_ms,
        acg_bin_ms=config.acg_bin_ms,
        acg_n_deciles=config.acg_n_deciles,
        acg_scale_factor=config.acg_scale_factor,
        super_region_ids=data["super_region_ids"],
        technology_ids=data["technology_ids"],
        layer_ids=data["layer_ids"],
    )

    # Build ACG cache path: <cache_dir>/acgs_w<win>_b<bin>_d<deciles>.npy
    _n_bins = int(round(config.acg_win_ms / config.acg_bin_ms)) + 1
    print(f"Precomputing 3D ACGs ({config.acg_n_deciles} deciles × {_n_bins} bins) ...")
    t_acg = time.time()

    _use_cache = args.acg_cache_dir != "none"
    _cache_path: str | None = None
    if _use_cache:
        _cache_dir = args.acg_cache_dir or os.path.dirname(h5_path)
        _cache_file = (
            f"acgs_w{int(config.acg_win_ms)}_b{config.acg_bin_ms}"
            f"_d{config.acg_n_deciles}.npy"
        )
        _cache_path = os.path.join(_cache_dir, _cache_file)

    if _cache_path and os.path.exists(_cache_path):
        print(f"  Loading ACGs from cache: {_cache_path}")
        full_ds.precomputed_acgs = np.load(_cache_path)
    else:
        full_ds.precompute_acgs()
        if _cache_path:
            np.save(_cache_path, full_ds.precomputed_acgs)
            print(f"  Saved ACG cache to: {_cache_path}")

    acg_time = time.time() - t_acg
    print(f"ACG precomputation done in {acg_time:.1f}s")

    # ── 5. Build model ────────────────────────────────────────────────────────
    model = HippieWF3DACGLightning(
        config=config,
        num_sources=1,
        num_classes=len(data["label_names"]),
        num_super_regions=len(SUPER_REGION_IDS) if args.use_region_conditioning else None,
        num_technologies=len(TECHNOLOGY_IDS),
        num_layers=len(LAYER_IDS) if args.use_region_conditioning else None,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # ── 6. Transductive training ──────────────────────────────────────────────
    eff_bs = min(args.batch_size, len(full_ds))
    train_loader = make_dataloader(full_ds, batch_size=eff_bs, shuffle=True, drop_last=False)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=0)],
        logger=False,
    )
    print(f"\nTraining for {args.epochs} epochs ...")
    t_train = time.time()
    trainer.fit(model, train_loader)
    train_time = time.time() - t_train
    print(f"Training done in {train_time:.1f}s ({train_time/args.epochs:.1f}s/epoch)")

    # ── 7. Extract embeddings ─────────────────────────────────────────────────
    labeled_ds = _make_subset(full_ds, labeled_idx)
    print("\nExtracting embeddings for labeled neurons...")
    t_emb      = time.time()
    labeled_emb = _extract_embeddings(model, labeled_ds)
    emb_time    = time.time() - t_emb

    if len(labeled_emb) != len(labeled_idx):
        print(
            f"WARNING: extracted {len(labeled_emb)} embeddings for "
            f"{len(labeled_idx)} labeled neurons — some __getitem__ calls failed",
            file=sys.stderr,
        )
        n_ok      = len(labeled_emb)
        train_pos = train_pos[train_pos < n_ok]
        test_pos  = test_pos[test_pos  < n_ok]

    print(f"Embeddings: {labeled_emb.shape}  ({emb_time:.1f}s)")
    print(f"NaN/Inf: {np.isnan(labeled_emb).sum()} / {np.isinf(labeled_emb).sum()}")

    train_emb    = labeled_emb[train_pos]
    test_emb     = labeled_emb[test_pos]
    train_labels = labeled_int[train_pos]
    test_labels  = labeled_int[test_pos]

    # ── 8. Probes ─────────────────────────────────────────────────────────────
    print("\nRunning probes...")
    t_probe = time.time()
    knn_preds, mlp_preds, lr_preds = _run_probes(
        train_emb, train_labels, test_emb, test_labels
    )
    probe_time = time.time() - t_probe

    # ── 10. Write prediction CSVs ─────────────────────────────────────────────
    original_test_str = le.inverse_transform(labeled_int[test_pos].astype(int))

    def _write_preds(preds: np.ndarray, fname: str):
        df = pd.DataFrame({
            "pred": le.inverse_transform(preds.astype(int)),
            "true": original_test_str,
        })
        df.to_csv(os.path.join(args.output_dir, fname), index=False)

    _write_preds(knn_preds, "transductive_predictions.csv")
    _write_preds(mlp_preds, "mlp_predictions.csv")
    _write_preds(lr_preds,  "linear_probe_predictions.csv")

    with open(os.path.join(args.output_dir, "cv_split.json"), "w") as f:
        json.dump(
            {
                "train_global_indices": labeled_idx[train_pos].tolist(),
                "test_global_indices":  labeled_idx[test_pos].tolist(),
                "fold":                  args.cv_fold,
                "n_folds":               args.n_cv_folds,
                "z_dim":                 args.z_dim,
                "beta":                  args.beta,
                "acg_win_ms":            args.acg_win_ms,
                "acg_bin_ms":            args.acg_bin_ms,
                "acg_n_deciles":         args.acg_n_deciles,
                "use_region_conditioning": args.use_region_conditioning,
            },
            f, indent=2,
        )

    # ── 11. Timings CSV ───────────────────────────────────────────────────────
    timings = [
        {"phase": "data_load",   "seconds": round(load_time,  1), "n_cells": N},
        {"phase": "acg_precomp", "seconds": round(acg_time,   1), "n_cells": N},
        {"phase": "train_cvae",  "seconds": round(train_time, 1), "n_cells": N,
         "epochs": args.epochs},
        {"phase": "embed+probe", "seconds": round(emb_time + probe_time, 1),
         "n_cells": len(labeled_idx)},
    ]
    _write_timings(args.timings_csv, timings)

    print(f"\nResults written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
