#!/usr/bin/env python3
"""
hippie-wf3dacg cross-dataset training script.

Mirrors hippie cross_dataset_script.py but uses NEMO's encoders (waveform MLP +
3D ACG ConvNet) inside the HIPPIE CVAE framework.

Protocol:
  1. Train hippie-wf3dacg CVAE on the SOURCE dataset (all neurons, unsupervised).
  2. Extract z-mean embeddings for all labeled SOURCE neurons.
  3. Extract z-mean embeddings for all labeled TARGET neurons (same model weights,
     no fine-tuning — target neurons are processed as if they were source neurons).
  4. KNN (cosine, best k in 1..20) from source embeddings → target labels.
  5. Write predictions.csv, train_embeddings.csv, predict_embeddings.csv.

The shared-cell-type filter is applied inside the evaluation so that only classes
present in BOTH datasets are scored.

Usage
-----
    python scripts/train_hippie_wf3dacg_cross_dataset.py \\
        --training-dataset hull_cell_type \\
        --predict-dataset  lisberger_labeled_cell_type \\
        --z-dim 100 \\
        --beta 0.5 \\
        --epochs 100 \\
        --data-root /data/datasets \\
        --output-dir /data/results/hippie-wf3dacg-cross-dataset/cross_hull_to_lisberger
"""
from __future__ import annotations

import argparse
import csv
import glob
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

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
    pattern = os.path.join(data_root, dataset, "*.h5")
    hits    = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No .h5 file found at {pattern}.")
    if len(hits) > 1:
        print(f"WARNING: multiple H5 files; using {hits[0]}", file=sys.stderr)
    return hits[0]


def _filter_data_to_classes(data: dict, keep_classes: set) -> dict:
    """Return a copy of data dict with only neurons whose str_label is in keep_classes."""
    mask = np.array([s in keep_classes for s in data["str_labels"]])
    if mask.sum() == 0:
        raise ValueError(f"No neurons remain after filtering to {keep_classes}")
    idx = np.where(mask)[0]
    wf = data["waveforms"]
    wf_filtered = wf[idx] if wf.dtype != object else np.array([wf[i] for i in idx], dtype=object)
    filtered = {
        "waveforms":        wf_filtered,
        "spike_times":      [data["spike_times"][i] for i in idx],
        "labels":           data["labels"][idx],
        "str_labels":       [data["str_labels"][i] for i in idx],
        "label_names":      sorted(keep_classes & set(data["str_labels"])),
        "source_ids":       data["source_ids"][idx],
    }
    for k in ("super_region_ids", "technology_ids", "layer_ids"):
        if data.get(k) is not None:
            filtered[k] = data[k][idx]
        else:
            filtered[k] = None
    return filtered


def _load_dataset(data_root: str, dataset: str, source_id: int) -> dict:
    h5 = _find_h5(data_root, dataset)
    print(f"\nLoading {h5} ...")
    t0   = time.time()
    data = load_c4_dataset(h5, dataset, source_id=source_id)
    print(f"Loaded {len(data['labels'])} neurons in {time.time()-t0:.1f}s")
    cnts: dict = {}
    for s in data["str_labels"]:
        cnts[s] = cnts.get(s, 0) + 1
    print(f"Label distribution: {cnts}")
    return data


def _build_dataset(data: dict, config: HippieWF3DACGConfig, cache_dir: str | None) -> HippieWF3DACGDataset:
    ds = HippieWF3DACGDataset(
        waveforms=data["waveforms"],
        spike_times=data["spike_times"],
        labels=data["labels"],
        source_ids=data["source_ids"],
        wave_len=config.wave_len,
        acg_win_ms=config.acg_win_ms,
        acg_bin_ms=config.acg_bin_ms,
        acg_n_deciles=config.acg_n_deciles,
        acg_scale_factor=config.acg_scale_factor,
        super_region_ids=data.get("super_region_ids"),
        technology_ids=data.get("technology_ids"),
        layer_ids=data.get("layer_ids"),
    )
    _n_bins     = int(round(config.acg_win_ms / config.acg_bin_ms)) + 1
    _cache_file = (f"acgs_w{int(config.acg_win_ms)}_b{config.acg_bin_ms}"
                   f"_d{config.acg_n_deciles}.npy")
    _cache_path = os.path.join(cache_dir, _cache_file) if cache_dir else None

    if _cache_path and os.path.exists(_cache_path):
        print(f"  Loading ACGs from cache: {_cache_path}")
        ds.precomputed_acgs = np.load(_cache_path)
    else:
        print(f"Precomputing 3D ACGs ({config.acg_n_deciles} deciles × {_n_bins} bins) ...")
        t0 = time.time()
        ds.precompute_acgs()
        print(f"ACG precomputation done in {time.time()-t0:.1f}s")
        if _cache_path:
            np.save(_cache_path, ds.precomputed_acgs)
            print(f"  Saved ACG cache: {_cache_path}")
    return ds


def _make_subset(ds: HippieWF3DACGDataset, indices: np.ndarray) -> HippieWF3DACGDataset:
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


def _concat_waveforms(arrays: list) -> np.ndarray:
    """Concatenate waveform arrays that may be 2D float or 1D object dtype."""
    if any(a.dtype == object for a in arrays):
        rows = [row for a in arrays for row in (a if a.dtype == object else a)]
        out = np.empty(len(rows), dtype=object)
        for i, r in enumerate(rows):
            out[i] = r
        return out
    return np.concatenate(arrays)


def _concat_datasets(datasets: list) -> HippieWF3DACGDataset:
    """Concatenate multiple HippieWF3DACGDatasets (all must have precomputed_acgs)."""
    ref = datasets[0]
    return HippieWF3DACGDataset(
        waveforms=_concat_waveforms([d.waveforms for d in datasets]),
        spike_times=sum([list(d.spike_times) for d in datasets], []),
        labels=np.concatenate([d.labels for d in datasets]),
        source_ids=np.concatenate([d.source_ids for d in datasets]),
        wave_len=ref.wave_len,
        acg_win_ms=ref.acg_win_ms,
        acg_bin_ms=ref.acg_bin_ms,
        acg_n_deciles=ref.acg_n_deciles,
        acg_scale_factor=ref.acg_scale_factor,
        precomputed_acgs=np.concatenate([d.precomputed_acgs for d in datasets]),
        super_region_ids=(np.concatenate([d.super_region_ids for d in datasets])
                          if all(d.super_region_ids is not None for d in datasets) else None),
        technology_ids=(np.concatenate([d.technology_ids for d in datasets])
                        if all(d.technology_ids is not None for d in datasets) else None),
        layer_ids=(np.concatenate([d.layer_ids for d in datasets])
                   if all(d.layer_ids is not None for d in datasets) else None),
    )


def _write_timings(path: str, rows: list):
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
        description="hippie-wf3dacg cross-dataset transfer: train on source, predict on target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-dataset", required=True,
                        help="Source dataset (train on this)")
    parser.add_argument("--predict-dataset", required=True,
                        help="Target dataset (predict on this)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Phase 1: pretrain on Hausser+source+target combined, "
                             "then fine-tune on source only (mirrors hippie-cross-dataset-allpretrain).")
    parser.add_argument("--pretrain-datasets", nargs="+",
                        default=["hausser_cell_type"],
                        help="Extra datasets included in Phase 1 pretraining corpus.")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="Epochs for Phase 1 pretraining. Defaults to --epochs.")
    parser.add_argument("--pretrain-shared-only", action="store_true",
                        help="Filter all pretrain datasets to only cell types shared between "
                             "source and target datasets (removes unlabeled/irrelevant neurons).")
    parser.add_argument("--z-dim",     type=int,   default=100)
    parser.add_argument("--beta",      type=float, default=0.5)
    parser.add_argument("--beta-acg",  type=float, default=1.0)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--batch-size", type=int,  default=256)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--acg-win-ms",     type=float, default=100.0)
    parser.add_argument("--acg-bin-ms",     type=float, default=1.0)
    parser.add_argument("--acg-n-deciles",  type=int,   default=10)
    parser.add_argument("--acg-scale-factor", type=float, default=10.0)
    parser.add_argument("--cross-modal-weight", type=float, default=0.0)
    parser.add_argument("--cross-modal-temperature", type=float, default=0.5)
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--output-dir", required=True,
                        help="Write predictions.csv and embeddings here")
    parser.add_argument("--timings-csv",
                        default="results/compute_parity/timings.csv")
    parser.add_argument("--unlabeled-strings", nargs="+", default=["", "unlabeled"])
    parser.add_argument("--min-class-cells", type=int, default=10,
                        help="Minimum labeled cells per class in BOTH source and target "
                             "for a class to be included in the shared evaluation set. "
                             "Matches the filter used in hyperparam tuning (default: 10).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    unlabeled = set(args.unlabeled_strings)

    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else args.epochs

    print("=" * 60)
    print("hippie-wf3dacg cross-dataset job")
    print(f"  training_dataset : {args.training_dataset}")
    print(f"  predict_dataset  : {args.predict_dataset}")
    print(f"  pretrain         : {args.pretrain}")
    if args.pretrain:
        print(f"  pretrain_datasets: {args.pretrain_datasets}")
        print(f"  pretrain_epochs  : {pretrain_epochs}")
    print(f"  z_dim            : {args.z_dim}")
    print(f"  beta_kl          : {args.beta}")
    print(f"  beta_acg         : {args.beta_acg}")
    print(f"  epochs           : {args.epochs}")
    print("=" * 60)

    # ── 1. Load source and target datasets ───────────────────────────────────
    t0 = time.time()
    src_data = _load_dataset(args.data_root, args.training_dataset, source_id=0)
    tgt_data = _load_dataset(args.data_root, args.predict_dataset,  source_id=0)

    # Load extra pretrain datasets if requested (name → data dict pairs)
    extra_named: list[tuple[str, dict]] = []
    if args.pretrain:
        for ds_name in args.pretrain_datasets:
            if ds_name in (args.training_dataset, args.predict_dataset):
                continue  # source and target are already included in pretrain corpus
            print(f"\nLoading extra pretrain dataset: {ds_name}")
            extra_named.append((ds_name, _load_dataset(args.data_root, ds_name, source_id=0)))

    load_time = time.time() - t0

    # ── 2. Build config and hippie-wf3dacg model ────────────────────────────────
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

    # ── 3. Build datasets and precompute ACGs ────────────────────────────────
    src_cache = os.path.dirname(_find_h5(args.data_root, args.training_dataset))
    tgt_cache = os.path.dirname(_find_h5(args.data_root, args.predict_dataset))

    print(f"\nBuilding source dataset ({len(src_data['labels'])} neurons) ...")
    src_ds = _build_dataset(src_data, config, src_cache)

    print(f"\nBuilding target dataset ({len(tgt_data['labels'])} neurons) ...")
    tgt_ds = _build_dataset(tgt_data, config, tgt_cache)

    # Build extra pretrain datasets
    extra_ds_list = []
    for ds_name, ed in extra_named:
        ed_cache = os.path.dirname(_find_h5(args.data_root, ds_name))
        print(f"\nBuilding extra pretrain dataset {ds_name} ({len(ed['labels'])} neurons) ...")
        extra_ds_list.append(_build_dataset(ed, config, ed_cache))

    # ── 4. Build model ────────────────────────────────────────────────────────
    model = HippieWF3DACGLightning(
        config=config,
        num_sources=1,
        num_classes=len(src_data["label_names"]),
        num_super_regions=None,
        num_technologies=len(TECHNOLOGY_IDS),
        num_layers=None,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}")

    pretrain_time = 0.0
    # ── 4a. Phase 1: unsupervised pretraining on all datasets (optional) ──────
    if args.pretrain:
        if args.pretrain_shared_only:
            # Keep only cell types with >= min_class_cells in BOTH source and target
            from collections import Counter as _Counter
            _src_cnt = _Counter(s for s in src_data["str_labels"] if s not in unlabeled)
            _tgt_cnt = _Counter(s for s in tgt_data["str_labels"] if s not in unlabeled)
            shared_classes = {
                c for c in (_src_cnt.keys() & _tgt_cnt.keys())
                if _src_cnt[c] >= args.min_class_cells and _tgt_cnt[c] >= args.min_class_cells
            }
            print(f"\n[pretrain] Filtering to shared classes ({len(shared_classes)}): {sorted(shared_classes)}")
            src_pt  = _filter_data_to_classes(src_data, shared_classes)
            tgt_pt  = _filter_data_to_classes(tgt_data, shared_classes)
            extra_named_pt = [(n, _filter_data_to_classes(d, shared_classes)) for n, d in extra_named]
            for n, d in [(args.training_dataset, src_pt), (args.predict_dataset, tgt_pt)] + extra_named_pt:
                print(f"  {n}: {len(d['labels'])} neurons after filtering")
            extra_ds_pt = [_build_dataset(d, config, None) for _, d in extra_named_pt]
            src_ds_pt   = _build_dataset(src_pt, config, None)
            tgt_ds_pt   = _build_dataset(tgt_pt, config, None)
            pretrain_corpus = extra_ds_pt + [src_ds_pt, tgt_ds_pt]
        else:
            pretrain_corpus = extra_ds_list + [src_ds, tgt_ds]
        combined_ds     = _concat_datasets(pretrain_corpus)
        # Class vocabularies differ across datasets — disable class conditioning
        # during pretraining to avoid out-of-bounds embedding lookups.
        combined_ds.labels[:] = -1
        n_combined      = len(combined_ds)
        print(f"\nPhase 1 pretraining on {n_combined} neurons "
              f"({'+'.join(args.pretrain_datasets + [args.training_dataset, args.predict_dataset])}) "
              f"for {pretrain_epochs} epochs ...")
        eff_bs_pt = min(args.batch_size, n_combined)
        pt_loader = make_dataloader(combined_ds, batch_size=eff_bs_pt, shuffle=True, drop_last=False)
        pt_trainer = pl.Trainer(
            max_epochs=pretrain_epochs,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            callbacks=[TQDMProgressBar(refresh_rate=0)],
            logger=False,
        )
        t_pt = time.time()
        pt_trainer.fit(model, pt_loader)
        pretrain_time = time.time() - t_pt
        print(f"Phase 1 pretraining done in {pretrain_time:.1f}s")

    # ── 4b. Phase 2: fine-tune on source dataset only ─────────────────────────
    eff_bs       = min(args.batch_size, len(src_ds))
    train_loader = make_dataloader(src_ds, batch_size=eff_bs, shuffle=True, drop_last=False)
    phase2_label = "Phase 2 fine-tuning" if args.pretrain else "Training"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=0)],
        logger=False,
    )
    print(f"\n{phase2_label} for {args.epochs} epochs on {args.training_dataset} ...")
    t_train = time.time()
    trainer.fit(model, train_loader)
    train_time = time.time() - t_train
    print(f"{phase2_label} done in {train_time:.1f}s")

    # ── 5. Extract embeddings ────────────────────────────────────────────────
    # Source: all labeled neurons
    src_is_labeled = np.array([s not in unlabeled for s in src_data["str_labels"]])
    src_labeled_idx = np.where(src_is_labeled)[0]
    src_labeled_ds  = _make_subset(src_ds, src_labeled_idx)
    src_labeled_str = [src_data["str_labels"][i] for i in src_labeled_idx]

    # Target: all labeled neurons (only shared classes will be evaluated)
    tgt_is_labeled = np.array([s not in unlabeled for s in tgt_data["str_labels"]])
    tgt_labeled_idx = np.where(tgt_is_labeled)[0]
    tgt_labeled_ds  = _make_subset(tgt_ds, tgt_labeled_idx)
    tgt_labeled_str = [tgt_data["str_labels"][i] for i in tgt_labeled_idx]

    print(f"\nExtracting source embeddings ({len(src_labeled_idx)} neurons) ...")
    t_emb = time.time()
    src_emb = _extract_embeddings(model, src_labeled_ds)

    print(f"Extracting target embeddings ({len(tgt_labeled_idx)} neurons) ...")
    tgt_emb = _extract_embeddings(model, tgt_labeled_ds)
    emb_time = time.time() - t_emb
    print(f"Embeddings done in {emb_time:.1f}s")

    # ── 6. Shared-class KNN ───────────────────────────────────────────────────
    from collections import Counter as _Counter
    src_counts = _Counter(src_labeled_str)
    tgt_counts = _Counter(tgt_labeled_str)
    src_classes = {c for c, n in src_counts.items() if n >= args.min_class_cells}
    tgt_classes = {c for c, n in tgt_counts.items() if n >= args.min_class_cells}
    shared      = src_classes & tgt_classes
    print(f"\nShared classes (min_cells={args.min_class_cells}, n={len(shared)}): {sorted(shared)}")

    if not shared:
        print("ERROR: no shared classes between source and target!", file=sys.stderr)
        return 1

    # Filter source to shared classes
    src_mask = np.array([s in shared for s in src_labeled_str])
    src_emb_sh  = src_emb[src_mask]
    src_str_sh  = [s for s in src_labeled_str if s in shared]

    # Filter target to shared classes
    tgt_mask = np.array([s in shared for s in tgt_labeled_str])
    tgt_emb_sh  = tgt_emb[tgt_mask]
    tgt_str_sh  = [s for s in tgt_labeled_str if s in shared]

    le = LabelEncoder()
    le.fit(sorted(shared))
    src_int = le.transform(src_str_sh)
    tgt_int = le.transform(tgt_str_sh)

    def _l2norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    src_l2 = _l2norm(src_emb_sh)
    tgt_l2 = _l2norm(tgt_emb_sh)

    print("\nRunning cross-dataset KNN...")
    t_probe = time.time()
    # k is selected by stratified inner CV on the SOURCE training fold only,
    # then evaluated once on the held-out target. See
    # hippie/utils.py:select_knn_k_by_train_cv for rationale.
    from hippie.utils import select_knn_k_by_train_cv
    knn_result = select_knn_k_by_train_cv(
        src_l2,
        src_int,
        tgt_l2,
        tgt_int,
        k_grid=range(1, 21),
        inner_cv_splits=5,
        metric="cosine",
        random_state=getattr(args, "seed", 42),
    )
    best_k = knn_result["best_k"]
    best_ba = knn_result["test_balanced_accuracy"]
    best_preds = knn_result["test_predictions"]
    probe_time = time.time() - t_probe
    print(f"  KNN best k={best_k} (selected by train-CV)  BA={best_ba:.4f}")

    # ── 7. Save predictions CSV ───────────────────────────────────────────────
    pred_str = le.inverse_transform(best_preds.astype(int))
    true_str = le.inverse_transform(tgt_int.astype(int))

    preds_df = pd.DataFrame({
        "predicted_label": pred_str,
        "true_label":      true_str,
        "true_label_original": tgt_str_sh,
    })
    preds_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    print(f"Saved predictions.csv ({len(preds_df)} rows)")

    # ── 8. Save embeddings CSVs ───────────────────────────────────────────────
    train_emb_df = pd.DataFrame(src_emb_sh,
                                 columns=[f"z{i}" for i in range(src_emb_sh.shape[1])])
    train_emb_df["label"] = src_str_sh
    train_emb_df.to_csv(os.path.join(args.output_dir, "train_embeddings.csv"), index=False)

    pred_emb_df = pd.DataFrame(tgt_emb_sh,
                                columns=[f"z{i}" for i in range(tgt_emb_sh.shape[1])])
    pred_emb_df["label"]       = tgt_str_sh
    pred_emb_df["true_label"]  = true_str
    pred_emb_df.to_csv(os.path.join(args.output_dir, "predict_embeddings.csv"), index=False)

    print(f"\nResults written to {args.output_dir}")

    # ── 9. Timings CSV ────────────────────────────────────────────────────────
    timings = [{"phase": "data_load", "seconds": round(load_time, 1)}]
    if args.pretrain:
        timings.append({"phase": "pretrain_cvae", "seconds": round(pretrain_time, 1),
                        "epochs": pretrain_epochs})
    timings += [
        {"phase": "train_cvae",  "seconds": round(train_time, 1),
         "epochs": args.epochs,
         "n_cells_src": len(src_data["labels"]),
         "n_cells_tgt": len(tgt_data["labels"])},
        {"phase": "embed+probe", "seconds": round(emb_time + probe_time, 1)},
    ]
    _write_timings(args.timings_csv, timings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
