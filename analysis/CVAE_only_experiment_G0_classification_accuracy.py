#!/usr/bin/env python3
"""G0 — Latent-space classification accuracy (sanity gate).

Encodes each labeled dataset, fits a leave-one-out cross-validated
LogisticRegression on the μ vectors, and reports balanced accuracy and
per-cell-type F1.  Run this immediately after training as a quality gate
before the generative experiments; abort the job if accuracy is too low.

Writes:
    G0_classification_accuracy.csv   — one row per (method, dataset)
    G0_per_class_accuracy.csv        — one row per (method, dataset, cell_type)

Usage:
    python CVAE_only_experiment_G0_classification_accuracy.py \\
        --registry /tmp/cvae_registry.json \\
        --data-dirs /data/datasets/hausser_cell_type /data/datasets/lisberger_labeled_cell_type \\
        --datasets  hausser_cell_type lisberger_labeled_cell_type \\
        --out-summary  /data/results/cvae_only/G0_classification_accuracy.csv \\
        --out-per-class /data/results/cvae_only/G0_per_class_accuracy.csv \\
        [--min-balanced-accuracy 0.50]   # abort if below this on any dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from CVAE_only_experiment_utils import (
    DEFAULT_RESULTS_DIR,
    build_loader,
    encode_batch,
    iter_registry,
    load_dataset_csv,
    load_registry,
)


def encode_dataset(lm, data: dict, batch_size: int = 256) -> np.ndarray:
    """Return (N, z_dim) array of μ vectors for the whole dataset."""
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    loader = build_loader(data, mods, batch_size=batch_size)
    mus = []
    for batch in loader:
        if batch is None:
            continue
        batch_data, _ = batch
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        mu, _ = encode_batch(lm, batch_data)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)


def cv_classify(Z: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Stratified k-fold LogisticRegression; returns summary dict."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    if len(classes) < 2:
        return dict(balanced_accuracy=np.nan, macro_f1=np.nan,
                    n_cells=len(y), n_classes=len(classes))

    # Use min(5, smallest_class_size) folds so rare classes are always represented
    min_class = np.bincount(y_enc).min()
    k = min(n_splits, int(min_class))
    if k < 2:
        # Too few samples for CV — fit on all, report train accuracy as lower bound
        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
        clf.fit(Z, y_enc)
        pred = clf.predict(Z)
        ba = balanced_accuracy_score(y_enc, pred)
        f1 = f1_score(y_enc, pred, average="macro", zero_division=0)
        per_class = {cls: float(f1_score(y_enc == i, pred == i, zero_division=0))
                     for i, cls in enumerate(classes)}
        return dict(balanced_accuracy=ba, macro_f1=f1, n_cells=len(y),
                    n_classes=len(classes), cv_note="train_only", **{f"f1_{c}": v for c, v in per_class.items()})

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    for train_idx, test_idx in skf.split(Z, y_enc):
        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
        clf.fit(Z[train_idx], y_enc[train_idx])
        all_pred.extend(clf.predict(Z[test_idx]).tolist())
        all_true.extend(y_enc[test_idx].tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    ba = balanced_accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    per_class = {cls: float(f1_score(all_true == i, all_pred == i, zero_division=0))
                 for i, cls in enumerate(classes)}
    return dict(balanced_accuracy=ba, macro_f1=f1, n_cells=len(y),
                n_classes=len(classes), cv_folds=k,
                **{f"f1_{c}": v for c, v in per_class.items()})


def run(args) -> tuple[list[dict], list[dict]]:
    registry = load_registry(args.registry)
    summary_rows, per_class_rows = [], []

    for lm in iter_registry(registry):
        for data_dir, dataset in zip(args.data_dirs, args.datasets):
            data = load_dataset_csv(Path(data_dir), lm.modalities.keys(),
                                    target_sizes=lm.modalities)
            labels = data.get("labels")
            if labels is None:
                print(f"[G0] {dataset}: no labels — skipping")
                continue

            # Only keep labeled cells (exclude "unlabeled")
            mask = np.array([str(l) not in ("unlabeled", "nan", "") for l in labels])
            if mask.sum() < 4:
                print(f"[G0] {dataset}: fewer than 4 labeled cells — skipping")
                continue

            Z = encode_dataset(lm, data)
            Z_labeled = Z[mask]
            y_labeled = labels[mask].astype(str)

            result = cv_classify(Z_labeled, y_labeled)
            ba = result["balanced_accuracy"]

            print(f"\n[G0] {lm.method} | {dataset}")
            print(f"  n_cells={result['n_cells']}  n_classes={result['n_classes']}")
            print(f"  balanced_accuracy = {ba:.3f}")
            print(f"  macro_f1          = {result['macro_f1']:.3f}")
            # Per-class F1
            for k, v in result.items():
                if k.startswith("f1_"):
                    print(f"    {k[3:]:>12s}: {v:.3f}")

            summary_rows.append(dict(method=lm.method, dataset=dataset,
                                     balanced_accuracy=ba,
                                     macro_f1=result["macro_f1"],
                                     n_cells=result["n_cells"],
                                     n_classes=result["n_classes"]))
            for k, v in result.items():
                if k.startswith("f1_"):
                    per_class_rows.append(dict(method=lm.method, dataset=dataset,
                                               cell_type=k[3:], f1=v))

    return summary_rows, per_class_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--datasets",  nargs="+", required=True)
    ap.add_argument("--out-summary",   type=Path,
                    default=DEFAULT_RESULTS_DIR / "G0_classification_accuracy.csv")
    ap.add_argument("--out-per-class", type=Path,
                    default=DEFAULT_RESULTS_DIR / "G0_per_class_accuracy.csv")
    ap.add_argument("--min-balanced-accuracy", type=float, default=0.0,
                    help="Abort with exit-code 1 if any dataset is below this threshold")
    args = ap.parse_args()

    if len(args.data_dirs) != len(args.datasets):
        ap.error("--data-dirs and --datasets must have the same number of entries")

    summary_rows, per_class_rows = run(args)

    if summary_rows:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(args.out_summary, index=False)
        print(f"\n[G0] summary → {args.out_summary}")

        args.out_per_class.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_class_rows).to_csv(args.out_per_class, index=False)
        print(f"[G0] per-class → {args.out_per_class}")

    # Gate: abort if any dataset is below the minimum threshold
    if args.min_balanced_accuracy > 0.0:
        failures = [r for r in summary_rows
                    if r["balanced_accuracy"] < args.min_balanced_accuracy]
        if failures:
            print("\n[G0] ACCURACY GATE FAILED — aborting job:")
            for f in failures:
                print(f"  {f['method']} | {f['dataset']}: "
                      f"balanced_acc={f['balanced_accuracy']:.3f} "
                      f"< threshold={args.min_balanced_accuracy}")
            sys.exit(1)
        else:
            print(f"\n[G0] Accuracy gate passed (threshold={args.min_balanced_accuracy:.2f}) ✓")


if __name__ == "__main__":
    main()
