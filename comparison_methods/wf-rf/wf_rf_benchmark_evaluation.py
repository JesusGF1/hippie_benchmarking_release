#!/usr/bin/env python3
"""
WF-RF — handcrafted waveform features + Random Forest baseline.

Referred to as "WF-RF" in the paper (Methods § Baselines). The four core
shape descriptors follow Jia et al. (2019, Journal of Neurophysiology);
the five additional features follow the Allen Institute ecephys waveform
metrics convention.

Implementation details for the single-channel variant: since the benchmark
datasets in this repository expose only the peak-channel waveform, we
extract nine scalar descriptors from that waveform plus PCA components
that retain ≥ 90 % of the raw waveform variance.

Features extracted on the fly from the 90-sample peak-channel waveform:
  1. PT ratio          — positive peak / abs(negative trough)
  2. Duration          — (peak_idx - trough_idx) / T  [signed, normalised]
  3. Halfwidth         — fraction of samples above half-max abs amplitude
  4. Repolarisation slope — (peak_val - trough_val) / |peak_idx - trough_idx|
  5. Recovery slope    — slope from later extremum to waveform end
  6. Peak amplitude    — raw positive peak value
  7. Trough amplitude  — raw negative trough value
  8. Total amplitude   — peak_val - trough_val
  9. Peak location     — index of abs-maximum / T

  plus PCA components that retain ≥ 90 % of variance (matching the paper).

All of the above are concatenated into a single feature vector and fed to a
Random Forest (primary), MLP, and Logistic Regression classifier, matching
the multi-probe evaluation protocol used by NEMO.

No neural network training required — pure sklearn, CPU-only.

Output format matches the benchmark standard:
  transductive_predictions.csv   — (pred, true) — Random Forest
  mlp_predictions.csv            — (pred, true) — MLP probe
  linear_probe_predictions.csv   — (pred, true) — Logistic Regression

Usage
-----
    python comparison_methods/wf-rf/wf_rf_benchmark_evaluation.py \\
        --dataset hull_cell_type \\
        --cv-fold 0 \\
        --data-root /data/datasets \\
        --output-dir /data/results/wf-rf/hull_cell_type/fold_0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Repo-relative import of the bundled hippie_wf3dacg H5 loader. This is a
# drop-in replacement for the upstream C4 H5 loader — the dict
# returned by load_c4_dataset has identical keys and semantics.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hippie_wf3dacg.load_c4 import load_c4_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Waveform feature extraction  (Buccino 2018 single-channel variant)
# ---------------------------------------------------------------------------

def extract_waveform_features(waveforms: np.ndarray) -> np.ndarray:
    """Compute 9 shape descriptors from a batch of peak-channel waveforms.

    Args:
        waveforms: (N, T) float32 array — peak-channel waveform for each unit.

    Returns:
        feats: (N, 9) float32 feature array.

    Feature index legend
    --------------------
    0  PT ratio          — peak_val / (|trough_val| + ε)
    1  duration          — (peak_idx – trough_idx) / T   [signed, normalised]
    2  halfwidth         — fraction of samples ≥ half max abs amplitude
    3  repol_slope       — (peak_val – trough_val) / |peak_idx – trough_idx|
    4  recovery_slope    — slope from later extremum to last sample
    5  peak_val          — positive peak amplitude
    6  trough_val        — negative trough amplitude
    7  amplitude         — peak_val – trough_val
    8  peak_loc          — index_of_abs_max / T
    """
    N, T = waveforms.shape
    feats = np.zeros((N, 9), dtype=np.float32)

    for i, wvf in enumerate(waveforms):
        trough_idx = int(np.argmin(wvf))
        peak_idx   = int(np.argmax(wvf))
        trough_val = float(wvf[trough_idx])
        peak_val   = float(wvf[peak_idx])

        # 0. PT ratio
        feats[i, 0] = peak_val / (abs(trough_val) + 1e-10)

        # 1. Duration (signed: positive if peak after trough → typical RS cell)
        feats[i, 1] = (peak_idx - trough_idx) / T

        # 2. Halfwidth — fraction of time spent ≥ half the max absolute amplitude
        abs_wvf = np.abs(wvf)
        half_max = abs_wvf.max() / 2.0
        feats[i, 2] = float(np.sum(abs_wvf >= half_max)) / T

        # 3. Repolarisation slope (trough → peak, regardless of order)
        dt = abs(peak_idx - trough_idx)
        feats[i, 3] = (peak_val - trough_val) / (dt + 1e-10)

        # 4. Recovery slope (later of the two extrema → end of waveform)
        later = max(trough_idx, peak_idx)
        remaining = T - 1 - later
        if remaining > 0:
            feats[i, 4] = (float(wvf[-1]) - float(wvf[later])) / remaining

        # 5–8. Amplitude stats
        feats[i, 5] = peak_val
        feats[i, 6] = trough_val
        feats[i, 7] = peak_val - trough_val
        feats[i, 8] = float(np.argmax(abs_wvf)) / T

    # Clip any NaN / Inf that could arise from degenerate (zero) waveforms
    np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return feats


def build_features(
    waveforms: np.ndarray,
    pca_variance: float = 0.90,
    pca: PCA | None = None,
    fit_pca: bool = True,
) -> tuple[np.ndarray, PCA]:
    """Build combined feature matrix: shape descriptors + PCA components.

    Replicates Buccino 2018 which combines:
      - Extracted waveform features (PT ratio, duration, …)
      - PCA on the raw waveform retaining 90 % of explained variance.

    Args:
        waveforms:    (N, T) raw waveform array.
        pca_variance: Fraction of variance to retain (default 0.90).
        pca:          Pre-fitted PCA object.  Used when fit_pca=False.
        fit_pca:      Fit PCA on this batch (True for training data).

    Returns:
        features: (N, n_shape + n_pca) combined feature matrix.
        pca:      Fitted PCA object (to apply identically to test data).
    """
    shape_feats = extract_waveform_features(waveforms)

    N, T = waveforms.shape
    if fit_pca:
        # Upper bound: never more components than min(N-1, T)
        n_max = min(N - 1, T)
        pca = PCA(n_components=n_max, random_state=42)
        pca_all = pca.fit_transform(waveforms)
        # Trim to the fewest components that capture ≥ pca_variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, pca_variance)) + 1
        pca_feats = pca_all[:, :n_keep]
    else:
        pca_all = pca.transform(waveforms)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cum_var, pca_variance)) + 1
        pca_feats = pca_all[:, :n_keep]

    return np.concatenate([shape_feats, pca_feats], axis=1), pca


# ---------------------------------------------------------------------------
# Holdout (inductive) split helpers
# ---------------------------------------------------------------------------

def _build_animal_splits(
    animal_ids: np.ndarray,
    holdout_fold: int,
    n_holdout_folds: int,
    min_cells: int,
) -> tuple[set, set]:
    """Partition animals into n_holdout_folds groups; return (train_set, test_set).

    Replicates the logic in the holdout training script::_build_animal_splits.
    Animals with < min_cells neurons are excluded entirely.
    """
    from collections import Counter
    counts = Counter(animal_ids.tolist())
    valid_animals = sorted(a for a, c in counts.items() if c >= min_cells and a)
    if not valid_animals:
        raise ValueError(
            f"No animals with ≥ {min_cells} cells. Counts: {dict(counts)}"
        )
    rng = np.random.default_rng(42)
    shuffled = list(rng.permutation(valid_animals))
    # Interleave assignment to balance group sizes
    fold_groups = [shuffled[i::n_holdout_folds] for i in range(n_holdout_folds)]
    test_set  = set(fold_groups[holdout_fold])
    train_set = set(a for a in valid_animals if a not in test_set)
    return train_set, test_set


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_h5(data_root: str, dataset: str) -> str:
    """Return the single .h5 file under <data_root>/<dataset>/."""
    pattern = os.path.join(data_root, dataset, "*.h5")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No .h5 file found at {pattern}. "
            "Download the C4 H5 file via `python scripts/download_c4_h5.py "
            "--dataset <key> --out-dir <data_root>/<dataset>/` and re-run. "
            "See docs/DATASETS.md for the available keys and the C4 access "
            "procedure."
        )
    if len(hits) > 1:
        print(f"WARNING: multiple H5 files found; using {hits[0]}", file=sys.stderr)
    return hits[0]


# ---------------------------------------------------------------------------
# Probe suite  (matches the HIPPIE training script / nemo_benchmark_evaluation.py)
# ---------------------------------------------------------------------------

def _run_probes(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    n_trees: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random Forest (primary) + MLP + Logistic Regression probes.

    Features are z-scored with a scaler fitted on the training split.

    Returns:
        rf_preds, mlp_preds, lr_preds — integer arrays aligned with test set.
    """
    sc = StandardScaler()
    Xtr = sc.fit_transform(train_feats)
    Xte = sc.transform(test_feats)

    # Guard against NaN / Inf after scaling
    np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # 1. Random Forest — primary classifier (Buccino 2018)
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xtr, train_labels)
    rf_preds = rf.predict(Xte)
    print(f"  RF   BA={balanced_accuracy_score(test_labels, rf_preds):.4f}")

    # 2. MLP probe (256→128, same as NEMO)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(Xtr, train_labels)
    mlp_preds = mlp.predict(Xte)
    print(f"  MLP  BA={balanced_accuracy_score(test_labels, mlp_preds):.4f}")

    # 3. Logistic Regression probe
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr, train_labels)
    lr_preds = lr.predict(Xte)
    print(f"  LR   BA={balanced_accuracy_score(test_labels, lr_preds):.4f}")

    return rf_preds, mlp_preds, lr_preds


# ---------------------------------------------------------------------------
# Holdout (inductive) evaluation
# ---------------------------------------------------------------------------

def _run_holdout(data: dict, args: argparse.Namespace) -> int:
    """Run Buccino in inductive holdout mode.

    Splits neurons by animal_id: trains PCA + classifiers on (N-1)/N animal
    groups and evaluates on the held-out group.  Mirrors the protocol in
    the holdout training script for cross-method consistency.

    Outputs holdout_predictions.csv, mlp_predictions.csv,
    linear_probe_predictions.csv (all with an animal_id column).
    """
    N = len(data["labels"])
    all_animal_ids = np.array(data.get("animal_ids", [""] * N), dtype=object)
    n_unique = len(set(a for a in all_animal_ids if a))
    if n_unique == 0:
        print(
            "ERROR: C4 H5 has no animal_id metadata.\n"
            "Rebuild the H5 with the C4-format preprocessing script after the ANIMAL_ID_FIELD update.",
            file=sys.stderr,
        )
        return 1
    print(f"Animal IDs: {n_unique} unique animals across {N} neurons")

    # ── Split animals into train / test ──────────────────────────────────────
    train_animals, test_animals = _build_animal_splits(
        all_animal_ids,
        holdout_fold=args.holdout_fold,
        n_holdout_folds=args.n_holdout_folds,
        min_cells=args.min_cells_per_animal,
    )
    print(f"Holdout fold {args.holdout_fold}: "
          f"train={len(train_animals)} animals  test={len(test_animals)} animals")

    train_mask = np.array([a in train_animals for a in all_animal_ids])
    test_mask  = np.array([a in test_animals  for a in all_animal_ids])

    # ── Filter to labeled neurons within each split ──────────────────────────
    unlabeled = set(args.unlabeled_strings)
    def _labeled_in_mask(mask):
        return np.where(
            mask & np.array([s not in unlabeled for s in data["str_labels"]])
        )[0]

    train_idx = _labeled_in_mask(train_mask)
    test_idx  = _labeled_in_mask(test_mask)
    print(f"  Labeled train : {len(train_idx)}  |  Labeled test : {len(test_idx)}")

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("ERROR: no labeled neurons in train or test split.", file=sys.stderr)
        return 1

    # ── Label encoding (fit on union so vocabulary is shared) ────────────────
    all_str = ([data["str_labels"][i] for i in train_idx] +
               [data["str_labels"][i] for i in test_idx])
    le = LabelEncoder()
    le.fit(all_str)
    train_labels_int = le.transform([data["str_labels"][i] for i in train_idx])
    test_labels_int  = le.transform([data["str_labels"][i] for i in test_idx])
    print(f"  Classes: {list(le.classes_)}")

    # ── Waveform arrays ──────────────────────────────────────────────────────
    waveforms_all = data["waveforms"]
    if waveforms_all.dtype == object:
        lengths = [len(waveforms_all[i]) for i in np.concatenate([train_idx, test_idx])]
        T = int(np.median(lengths))
        def _to_fixed(idx_arr):
            out = np.zeros((len(idx_arr), T), dtype=np.float32)
            for j, i in enumerate(idx_arr):
                w = waveforms_all[i]
                l = min(len(w), T)
                out[j, :l] = w[:l]
            return out
        train_wvf = _to_fixed(train_idx)
        test_wvf  = _to_fixed(test_idx)
    else:
        train_wvf = waveforms_all[train_idx].astype(np.float32)
        test_wvf  = waveforms_all[test_idx].astype(np.float32)

    print(f"Waveform shape: train={train_wvf.shape}  test={test_wvf.shape}")

    # ── Feature extraction (fit PCA on training neurons only) ────────────────
    print("\nExtracting waveform features ...")
    t_feat = time.time()
    train_feats, pca_obj = build_features(
        train_wvf, pca_variance=args.pca_variance, fit_pca=True)
    test_feats, _ = build_features(
        test_wvf, pca_variance=args.pca_variance, pca=pca_obj, fit_pca=False)
    feat_time = time.time() - t_feat
    print(f"Feature shape: train={train_feats.shape}  test={test_feats.shape}  "
          f"({feat_time:.2f}s)")

    # ── Probes ───────────────────────────────────────────────────────────────
    print("\nRunning probes ...")
    t_probe = time.time()
    rf_preds, mlp_preds, lr_preds = _run_probes(
        train_feats, train_labels_int,
        test_feats,  test_labels_int,
        n_trees=args.n_trees,
    )
    probe_time = time.time() - t_probe
    print(f"Probes done in {probe_time:.1f}s")

    # ── Write prediction CSVs ────────────────────────────────────────────────
    test_true_str    = le.inverse_transform(test_labels_int)
    test_animal_ids_ = all_animal_ids[test_idx]

    def _write(preds, fname):
        df = pd.DataFrame({
            "pred":      le.inverse_transform(preds.astype(int)),
            "true":      test_true_str,
            "animal_id": test_animal_ids_,
        })
        df.to_csv(os.path.join(args.output_dir, fname), index=False)
        print(f"  → {os.path.join(args.output_dir, fname)}")

    _write(rf_preds,  "holdout_predictions.csv")
    _write(mlp_preds, "mlp_predictions.csv")
    _write(lr_preds,  "linear_probe_predictions.csv")

    with open(os.path.join(args.output_dir, "animal_split.json"), "w") as f:
        json.dump({
            "holdout_fold":         args.holdout_fold,
            "n_holdout_folds":      args.n_holdout_folds,
            "train_animals":        sorted(train_animals),
            "test_animals":         sorted(test_animals),
            "min_cells_per_animal": args.min_cells_per_animal,
            "pca_variance":         args.pca_variance,
            "n_pca_components":     int(
                np.searchsorted(
                    np.cumsum(pca_obj.explained_variance_ratio_),
                    args.pca_variance,
                ) + 1
            ),
            "n_trees":   args.n_trees,
            "n_features": int(train_feats.shape[1]),
        }, f, indent=2)

    total = feat_time + probe_time
    print(f"\nDone in {total:.1f}s  (feat={feat_time:.1f}s  probe={probe_time:.1f}s)")
    print(f"Results written to {args.output_dir}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Buccino 2018 single-channel waveform feature baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",      required=True,
                        help="Dataset name, e.g. hull_cell_type")
    parser.add_argument("--cv-fold",      type=int, default=0,
                        help="CV fold index (0 to n-cv-folds-1)")
    parser.add_argument("--n-cv-folds",   type=int, default=5)
    parser.add_argument("--data-root",    default="./datasets",
                        help="Root directory containing downloaded H5 files")
    parser.add_argument("--output-dir",   required=True,
                        help="Write prediction CSVs and cv_split.json here")
    parser.add_argument("--pca-variance", type=float, default=0.90,
                        help="Fraction of variance retained by PCA (Buccino 2018: 0.90)")
    parser.add_argument("--n-trees",      type=int, default=200,
                        help="Number of trees in the Random Forest")
    parser.add_argument("--unlabeled-strings", nargs="+",
                        default=["", "unlabeled"],
                        help="str_label values treated as unlabeled/excluded from probes")
    # ── Holdout (inductive) mode ──────────────────────────────────────────────
    parser.add_argument("--holdout-fold",   type=int, default=None,
                        help="Enable holdout mode: index of animal fold to hold out "
                             "(0 to n-holdout-folds-1).  When set, --cv-fold is ignored.")
    parser.add_argument("--n-holdout-folds", type=int, default=5,
                        help="Number of animal holdout folds (default: 5)")
    parser.add_argument("--min-cells-per-animal", type=int, default=50,
                        help="Minimum cells per animal to include in holdout splits "
                             "(default: 50, matching HIPPIE)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mode = "holdout" if args.holdout_fold is not None else "transductive"
    print("=" * 60)
    print("Buccino 2018 Waveform Feature Baseline")
    print(f"  dataset      : {args.dataset}")
    print(f"  mode         : {mode}")
    if mode == "holdout":
        print(f"  holdout_fold : {args.holdout_fold}/{args.n_holdout_folds}")
        print(f"  min_cells    : {args.min_cells_per_animal}")
    else:
        print(f"  cv_fold      : {args.cv_fold}/{args.n_cv_folds}")
    print(f"  pca_variance : {args.pca_variance}")
    print(f"  n_trees      : {args.n_trees}")
    print("=" * 60)

    # ── 1. Load data from C4 H5 format ─────────────────────────────────
    t_load = time.time()
    h5_path = _find_h5(args.data_root, args.dataset)
    print(f"\nLoading {h5_path} ...")
    data = load_c4_dataset(h5_path, args.dataset, source_id=0)
    N = len(data["labels"])
    load_time = time.time() - t_load
    print(f"Loaded {N} neurons in {load_time:.1f}s")

    label_counts: dict[str, int] = {}
    for s in data["str_labels"]:
        label_counts[s] = label_counts.get(s, 0) + 1
    print(f"Label distribution: {label_counts}")

    # ── 2. Dispatch to holdout mode if requested ─────────────────────────────
    if args.holdout_fold is not None:
        return _run_holdout(data, args)

    # ── 3. (Transductive) Identify labeled neurons ───────────────────────────
    unlabeled = set(args.unlabeled_strings)
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

    # ── 3. Label encoder + CV split (same seed as NEMO) ──────────
    labeled_str = [data["str_labels"][i] for i in labeled_idx]
    le = LabelEncoder()
    le.fit(labeled_str)
    labeled_int = le.transform(labeled_str)

    skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=42)
    splits = list(skf.split(labeled_idx, labeled_int))
    train_pos, test_pos = splits[args.cv_fold]
    print(f"CV fold {args.cv_fold}: train={len(train_pos)}  test={len(test_pos)}")
    print(f"Classes: {list(le.classes_)}")

    # ── 4. Extract waveforms for labeled neurons ──────────────────────────────
    waveforms_all = data["waveforms"]

    # Ensure uniform shape — load_c4_dataset may return object array if ragged
    waveforms_labeled = waveforms_all[labeled_idx]
    if waveforms_labeled.dtype == object:
        # Pad / truncate to the most common length
        lengths = [len(w) for w in waveforms_labeled]
        T = int(np.median(lengths))
        wvf_fixed = np.zeros((len(labeled_idx), T), dtype=np.float32)
        for j, w in enumerate(waveforms_labeled):
            l = min(len(w), T)
            wvf_fixed[j, :l] = w[:l]
        waveforms_labeled = wvf_fixed
    else:
        waveforms_labeled = waveforms_labeled.astype(np.float32)

    train_wvf = waveforms_labeled[train_pos]
    test_wvf  = waveforms_labeled[test_pos]

    train_labels_int = labeled_int[train_pos]
    test_labels_int  = labeled_int[test_pos]

    print(f"\nWaveform shape: {waveforms_labeled.shape}  (dtype {waveforms_labeled.dtype})")

    # ── 5. Feature extraction + PCA ──────────────────────────────────────────
    print("\nExtracting waveform features ...")
    t_feat = time.time()
    train_feats, pca_obj = build_features(
        train_wvf,
        pca_variance=args.pca_variance,
        fit_pca=True,
    )
    test_feats, _ = build_features(
        test_wvf,
        pca_variance=args.pca_variance,
        pca=pca_obj,
        fit_pca=False,
    )
    feat_time = time.time() - t_feat
    print(f"Feature shape: train={train_feats.shape}  test={test_feats.shape}  "
          f"({feat_time:.2f}s)")

    # ── 6. Probes ─────────────────────────────────────────────────────────────
    print("\nRunning probes ...")
    t_probe = time.time()
    rf_preds, mlp_preds, lr_preds = _run_probes(
        train_feats, train_labels_int,
        test_feats,  test_labels_int,
        n_trees=args.n_trees,
    )
    probe_time = time.time() - t_probe
    print(f"Probes done in {probe_time:.1f}s")

    # ── 7. Write prediction CSVs ─────────────────────────────────────────────
    test_true_str = le.inverse_transform(test_labels_int)

    def _write_preds(preds: np.ndarray, fname: str):
        df = pd.DataFrame({
            "pred": le.inverse_transform(preds.astype(int)),
            "true": test_true_str,
        })
        df.to_csv(os.path.join(args.output_dir, fname), index=False)
        print(f"  → {os.path.join(args.output_dir, fname)}")

    _write_preds(rf_preds,  "transductive_predictions.csv")
    _write_preds(mlp_preds, "mlp_predictions.csv")
    _write_preds(lr_preds,  "linear_probe_predictions.csv")

    # ── 8. CV split metadata ──────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "cv_split.json"), "w") as f:
        json.dump(
            {
                "train_global_indices": labeled_idx[train_pos].tolist(),
                "test_global_indices":  labeled_idx[test_pos].tolist(),
                "fold":       args.cv_fold,
                "n_folds":    args.n_cv_folds,
                "pca_variance": args.pca_variance,
                "n_pca_components": int(
                    np.searchsorted(
                        np.cumsum(pca_obj.explained_variance_ratio_),
                        args.pca_variance,
                    ) + 1
                ),
                "n_trees":    args.n_trees,
                "n_features": int(train_feats.shape[1]),
            },
            f,
            indent=2,
        )

    total_time = load_time + feat_time + probe_time
    print(f"\nDone in {total_time:.1f}s total")
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
