#!/usr/bin/env python3
"""
NEMO Benchmark Evaluation Script for HIPPIE Pipeline

This script runs NEMO (Neural Multi-modal Embedding) evaluation on C4 database
H5 files using the same CV protocol as HIPPIE and PhysMAP for fair comparison.

IMPORTANT: NEMO requires data in H5 format with:
- Waveforms: (N, 90) interpolated to 90 samples
- 3D ACGs: (N, 10, 101) computed from spike times via fast_acg3d

This script loads directly from C4 database H5 files - NOT from HIPPIE CSV format.

Evaluation methods:
1. NEMO Native: Linear probe and MLP probe on frozen embeddings
2. HIPPIE-style: KNN (k=1,3,5,10,20) on embeddings

Output format matches HIPPIE/PhysMAP:
- transductive_predictions.csv: true,pred,fold columns
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Also add the top-level hippie/ package so we can import the unified
# compute-parity timing helper. NEMO lives under comparison_methods/nemo/,
# so the hippie package is four directories up from this script.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hippie')),
)

from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.utils.c4_data_utils import get_dataset_from_file
from celltype_ibl.params.config import DATASETS_DIRECTORY

import compute_parity as _cp  # unified per-phase timing


def load_c4_dataset(dataset_file: str, normalise: bool = True) -> tuple:
    """
    Load dataset from C4 database H5 file using NEMO's native loading.

    Args:
        dataset_file: Filename of H5 file (e.g., 'C4_database_hull_labelled.h5')
        normalise: Whether to normalize waveforms

    Returns:
        Tuple of (waveforms, acg_3d, labels, label_names)
    """
    print(f"Loading C4 dataset: {dataset_file}")

    # Use NEMO's native data loading
    result = get_dataset_from_file(
        dataset_file,
        is_labeled=True,
        normalise=normalise,
        use_cache=True
    )

    waveforms, acg_3d, label_idx, label_names, labelling_dict, correspondence_dict = result

    print(f"Loaded waveforms: {waveforms.shape}")
    print(f"Loaded ACG 3D: {acg_3d.shape}")
    print(f"Labels: {len(np.unique(label_idx))} classes")

    # Filter out unlabeled samples
    label_names_array = np.array(label_names)
    unlabeled_mask = np.isin(label_names_array, ['unlabelled', 'unlabeled', 'unknown', 'Unknown'])
    labeled_mask = ~unlabeled_mask

    if np.sum(unlabeled_mask) > 0:
        print(f"Removing {np.sum(unlabeled_mask)} unlabeled samples")
        waveforms = waveforms[labeled_mask]
        acg_3d = acg_3d[labeled_mask]
        label_names = label_names_array[labeled_mask]

    return waveforms, acg_3d, label_names


def load_c4_dataset_from_s3(s3_path: str, local_cache_dir: str = None) -> tuple:
    """
    Load C4 dataset from S3 bucket.

    Args:
        s3_path: Optional S3 path (e.g., 's3://<bucket>/<prefix>/C4_database_hull_labelled.h5')
        local_cache_dir: Local directory to cache downloaded files

    Returns:
        Tuple of (waveforms, acg_3d, labels)
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError("boto3 required for S3 access. Install with: pip install boto3")

    if local_cache_dir is None:
        local_cache_dir = os.path.join(os.path.expanduser("~"), ".c4_cache")
    os.makedirs(local_cache_dir, exist_ok=True)

    # Parse S3 path
    # Expected format: s3://bucket/prefix/filename.h5
    s3_path = s3_path.replace("s3://", "")
    parts = s3_path.split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    filename = parts[-1]

    local_path = os.path.join(local_cache_dir, filename)

    # Download if not cached
    if not os.path.exists(local_path):
        print(f"Downloading from S3: {s3_path}")

        s3_client = boto3.client(
            's3',
            endpoint_url="https://s3-west.nrp-nautilus.io",
            config=Config(signature_version='s3v4')
        )

        s3_client.download_file(bucket, key, local_path)
        print(f"Downloaded to: {local_path}")

    # Now load using standard method
    # Temporarily override DATASETS_DIRECTORY
    import celltype_ibl.params.config as config_module
    original_dir = config_module.DATASETS_DIRECTORY
    config_module.DATASETS_DIRECTORY = local_cache_dir

    try:
        result = load_c4_dataset(filename)
    finally:
        config_module.DATASETS_DIRECTORY = original_dir

    return result


def get_embeddings_nemo(wvf_data: np.ndarray,
                        acg_data: np.ndarray,
                        model: BimodalEmbeddingModel,
                        device: torch.device,
                        batch_size: int = 512) -> np.ndarray:
    """
    Extract embeddings from NEMO model.

    Args:
        wvf_data: (N, 90) waveform array
        acg_data: (N, 10, 101) 3D ACG array
        model: Trained NEMO model
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Joint embeddings of shape (N, latent_dim)
    """
    model.eval()
    embeddings = []

    wvf_tensor = torch.from_numpy(wvf_data.astype("float32")).to(device)
    acg_tensor = torch.from_numpy(acg_data.astype("float32")).to(device)

    n_samples = len(wvf_data)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            wvf_batch = wvf_tensor[i:end_idx]
            # NEMO expects ACG in shape (batch, 1, 10, 101) and scaled by 10
            acg_batch = acg_tensor[i:end_idx].reshape(-1, 1, 10, 101) * 10

            wvf_emb, acg_emb = model.embed(wvf_batch, acg_batch)
            joint_emb = (wvf_emb + acg_emb) / 2

            embeddings.append(joint_emb.cpu().numpy())

    return np.vstack(embeddings)


def train_nemo_model(wvf_all: np.ndarray,
                     acg_all: np.ndarray,
                     args: argparse.Namespace,
                     device: torch.device) -> BimodalEmbeddingModel:
    """
    Train NEMO bimodal embedding model (transductive - uses all data).

    Args:
        wvf_all: All waveforms (train + test)
        acg_all: All 3D ACGs (train + test)
        args: Training arguments
        device: Torch device

    Returns:
        Trained NEMO model
    """
    model = BimodalEmbeddingModel(
        temperature=args.temperature,
        latent_dim=args.z_dim,
        similarity_adjust=args.similarity_adjust,
        l2_norm=args.l2_norm,
        activation=args.activation,
        batch_norm=args.batch_norm,
        layer_norm=True,
        adjust_to_ce=False,
        adjust_to_ultra=False,
        acg_dropout=None,
        wvf_dropout=None
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    wvf_tensor = torch.from_numpy(wvf_all.astype("float32")).to(device)
    acg_tensor = torch.from_numpy(acg_all.astype("float32")).to(device)

    n_samples = len(wvf_all)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, args.batch_size):
            idx = perm[i:min(i + args.batch_size, n_samples)]
            wvf_batch = wvf_tensor[idx]
            # NEMO expects ACG in shape (batch, 1, 10, 101) and scaled by 10
            acg_batch = acg_tensor[idx].reshape(-1, 1, 10, 101) * 10

            optimizer.zero_grad()
            loss = model(wvf_batch, acg_batch)
            loss.backward()

            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    return model


def train_nemo_model_inductive(wvf_train: np.ndarray,
                               acg_train: np.ndarray,
                               args: argparse.Namespace,
                               device: torch.device) -> BimodalEmbeddingModel:
    """Train NEMO in inductive (holdout) mode — only on training animals.

    Unlike the transductive version, this does NOT concatenate train+test data.
    The model learns representations from training animals only and is then
    evaluated on held-out animals.

    Args:
        wvf_train: Training-split waveforms (N_train, 90).
        acg_train: Training-split 3D ACGs (N_train, 10, 101).
        args: Training arguments.
        device: Torch device.

    Returns:
        Trained NEMO model.
    """
    model = BimodalEmbeddingModel(
        temperature=args.temperature,
        latent_dim=args.z_dim,
        similarity_adjust=args.similarity_adjust,
        l2_norm=args.l2_norm,
        activation=args.activation,
        batch_norm=args.batch_norm,
        layer_norm=True,
        adjust_to_ce=False,
        adjust_to_ultra=False,
        acg_dropout=None,
        wvf_dropout=None
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    wvf_tensor = torch.from_numpy(wvf_train.astype("float32")).to(device)
    acg_tensor = torch.from_numpy(acg_train.astype("float32")).to(device)

    n_samples = len(wvf_train)
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, args.batch_size):
            idx = perm[i:min(i + args.batch_size, n_samples)]
            wvf_batch = wvf_tensor[idx]
            acg_batch = acg_tensor[idx].reshape(-1, 1, 10, 101) * 10

            optimizer.zero_grad()
            loss = model(wvf_batch, acg_batch)
            loss.backward()

            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    return model


def make_holdout_splits(waveforms: np.ndarray,
                        acg_3d: np.ndarray,
                        labels_encoded: np.ndarray,
                        group_ids: np.ndarray,
                        holdout_fold: int,
                        n_holdout_folds: int,
                        min_cells_per_group: int = 50,
                        seed: int = 42):
    """Create inductive train/test splits by grouping on animal/session IDs.

    Replicates the logic in train_multimodal_holdout.py::main() and
    physmap_script_holdout.r for cross-method consistency.

    Args:
        waveforms: (N, 90) waveform array.
        acg_3d: (N, 10, 101) 3D ACG array.
        labels_encoded: (N,) integer label array.
        group_ids: (N,) array of group identifiers (e.g., specimen_id from
            the H5 file).  Read directly from the NEMO H5 — no metadata.csv.
        holdout_fold: Which fold index to hold out (0-indexed).
        n_holdout_folds: Total number of folds.
        min_cells_per_group: Minimum cells per group to include.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_idx, test_idx, group_ids_train, group_ids_test).
    """
    assert len(group_ids) == len(waveforms), (
        f"group_ids length ({len(group_ids)}) != waveforms ({len(waveforms)})"
    )

    # Filter groups with sufficient cells
    unique, counts = np.unique(group_ids, return_counts=True)
    group_counts = dict(zip(unique, counts))
    valid_groups = [g for g, c in group_counts.items() if c >= min_cells_per_group]
    valid_mask = np.isin(group_ids, valid_groups)
    print(f"[holdout] Using {len(valid_groups)} groups with >= {min_cells_per_group} cells "
          f"({valid_mask.sum()} / {len(group_ids)} cells retained)")

    # Reproducible shuffle of group IDs (matches HIPPIE/PhysMAP logic)
    rng = np.random.RandomState(seed)
    shuffled_groups = np.array(valid_groups)
    rng.shuffle(shuffled_groups)

    # Assign groups to folds (same formula as train_multimodal_holdout.py)
    fold_size = len(valid_groups) // n_holdout_folds
    fold_start = holdout_fold * fold_size
    fold_end = (
        fold_start + fold_size
        if holdout_fold < n_holdout_folds - 1
        else len(valid_groups)
    )
    test_groups = shuffled_groups[fold_start:fold_end]
    train_groups = np.concatenate(
        [shuffled_groups[:fold_start], shuffled_groups[fold_end:]]
    )
    print(f"[holdout] Fold {holdout_fold}/{n_holdout_folds}: "
          f"train_groups={len(train_groups)}, test_groups={len(test_groups)}")

    # Build index arrays (only over valid-mask positions)
    group_col_values = group_ids
    train_idx = np.where(np.isin(group_col_values, train_groups))[0]
    test_idx = np.where(np.isin(group_col_values, test_groups))[0]
    print(f"[holdout] train_cells={len(train_idx)}, test_cells={len(test_idx)}")

    return (train_idx, test_idx,
            train_groups.tolist(), test_groups.tolist())


def evaluate_with_classifiers(train_embeddings: np.ndarray,
                              test_embeddings: np.ndarray,
                              train_labels: np.ndarray,
                              test_labels: np.ndarray) -> dict:
    """
    Evaluate embeddings with multiple classifiers.

    Returns dict with predictions and metrics for each classifier type.
    """
    results = {}

    # Scale embeddings for linear/MLP classifiers
    scaler = StandardScaler()
    train_emb_scaled = scaler.fit_transform(train_embeddings)
    test_emb_scaled = scaler.transform(test_embeddings)

    # Replace NaN/Inf embeddings with zeros first (can arise from zero-waveform
    # units where the contrastive encoder produces undefined outputs).
    # Must happen BEFORE L2 normalization — otherwise NaN rows survive normalization
    # and cause KNN to crash with "Input X contains NaN".
    for _arr in (train_embeddings, test_embeddings):
        _nan_mask = ~np.isfinite(_arr)
        if _nan_mask.any():
            n_bad = int(_nan_mask.any(axis=1).sum())
            print(f"[evaluate] WARNING: {n_bad} embeddings contain NaN/Inf — replacing with 0")
            _arr[_nan_mask] = 0.0

    # L2-normalize embeddings for KNN to match HIPPIE's protocol
    # (train_multimodal_transductive.py ~line 1240). Pairs with metric='cosine'.
    # Zero-norm rows (from the NaN→0 replacement above) get norm=1e-12, producing
    # all-zero unit vectors — harmless for cosine KNN.
    train_emb_l2 = train_embeddings / np.maximum(
        np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12
    )
    test_emb_l2 = test_embeddings / np.maximum(
        np.linalg.norm(test_embeddings, axis=1, keepdims=True), 1e-12
    )

    # 1. KNN evaluation — k=1..20. The SELECTED k (results['knn_best']) is
    # chosen by stratified inner CV on the TRAINING fold only (matches the
    # train-CV protocol used by PhysMAP and the HIPPIE benchmark scripts) and
    # is the single test evaluation reported in the paper.
    # The per-k entries results['knn_k{k}'] retain test-set predictions purely
    # as diagnostic logging — they are NOT used for selection.
    # See hippie/utils.py:select_knn_k_by_train_cv.
    from hippie.utils import select_knn_k_by_train_cv

    # Diagnostic per-k test predictions (not used for selection).
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_emb_l2, train_labels)
        preds = knn.predict(test_emb_l2)
        results[f'knn_k{k}'] = {
            'predictions': preds,
            'accuracy': accuracy_score(test_labels, preds),
            'balanced_accuracy': balanced_accuracy_score(test_labels, preds),
            'f1_macro': f1_score(test_labels, preds, average='macro'),
        }

    # SELECTED k via inner train-CV; single test evaluation.
    knn_result = select_knn_k_by_train_cv(
        train_emb_l2,
        train_labels,
        test_emb_l2,
        test_labels,
        k_grid=range(1, 21),
        inner_cv_splits=5,
        metric="cosine",
        random_state=42,
    )
    best_k = knn_result["best_k"]
    best_knn_acc = knn_result["test_balanced_accuracy"]
    best_knn_preds = knn_result["test_predictions"]

    # Fold the train-CV diagnostic into the per-k entries.
    for k, cv_score in knn_result["per_k_train_cv"].items():
        if f'knn_k{k}' in results:
            results[f'knn_k{k}']['train_cv_balanced_accuracy'] = cv_score

    results['knn_best'] = {
        'predictions': best_knn_preds,
        'k': best_k,
        'balanced_accuracy': best_knn_acc,
        'train_cv_balanced_accuracy': knn_result["best_train_cv_score"],
    }

    # 2. Linear probe (NEMO native)
    try:
        lr = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
        lr.fit(train_emb_scaled, train_labels)
        lr_preds = lr.predict(test_emb_scaled)
        results['linear_probe'] = {
            'predictions': lr_preds,
            'accuracy': accuracy_score(test_labels, lr_preds),
            'balanced_accuracy': balanced_accuracy_score(test_labels, lr_preds),
            'f1_macro': f1_score(test_labels, lr_preds, average='macro')
        }
    except Exception as e:
        print(f"Linear probe failed: {e}")
        results['linear_probe'] = None

    # 3. MLP probe (NEMO native / HIPPIE-style)
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp.fit(train_emb_scaled, train_labels)
        mlp_preds = mlp.predict(test_emb_scaled)
        results['mlp_probe'] = {
            'predictions': mlp_preds,
            'accuracy': accuracy_score(test_labels, mlp_preds),
            'balanced_accuracy': balanced_accuracy_score(test_labels, mlp_preds),
            'f1_macro': f1_score(test_labels, mlp_preds, average='macro')
        }
    except Exception as e:
        print(f"MLP probe failed: {e}")
        results['mlp_probe'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="NEMO Benchmark Evaluation")

    # Data arguments - load from H5 files, NOT HIPPIE format
    parser.add_argument("--dataset-file", type=str, required=True,
                        help="C4 database H5 filename (e.g., 'C4_database_hull_labelled.h5'). "
                             "In cross-dataset mode this is the TRAINING dataset H5.")
    parser.add_argument("--predict-dataset-file", type=str, default=None,
                        help="C4 database H5 filename for the PREDICT (test) dataset. "
                             "When set, enables cross-dataset transfer mode: "
                             "NEMO is trained on --dataset-file and evaluated on "
                             "--predict-dataset-file. The DATASETS_DIRECTORY env var "
                             "must point to a directory containing BOTH H5 files, or "
                             "use --predict-datasets-directory to set a separate root.")
    parser.add_argument("--predict-datasets-directory", type=str, default=None,
                        help="Directory that contains the predict dataset H5 file. "
                             "Defaults to the same DATASETS_DIRECTORY used for training.")
    parser.add_argument("--s3-path", type=str, default=None,
                        help="S3 path to H5 file (overrides --dataset-file for S3 loading)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for results")

    # CV arguments (transductive mode)
    parser.add_argument("--cv-fold", type=int, default=0,
                        help="Which CV fold to run (0 to n_cv_folds-1)")
    parser.add_argument("--n-cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Holdout / inductive mode arguments
    parser.add_argument("--holdout-fold", type=int, default=None,
                        help="Enable holdout (inductive) mode: which animal/session fold to hold out "
                             "(0 to n-holdout-folds-1). When set, --metadata-csv must also be provided.")
    parser.add_argument("--n-holdout-folds", type=int, default=5,
                        help="Number of animal/session holdout folds (default: 5)")
    parser.add_argument("--metadata-csv", type=str, default=None,
                        help="Path to HIPPIE-format metadata.csv used for animal/session holdout splits. "
                             "Expected columns: 'specimen_id' (Allen) or 'subject' (IBL). "
                             "Row order must match the loaded H5 data.")
    parser.add_argument("--grouping-col", type=str, default=None,
                        help="Column in metadata-csv to group by for holdout splits. "
                             "Auto-detected from dataset name if not set "
                             "('subject' for IBL, 'specimen_id' for Allen).")
    parser.add_argument("--min-cells-per-group", type=int, default=50,
                        help="Minimum cells per animal/session to include in holdout splits (default: 50)")

    # Model arguments
    parser.add_argument("--z-dim", type=int, default=256,
                        help="Latent dimension. Default 256 is the locked configuration "
                             "reported in the paper (Methods § Baselines), selected by "
                             "a 42-config sweep on the Hausser dataset.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for contrastive loss")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--activation", type=str, default="gelu",
                        help="Activation function")
    parser.add_argument("--batch-norm", action="store_true",
                        help="Use batch normalization")
    parser.add_argument("--similarity-adjust", action="store_true",
                        help="Use similarity-adjusted targets")
    parser.add_argument("--l2-norm", action="store_true", default=True,
                        help="Apply L2 normalization to embeddings")
    parser.add_argument("--extra-pretrain-files", type=str, nargs="*", default=None,
                        help="Additional H5 files to include in contrastive pretraining "
                             "(pooled with --dataset-file for encoder training only; "
                             "KNN is still trained on --dataset-file embeddings alone).")
    parser.add_argument("--exclude-classes", type=str, nargs="*", default=None,
                        help="Class names to remove from both training and test sets before "
                             "CV splitting (e.g. GoC for Hull to match other methods' splits).")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Compute parity timing setup ----
    # Derive a stable dataset label from the H5 filename, e.g.
    #   "C4_database_hull_labelled.h5" -> "hull_cell_type"-ish.
    # We just strip prefix/suffix; the aggregator can normalise further.
    _cp_dataset = (
        os.path.basename(args.dataset_file)
        .replace("C4_database_", "")
        .replace(".h5", "")
    )
    _cp_run_id = _cp.make_run_id(prefix="nemo-bench-")
    _cp_t0_total = time.perf_counter()
    _cp_total_started_at = __import__("datetime").datetime.utcnow().isoformat(
        timespec="seconds") + "Z"

    # _cp_fold_ref is a mutable container so the closure below can see the
    # updated fold label after _fold_label is determined (holdout vs transductive).
    _cp_fold_ref = [args.cv_fold]

    def _timed(phase_name, *, n_cells=0, n_classes=0, extra=None):
        return _cp.phase(
            method="nemo",
            dataset=_cp_dataset,
            phase=phase_name,
            fold=_cp_fold_ref[0],
            n_cells=n_cells,
            n_classes=n_classes,
            control="",  # NEMO has no shuffle/shift flags
            config="bimodal",
            run_id=_cp_run_id,
            extra={
                "z_dim": args.z_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "script": "nemo_benchmark_evaluation",
                **(extra or {}),
            },
        )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset from H5 file (NOT HIPPIE format)
    print(f"\nLoading C4 database H5 file...")
    if args.s3_path:
        waveforms, acg_3d, labels = load_c4_dataset_from_s3(args.s3_path)
    else:
        waveforms, acg_3d, labels = load_c4_dataset(args.dataset_file)

    print(f"Waveforms shape: {waveforms.shape}")
    print(f"ACG 3D shape: {acg_3d.shape}")

    # Sanitize data: replace NaN/Inf with 0 to prevent NaN loss during training.
    # Also remove neurons whose waveform is all zeros (they contribute nothing).
    n_nan_wvf = int(np.sum(~np.isfinite(waveforms)))
    n_nan_acg = int(np.sum(~np.isfinite(acg_3d)))
    if n_nan_wvf or n_nan_acg:
        print(f"[data-clean] Replacing {n_nan_wvf} NaN/Inf waveform values, "
              f"{n_nan_acg} NaN/Inf ACG values with 0.")
        waveforms = np.nan_to_num(waveforms, nan=0.0, posinf=0.0, neginf=0.0)
        acg_3d = np.nan_to_num(acg_3d, nan=0.0, posinf=0.0, neginf=0.0)

    zero_wvf_mask = np.max(np.abs(waveforms), axis=1) < 1e-8
    if np.any(zero_wvf_mask):
        n_zero = int(np.sum(zero_wvf_mask))
        print(f"[data-clean] Removing {n_zero} neurons with zero/flat waveforms.")
        keep = ~zero_wvf_mask
        waveforms = waveforms[keep]
        acg_3d = acg_3d[keep]
        labels = np.array(labels)[keep]

    # Exclude specified classes from both train and test before any splitting
    if args.exclude_classes:
        keep_mask = ~np.isin(labels, args.exclude_classes)
        n_removed = int((~keep_mask).sum())
        waveforms = waveforms[keep_mask]
        acg_3d    = acg_3d[keep_mask]
        labels    = np.array(labels)[keep_mask]
        print(f"[exclude-classes] Removed {n_removed} neurons from classes "
              f"{args.exclude_classes}. Remaining: {len(waveforms)}")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    label_names = le.classes_

    print(f"\nTotal samples: {len(waveforms)}")
    print(f"Number of classes: {len(label_names)}")
    print(f"Classes: {label_names}")

    # -------------------------------------------------------------------------
    # Cross-dataset transfer mode
    # -------------------------------------------------------------------------
    # When --predict-dataset-file is set we take a completely different path:
    #   1. Train NEMO contrastively on ALL cells from --dataset-file (training)
    #   2. Load --predict-dataset-file as the test dataset
    #   3. Extract embeddings for both sets
    #   4. Train KNN on training embeddings, evaluate on predict embeddings
    # This branch returns early so the holdout/transductive split code below
    # is never reached.
    if args.predict_dataset_file is not None:
        print("\n" + "=" * 60)
        print("CROSS-DATASET TRANSFER MODE")
        print(f"  Training dataset : {args.dataset_file}")
        print(f"  Predict dataset  : {args.predict_dataset_file}")
        print("=" * 60)

        # Reload the predict dataset, possibly from a different directory
        predict_dir = (
            args.predict_datasets_directory
            or os.environ.get('PREDICT_DATASETS_DIRECTORY')
            or os.environ.get('DATASETS_DIRECTORY', '/data/datasets')
        )
        print(f"\nLoading predict dataset from: {predict_dir}/{args.predict_dataset_file}")

        import celltype_ibl.params.config as _config_module
        _orig_dir = _config_module.DATASETS_DIRECTORY
        _config_module.DATASETS_DIRECTORY = predict_dir
        try:
            wvf_predict, acg_predict, labels_predict = load_c4_dataset(
                args.predict_dataset_file
            )
        finally:
            _config_module.DATASETS_DIRECTORY = _orig_dir

        # Encode predict labels using a fresh encoder (independent label space)
        le_predict = LabelEncoder()
        labels_predict_encoded = le_predict.fit_transform(labels_predict)
        predict_label_names = le_predict.classes_

        print(f"\nTraining dataset : {len(waveforms)} cells, {len(label_names)} classes")
        print(f"Predict dataset  : {len(wvf_predict)} cells, {len(predict_label_names)} classes")

        # --- Optionally pool extra datasets for contrastive pretraining ---
        # Extra files expand the encoder's pretraining pool but are excluded from
        # the KNN training step — KNN is always trained on waveforms/acg_3d only.
        if args.extra_pretrain_files:
            wvf_for_training = [waveforms]
            acg_for_training = [acg_3d]
            for extra_h5 in args.extra_pretrain_files:
                ex_wvf, ex_acg, _ = load_c4_dataset(extra_h5)
                wvf_for_training.append(ex_wvf)
                acg_for_training.append(ex_acg)
                print(f"[extra-pretrain] Added {len(ex_wvf)} cells from {extra_h5}")
            wvf_pretrain = np.concatenate(wvf_for_training, axis=0)
            acg_pretrain = np.concatenate(acg_for_training, axis=0)
            print(f"[extra-pretrain] Total pretrain pool: {len(wvf_pretrain)} cells")
        else:
            wvf_pretrain = waveforms
            acg_pretrain = acg_3d

        # --- Train NEMO on ALL training data (inductive: training set only) ---
        print("\n" + "=" * 50)
        print("TRAINING PHASE (Cross-Dataset / Inductive on train)")
        print("=" * 50)
        _fold_label = "cross_dataset"
        _cp_fold_ref[0] = _fold_label
        start_time = time.time()
        with _timed("train", n_cells=len(wvf_pretrain), n_classes=len(label_names)):
            model = train_nemo_model_inductive(wvf_pretrain, acg_pretrain, args, device)
        train_time = time.time() - start_time
        print(f"Cross-dataset training completed in {train_time:.2f}s")

        # --- Extract embeddings ---
        # KNN always trained on the original training dataset only (waveforms/acg_3d),
        # regardless of how large the pretraining pool was.
        print("\n" + "=" * 50)
        print("EMBEDDING PHASE")
        print("=" * 50)
        with _timed("embedding",
                    n_cells=len(waveforms) + len(wvf_predict),
                    n_classes=len(label_names)):
            train_embeddings = get_embeddings_nemo(waveforms, acg_3d, model, device)
            test_embeddings  = get_embeddings_nemo(wvf_predict, acg_predict, model, device)

        print(f"Train embeddings : {train_embeddings.shape}")
        print(f"Predict embeddings: {test_embeddings.shape}")

        # --- Evaluate ---
        print("\n" + "=" * 50)
        print("EVALUATION PHASE")
        print("=" * 50)
        with _timed("evaluate",
                    n_cells=len(waveforms) + len(wvf_predict),
                    n_classes=len(predict_label_names),
                    extra={"k_grid": "1..20"}) as _eval_info:
            results = evaluate_with_classifiers(
                train_embeddings, test_embeddings,
                labels_encoded, labels_predict_encoded
            )
            _eval_info["best_knn_k"] = int(results["knn_best"]["k"])
            _eval_info["best_knn_balanced_accuracy"] = float(
                results["knn_best"]["balanced_accuracy"]
            )

        print("\nCross-dataset Results:")
        for meth, res in results.items():
            if res is not None and 'balanced_accuracy' in res:
                print(f"  {meth}: Balanced Accuracy = {res['balanced_accuracy']:.4f}")

        # --- Save outputs ---
        fold_dir = os.path.join(args.out_dir, "cross_dataset")
        os.makedirs(fold_dir, exist_ok=True)

        best_preds = results['knn_best']['predictions']
        predictions_df = pd.DataFrame({
            'true': le_predict.inverse_transform(labels_predict_encoded),
            'pred': le.inverse_transform(
                # KNN was trained on train label space; re-map to predict label names
                # Predictions are label indices from the TRAIN encoder.
                # We use the train encoder's classes directly.
                best_preds
            ),
            'fold': 'cross_dataset'
        })
        predictions_df.to_csv(
            os.path.join(fold_dir, 'cross_dataset_predictions.csv'), index=False
        )
        # Also write as mlp_predictions.csv for aggregator compatibility
        predictions_df.to_csv(
            os.path.join(fold_dir, 'mlp_predictions.csv'), index=False
        )

        if results.get('linear_probe'):
            pd.DataFrame({
                'true': le_predict.inverse_transform(labels_predict_encoded),
                'pred': le.inverse_transform(results['linear_probe']['predictions']),
                'fold': 'cross_dataset'
            }).to_csv(os.path.join(fold_dir, 'linear_probe_predictions.csv'), index=False)

        if results.get('mlp_probe'):
            pd.DataFrame({
                'true': le_predict.inverse_transform(labels_predict_encoded),
                'pred': le.inverse_transform(results['mlp_probe']['predictions']),
                'fold': 'cross_dataset'
            }).to_csv(os.path.join(fold_dir, 'mlp_probe_predictions.csv'), index=False)

        # Save embeddings
        emb_dir = os.path.join(fold_dir, 'embeddings')
        os.makedirs(emb_dir, exist_ok=True)
        pd.DataFrame(train_embeddings).assign(
            label=le.inverse_transform(labels_encoded)
        ).to_csv(os.path.join(emb_dir, 'train_embeddings.csv'), index=False)
        pd.DataFrame(test_embeddings).assign(
            label=le_predict.inverse_transform(labels_predict_encoded)
        ).to_csv(os.path.join(emb_dir, 'test_embeddings.csv'), index=False)

        # Save metrics
        metrics = {
            'fold': 'cross_dataset',
            'mode': 'cross_dataset',
            'train_dataset': args.dataset_file,
            'predict_dataset': args.predict_dataset_file,
            'train_time_s': train_time,
            'n_train': len(waveforms),
            'n_predict': len(wvf_predict),
            'knn_best_k': results['knn_best']['k'],
            'knn_best_balanced_accuracy': results['knn_best']['balanced_accuracy'],
        }
        for meth in ['knn_k1', 'knn_k5', 'linear_probe', 'mlp_probe']:
            if results.get(meth):
                metrics[f'{meth}_balanced_accuracy'] = results[meth]['balanced_accuracy']
                metrics[f'{meth}_accuracy'] = results[meth]['accuracy']

        pd.DataFrame([metrics]).to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)

        with open(os.path.join(fold_dir, 'cross_dataset_info.json'), 'w') as _f:
            json.dump({
                'train_dataset': args.dataset_file,
                'predict_dataset': args.predict_dataset_file,
                'train_n_cells': int(len(waveforms)),
                'predict_n_cells': int(len(wvf_predict)),
                'train_label_mapping': {name: int(idx) for idx, name in enumerate(label_names)},
                'predict_label_mapping': {
                    name: int(idx) for idx, name in enumerate(predict_label_names)
                },
            }, _f, indent=2)

        print(f"\nCross-dataset results saved to: {fold_dir}")

        # Compute parity total row
        _cp_total_wall = time.perf_counter() - _cp_t0_total
        try:
            _cp.append_row(
                method="nemo",
                dataset=_cp_dataset,
                fold="cross_dataset",
                phase="total",
                wall_seconds=_cp_total_wall,
                peak_gpu_mb=_cp._peak_gpu_mb_and_reset(),
                peak_cpu_mb=_cp._peak_cpu_mb(),
                n_cells=len(waveforms),
                n_classes=len(label_names),
                control="",
                config="bimodal",
                run_id=_cp_run_id,
                started_at=_cp_total_started_at,
                extra={
                    "z_dim": args.z_dim,
                    "epochs": args.epochs,
                    "best_knn_k": int(results["knn_best"]["k"]),
                    "best_knn_balanced_accuracy": float(
                        results["knn_best"]["balanced_accuracy"]
                    ),
                    "script": "nemo_benchmark_evaluation",
                    "cross_dataset": True,
                    "predict_dataset": args.predict_dataset_file,
                },
            )
        except Exception as _e:
            print(f"[compute_parity] WARNING: failed to write total row: {_e}")

        print("Done (cross-dataset)!")
        return

    # Determine run mode: holdout (inductive) or transductive
    _holdout_mode = args.holdout_fold is not None

    if _holdout_mode:
        # --- Holdout (inductive) mode: split by animal/session ---
        # Read group IDs directly from the H5 file (flat-array datasets
        # store specimen_id / subject as top-level datasets).
        import h5py as _h5
        _h5_path = os.path.join(
            os.environ.get('DATASETS_DIRECTORY', '/data/datasets'),
            args.dataset_file
        )
        _group_ids = None
        _grouping_col = None
        with _h5.File(_h5_path, 'r') as _f:
            for _candidate in ('animal_id', 'specimen_id', 'subject'):
                if _candidate in _f:
                    _raw = _f[_candidate][()]
                    _group_ids = np.array([
                        x.decode('utf-8') if isinstance(x, bytes) else str(x)
                        for x in _raw
                    ])
                    _grouping_col = _candidate
                    print(f"[holdout] Read group IDs from H5 field '{_candidate}' "
                          f"({len(np.unique(_group_ids))} unique groups)")
                    break
        if _group_ids is None:
            # Per-neuron-group H5 (e.g., IBL brainwide_good): no top-level
            # subject field.  Read insertion_id from each neuron group (same
            # iteration order as the data loader), then map to subject via
            # metadata.csv (insertion_id → subject is always 1-to-1).
            if args.metadata_csv is None or not os.path.exists(args.metadata_csv):
                raise ValueError(
                    f"NEMO holdout for {args.dataset_file}: H5 has per-neuron-group "
                    f"format with no top-level subject field. Provide --metadata-csv "
                    f"with columns 'insertion_id' and 'subject'."
                )
            import pandas as _pd
            _meta_df = _pd.read_csv(args.metadata_csv)
            _ins_to_subject = dict(zip(
                _meta_df['insertion_id'].astype(str),
                _meta_df['subject'].astype(str),
            ))
            # Iterate neuron groups in the same order and with the same skip
            # condition as load_dandi_ibl_dataset_direct (c4_data_utils.py).
            _group_ids_raw = []
            _labels_raw = []
            with _h5.File(_h5_path, 'r') as _f:
                _top_keys = list(_f.keys())
                _neuron_keys = [k for k in _top_keys if 'neuron' in k.lower()]
                for _nk in _neuron_keys:
                    _grp = _f[_nk]
                    if 'mean_waveform_preprocessed' not in _grp:
                        continue  # same skip as data loader
                    _ins_id = _grp['insertion_id'][()] if 'insertion_id' in _grp else b'unknown'
                    if isinstance(_ins_id, bytes):
                        _ins_id = _ins_id.decode('utf-8')
                    _group_ids_raw.append(_ins_to_subject.get(_ins_id, f'unknown_{_ins_id}'))
                    _lbl = _grp['ground_truth_label'][()] if 'ground_truth_label' in _grp else b'unknown'
                    if isinstance(_lbl, bytes):
                        _lbl = _lbl.decode('utf-8')
                    _labels_raw.append(_lbl)
            # Apply the same unlabeled filter as load_c4_dataset
            _labels_arr = np.array(_labels_raw)
            _labeled_mask = ~np.isin(_labels_arr, ['unlabelled', 'unlabeled', 'unknown', 'Unknown'])
            _group_ids = np.array(_group_ids_raw)[_labeled_mask]
            _grouping_col = 'subject'
            print(f"[holdout] Mapped {len(_group_ids)} neurons via insertion_id→subject "
                  f"({len(np.unique(_group_ids))} unique subjects) from {args.metadata_csv}")
        # Align to data length (H5 may have dropped units during load)
        if len(_group_ids) > len(waveforms):
            _group_ids = _group_ids[:len(waveforms)]

        train_indices, test_indices, _train_groups, _test_groups = make_holdout_splits(
            waveforms=waveforms,
            acg_3d=acg_3d,
            labels_encoded=labels_encoded,
            group_ids=_group_ids,
            holdout_fold=args.holdout_fold,
            n_holdout_folds=args.n_holdout_folds,
            min_cells_per_group=args.min_cells_per_group,
            seed=args.seed,
        )

        print(f"\nHoldout fold {args.holdout_fold}/{args.n_holdout_folds}")
        print(f"Train set: {len(train_indices)} cells from {len(_train_groups)} groups")
        print(f"Test set : {len(test_indices)} cells from {len(_test_groups)} groups")

        # Encode holdout fold for output paths and compute-parity logging
        _fold_label = args.holdout_fold
        _cp_fold_ref[0] = _fold_label
    else:
        # --- Transductive mode: stratified k-fold ---
        skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(waveforms, labels_encoded))
        train_indices, test_indices = splits[args.cv_fold]

        print(f"\nCross-validation fold {args.cv_fold}/{args.n_cv_folds}")
        print(f"Train set: {len(train_indices)} samples")
        print(f"Test set: {len(test_indices)} samples")

        _fold_label = args.cv_fold
        _cp_fold_ref[0] = _fold_label

    # Split data
    wvf_train = waveforms[train_indices]
    wvf_test = waveforms[test_indices]
    acg_train = acg_3d[train_indices]
    acg_test = acg_3d[test_indices]
    label_train = labels_encoded[train_indices]
    label_test = labels_encoded[test_indices]

    if _holdout_mode:
        # Inductive training: only training animals see contrastive loss
        print("\n" + "=" * 50)
        print("TRAINING PHASE (Inductive / Holdout)")
        print("=" * 50)
        start_time = time.time()
        with _timed("train",
                    n_cells=len(wvf_train),
                    n_classes=len(label_names)):
            model = train_nemo_model_inductive(wvf_train, acg_train, args, device)
        train_time = time.time() - start_time
        print(f"Inductive training completed in {train_time:.2f}s")
    else:
        # Transductive training (use all data for representation learning)
        print("\n" + "=" * 50)
        print("TRAINING PHASE (Transductive)")
        print("=" * 50)

        wvf_all = np.vstack([wvf_train, wvf_test])
        acg_all = np.vstack([acg_train, acg_test])

        start_time = time.time()
        with _timed("train",
                    n_cells=len(wvf_all),
                    n_classes=len(label_names)):
            model = train_nemo_model(wvf_all, acg_all, args, device)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f}s")

    # Extract embeddings
    print("\n" + "=" * 50)
    print("EVALUATION PHASE")
    print("=" * 50)

    print("Extracting embeddings...")
    with _timed("embedding",
                n_cells=len(wvf_train) + len(wvf_test),
                n_classes=len(label_names)):
        train_embeddings = get_embeddings_nemo(wvf_train, acg_train, model, device)
        test_embeddings = get_embeddings_nemo(wvf_test, acg_test, model, device)

    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")

    # Evaluate with multiple classifiers
    print("\nRunning evaluation...")
    with _timed("evaluate",
                n_cells=len(wvf_train) + len(wvf_test),
                n_classes=len(label_names),
                extra={"k_grid": "1..20"}) as _eval_info:
        results = evaluate_with_classifiers(
            train_embeddings, test_embeddings,
            label_train, label_test
        )
        _eval_info["best_knn_k"] = int(results["knn_best"]["k"])
        _eval_info["best_knn_balanced_accuracy"] = float(
            results["knn_best"]["balanced_accuracy"]
        )

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    for method, res in results.items():
        if res is not None and 'balanced_accuracy' in res:
            print(f"{method}: Balanced Accuracy = {res['balanced_accuracy']:.4f}")

    # Create output directory (use _fold_label so holdout and transductive
    # jobs don't collide in the same out-dir when fold numbers overlap)
    fold_dir = os.path.join(args.out_dir, f'fold_{_fold_label}')
    os.makedirs(fold_dir, exist_ok=True)

    # Save predictions (HIPPIE format: true,pred,fold)
    # Use 'holdout_predictions.csv' in holdout mode, 'transductive_predictions.csv'
    # in transductive mode so upload_smoke_results.py can find them by glob.
    best_preds = results['knn_best']['predictions']
    predictions_df = pd.DataFrame({
        'true': le.inverse_transform(label_test),
        'pred': le.inverse_transform(best_preds),
        'fold': _fold_label
    })
    _preds_filename = 'holdout_predictions.csv' if _holdout_mode else 'transductive_predictions.csv'
    predictions_df.to_csv(os.path.join(fold_dir, _preds_filename), index=False)
    # Always write a copy as mlp_predictions.csv for aggregator compatibility
    predictions_df.to_csv(os.path.join(fold_dir, 'mlp_predictions.csv'), index=False)

    # Save linear probe predictions
    if results.get('linear_probe'):
        linear_df = pd.DataFrame({
            'true': le.inverse_transform(label_test),
            'pred': le.inverse_transform(results['linear_probe']['predictions']),
            'fold': _fold_label
        })
        linear_df.to_csv(os.path.join(fold_dir, 'linear_probe_predictions.csv'), index=False)

    # Save MLP probe predictions
    if results.get('mlp_probe'):
        mlp_df = pd.DataFrame({
            'true': le.inverse_transform(label_test),
            'pred': le.inverse_transform(results['mlp_probe']['predictions']),
            'fold': _fold_label
        })
        mlp_df.to_csv(os.path.join(fold_dir, 'mlp_probe_predictions.csv'), index=False)

    # Save embeddings
    emb_dir = os.path.join(fold_dir, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)

    train_emb_df = pd.DataFrame(train_embeddings)
    train_emb_df['label'] = le.inverse_transform(label_train)
    train_emb_df.to_csv(os.path.join(emb_dir, 'train_embeddings.csv'), index=False)

    test_emb_df = pd.DataFrame(test_embeddings)
    test_emb_df['label'] = le.inverse_transform(label_test)
    test_emb_df.to_csv(os.path.join(emb_dir, 'test_embeddings.csv'), index=False)

    # Save split info (CV or holdout)
    split_info = {
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'fold': _fold_label,
        'n_folds': args.n_holdout_folds if _holdout_mode else args.n_cv_folds,
        'mode': 'holdout' if _holdout_mode else 'transductive',
        'dataset_file': args.dataset_file,
        'label_mapping': {name: int(idx) for idx, name in enumerate(label_names)}
    }
    if _holdout_mode:
        split_info['train_groups'] = _train_groups
        split_info['test_groups'] = _test_groups
        split_info['grouping_col'] = _grouping_col
        split_info['group_source'] = 'h5_direct'
    with open(os.path.join(fold_dir, 'cv_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Save metrics
    metrics = {
        'fold': _fold_label,
        'mode': 'holdout' if _holdout_mode else 'transductive',
        'train_time_s': train_time,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'knn_best_k': results['knn_best']['k'],
        'knn_best_balanced_accuracy': results['knn_best']['balanced_accuracy'],
    }

    for method in ['knn_k1', 'knn_k5', 'linear_probe', 'mlp_probe']:
        if results.get(method):
            metrics[f'{method}_balanced_accuracy'] = results[method]['balanced_accuracy']
            metrics[f'{method}_accuracy'] = results[method]['accuracy']

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)

    print(f"\nResults saved to: {fold_dir}")

    # ---- Compute parity: emit final "total" row for the whole launch ----
    _cp_total_wall = time.perf_counter() - _cp_t0_total
    try:
        _cp.append_row(
            method="nemo",
            dataset=_cp_dataset,
            fold=_fold_label,
            phase="total",
            wall_seconds=_cp_total_wall,
            peak_gpu_mb=_cp._peak_gpu_mb_and_reset(),
            peak_cpu_mb=_cp._peak_cpu_mb(),
            n_cells=len(waveforms),
            n_classes=len(label_names),
            control="",
            config="bimodal",
            run_id=_cp_run_id,
            started_at=_cp_total_started_at,
            extra={
                "z_dim": args.z_dim,
                "epochs": args.epochs,
                "best_knn_k": int(results["knn_best"]["k"]),
                "best_knn_balanced_accuracy": float(
                    results["knn_best"]["balanced_accuracy"]
                ),
                "script": "nemo_benchmark_evaluation",
            },
        )
        print(f"[compute_parity] total wall_seconds={_cp_total_wall:.1f}")
    except Exception as e:
        print(f"[compute_parity] WARNING: failed to write total row: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
