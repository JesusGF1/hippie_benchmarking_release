"""
Holdout Training Script for HIPPIE - WAVEFORM + ISI ONLY (Bimodal)

This script trains HIPPIE using only two modalities:
  - Waveform (wave): Spike waveforms
  - ISI Distribution (isi): Interspike interval histograms

The ACG (autocorrelogram) modality is excluded from this version.
This is the inductive/holdout evaluation variant (train on subset of animals,
test on held-out animals).

Usage:
    python train_multimodal_holdout_waveisi.py \\
        --config augmentation_ablation \\
        --dataset allen_scope_neuropixel \\
        --holdout_fold 0
"""

import sys
import os
import time

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hippie'))
sys.path.append(code_dir)

# Now import directly from the modules (no 'code.' prefix)
from dataloading import MultiModalEphysDataset, none_safe_collate
from multimodal_model import MultiModalCVAE, MultiModalCVAETrainModule
from utils import make_confmat, get_embeddings
from augmentations import AugmentedMultiModalEphysDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
import pandas as pd
import argparse
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as SKLearnMLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from torch.utils.data import random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from multimodal_model import CVAEConfig, ExperimentConfigs
from torch.utils.data import TensorDataset

# -------------------------------
# Resource monitoring callback
# -------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


class ResourceMonitor(pl.Callback):
    """Logs GPU/CPU memory and average step time to W&B every N steps."""
    def __init__(self, log_every_n_steps: int = 50, namespace: str = "resources"):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.ns = namespace
        self._last_time = None
        self._accum_step_time = 0.0
        self._accum_steps = 0

    def on_train_start(self, trainer, pl_module):
        self._reset_cuda_peaks()
        self._last_time = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._reset_cuda_peaks()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Step timing
        now = time.perf_counter()
        if self._last_time is not None:
            self._accum_step_time += (now - self._last_time)
            self._accum_steps += 1
        self._last_time = now

        # Periodic log
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0 and global_step > 0:
            metrics = {}
            # Avg step time over the interval
            if self._accum_steps > 0:
                metrics[f"{self.ns}/avg_step_time_s"] = self._accum_step_time / self._accum_steps
                self._accum_step_time, self._accum_steps = 0.0, 0

            # CPU memory (RSS)
            if _HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                rss_mb = process.memory_info().rss / (1024 ** 2)
                metrics[f"{self.ns}/cpu_rss_mb"] = rss_mb

            # GPU memory for each device
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    device = torch.device(f"cuda:{d}")
                    curr = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    metrics[f"{self.ns}/gpu{d}_mem_alloc_mb"] = curr
                    metrics[f"{self.ns}/gpu{d}_mem_reserved_mb"] = reserved
                    metrics[f"{self.ns}/gpu{d}_mem_peak_mb"] = peak

            if metrics:
                wandb.log(metrics, step=global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log peaks at epoch boundary, then reset peaks.
        if torch.cuda.is_available():
            metrics = {}
            for d in range(torch.cuda.device_count()):
                peak = torch.cuda.max_memory_allocated(d) / (1024 ** 2)
                metrics[f"{self.ns}/gpu{d}_mem_epoch_peak_mb"] = peak
            if metrics:
                wandb.log(metrics, step=trainer.global_step)
        self._reset_cuda_peaks()

    def _reset_cuda_peaks(self):
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)


def get_embeddings_multimodal(loader, model, mask_class_labels=False):
    """Get embeddings from multimodal model.

    Args:
        loader: DataLoader for the dataset
        model: The CVAE model
        mask_class_labels: If True, use zero embeddings instead of true class labels
                          to prevent data leakage during test set evaluation
    """
    model.eval()
    # Ensure model is on the correct device
    if torch.cuda.is_available():
        model = model.cuda()

    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            data_dict, batch_labels = batch
            if torch.cuda.is_available():
                data_dict = {k: v.cuda() for k, v in data_dict.items()}
                batch_labels = batch_labels.cuda()

            print(f"Data dict shapes: {[(k, v.shape) for k, v in data_dict.items()]}")
            print(f"Batch labels shape: {batch_labels.shape}")

            # Extract class labels, source labels, and super_region labels if available
            if batch_labels.dim() > 1 and batch_labels.shape[1] > 1:
                class_labels = batch_labels[:, 0]  # First column is class labels
                source_labels = batch_labels[:, 1]  # Second column is source labels
                # Third column is super_region labels if available
                super_region_labels = batch_labels[:, 2] if batch_labels.shape[1] > 2 else None

                print(f"Extracted class labels shape: {class_labels.shape}")
                print(f"Extracted source labels shape: {source_labels.shape}")
                if super_region_labels is not None:
                    print(f"Extracted super_region labels shape: {super_region_labels.shape}")
                print(f"Class label range: {class_labels.min().item()} to {class_labels.max().item()}")

                # Check for invalid labels (-1) and handle them
                invalid_mask = class_labels == -1
                if invalid_mask.any():
                    print(f"Found {invalid_mask.sum().item()} invalid labels (-1), setting to 0")
                    class_labels = class_labels.clone()
                    class_labels[invalid_mask] = 0
            else:
                class_labels = batch_labels
                source_labels = None
                super_region_labels = None

            # Mask class labels if requested (for test set to prevent leakage)
            class_labels_for_encoding = None if mask_class_labels else class_labels

            # Get latent embeddings with proper labels (including super_regions)
            h, mu, log_var = model.encode(
                data_dict,
                source_labels=source_labels,
                class_labels=class_labels_for_encoding,
                super_region_labels=super_region_labels
            )
            embeddings.append(mu.cpu().numpy())
            labels.append(class_labels.cpu().numpy())

    return np.vstack(embeddings), np.hstack(labels)


def _normalize_label(lbl):
    """Mirror of train_multimodal_transductive._normalize_label.

    Undoes both real bytestrings and str(bytes) artifacts (e.g. the literal
    3-char string ``b''`` that appears in the hausser unified-layer
    labels.csv). Returns a plain str (or "" for missing/empty).
    """
    try:
        if pd.isna(lbl):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(lbl, bytes):
        try:
            return lbl.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    s = str(lbl).strip()
    if len(s) >= 3 and s[0] == "b" and s[-1] in ("'", '"') and s[1] == s[-1]:
        s = s[2:-1]
    return s


def _is_unlabeled(lbl) -> bool:
    """Mirror of train_multimodal_transductive._is_unlabeled.

    Drops NaN, empty strings, empty bytestrings, the literal 3-char string
    ``b''``, and labels containing 'unlabeled', 'unknown', 'juxt', 'axo',
    or 'vgat'.
    """
    s = _normalize_label(lbl)
    if not s:
        return True
    s_low = s.lower()
    return any(x in s_low for x in ("unlabeled", "unlabelled", "unknown",
                                    "juxt", "axo", "vgat"))


def shuffle_labels_independently(train_labels, test_labels, seed=42):
    """Randomly shuffle train and test labels independently."""
    np.random.seed(seed)

    # Shuffle train labels
    shuffled_train = train_labels.copy()
    np.random.shuffle(shuffled_train)

    # Use different seed for test to ensure independence
    np.random.seed(seed + 1)
    shuffled_test = test_labels.copy()
    np.random.shuffle(shuffled_test)

    return shuffled_train, shuffled_test


def shift_labels_train_only(train_labels, test_labels, shift=1):
    """Shift training labels by N positions (wrapping around), keep test labels unchanged.

    With shift=1: class 0 -> 1, 1 -> 2, ..., N-1 -> 0. Test labels are returned unchanged
    so the resulting confusion matrix should show a diagonal shifted by `shift` positions
    if the model has truly learned to fit the (mislabeled) training data.
    """
    unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
    num_classes = len(unique_classes)
    shift_mapping = {old: unique_classes[(i + shift) % num_classes]
                     for i, old in enumerate(unique_classes)}
    shifted_train = np.array([shift_mapping[l] for l in train_labels])
    return shifted_train, test_labels.copy()


def _run_simple_probes(train_embeddings, train_labels, test_embeddings, test_labels):
    """Evaluate HIPPIE embeddings with sklearn's LogisticRegression + MLPClassifier.

    Matches NEMO's probe setup in `comparison_methods/nemo/scripts/nemo_benchmark_evaluation.py`
    (`evaluate_with_classifiers`). Replaces a previous PyTorch-Lightning MLP head whose BN +
    LayerNorm + label-smoothing × class-weights training loop collapsed toward the majority
    class on small hausser-scale folds (see Phase 2 investigation, 2026-04-11).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_embeddings)
    Xte = scaler.transform(test_embeddings)

    lr = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    lr.fit(Xtr, train_labels)
    lr_preds = lr.predict(Xte)
    lr_ba = balanced_accuracy_score(test_labels, lr_preds)

    mlp = SKLearnMLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(Xtr, train_labels)
    mlp_preds = mlp.predict(Xte)
    mlp_ba = balanced_accuracy_score(test_labels, mlp_preds)

    return lr_preds, lr_ba, mlp_preds, mlp_ba


def split_by_animals(metadata_df, train_animals, test_animals):
    """Split data indices based on animal IDs."""
    train_indices = metadata_df[metadata_df['specimen_id'].isin(train_animals)].index.tolist()
    test_indices = metadata_df[metadata_df['specimen_id'].isin(test_animals)].index.tolist()
    return train_indices, test_indices


def create_balanced_sampler(dataset, labels):
    """
    Create a WeightedRandomSampler for class-balanced sampling.

    Args:
        dataset: The dataset to sample from
        labels: numpy array of class labels

    Returns:
        WeightedRandomSampler configured for balanced sampling
    """
    # Count samples per class
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Calculate class weights (inverse frequency)
    class_weights = 1.0 / label_counts

    # Create a weight for each sample based on its class
    sample_weights = np.zeros(len(labels))
    for label_idx, label in enumerate(unique_labels):
        mask = labels == label
        sample_weights[mask] = class_weights[label_idx]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True  # Sample with replacement to ensure balanced batches
    )

    print(f"\n{'='*60}")
    print("Class-Balanced Sampling Enabled")
    print(f"{'='*60}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Class distribution before balancing:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  Class {label}: {count} samples ({100*count/len(labels):.2f}%)")
    print(f"Class weights (1/frequency): {class_weights}")
    print(f"{'='*60}\n")

    return sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="class_decoder_source_bn_aug_reg")
    parser.add_argument("--dataset", type=str, default="allen_s_n_a_subset_no_superregions")
    parser.add_argument("--z_dim", type=int, default=None,
                        help="Total latent dimension. If unset, computed as "
                             "z_dim_per_modality * num_modalities (e.g., 15 for trimodal, 10 for bimodal).")
    parser.add_argument("--z_dim_per_modality", type=int, default=10,
                        help="Latent dimensions per input modality. Locked default 10 (z=20 bimodal) "
                             "selected by Hausser hyperparameter sweep (42 configs; frozen before "
                             "any other benchmark dataset was evaluated).")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for KL divergence loss. Locked default 1.0 selected by "
                             "Hausser hyperparameter sweep (42 configs; frozen before any other "
                             "benchmark dataset was evaluated).")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--supervised_batch_size", type=int, default=64)
    parser.add_argument("--pretrain_max_epochs", type=int, default=50)
    parser.add_argument("--finetune_max_epochs", type=int, default=20)
    parser.add_argument("--supervised_max_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--wandb_tag", type=str, default="hippie_holdout")
    parser.add_argument("--project", type=str, default="HIPPIE")
    parser.add_argument("--finetune_without_labels", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="multimodal")
    parser.add_argument("--mod1_weight", type=float, default=1.0)
    parser.add_argument("--mod2_weight", type=float, default=1.0)
    parser.add_argument("--wave_weight", type=float, default=1.0)
    parser.add_argument("--isi_weight", type=float, default=1.0)
    # ACG not used in bimodal (waveisi) version

    # New arguments for holdout testing
    parser.add_argument("--n_holdout_folds", type=int, default=5, help="Number of animal holdout folds")
    parser.add_argument("--holdout_fold", type=int, default=0, help="Which holdout fold to run (0 to n_holdout_folds-1)")
    parser.add_argument("--min_cells_per_animal", type=int, default=50, help="Minimum cells per animal to include")
    parser.add_argument("--shuffle_labels", action="store_true", help="Randomly shuffle labels for control experiments")
    parser.add_argument("--shift_labels", action="store_true",
                        help="Shift training labels by 1 position (class 0->1, 1->2, ...) "
                             "for control experiments. Test labels stay unchanged.")

    # Class balancing argument.
    # NOTE: This flag exists as an opt-in but was NOT used in the published runs.
    # The cached predictions in results/ were produced without --use_balanced_sampling
    # (i.e., default uniform random sampling). Kept for users who want to experiment.
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use class-balanced sampling via WeightedRandomSampler during supervised training. "
                             "Off by default; was NOT used in the published runs.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (seeds python/numpy/torch/CUDA and DataLoader workers via pl.seed_everything).")
    args = parser.parse_args()

    # Resolve total z_dim from per-modality default unless an explicit override was passed.
    # train_multimodal_holdout_waveisi uses 2 modalities {wave, isi} (no ACG), so we
    # can resolve immediately after parsing (before z_dim is used in run_name / paths).
    # Defaults (z_dim_per_modality=5, beta=0.9) come from systematic architectural ablation
    # on cellexplorer_cell_type + lisberger_labeled_cell_type only.
    if args.z_dim is None:
        _NUM_MODALITIES = 2  # bimodal: wave + isi only
        args.z_dim = args.z_dim_per_modality * _NUM_MODALITIES
        print(f"[z_dim auto] z_dim_per_modality={args.z_dim_per_modality} * "
              f"num_modalities={_NUM_MODALITIES} -> z_dim={args.z_dim}")

    # wandb is opt-in: default to disabled unless WANDB_MODE is set explicitly.
    os.environ.setdefault("WANDB_MODE", "disabled")
    if os.environ["WANDB_MODE"].lower() not in ("disabled", "dryrun"):
        wandb.login()

    # Initialize wandb logger
    shuffle_suffix = "_shuffled" if args.shuffle_labels else ("_shifted" if args.shift_labels else "")
    wandb_logger = pl.loggers.WandbLogger(
        project=args.project,
        name=f"{args.wandb_tag}-{args.dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}",
        tags=[args.wandb_tag, f"holdout_fold_{args.holdout_fold}"],
    )
    # Initialize run via logger and log phase before training
    _ = wandb_logger.experiment
    wandb_logger.experiment.log({"phase": "holdout_start"})

    print(f"Running holdout fold {args.holdout_fold}/{args.n_holdout_folds}")

    # Helper function to find dataset files
    def find_dataset_file(dataset, filename):
        env_root = os.environ.get("HIPPIE_DATA_ROOT")
        possible_paths = []
        if env_root:
            possible_paths.append(f"{env_root.rstrip('/')}/{dataset}/{filename}")
        possible_paths.extend([
            # Local checkout (default for reviewers): hippie_benchmarking_release/datasets/
            f"./datasets/{dataset}/{filename}",
            f"datasets/{dataset}/{filename}",
            f"./datasets_hippie/{dataset}/{filename}",
            f"datasets_hippie/{dataset}/{filename}",
            # Container-image fallbacks (used only when no local copy is present).
            f"/data/datasets/{dataset}/{filename}",
            f"/datasets_hippie/{dataset}/{filename}",
            f"/src/datasets_hippie/{dataset}/{filename}",
            f"/src/datasets/{dataset}/{filename}",
        ])
        local_roots = ("./datasets/", "datasets/", "./datasets_hippie/", "datasets_hippie/")
        for path in possible_paths:
            if os.path.exists(path):
                outside_checkout = not path.startswith(local_roots) and not (
                    env_root and path.startswith(env_root.rstrip("/"))
                )
                if outside_checkout:
                    # Resolving to a container mount rather than the checkout means
                    # the run may not be reproducible from this repository, and any
                    # cache it writes can silently disagree with datasets/.
                    print(
                        f"WARNING: {filename} for '{dataset}' resolved to {path}, "
                        f"which is outside this repository. Results computed here "
                        f"may not match datasets/{dataset}/. Set HIPPIE_DATA_ROOT "
                        f"to pin the source.",
                        file=sys.stderr,
                    )
                print(f"Found {filename} at: {path}")
                return path
        raise FileNotFoundError(f"Could not find {filename} in any of these locations: {possible_paths}")

    # Load dataset and metadata
    dataset = args.dataset
    # Accept any Allen dataset variant with animal metadata
    allen_datasets = [
        "allen_s_n_a_subset_no_superregions",
        "allen_scope_neuropixel_area_subset",
        "allen_scope_neuropixel_area"
    ]
    if not any(dataset.startswith(prefix) for prefix in ["allen_", "Allen_"]):
        raise ValueError(f"This script is designed for Allen datasets with animal metadata. Got: {dataset}")

    # Load metadata to get animal information
    metadata_path = find_dataset_file(dataset, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path)

    # Filter animals with sufficient cells
    animal_counts = metadata_df['specimen_id'].value_counts()
    valid_animals = animal_counts[animal_counts >= args.min_cells_per_animal].index.tolist()
    metadata_df = metadata_df[metadata_df['specimen_id'].isin(valid_animals)]

    print(f"Using {len(valid_animals)} animals with >= {args.min_cells_per_animal} cells each")
    print(f"Total cells after filtering: {len(metadata_df)}")

    # Create animal-based splits — seed python random, numpy, torch (CPU+CUDA), and workers
    pl.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    shuffled_animals = np.array(valid_animals)
    np.random.shuffle(shuffled_animals)

    # Split animals into folds
    fold_size = len(valid_animals) // args.n_holdout_folds
    fold_start = args.holdout_fold * fold_size
    fold_end = fold_start + fold_size if args.holdout_fold < args.n_holdout_folds - 1 else len(valid_animals)

    test_animals = shuffled_animals[fold_start:fold_end]
    train_animals = np.concatenate([shuffled_animals[:fold_start], shuffled_animals[fold_end:]])

    print(f"Train animals: {len(train_animals)}, Test animals: {len(test_animals)}")

    # Get indices for train/test split
    train_indices, test_indices = split_by_animals(metadata_df, train_animals, test_animals)

    # Load the actual data (bimodal: wave + isi only, no ACG)
    supervised_wf = pd.read_csv(find_dataset_file(dataset, "waveforms.csv")).to_numpy()
    supervised_isi = pd.read_csv(find_dataset_file(dataset, "isi_dist.csv")).to_numpy()

    # Load labels and drop unlabeled rows BEFORE encoding (and BEFORE the
    # animal-based train/test split is materialized into modality arrays).
    # This script is Allen-only in practice and Allen labels don't currently
    # contain b'' or 'unlabeled' rows, but we apply the filter defensively
    # so a future dataset addition can't silently corrupt training the way
    # the hausser b'' rows did in the transductive script. See _is_unlabeled.
    try:
        labels_path = find_dataset_file(dataset, "labels.csv")
        labels = pd.read_csv(labels_path)
        supervised_labels = labels[labels.columns[0]].values

        keep_mask = np.array([not _is_unlabeled(lbl) for lbl in supervised_labels])
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            print(f"[label filter] dropping {n_dropped}/{len(supervised_labels)} "
                  f"unlabeled rows from {dataset}")
            # Drop from modality arrays atomically
            supervised_wf = supervised_wf[keep_mask]
            supervised_isi = supervised_isi[keep_mask]
            supervised_labels = supervised_labels[keep_mask]
            # Re-map train_indices/test_indices from original-CSV row positions
            # to post-filter positions. cumsum(keep_mask)-1 maps original i to
            # its new index iff keep_mask[i] is True.
            new_index_for_old = np.cumsum(keep_mask) - 1
            train_indices = [int(new_index_for_old[i]) for i in train_indices if keep_mask[i]]
            test_indices = [int(new_index_for_old[i]) for i in test_indices if keep_mask[i]]
            print(f"[label filter] post-filter train={len(train_indices)} "
                  f"test={len(test_indices)}")
            # NOTE: metadata_df is intentionally NOT touched here. It was
            # already filtered to valid_animals earlier (different criterion,
            # different length) and is only used for the train/test split,
            # which we've already remapped via train_indices/test_indices.

        # Normalize bytestrings AND str(bytes) artifacts -> plain str
        supervised_labels = np.array([_normalize_label(lbl) for lbl in supervised_labels])
        le = LabelEncoder().fit(supervised_labels)
        supervised_labels = le.transform(supervised_labels)

        # Drop classes with fewer than MIN_CELLS_PER_CLASS samples. Mirrors the
        # filter in train_multimodal_transductive_waveisi.py — mandatory per the
        # task doc's "Dataset Filtering Rules" section.
        MIN_CELLS_PER_CLASS = 10
        unique, counts = np.unique(supervised_labels, return_counts=True)
        rare_encoded = {lbl for lbl, cnt in zip(unique, counts) if cnt < MIN_CELLS_PER_CLASS}
        if rare_encoded:
            keep_mask_rare = np.array([lbl not in rare_encoded for lbl in supervised_labels])
            n_rare = int((~keep_mask_rare).sum())
            rare_names = le.inverse_transform(sorted(rare_encoded))
            print(f"[min_cells filter] dropping {n_rare} cells from {len(rare_encoded)} "
                  f"rare classes (< {MIN_CELLS_PER_CLASS} samples): {list(rare_names)}")
            supervised_wf = supervised_wf[keep_mask_rare]
            supervised_isi = supervised_isi[keep_mask_rare]
            supervised_labels = supervised_labels[keep_mask_rare]
            new_index_for_old = np.cumsum(keep_mask_rare) - 1
            train_indices = [int(new_index_for_old[i]) for i in train_indices if keep_mask_rare[i]]
            test_indices = [int(new_index_for_old[i]) for i in test_indices if keep_mask_rare[i]]
            remaining_label_strings = le.inverse_transform(supervised_labels)
            le = LabelEncoder().fit(remaining_label_strings)
            supervised_labels = le.transform(remaining_label_strings)
            print(f"[min_cells filter] post-filter train={len(train_indices)} "
                  f"test={len(test_indices)} classes={len(le.classes_)}")
    except FileNotFoundError:
        print(f"No labels.csv found for {dataset}")
        supervised_labels = np.zeros(len(supervised_wf))
        le = LabelEncoder().fit(supervised_labels)

    # Get config first
    config = getattr(ExperimentConfigs, args.config)()
    # Apply --beta override so the CLI flag is not silently ignored
    config.beta = args.beta
    print(f"[CVAEConfig] beta={config.beta}")

    # Load super_regions if available (hierarchical labels for region conditioning)
    supervised_super_regions = None
    super_region_le = None
    num_super_regions = None
    try:
        super_regions_path = find_dataset_file(dataset, "super_regions.csv")
        super_regions_df = pd.read_csv(super_regions_path)
        supervised_super_regions = super_regions_df[super_regions_df.columns[0]].values
        super_region_le = LabelEncoder().fit(supervised_super_regions)
        supervised_super_regions = super_region_le.transform(supervised_super_regions)
        num_super_regions = len(np.unique(supervised_super_regions))
        print(f"Loaded super_regions with {num_super_regions} unique regions: {super_region_le.classes_}")
        config.use_super_region_embedding = True
    except FileNotFoundError:
        print(f"No super_regions.csv found for {dataset}, proceeding without super_region conditioning")

    # Apply the animal-based split (bimodal: no ACG)
    wf_train = supervised_wf[train_indices]
    wf_test = supervised_wf[test_indices]
    isi_train = supervised_isi[train_indices]
    isi_test = supervised_isi[test_indices]
    label_train = supervised_labels[train_indices]
    label_test = supervised_labels[test_indices]

    # Split super_regions if available
    if supervised_super_regions is not None:
        super_region_train = supervised_super_regions[train_indices]
        super_region_test = supervised_super_regions[test_indices]
    else:
        super_region_train = None
        super_region_test = None

    print(f"Train set: {len(wf_train)} cells from {len(train_animals)} animals")
    print(f"Test set: {len(wf_test)} cells from {len(test_animals)} animals")

    # Apply label shuffling if requested (after the train/test split)
    if args.shuffle_labels:
        print("Shuffling train and test labels independently for control experiment...")
        label_train, label_test = shuffle_labels_independently(label_train, label_test, seed=42)
        print(f"Labels shuffled! Train classes: {np.unique(label_train)}, Test classes: {np.unique(label_test)}")

    # Apply label shifting if requested (after the train/test split)
    if args.shift_labels:
        print("Shifting training labels by 1 position for control experiment...")
        print(f"Original train label distribution: {dict(zip(*np.unique(label_train, return_counts=True)))}")
        label_train, label_test = shift_labels_train_only(label_train, label_test, shift=1)
        print(f"Shifted train label distribution: {dict(zip(*np.unique(label_train, return_counts=True)))}")
        print("Test labels remain unchanged - expect shifted diagonal in confusion matrix")

    # Define modalities and weights (bimodal: wave + isi only, no ACG)
    modalities = {
        "wave": 50,
        "isi": 100,
    }

    assert args.z_dim is not None, "args.z_dim should have been resolved right after argparse"
    modality_weights = {
        "wave": args.wave_weight,
        "isi": args.isi_weight,
    }

    # -------------------------------
    # PRETRAINING PHASE (on training animals only)
    # -------------------------------

    # Unified-layer dataset mapping. Names match
    # data_wrangling_scripts/manifest.json so the deploy
    # helpers (which download into /data/datasets/<name>/) produce paths that
    # find_dataset_file() resolves correctly. Holdout uses this dict only to
    # look up the source ID for the target dataset (it pretrains single-dataset
    # on training animals, not multi-dataset like the transductive script).
    # Stable IDs match train_multimodal_transductive.ALL_DATASETS so cached
    # source-embedding rows are compatible across the two entry points.
    all_dataset_files = {
        "hull_cell_type": 1,
        "cellexplorer_cell_type": 2,
        "hausser_cell_type": 3,
        "lisberger_labeled_cell_type": 4,
        "dandi_000041_cell_type": 5,
        "dandi_000473_cell_type": 6,
        "dandi_000955_cell_type": 7,
        "allen_scope_neuropixel_area_subset": 8,
        "ibl_brainwide": 9,
    }

    num_sources = max(all_dataset_files.values()) + 1

    # Re-encode labels to ensure they start from 0 and are contiguous for training set
    unique_train_labels = np.unique(label_train)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_train_labels)}
    label_train_reencoded = np.array([label_mapping[label] for label in label_train])
    label_test_reencoded = np.array([label_mapping.get(label, -1) for label in label_test])  # -1 for unseen labels

    num_class_labels = len(unique_train_labels)
    print(f"Training with {num_class_labels} classes: {unique_train_labels}")
    print(f"Label mapping: {label_mapping}")

    # Update labels for training
    label_train = label_train_reencoded
    label_test = label_test_reencoded

    # Create source labels for embedding (only train animals)
    source_labels_train = all_dataset_files[dataset] * np.ones(len(label_train))

    # Create pretraining datasets (only train animals, bimodal: no ACG)
    train_data_dict = {
        "wave": wf_train,
        "isi": isi_train,
    }

    # Stack labels: (class, source) or (class, source, super_region)
    if super_region_train is not None:
        train_labels_stacked = np.vstack((np.zeros_like(label_train), source_labels_train, super_region_train)).T
    else:
        train_labels_stacked = np.vstack((np.zeros_like(label_train), source_labels_train)).T

    dataset_train_multi = MultiModalEphysDataset(
        train_data_dict,
        train_labels_stacked,  # No class labels for pretraining, but include super_region if available
        mode="multi"
    )

    # Create augmented dataset if needed
    if "augmentation" in args.config:
        dataset_train_multi = AugmentedMultiModalEphysDataset(dataset_train_multi, config)

    train_loader_multi = torch.utils.data.DataLoader(
        dataset_train_multi,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=none_safe_collate,
        num_workers=0
    )

    # Initialize model
    joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_super_regions,  # Add super_region conditioning
        config=config,
    )

    joint_model = MultiModalCVAETrainModule(
        joint_model,
        modality_weights=modality_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        config=config,
    )

    # Set up trainer
    callbacks = [ResourceMonitor()]
    if args.early_stopping_patience > 0:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="train_loss",
            patience=args.early_stopping_patience,
            mode="min"
        )
        callbacks.append(early_stop_callback)

    # Phase marker for pretraining start
    wandb_logger.experiment.log({"phase": "pretrain_start"})

    trainer = pl.Trainer(
        max_epochs=args.pretrain_max_epochs,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    print("Starting pretraining...")
    trainer.fit(joint_model, train_loader_multi)

    # Save pretrained model
    joint_path = f"../results_hippie_holdout/{dataset}/fold_{args.holdout_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/joint_model.ckpt"
    os.makedirs(os.path.dirname(joint_path), exist_ok=True)
    trainer.save_checkpoint(joint_path)

    print("Pretraining completed!")

    # -------------------------------
    # SUPERVISED PHASE
    # -------------------------------

    supervised_joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_super_regions,  # Add super_region conditioning
        config=config,
    )

    # Load pretrained weights but skip the class embedding layer
    joint_seq = torch.load(joint_path)
    if "model.class_embedding.weight" in joint_seq["state_dict"]:
        joint_seq["state_dict"].pop("model.class_embedding.weight")

    # If we're using super_region embeddings, we need to skip incompatible layers
    # because the embedding dimension changed (affects fusion_encoder and decoder_fcs input sizes)
    if num_super_regions is not None and config.use_super_region_embedding:
        print(f"Detected super_region embedding addition - will skip loading super_region_embedding, fusion_encoder and decoder_fcs layers")
        keys_to_remove = [k for k in joint_seq["state_dict"].keys()
                         if k.startswith("model.super_region_embedding.") or
                            k.startswith("model.fusion_encoder.") or
                            k.startswith("model.decoder_fcs.")]
        for key in keys_to_remove:
            joint_seq["state_dict"].pop(key)

    supervised_joint_model = MultiModalCVAETrainModule(
        supervised_joint_model,
        modality_weights=modality_weights,
        learning_rate=(1/10)*args.learning_rate,
        weight_decay=args.weight_decay,
        config=config,
    )
    supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)

    # Create supervised training datasets (only train animals)
    source_labels_train_supervised = all_dataset_files[dataset] * np.ones_like(label_train)

    # Stack labels: (class, source) or (class, source, super_region)
    if super_region_train is not None:
        train_labels_supervised_stacked = np.vstack((label_train, source_labels_train_supervised, super_region_train)).T
    else:
        train_labels_supervised_stacked = np.vstack((label_train, source_labels_train_supervised)).T

    dataset_train_multi_supervised = MultiModalEphysDataset(
        train_data_dict,
        train_labels_supervised_stacked,
        mode="multi"
    )

    # Create DataLoader with optional class-balanced sampling
    if args.use_balanced_sampling:
        # Create balanced sampler for training data
        train_sampler = create_balanced_sampler(dataset_train_multi_supervised, label_train)

        train_loader_multi_supervised = torch.utils.data.DataLoader(
            dataset_train_multi_supervised,
            batch_size=args.supervised_batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=none_safe_collate,
            num_workers=0
        )
    else:
        train_loader_multi_supervised = torch.utils.data.DataLoader(
            dataset_train_multi_supervised,
            batch_size=args.supervised_batch_size,
            shuffle=True,
            collate_fn=none_safe_collate,
            num_workers=0
        )

    # Phase marker for supervised start
    wandb_logger.experiment.log({"phase": "supervised_start"})

    # Train supervised model
    supervised_trainer = pl.Trainer(
        max_epochs=args.supervised_max_epochs,
        callbacks=[ResourceMonitor()],
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    print("Starting supervised training...")
    supervised_trainer.fit(supervised_joint_model, train_loader_multi_supervised)

    # Save supervised model
    supervised_path = f"../results_hippie_holdout/{dataset}/fold_{args.holdout_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}/supervised_model.ckpt"
    supervised_trainer.save_checkpoint(supervised_path)

    print("Supervised training completed!")

    # -------------------------------
    # EVALUATION ON HELD-OUT ANIMALS
    # -------------------------------

    # Create test dataset (bimodal: no ACG)
    test_data_dict = {
        "wave": wf_test,
        "isi": isi_test,
    }

    source_labels_test = all_dataset_files[dataset] * np.ones_like(label_test)

    # Stack labels: (class, source) or (class, source, super_region)
    if super_region_test is not None:
        test_labels_stacked = np.vstack((label_test, source_labels_test, super_region_test)).T
    else:
        test_labels_stacked = np.vstack((label_test, source_labels_test)).T

    dataset_test_multi = MultiModalEphysDataset(
        test_data_dict,
        test_labels_stacked,
        mode="multi"
    )

    test_loader_multi = torch.utils.data.DataLoader(
        dataset_test_multi,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=0
    )

    # Get embeddings for both train and test
    print("Extracting embeddings...")
    # Train embeddings: use true labels (no masking)
    train_embeddings, train_labels_extracted = get_embeddings_multimodal(
        train_loader_multi_supervised,
        supervised_joint_model.model,
        mask_class_labels=False
    )
    # Test embeddings: mask class labels to prevent data leakage
    test_embeddings, test_labels_extracted = get_embeddings_multimodal(
        test_loader_multi,
        supervised_joint_model.model,
        mask_class_labels=True
    )

    # Ensure results directory exists
    results_dir = f"../results_hippie_holdout/{dataset}/fold_{args.holdout_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}"
    os.makedirs(results_dir, exist_ok=True)

    # Save and log pretraining embeddings (before supervised training results)
    pretraining_embeddings_df = pd.DataFrame(train_embeddings)
    pretraining_embeddings_path = f"{results_dir}/pretraining_embeddings.csv"
    pretraining_embeddings_df.to_csv(pretraining_embeddings_path, index=False)

    # Log pretraining embeddings as wandb artifact
    wandb_logger.experiment.log_artifact(
        pretraining_embeddings_path,
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-pretraining_embeddings.csv",
        type="pretraining_embeddings"
    )

    # K-NN classification — k selected by stratified inner CV on the TRAINING
    # animals only, then evaluated once on the held-out test animals.
    # See hippie/utils.py:select_knn_k_by_train_cv.
    from hippie.utils import select_knn_k_by_train_cv
    print("Performing K-NN classification...")
    knn_result = select_knn_k_by_train_cv(
        train_embeddings,
        train_labels_extracted,
        test_embeddings,
        test_labels_extracted,
        k_grid=range(1, 21),
        inner_cv_splits=5,
        metric="cosine",
        random_state=args.seed if hasattr(args, "seed") else 42,
    )
    best_k = knn_result["best_k"]
    best_accuracy = knn_result["test_balanced_accuracy"]
    best_predictions = knn_result["test_predictions"]
    for k, cv_score in knn_result["per_k_train_cv"].items():
        wandb_logger.experiment.log({
            f"knn_train_cv_balanced_accuracy_k{k}": cv_score,
            "holdout_fold": args.holdout_fold,
            "n_train_animals": len(train_animals),
            "n_test_animals": len(test_animals),
            "n_train_cells": len(train_indices),
            "n_test_cells": len(test_indices),
        })
        print(f"Inner train-CV balanced accuracy (k={k}): {cv_score:.4f}")

    print(f"Best K-NN performance: k={best_k}, accuracy={best_accuracy:.4f}")

    # Log best accuracy as summary metric
    wandb_logger.experiment.log({
        "best_holdout_knn_accuracy": best_accuracy,
        "best_k": best_k
    })

    # -------------------------------
    # SIMPLE PROBES (linear + MLP, sklearn)
    # -------------------------------
    # Matches NEMO's probe setup (see
    # comparison_methods/nemo/scripts/nemo_benchmark_evaluation.py
    # ::evaluate_with_classifiers). Replaces a prior PyTorch-Lightning MLP
    # head whose BN / LayerNorm / label-smoothing training loop collapsed
    # toward the majority class on small folds (2026-04-11 investigation).
    print("\nRunning simple probes (sklearn LR + MLP) on embeddings...")
    wandb_logger.experiment.log({"phase": "simple_probes_start"})

    label_names = le.classes_

    lr_preds, lr_ba, mlp_preds, mlp_ba = _run_simple_probes(
        train_embeddings, train_labels_extracted,
        test_embeddings, test_labels_extracted,
    )
    print(f"Linear probe (LogisticRegression) balanced accuracy: {lr_ba:.4f}")
    print(f"MLP probe (sklearn MLPClassifier)   balanced accuracy: {mlp_ba:.4f}")

    wandb_logger.experiment.log({
        "linear_probe_holdout_accuracy": lr_ba,
        "mlp_holdout_accuracy": mlp_ba,
        "phase": "simple_probes_completed",
    })

    lr_conf_matrix = confusion_matrix(test_labels_extracted, lr_preds)
    mlp_conf_matrix = confusion_matrix(test_labels_extracted, mlp_preds)
    figure_lr = make_confmat(lr_conf_matrix, label_names, "LinearProbe")
    figure_mlp = make_confmat(mlp_conf_matrix, label_names, "MLP")
    wandb_logger.experiment.log({
        f"{dataset}_linear_probe_confusion_matrix_fold_{args.holdout_fold}": wandb.Image(figure_lr),
        f"{dataset}_mlp_confusion_matrix_fold_{args.holdout_fold}": wandb.Image(figure_mlp),
    })

    # Prepare animal IDs for saving with predictions
    test_animal_ids = [metadata_df.iloc[i]['specimen_id'] for i in test_indices[:len(test_embeddings)]]

    # Save predictions (mlp_predictions.csv filename kept stable for pick_locked_configs.py)
    pd.DataFrame({
        "pred": le.inverse_transform(mlp_preds.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
        "animal_id": test_animal_ids,
    }).to_csv(f"{results_dir}/mlp_predictions.csv", index=False)
    pd.DataFrame({
        "pred": le.inverse_transform(lr_preds.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
        "animal_id": test_animal_ids,
    }).to_csv(f"{results_dir}/linear_probe_predictions.csv", index=False)

    wandb_logger.experiment.log_artifact(
        f"{results_dir}/mlp_predictions.csv",
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-mlp_predictions.csv",
        type="mlp_predictions"
    )
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/linear_probe_predictions.csv",
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-linear_probe_predictions.csv",
        type="linear_probe_predictions"
    )

    # Create confusion matrix for best performance
    conf_matrix = confusion_matrix(test_labels_extracted, best_predictions)
    figure_holdout = make_confmat(conf_matrix, label_names, best_k)

    # Log confusion matrix
    wandb_logger.experiment.log({
        f"{dataset}_holdout_confusion_matrix_fold_{args.holdout_fold}": wandb.Image(figure_holdout),
    })


    # Save embeddings and results
    results_dir = f"../results_hippie_holdout/{dataset}/fold_{args.holdout_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}"

    # Save train embeddings
    train_df = pd.DataFrame(train_embeddings)
    train_df['label'] = le.inverse_transform(train_labels_extracted.astype(int))
    # Ensure we have the right number of animal IDs by using only the first len(train_embeddings) indices
    train_animal_ids = [metadata_df.iloc[i]['specimen_id'] for i in train_indices[:len(train_embeddings)]]
    train_df['animal_id'] = train_animal_ids
    train_df.to_csv(f"{results_dir}/train_embeddings.csv", index=False)

    # Log train embeddings as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/train_embeddings.csv",
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-train_embeddings.csv",
        type="train_embeddings"
    )

    # Save test embeddings
    test_df = pd.DataFrame(test_embeddings)
    test_df['label'] = le.inverse_transform(test_labels_extracted.astype(int))
    test_df['animal_id'] = test_animal_ids
    test_df.to_csv(f"{results_dir}/test_embeddings.csv", index=False)

    # Log test embeddings as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/test_embeddings.csv",
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-test_embeddings.csv",
        type="test_embeddings"
    )

    # Save predictions and true labels for best K
    predictions_df = pd.DataFrame({
        "pred": le.inverse_transform(best_predictions.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
        "animal_id": test_animal_ids  # Reuse the same animal_ids we calculated above
    })
    predictions_df.to_csv(f"{results_dir}/holdout_predictions.csv", index=False)


    # Log predictions as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/holdout_predictions.csv",
        name=f"{dataset}-fold_{args.holdout_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-holdout_predictions.csv",
        type="predictions"
    )

    # Save animal split info
    split_info = {
        'train_animals': train_animals.tolist(),
        'test_animals': test_animals.tolist(),
        'fold': args.holdout_fold,
        'n_folds': args.n_holdout_folds
    }


    
    import json
    with open(f"{results_dir}/animal_split.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # Log hyperparameters
    wandb_logger.experiment.config.update(vars(args))

    # Phase marker for completion
    wandb_logger.experiment.log({"phase": "holdout_completed"})

    wandb_logger.experiment.finish()
    print("Holdout evaluation completed!")


if __name__ == "__main__":
    main()