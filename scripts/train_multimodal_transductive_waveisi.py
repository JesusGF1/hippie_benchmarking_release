"""
Transductive Training Script for HIPPIE - WAVEFORM + ISI ONLY

This script trains HIPPIE using only two modalities:
  - Waveform (wave): Spike waveforms
  - ISI Distribution (isi): Interspike interval histograms

The ACG (autocorrelogram) modality is excluded from this version.

Usage:
    python train_multimodal_transductive_waveisi.py \\
        --config full_model \\
        --dataset allen_scope_neuropixel_area_subset \\
        --cv_fold 0 \\
        --use_balanced_sampling
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
import compute_parity as _cp  # unified per-phase timing -> results/compute_parity/timings.csv

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
from sklearn.model_selection import StratifiedKFold
from multimodal_model import CVAEConfig, ExperimentConfigs
from torch.utils.data import TensorDataset

# -------------------------------
# Helper function to find dataset files
# -------------------------------
def find_dataset_file(dataset, filename):
    """Find dataset file in multiple possible locations.

    Search order: env-var override, local checkout next to the script, then a
    few container-image fallbacks for the case where datasets are mounted
    under /data or /src.
    """
    # Allow an explicit override via env var.
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

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found {filename} at: {path}")
            return path

    raise FileNotFoundError(f"Could not find {filename} in any of these locations: {possible_paths}")

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
            try:
                if torch.cuda.is_available():
                    for d in range(torch.cuda.device_count()):
                        device = torch.device(f"cuda:{d}")
                        curr = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        metrics[f"{self.ns}/gpu{d}_mem_alloc_mb"] = curr
                        metrics[f"{self.ns}/gpu{d}_mem_reserved_mb"] = reserved
                        metrics[f"{self.ns}/gpu{d}_mem_peak_mb"] = peak
            except RuntimeError:
                # CUDA not initialized yet, skip GPU metrics
                pass

            if metrics:
                wandb.log(metrics, step=global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log peaks at epoch boundary, then reset peaks.
        try:
            if torch.cuda.is_available():
                metrics = {}
                for d in range(torch.cuda.device_count()):
                    peak = torch.cuda.max_memory_allocated(d) / (1024 ** 2)
                    metrics[f"{self.ns}/val_gpu{d}_mem_peak_mb"] = peak
                if metrics:
                    wandb.log(metrics, step=trainer.global_step)
        except RuntimeError:
            # CUDA not initialized yet, skip
            pass
        self._reset_cuda_peaks()

    def _reset_cuda_peaks(self):
        try:
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)
        except RuntimeError:
            # CUDA not initialized yet, skip
            pass


def _normalize_label(lbl):
    """Normalize a raw label into a plain str, undoing common encoding artifacts.

    Cases handled:
      - Python `bytes` -> decoded str
      - Python `str` that LOOKS like a `repr(bytes)`: e.g. ``b''`` (3 bytes:
        b, ', ') or ``b'PkC_ss'`` -> the inner string. This is the dominant
        case in the hausser unified-layer labels.csv: whoever wrote the file
        called ``str(some_bytestring)`` and got ``b'...'`` written to disk
        verbatim, so pandas reads back the literal characters ``b''`` rather
        than an empty bytestring.
      - NaN / None -> the empty string
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
    # Undo str(bytes) if present: matches b'...' and b"..."
    if len(s) >= 3 and s[0] == "b" and s[-1] in ("'", '"') and s[1] == s[-1]:
        s = s[2:-1]
    return s


def _is_unlabeled(lbl) -> bool:
    """Return True if a row's label should be dropped from supervised training.

    Catches:
      - NaN / None
      - Empty strings, empty bytestrings, AND empty `str(bytes)` artifacts
        (e.g. the literal 3-character string ``b''`` that hausser's
        unified-layer labels.csv contains for 3770/3996 rows)
      - Labels whose .lower() contains 'unlabeled', 'unknown', 'juxt', 'axo',
        or 'vgat' (catches both ``b'unlabeled'`` from the macaque dataset
        and the juxtacellular/axo/vgat filters used elsewhere).
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

    This creates a systematic mislabeling where:
    - Class 0 becomes Class 1
    - Class 1 becomes Class 2
    - ...
    - Class N-1 becomes Class 0 (wrap around)

    The test labels stay the same, so the confusion matrix should show
    a diagonal shifted by `shift` positions.
    """
    # Get unique classes and create mapping
    unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
    num_classes = len(unique_classes)

    # Create shift mapping: old_label -> new_label
    # shift=1 means class 0->1, 1->2, ..., N-1->0
    shift_mapping = {old: unique_classes[(i + shift) % num_classes]
                     for i, old in enumerate(unique_classes)}

    # Apply shift only to training labels
    shifted_train = np.array([shift_mapping[label] for label in train_labels])

    # Test labels remain unchanged
    return shifted_train, test_labels


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


def get_embeddings_multimodal(loader, model, mask_class_labels=False):
    """Get embeddings from multimodal model.

    Args:
        loader: DataLoader for the dataset
        model: The CVAE model
        mask_class_labels: If True, use zero embeddings instead of true class labels
                          to prevent data leakage during test set evaluation
    """
    model.eval()
    try:
        if torch.cuda.is_available():
            model = model.cuda()
    except RuntimeError:
        pass  # CUDA not initialized, use CPU

    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            data_dict, batch_labels = batch
            try:
                if torch.cuda.is_available():
                    data_dict = {k: v.cuda() for k, v in data_dict.items()}
                    batch_labels = batch_labels.cuda()
            except RuntimeError:
                pass  # CUDA not initialized, use CPU

            # Extract class labels, source labels, and optional super_region labels
            if batch_labels.dim() > 1 and batch_labels.shape[1] > 1:
                class_labels = batch_labels[:, 0]  # First column is class labels
                source_labels = batch_labels[:, 1]  # Second column is source labels

                # Third column is super_region labels if available
                super_region_labels = batch_labels[:, 2] if batch_labels.shape[1] > 2 else None

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
            # super_region labels are kept unconditionally (brain region is
            # observable at test time and does not leak the cell-type label)
            class_labels_for_encoding = None if mask_class_labels else class_labels

            # Get latent embeddings with proper labels
            h, mu, log_var = model.encode(data_dict, source_labels=source_labels,
                                          class_labels=class_labels_for_encoding,
                                          super_region_labels=super_region_labels)
            embeddings.append(mu.cpu().numpy())
            labels.append(class_labels.cpu().numpy())

    return np.vstack(embeddings), np.hstack(labels)


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
    # Initialize CUDA early to avoid runtime errors
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception as e:
        print(f"Warning: CUDA initialization failed: {e}")
        print("Continuing without GPU acceleration...")

    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument("--z_dim", type=int, default=20, help="Dimension of latent space (10/modality × 2 modalities)")
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0, help="Weight for KL divergence loss (locked default from Hausser sweep)")
    parser.add_argument('--dataset', type=str, default="allen_scope_neuropixel_area_subset")
    parser.add_argument('--wandb-tag', type=str, default="Hippie_transductive")
    parser.add_argument('--project', type=str, default="HIPPIE")
    parser.add_argument('--pretrain-max-epochs', type=int, default=100)
    parser.add_argument('--finetune-max-epochs', type=int, default=10)
    parser.add_argument('--supervised-max-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--supervised-batch-size', type=int, default=128)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--train-val-split', type=float, default=0.8)

    # New arguments for multimodal approach (WAVEFORM + ISI ONLY)
    parser.add_argument('--wave-weight', type=float, default=1.0, help='Weight for the waveform modality loss')
    parser.add_argument('--isi-weight', type=float, default=1.0, help='Weight for the ISI modality loss')

    # Pretraining pool control. Mirrors the trimodal script: empty default =
    # target-only single-dataset pretraining. "all" = every dataset in
    # all_dataset_files. Comma-separated list = whitelist
    # with the --dataset target auto-added.
    parser.add_argument('--pretrain-datasets', type=str, default="",
                        help="Pretraining pool control. Empty (default) = "
                             "target-only single-dataset pretraining. 'all' = "
                             "every dataset in all_dataset_files. "
                             "Comma-separated list = whitelist with the "
                             "--dataset target auto-added.")

    parser.add_argument('--config', type=str, default="class_decoder_source_bn_aug_reg",
                       choices=["baseline", "with_source", "with_class", "with_both_embeddings",
                               "with_batch_norm", "no_augmentations", "no_fusion",
                               "with_light_augmentations", "with_heavy_augmentations",
                               "full_architecture", "conditional_decoder_only",
                               "full_architecture_heavy_reg", "class_decoder_source",
                               "class_decoder_source_bn", "class_decoder_source_bn_strong_aug",
                               "class_decoder_source_bn_aug_reg"])

    # Cross validation arguments
    parser.add_argument("--n_cv_folds", type=int, default=5, help="Number of cross validation folds")
    parser.add_argument("--cv_fold", type=int, default=0, help="Which CV fold to run (0 to n_cv_folds-1)")
    parser.add_argument("--shuffle_labels", action="store_true", help="Randomly shuffle labels for control experiments")
    parser.add_argument("--shift_labels", action="store_true", help="Shift training labels by 1 position (class 0->1, 1->2, etc.) for control experiments")

    # Class balancing argument.
    # NOTE: This flag exists as an opt-in but was NOT used in the published runs.
    # The cached predictions in results/ were produced without --use_balanced_sampling
    # (i.e., default uniform random sampling). Kept for users who want to experiment.
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use class-balanced sampling via WeightedRandomSampler during supervised training. "
                             "Off by default; was NOT used in the published runs.")

    # Region conditioning flag
    parser.add_argument("--use-region-conditioning", action="store_true",
                        help="Enable layer/region conditioning via super_regions.csv or layer_labels.csv")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (seeds python/numpy/torch/CUDA and DataLoader workers via pl.seed_everything).")
    args = parser.parse_args()

    # wandb is opt-in: default to disabled unless WANDB_MODE is set explicitly.
    os.environ.setdefault("WANDB_MODE", "disabled")
    if os.environ["WANDB_MODE"].lower() not in ("disabled", "dryrun"):
        wandb.login()

    # Initialize wandb logger
    shuffle_suffix = "_shuffled" if args.shuffle_labels else ("_shifted" if args.shift_labels else "")
    wandb_logger = pl.loggers.WandbLogger(
        project=args.project,
        name=f"{args.wandb_tag}-{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}",
        tags=[args.wandb_tag, f"transductive_cv_fold_{args.cv_fold}"],
    )
    # Initialize run via logger and log phase before training
    _ = wandb_logger.experiment
    wandb_logger.experiment.log({"phase": "transductive_start"})

    print(f"Running transductive CV fold {args.cv_fold}/{args.n_cv_folds}")

    # Seed python random, numpy, torch (CPU+CUDA), and DataLoader workers
    pl.seed_everything(args.seed, workers=True)

    # -------------------------------
    # Load target dataset
    # -------------------------------
    # Load the actual data (WAVEFORM + ISI ONLY)
    supervised_wf = pd.read_csv(find_dataset_file(args.dataset, "waveforms.csv")).to_numpy()
    supervised_isi = pd.read_csv(find_dataset_file(args.dataset, "isi_dist.csv")).to_numpy()
    # ACG not used in this version

    # Load labels
    try:
        labels_path = find_dataset_file(args.dataset, "labels.csv")
        labels = pd.read_csv(labels_path)
        supervised_labels = labels[labels.columns[0]].values

        # Remove samples with unlabeled/invalid labels BEFORE encoding
        # _is_unlabeled() handles: NaN, empty strings, b'' artifacts (hausser),
        # and labels containing "unlabeled", "unknown", "juxt", "axo", "vgat"
        indices_to_keep = [i for i, lbl in enumerate(supervised_labels)
                           if not _is_unlabeled(lbl)]
        _unlabeled_keep = np.array(indices_to_keep, dtype=np.intp)

        supervised_wf = supervised_wf[indices_to_keep]
        supervised_isi = supervised_isi[indices_to_keep]
        supervised_labels = supervised_labels[indices_to_keep]

        # Normalize labels to clean strings before encoding
        supervised_labels = np.array([_normalize_label(lbl) for lbl in supervised_labels])

        # Now encode the filtered labels
        le = LabelEncoder().fit(supervised_labels)
        supervised_labels = le.transform(supervised_labels)

        # Remove classes with fewer than MIN_CELLS_PER_CLASS samples (after encoding).
        # min_cells=10 per the "Dataset Filtering Rules" section of the task doc —
        # statistical reliability floor shared across HIPPIE / PhysMAP / NEMO.
        MIN_CELLS_PER_CLASS = 10
        unique, counts = np.unique(supervised_labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        labels_to_remove = [label for label, count in label_counts.items() if count < MIN_CELLS_PER_CLASS]

        indices_to_keep = [i for i, label in enumerate(supervised_labels) if label not in labels_to_remove]
        _rare_keep = np.array(indices_to_keep, dtype=np.intp)
        supervised_wf = supervised_wf[indices_to_keep]
        supervised_isi = supervised_isi[indices_to_keep]
        supervised_labels = supervised_labels[indices_to_keep]

        # Re-fit encoder after removing small classes to ensure sequential labels
        remaining_label_strings = le.inverse_transform(supervised_labels)
        le = LabelEncoder().fit(remaining_label_strings)
        supervised_labels = le.transform(remaining_label_strings)

    except FileNotFoundError:
        print(f"No labels.csv found for {args.dataset}")
        supervised_labels = np.zeros(len(supervised_wf))
        le = LabelEncoder().fit(supervised_labels)
        _unlabeled_keep = np.arange(len(supervised_labels), dtype=np.intp)
        _rare_keep = np.arange(len(supervised_labels), dtype=np.intp)

    # Load super_regions if available AND --use-region-conditioning was requested.
    # Applies the same two-stage row filter (_unlabeled_keep, then _rare_keep)
    # that was applied to labels/waveforms above so all arrays stay aligned.
    supervised_super_regions = None
    super_region_le = None
    num_super_regions = None
    if args.use_region_conditioning:
        # Try super_regions.csv first (Allen brain region hierarchy),
        # then layer_labels.csv (cerebellar/cortical layer for hull/lisberger/juxtacellular).
        conditioning_file = None
        conditioning_fname = None
        for fname in ("super_regions.csv", "layer_labels.csv"):
            try:
                conditioning_file = find_dataset_file(args.dataset, fname)
                conditioning_fname = fname
                print(f"[region-conditioning] Found {fname}")
                break
            except FileNotFoundError:
                continue
        if conditioning_file is None:
            print(f"[region-conditioning] No conditioning file found for {args.dataset} — disabling")
            _super_region_conditioning_enabled = False
        else:
            try:
                super_regions_df = pd.read_csv(conditioning_file)
                _raw_super_regions = super_regions_df[super_regions_df.columns[0]].values
                # Apply the same filters used for labels (unlabeled rows, then rare-class rows)
                _raw_super_regions = _raw_super_regions[_unlabeled_keep]
                _raw_super_regions = _raw_super_regions[_rare_keep]
                super_region_le = LabelEncoder().fit(_raw_super_regions)
                supervised_super_regions = super_region_le.transform(_raw_super_regions)
                num_super_regions = len(np.unique(supervised_super_regions))
                _super_region_conditioning_enabled = True
                print(f"Loaded {conditioning_fname}: {num_super_regions} regions: {super_region_le.classes_}")
            except Exception as e:
                print(f"[region-conditioning] Error loading {conditioning_fname}: {e} — disabling")
                _super_region_conditioning_enabled = False
    else:
        _super_region_conditioning_enabled = False

    num_class_labels = len(np.unique(supervised_labels))
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {len(supervised_wf)}")
    print(f"Number of classes: {num_class_labels}")
    
    print(f"Samples after removing rare classes: {len(supervised_wf)}")
    num_class_labels = len(np.unique(supervised_labels))
    print(f"Classes after removing rare classes: {np.unique(supervised_labels)}")
    print(f"Number of classes after removing rare classes: {num_class_labels}")
    print(f"Waveform shape: {supervised_wf.shape}")
    print(f"ISI shape: {supervised_isi.shape}")
    print(f"Labels shape: {supervised_labels.shape}")
    print(f"Label distribution: {np.unique(supervised_labels, return_counts=True)}")
    # Sanity check
    assert len(supervised_wf) == len(supervised_isi) == len(supervised_labels), "Data and labels must have the same number of samples"


    # -------------------------------
    # Cross-validation split
    # -------------------------------
    # Create stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=42)
    splits = list(skf.split(supervised_wf, supervised_labels))
    train_indices, test_indices = splits[args.cv_fold]

    # Apply the CV split
    wf_train = supervised_wf[train_indices]
    wf_test = supervised_wf[test_indices]
    isi_train = supervised_isi[train_indices]
    isi_test = supervised_isi[test_indices]
    label_train = supervised_labels[train_indices]
    label_test = supervised_labels[test_indices]

    print(f"Train set: {len(wf_train)} cells")
    print(f"Test set: {len(wf_test)} cells")
    print(f"Train classes: {np.unique(label_train)}")
    print(f"Test classes: {np.unique(label_test)}")

    # Split super_regions if available (same indices as labels)
    if supervised_super_regions is not None:
        super_region_train = supervised_super_regions[train_indices]
        super_region_test = supervised_super_regions[test_indices]
    else:
        super_region_train = None
        super_region_test = None

    # Apply label shuffling if requested (after the CV split)
    if args.shuffle_labels:
        print("Shuffling train and test labels independently for control experiment...")
        label_train, label_test = shuffle_labels_independently(label_train, label_test, seed=42)
        print(f"Labels shuffled! Train classes: {np.unique(label_train)}, Test classes: {np.unique(label_test)}")

    # Apply label shifting if requested (after the CV split)
    if args.shift_labels:
        print("Shifting training labels by 1 position for control experiment...")
        print(f"Original train label distribution: {dict(zip(*np.unique(label_train, return_counts=True)))}")
        label_train, label_test = shift_labels_train_only(label_train, label_test, shift=1)
        print(f"Shifted train label distribution: {dict(zip(*np.unique(label_train, return_counts=True)))}")
        print("Test labels remain unchanged - expect shifted diagonal in confusion matrix")

    # Get config
    config = getattr(ExperimentConfigs, args.config)()

    # Enable super_region conditioning in config if the file was found
    if _super_region_conditioning_enabled:
        config.use_super_region_embedding = True

    # Define modalities and weights (WAVEFORM + ISI ONLY)
    modalities = {
        "wave": 50,
        "isi": 100
    }
    modality_weights = {
        "wave": args.wave_weight,
        "isi": args.isi_weight
    }

    # -------------------------------
    # PRETRAINING PHASE (multi-dataset: load ALL datasets for pretraining)
    # -------------------------------

    # Create dataset mappings for sources
    all_dataset_files = {
        "cellexplorer_area": 2,
        "cellexplorer_cell_type": 2,
        "hausser_cell_type": 3,
        "hull_cell_type": 4,
        "lisberger_labeled_cell_type": 5,
        "mouse_slice_area": 1,
        "juxtacellular_mouse_s1_area": 6,
        "allen_scope_neuropixel_area_subset": 7,
        "a1data_remove_undef": 8,
    }

    num_sources = max(all_dataset_files.values()) + 1

    # Resolve which datasets to include in the pretraining pool. Mirrors the
    # trimodal script's --pretrain-datasets logic: empty default = target-only.
    pretrain_arg = args.pretrain_datasets.strip() if args.pretrain_datasets else ""
    if pretrain_arg == "":
        if args.dataset not in all_dataset_files:
            raise ValueError(
                f"--dataset '{args.dataset}' is not in all_dataset_files and "
                f"--pretrain-datasets is empty (default = target-only). "
                f"Options: (1) use one of "
                f"{sorted(all_dataset_files.keys())}; (2) pass --pretrain-datasets all; "
                f"or (3) pass an explicit comma-separated whitelist."
            )
        pretrain_pool = {args.dataset: all_dataset_files[args.dataset]}
    elif pretrain_arg.lower() == "all":
        pretrain_pool = dict(all_dataset_files)
    else:
        requested = [d.strip() for d in pretrain_arg.split(",") if d.strip()]
        unknown = [d for d in requested if d not in all_dataset_files]
        if unknown:
            raise ValueError(
                f"--pretrain-datasets contains unknown dataset(s): {unknown}. "
                f"Allowed: {sorted(all_dataset_files.keys())}, or 'all'."
            )
        # Always include the target so the encoder sees it during pretraining.
        if args.dataset in all_dataset_files and args.dataset not in requested:
            requested.append(args.dataset)
        pretrain_pool = {d: all_dataset_files[d] for d in requested}

    # Load datasets for pretraining (multi-dataset learning)
    print("\n" + "="*60)
    print("Loading datasets for multi-dataset pretraining...")
    print(f"Pretrain pool: {list(pretrain_pool.keys())}")
    print("="*60)

    datasets_pretrain = []
    all_source_labels = []

    for dataset_name, source_id in pretrain_pool.items():
        try:
            # Load dataset (WAVEFORM + ISI ONLY)
            dataset_wf = pd.read_csv(find_dataset_file(dataset_name, "waveforms.csv")).to_numpy()
            dataset_isi = pd.read_csv(find_dataset_file(dataset_name, "isi_dist.csv")).to_numpy()

            # Filter out unlabeled/rare samples if labels exist
            try:
                labels_path = find_dataset_file(dataset_name, "labels.csv")
                dataset_labels = pd.read_csv(labels_path)[pd.read_csv(labels_path).columns[0]].values

                # Remove unlabeled/unwanted samples
                indices_to_keep = [i for i, lbl in enumerate(dataset_labels)
                                if not any(x in str(lbl).lower() for x in ["unlabeled", "juxt", "axo", "vgat"])]

                dataset_wf = dataset_wf[indices_to_keep]
                dataset_isi = dataset_isi[indices_to_keep]
            except FileNotFoundError:
                pass  # No labels, use all data

            # Create source labels (zeros for class labels in pretraining)
            source_labels = np.full(len(dataset_wf), source_id)
            zero_labels = np.zeros(len(dataset_wf))

            # Create multimodal dataset for this source (WAVEFORM + ISI ONLY)
            data_dict = {
                "wave": dataset_wf,
                "isi": dataset_isi
            }

            dataset_multi = MultiModalEphysDataset(
                data_dict,
                np.vstack((zero_labels, source_labels)).T,  # No class labels for pretraining
                mode="multi"
            )
            datasets_pretrain.append(dataset_multi)
            all_source_labels.extend(source_labels.tolist())

            print(f"  ✓ {dataset_name}: {len(dataset_wf)} samples (source_id={source_id})")

        except Exception as e:
            print(f"  ✗ {dataset_name}: Failed to load ({e})")
            continue

    # Concatenate all datasets (each dataset handles its own dimensions)
    dataset_pretrain_multi = torch.utils.data.ConcatDataset(datasets_pretrain)
    all_source_labels = np.array(all_source_labels)

    print(f"\nTotal pretraining samples: {len(all_source_labels)}")
    print(f"Source distribution: {np.unique(all_source_labels, return_counts=True)}")
    print("="*60 + "\n")

    # Create augmented dataset if needed
    if config.use_augmentations and config.augment_pretraining:
        dataset_pretrain_multi = AugmentedMultiModalEphysDataset(dataset_pretrain_multi, config, phase="pretraining")

    # Split for train/val during pretraining
    pretrain_train_size = int(args.train_val_split * len(dataset_pretrain_multi))
    pretrain_val_size = len(dataset_pretrain_multi) - pretrain_train_size
    pretrain_train_dataset, pretrain_val_dataset = random_split(
        dataset_pretrain_multi, [pretrain_train_size, pretrain_val_size]
    )

    pretrain_train_loader = torch.utils.data.DataLoader(
        pretrain_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    pretrain_val_loader = torch.utils.data.DataLoader(
        pretrain_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize model
    joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_super_regions,
        config=config,
    )

    joint_model = MultiModalCVAETrainModule(
        joint_model,
        modality_weights=modality_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        config=config,
    )

    # Set up trainer with checkpointing for pretraining
    pretrain_callbacks = [ResourceMonitor()]
    pretrain_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    pretrain_callbacks.append(pretrain_checkpoint)

    if args.early_stopping_patience > 0:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min"
        )
        pretrain_callbacks.append(early_stop_callback)

    # Phase marker for pretraining start
    wandb_logger.experiment.log({"phase": "pretrain_start"})

    pretrain_trainer = pl.Trainer(
        max_epochs=args.pretrain_max_epochs,
        callbacks=pretrain_callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        precision=("16-mixed" if torch.cuda.is_available() else "32-true"),  # Mixed precision for faster training
    )

    print("Starting pretraining...")
    pretrain_trainer.fit(joint_model, pretrain_train_loader, pretrain_val_loader)

    # Load best pretrained model
    joint_path = pretrain_checkpoint.best_model_path
    joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    print("Pretraining completed!")

    # -------------------------------
    # FINE-TUNING WITHOUT LABELS PHASE (on target dataset)
    # -------------------------------
    print("\n" + "="*60)
    print("Starting fine-tuning without labels on target dataset...")
    print("="*60)

    # Load target dataset for fine-tuning (WAVEFORM + ISI ONLY)
    finetune_wf = supervised_wf  # Use the full target dataset
    finetune_isi = supervised_isi

    # Create source labels (no class labels for this phase)
    finetune_source_labels = all_dataset_files[args.dataset] * np.ones(len(finetune_wf))
    finetune_zero_labels = np.zeros(len(finetune_wf))

    finetune_data_dict = {
        "wave": finetune_wf,
        "isi": finetune_isi
    }

    # Include super_region labels in finetune dataset if available
    if supervised_super_regions is not None:
        finetune_labels_stacked = np.vstack(
            (finetune_zero_labels, finetune_source_labels, supervised_super_regions)
        ).T
    else:
        finetune_labels_stacked = np.vstack(
            (finetune_zero_labels, finetune_source_labels)
        ).T

    finetune_dataset_multi = MultiModalEphysDataset(
        finetune_data_dict,
        finetune_labels_stacked,
        mode="multi"
    )

    # Create augmented dataset if needed
    if config.use_augmentations and config.augment_finetuning:
        finetune_dataset_multi = AugmentedMultiModalEphysDataset(finetune_dataset_multi, config, phase="finetuning")

    # Split for train/val during fine-tuning
    finetune_train_size = int(args.train_val_split * len(finetune_dataset_multi))
    finetune_val_size = len(finetune_dataset_multi) - finetune_train_size
    finetune_train_dataset, finetune_val_dataset = random_split(
        finetune_dataset_multi, [finetune_train_size, finetune_val_size]
    )

    finetune_train_loader = torch.utils.data.DataLoader(
        finetune_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    finetune_val_loader = torch.utils.data.DataLoader(
        finetune_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Create new model instance for fine-tuning with lower learning rate
    finetune_joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_super_regions,
        config=config,
    )

    # Load pretrained weights
    finetune_joint_model = MultiModalCVAETrainModule(
        finetune_joint_model,
        modality_weights=modality_weights,
        learning_rate=(1/10)*args.learning_rate,  # Lower LR for fine-tuning
        weight_decay=args.weight_decay,
        config=config,
    )
    finetune_joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    # Set up fine-tuning trainer with checkpointing
    finetune_callbacks = [ResourceMonitor()]
    finetune_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    finetune_callbacks.append(finetune_checkpoint)

    if args.early_stopping_patience > 0:
        finetune_early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min"
        )
        finetune_callbacks.append(finetune_early_stop)

    # Phase marker for fine-tuning start
    wandb_logger.experiment.log({"phase": "finetune_without_labels_start"})

    finetune_trainer = pl.Trainer(
        max_epochs=args.finetune_max_epochs,
        callbacks=finetune_callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        precision=("16-mixed" if torch.cuda.is_available() else "32-true"),  # Mixed precision for faster training
    )

    print("Starting fine-tuning without labels...")
    finetune_trainer.fit(finetune_joint_model, finetune_train_loader, finetune_val_loader)

    # Load best fine-tuned model to use for supervised phase
    joint_path = finetune_checkpoint.best_model_path
    finetune_joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    print("Fine-tuning without labels completed!")

    # -------------------------------
    # SUPERVISED PHASE (only on training data/labels)
    # -------------------------------

    supervised_joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_super_regions,
        config=config,
    )

    # Load pretrained weights but skip the class embedding layer and, when
    # super_region conditioning is enabled, skip the layers whose input size
    # changed (fusion_encoder and decoder_fcs) to avoid shape mismatches.
    joint_seq = torch.load(joint_path)
    if "model.class_embedding.weight" in joint_seq["state_dict"]:
        joint_seq["state_dict"].pop("model.class_embedding.weight")

    if num_super_regions is not None and config.use_super_region_embedding:
        print("Detected super_region embedding — skipping fusion_encoder and "
              "decoder_fcs keys from finetuned checkpoint (shape mismatch expected).")
        keys_to_remove = [k for k in joint_seq["state_dict"]
                          if k.startswith("model.super_region_embedding.") or
                             k.startswith("model.fusion_encoder.") or
                             k.startswith("model.decoder_fcs.")]
        for key in keys_to_remove:
            joint_seq["state_dict"].pop(key)
        print(f"  Removed {len(keys_to_remove)} incompatible keys; will re-initialise from scratch.")

    supervised_joint_model = MultiModalCVAETrainModule(
        supervised_joint_model,
        modality_weights=modality_weights,
        learning_rate=(1/10)*args.learning_rate,
        weight_decay=args.weight_decay,
        config=config,
    )
    supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)

    # Supervised dataset construction.
    # TRANSDUCTIVE: training loader uses train + (masked) test cells; the
    # VALIDATION loader used for early-stopping is built from training cells
    # only — the test fold is never used for supervised model selection.
    from sklearn.model_selection import train_test_split as _sk_train_test_split

    _source_id = all_dataset_files[args.dataset]
    train_source_labels = _source_id * np.ones(len(train_indices))
    test_source_labels = _source_id * np.ones(len(test_indices))
    _has_sr = super_region_train is not None and super_region_test is not None

    n_train_cells = len(label_train)
    val_size = max(1, n_train_cells - int(args.train_val_split * n_train_cells))
    try:
        sup_tr_idx, sup_val_idx = _sk_train_test_split(
            np.arange(n_train_cells),
            test_size=val_size,
            stratify=label_train,
            random_state=args.seed if hasattr(args, "seed") else 42,
        )
    except ValueError:
        sup_tr_idx, sup_val_idx = _sk_train_test_split(
            np.arange(n_train_cells),
            test_size=val_size,
            random_state=args.seed if hasattr(args, "seed") else 42,
        )

    # TRAIN dataset: labeled-train subset + masked test cells.
    sup_train_data = {
        "wave": np.vstack([wf_train[sup_tr_idx], wf_test]),
        "isi": np.vstack([isi_train[sup_tr_idx], isi_test]),
    }
    sup_train_class_labels = np.concatenate([
        label_train[sup_tr_idx],
        np.zeros_like(label_test),
    ])
    sup_train_source_labels = np.concatenate([
        train_source_labels[sup_tr_idx],
        test_source_labels,
    ])
    if _has_sr:
        sup_train_super_regions = np.concatenate([
            super_region_train[sup_tr_idx],
            super_region_test,
        ])
        sup_train_labels_stacked = np.vstack(
            (sup_train_class_labels, sup_train_source_labels, sup_train_super_regions)
        ).T
    else:
        sup_train_labels_stacked = np.vstack(
            (sup_train_class_labels, sup_train_source_labels)
        ).T
    supervised_train_dataset = MultiModalEphysDataset(
        sup_train_data, sup_train_labels_stacked, mode="multi"
    )

    # VAL dataset: training-fold subset ONLY (no test cells).
    sup_val_data = {
        "wave": wf_train[sup_val_idx],
        "isi": isi_train[sup_val_idx],
    }
    sup_val_class_labels = label_train[sup_val_idx]
    sup_val_source_labels = train_source_labels[sup_val_idx]
    if _has_sr:
        sup_val_super_regions = super_region_train[sup_val_idx]
        sup_val_labels_stacked = np.vstack(
            (sup_val_class_labels, sup_val_source_labels, sup_val_super_regions)
        ).T
    else:
        sup_val_labels_stacked = np.vstack(
            (sup_val_class_labels, sup_val_source_labels)
        ).T
    supervised_val_dataset = MultiModalEphysDataset(
        sup_val_data, sup_val_labels_stacked, mode="multi"
    )

    # Alias for backward compatibility with downstream code.
    all_labels_supervised = sup_train_class_labels

    # Create DataLoader with optional class-balanced sampling
    if args.use_balanced_sampling:
        train_labels_subset = sup_train_class_labels
        train_sampler = create_balanced_sampler(supervised_train_dataset, train_labels_subset)

        supervised_train_loader = torch.utils.data.DataLoader(
            supervised_train_dataset,
            batch_size=args.supervised_batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=none_safe_collate,
            num_workers=2,
        pin_memory=True,
        persistent_workers=True
        )
    else:
        supervised_train_loader = torch.utils.data.DataLoader(
            supervised_train_dataset,
            batch_size=args.supervised_batch_size,
            shuffle=True,
            collate_fn=none_safe_collate,
            num_workers=2,
        pin_memory=True,
        persistent_workers=True
        )

    supervised_val_loader = torch.utils.data.DataLoader(
        supervised_val_dataset,
        batch_size=args.supervised_batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Phase marker for supervised start
    wandb_logger.experiment.log({"phase": "supervised_start"})

    # Set up supervised trainer with checkpointing
    supervised_callbacks = [ResourceMonitor()]
    supervised_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    supervised_callbacks.append(supervised_checkpoint)

    if args.early_stopping_patience > 0:
        supervised_early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min"
        )
        supervised_callbacks.append(supervised_early_stop)

    # Train supervised model
    supervised_trainer = pl.Trainer(
        max_epochs=args.supervised_max_epochs,
        callbacks=supervised_callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        precision=("16-mixed" if torch.cuda.is_available() else "32-true"),  # Mixed precision for faster training
    )

    print("Starting supervised training...")
    supervised_trainer.fit(supervised_joint_model, supervised_train_loader, supervised_val_loader)

    # Load best supervised model
    supervised_path = supervised_checkpoint.best_model_path
    supervised_joint_model.load_state_dict(torch.load(supervised_path)["state_dict"])

    print("Supervised training completed!")

    # -------------------------------
    # EVALUATION ON HELD-OUT TEST SET
    # -------------------------------

    # Create test dataset
    test_data_dict = {
        "wave": wf_test,
        "isi": isi_test
    }

    source_labels_test = all_dataset_files[args.dataset] * np.ones_like(label_test)

    # Include super_region labels in test/train-only datasets if available
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
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Create loader for TRAIN data only for embedding extraction (for K-NN training)
    train_data_only_dict = {
        "wave": wf_train,
        "isi": isi_train
    }
    source_labels_train_only = all_dataset_files[args.dataset] * np.ones_like(label_train)
    if super_region_train is not None:
        train_only_labels_stacked = np.vstack(
            (label_train, source_labels_train_only, super_region_train)
        ).T
    else:
        train_only_labels_stacked = np.vstack(
            (label_train, source_labels_train_only)
        ).T
    dataset_train_only = MultiModalEphysDataset(
        train_data_only_dict,
        train_only_labels_stacked,
        mode="multi"
    )

    full_train_loader = torch.utils.data.DataLoader(
        dataset_train_only,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    # Get embeddings for both train and test
    print("Extracting embeddings...")
    # Train embeddings: use true labels (no masking)
    train_embeddings, train_labels_extracted = get_embeddings_multimodal(
        full_train_loader,
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
    results_dir = f"../results_hippie_transductive/{args.dataset}/fold_{args.cv_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}"
    os.makedirs(results_dir, exist_ok=True)

    # K-NN classification — k selected by stratified inner CV on the
    # TRAINING fold only, then evaluated once on the held-out test fold.
    # See hippie/utils.py:select_knn_k_by_train_cv.
    from hippie.utils import select_knn_k_by_train_cv
    print("Performing K-NN classification...")
    knn_result = select_knn_k_by_train_cv(
        train_embeddings,
        train_labels_extracted,
        test_embeddings,
        test_labels_extracted,
        k_grid=[1, 3, 5, 10],
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
            "cv_fold": args.cv_fold,
            "n_train_cells": len(train_indices),
            "n_test_cells": len(test_indices),
        })
        print(f"Inner train-CV balanced accuracy (k={k}): {cv_score:.4f}")

    print(f"Best K-NN performance: k={best_k}, accuracy={best_accuracy:.4f}")

    # Log best accuracy as summary metric
    wandb_logger.experiment.log({
        "best_transductive_knn_accuracy": best_accuracy,
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
        "linear_probe_transductive_accuracy": lr_ba,
        "mlp_transductive_accuracy": mlp_ba,
        "phase": "simple_probes_completed",
    })

    lr_conf_matrix = confusion_matrix(test_labels_extracted, lr_preds)
    mlp_conf_matrix = confusion_matrix(test_labels_extracted, mlp_preds)
    figure_lr = make_confmat(lr_conf_matrix, label_names, "LinearProbe")
    figure_mlp = make_confmat(mlp_conf_matrix, label_names, "MLP")
    wandb_logger.experiment.log({
        f"{args.dataset}_linear_probe_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_lr),
        f"{args.dataset}_mlp_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_mlp),
    })

    # Save predictions (mlp_predictions.csv filename kept stable for pick_locked_configs.py)
    pd.DataFrame({
        "pred": le.inverse_transform(mlp_preds.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
    }).to_csv(f"{results_dir}/mlp_predictions.csv", index=False)
    pd.DataFrame({
        "pred": le.inverse_transform(lr_preds.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
    }).to_csv(f"{results_dir}/linear_probe_predictions.csv", index=False)

    wandb_logger.experiment.log_artifact(
        f"{results_dir}/mlp_predictions.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-mlp_predictions.csv",
        type="mlp_predictions"
    )
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/linear_probe_predictions.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-linear_probe_predictions.csv",
        type="linear_probe_predictions"
    )

    # Create confusion matrix for best performance
    conf_matrix = confusion_matrix(test_labels_extracted, best_predictions)
    figure_transductive = make_confmat(conf_matrix, label_names, best_k)

    # Log confusion matrix
    wandb_logger.experiment.log({
        f"{args.dataset}_transductive_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_transductive),
    })

    # Save embeddings and results
    # Save train embeddings
    train_df = pd.DataFrame(train_embeddings)
    train_df['label'] = le.inverse_transform(train_labels_extracted.astype(int))
    train_df.to_csv(f"{results_dir}/train_embeddings.csv", index=False)

    # Log train embeddings as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/train_embeddings.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-train_embeddings.csv",
        type="train_embeddings"
    )

    # Save test embeddings
    test_df = pd.DataFrame(test_embeddings)
    test_df['label'] = le.inverse_transform(test_labels_extracted.astype(int))
    test_df.to_csv(f"{results_dir}/test_embeddings.csv", index=False)

    # Log test embeddings as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/test_embeddings.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-test_embeddings.csv",
        type="test_embeddings"
    )

    # Save test waveforms with predictions and true labels
    wf_df = pd.DataFrame(wf_test)
    wf_df['prediction'] = le.inverse_transform(best_predictions.astype(int))
    wf_df['true_label'] = le.inverse_transform(test_labels_extracted.astype(int))
    wf_df.to_csv(f"{results_dir}/test_waveforms.csv", index=False)

    # Log test waveforms as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/test_waveforms.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-test_waveforms.csv",
        type="test_waveforms"
    )

    # Save test ISI distributions with predictions and true labels
    isi_df = pd.DataFrame(isi_test)
    isi_df['prediction'] = le.inverse_transform(best_predictions.astype(int))
    isi_df['true_label'] = le.inverse_transform(test_labels_extracted.astype(int))
    isi_df.to_csv(f"{results_dir}/test_isi.csv", index=False)

    # Log test ISI as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/test_isi.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-test_isi.csv",
        type="test_isi"
    )

    # Save predictions and true labels for best K
    predictions_df = pd.DataFrame({
        "pred": le.inverse_transform(best_predictions.astype(int)),
        "true": le.inverse_transform(test_labels_extracted.astype(int)),
    })
    predictions_df.to_csv(f"{results_dir}/transductive_predictions.csv", index=False)

    # Log predictions as wandb artifact
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/transductive_predictions.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-transductive_predictions.csv",
        type="predictions"
    )

    # Save CV split info
    split_info = {
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'fold': args.cv_fold,
        'n_folds': args.n_cv_folds
    }

    import json
    with open(f"{results_dir}/cv_split.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # Log hyperparameters
    wandb_logger.experiment.config.update(vars(args))

    # Phase marker for completion
    wandb_logger.experiment.log({"phase": "transductive_completed"})

    wandb_logger.experiment.finish()
    print("Transductive evaluation completed!")


if __name__ == "__main__":
    main()