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
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from torch.utils.data import random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from multimodal_model import CVAEConfig, ExperimentConfigs

# -------------------------------
# Helper function to find dataset files
# -------------------------------
def find_dataset_file(dataset, filename):
    """Find dataset file in multiple possible locations."""
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
        f"../datasets_hippie/{dataset}/{filename}",
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
    raise FileNotFoundError(f"Could not find {filename} for dataset {dataset} in any of these locations: {possible_paths}")

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
                metrics[f"{self.ns}/val_gpu{d}_mem_peak_mb"] = peak
            if metrics:
                wandb.log(metrics, step=trainer.global_step)
        self._reset_cuda_peaks()

    def _reset_cuda_peaks(self):
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)


def _load_checkpoint_from_path(path: str) -> dict:
    """Load a Lightning checkpoint from a local path or s3:// URI."""
    if path.startswith("s3://"):
        import tempfile
        import boto3
        without_scheme = path[5:]
        bucket_name, _, key = without_scheme.partition("/")
        s3_endpoint = os.environ.get("S3_ENDPOINT", "https://s3.example.com")
        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            tmp_path = tmp.name
        s3.download_file(bucket_name, key, tmp_path)
        ckpt = torch.load(tmp_path, map_location="cpu")
        os.unlink(tmp_path)
        return ckpt
    else:
        return torch.load(path, map_location="cpu")


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


def _log_timer(timer_obj: Timer, prefix: str):
    """Helper to robustly pull timings from Lightning Timer across versions."""
    def _safe_elapsed(phase):
        try:
            val = timer_obj.time_elapsed(phase)
            return float(val) if val is not None else None
        except Exception:
            return None

    elapsed_fit = _safe_elapsed("fit")
    elapsed_train = _safe_elapsed("train")
    elapsed_validate = _safe_elapsed("validate")

    payload = {}
    if elapsed_fit is not None:
        payload[f"time/{prefix}_fit_s"] = elapsed_fit
    if elapsed_train is not None:
        payload[f"time/{prefix}_train_s"] = elapsed_train
    if elapsed_validate is not None:
        payload[f"time/{prefix}_val_s"] = elapsed_validate

    if payload:
        wandb.log(payload)


# ---------- FIX PART 1: Robust embedding extraction ----------
def get_embeddings_multimodal(loader, model, mask_class_labels=False):
    """Extract embeddings from a multimodal model.

    Args:
        loader: DataLoader for the dataset
        model: The CVAE model (can be TrainModule wrapper or raw model)
        mask_class_labels: If True, use None instead of true class labels during encoding
                          to prevent data leakage at test time. Source labels are still passed.

    Returns:
        embeddings: numpy array of shape (N, z_dim)
        labels: numpy array of class labels
    """
    model.eval()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Get the underlying model if wrapped in TrainModule
    underlying_model = model.model if hasattr(model, 'model') else model

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for sample in loader:
            data_dict, labels = sample

            # Move data to GPU if available
            if torch.cuda.is_available():
                data_dict = {k: v.cuda() for k, v in data_dict.items()}
                labels = labels.cuda()

            # Extract labels from batch (format: [class_label, source_id] or
            # [class_label, source_id, super_region_label] when region conditioning is on)
            if labels.ndim > 1 and labels.shape[1] > 1:
                class_labels = labels[:, 0]  # First column: class labels
                source_labels = labels[:, 1]  # Second column: source labels
                # Third column is super_region labels if available
                super_region_labels = labels[:, 2] if labels.shape[1] > 2 else None
            else:
                class_labels = labels
                source_labels = None
                super_region_labels = None

            # Mask class labels if requested (prevent leakage during test evaluation).
            # super_region labels are kept unconditionally (brain region is
            # realistic inference-time metadata, not a label to be predicted).
            class_labels_for_encoding = None if mask_class_labels else class_labels

            # Get embeddings (mu from latent space)
            h, mu, log_var = underlying_model.encode(
                data_dict,
                source_labels=source_labels,
                class_labels=class_labels_for_encoding,
                super_region_labels=super_region_labels,
            )

            embedding = mu.detach().cpu().numpy()
            all_embeddings.extend(embedding)

            all_labels.extend(class_labels.detach().cpu().numpy())

    return np.array(all_embeddings), np.array(all_labels)


# ---------- FIX PART 2: NaN sanitize ----------
def _nan_sanitize(x: np.ndarray, name: str, dataset: str):
    if np.isnan(x).any():
        print(f"NaN values detected in dataset '{dataset}', modality '{name}'. Replacing with 0.")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _normalize_label(lbl):
    """Mirror of train_multimodal_transductive._normalize_label.

    Undoes both real bytestrings and str(bytes) artifacts (e.g. the literal
    3-char string ``b''`` from hausser unified-layer labels.csv). Returns a
    plain str (or "" for missing/empty).
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


def load_dataset_data(dataset_name, dataset_files):
    """Load waveform, ISI, and ACG data for a dataset."""
    wf = pd.read_csv(find_dataset_file(dataset_name, "waveforms.csv")).to_numpy()
    isi = pd.read_csv(find_dataset_file(dataset_name, "isi_dist.csv")).to_numpy()

    # Load ACG data if it exists, otherwise use zeros
    try:
        acg_path = find_dataset_file(dataset_name, "acg.csv")
        acg = pd.read_csv(acg_path).to_numpy()
    except FileNotFoundError:
        acg = np.zeros_like(isi)

    # Sanitize NaNs
    wf = _nan_sanitize(wf, "wave", dataset_name)
    isi = _nan_sanitize(isi, "isi", dataset_name)
    acg = _nan_sanitize(acg, "acg", dataset_name)

    # Load labels if they exist
    labels = None
    try:
        labels_path = find_dataset_file(dataset_name, "labels.csv")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df[labels_df.columns[0]].values
    except FileNotFoundError:
        try:
            labels_path = find_dataset_file(dataset_name, "celltypes.csv")
            labels_df = pd.read_csv(labels_path)
            labels = labels_df[labels_df.columns[0]].values
        except FileNotFoundError:
            pass  # No labels file found

    # Drop unlabeled rows BEFORE returning. This applies to BOTH the training
    # and the predict dataset because cross_dataset_script calls load_dataset
    # for each. Without this, hausser b'' rows leak through and the encoder
    # creates a phantom "no label" class. See _is_unlabeled docstring.
    if labels is not None and len(labels) > 0:
        keep_mask = np.array([not _is_unlabeled(lbl) for lbl in labels])
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            print(f"[label filter] dropping {n_dropped}/{len(labels)} unlabeled "
                  f"rows from {dataset_name}")
            wf = wf[keep_mask]
            isi = isi[keep_mask]
            acg = acg[keep_mask]
            labels = labels[keep_mask]
        # Normalize bytestrings AND str(bytes) artifacts -> plain str
        labels = np.array([_normalize_label(lbl) for lbl in labels])

        # Drop classes with fewer than MIN_CELLS_PER_CLASS samples. Per the task
        # doc "Dataset Filtering Rules" — mandatory across all benchmarks.
        MIN_CELLS_PER_CLASS = 10
        uniq, counts = np.unique(labels, return_counts=True)
        rare = set(uniq[counts < MIN_CELLS_PER_CLASS].tolist())
        if rare:
            keep_rare = np.array([lbl not in rare for lbl in labels])
            n_rare = int((~keep_rare).sum())
            print(f"[min_cells filter] dropping {n_rare} cells from {len(rare)} "
                  f"rare classes (< {MIN_CELLS_PER_CLASS}) in {dataset_name}: {sorted(rare)}")
            wf = wf[keep_rare]
            isi = isi[keep_rare]
            acg = acg[keep_rare]
            labels = labels[keep_rare]

    source_id = dataset_files[dataset_name]

    print(f"Dataset {dataset_name} has shapes - waveform: {wf.shape}, isi: {isi.shape}, acg: {acg.shape}")

    return wf, isi, acg, labels, source_id


# ---------- FIX PART 3: Map predict labels to training encoder space ----------
def map_labels_to_training_encoder(le_train: LabelEncoder, labels: np.ndarray, fallback: int = 0):
    """Map string labels to indices in le_train; unseen labels map to `fallback` (default 0)."""
    train_set = set(le_train.classes_)
    out = np.empty(labels.shape[0], dtype=int)
    for i, lbl in enumerate(labels):
        if lbl in train_set:
            out[i] = le_train.transform([lbl])[0]
        else:
            out[i] = fallback
    return out


def shuffle_labels_independently(train_labels, test_labels, seed=42):
    """Randomly shuffle train and test labels independently.

    This is used for control experiments to establish a null distribution
    (what accuracy would be expected by chance).

    Args:
        train_labels: numpy array of training labels
        test_labels: numpy array of test labels
        seed: random seed for reproducibility

    Returns:
        shuffled_train: shuffled training labels
        shuffled_test: shuffled test labels
    """
    np.random.seed(seed)
    shuffled_train = train_labels.copy()
    np.random.shuffle(shuffled_train)
    np.random.seed(seed + 1)  # Different seed for test to ensure independence
    shuffled_test = test_labels.copy()
    np.random.shuffle(shuffled_test)
    return shuffled_train, shuffled_test


def shift_labels_train_only(train_labels, test_labels, shift=1):
    """Shift training labels by N positions (wrapping around), keep test labels unchanged.

    With shift=1: class 0 -> 1, 1 -> 2, ..., N-1 -> 0. The shift is computed over the
    union of train+test classes so it is well-defined for cross-dataset settings where
    train and predict label sets may differ.
    """
    unique_classes = np.unique(np.concatenate([np.asarray(train_labels), np.asarray(test_labels)]))
    num_classes = len(unique_classes)
    shift_mapping = {old: unique_classes[(i + shift) % num_classes]
                     for i, old in enumerate(unique_classes)}
    shifted_train = np.array([shift_mapping[l] for l in train_labels])
    return shifted_train, np.asarray(test_labels).copy()


if __name__ == '__main__':
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument("--z_dim", type=int, default=None,
                        help="Total latent dimension. If unset, computed as "
                             "z_dim_per_modality * num_modalities (e.g., 30 for trimodal, 20 for bimodal).")
    parser.add_argument("--z_dim_per_modality", type=int, default=10,
                        help="Latent dimensions per input modality. Locked default 10 (z=30 trimodal) "
                             "selected by Hausser hyperparameter sweep (42 configs; frozen before "
                             "any other benchmark dataset was evaluated).")
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Weight for KL divergence loss. Locked default 1.0 selected by "
                             "Hausser hyperparameter sweep (42 configs; frozen before any other "
                             "benchmark dataset was evaluated).")
    parser.add_argument('--training-dataset', type=str, required=True, help="Dataset to train on")
    parser.add_argument('--predict-dataset', type=str, required=True, help="Dataset to predict on")
    parser.add_argument('--pretrain-checkpoint', type=str, default=None,
                        help="Path to a pretrained checkpoint (.ckpt) to load "
                             "INSTEAD of running pretraining. Skips Phase 1 "
                             "entirely, starts at finetuning. Supports local "
                             "paths and s3:// URIs.")
    parser.add_argument('--upload-model', action='store_true')
    parser.add_argument('--wandb-tag', type=str, default="Hippie_cross_dataset")
    parser.add_argument('--project', type=str, default="HIPPIE")
    parser.add_argument('--finetune-without-labels', type=bool, default=True)
    # Production defaults (locked by user 2026-04-08, see
    # memory/project_scaling_benchmark.md): 100 / 20 / 10. Was 100/10/5.
    parser.add_argument('--keep-all-in-pretrain', action='store_true', default=False,
                        help="Include training and predict datasets in the pretraining corpus "
                             "(unsupervised). Useful for testing whether multi-species pretraining "
                             "improves cross-species label transfer.")
    parser.add_argument('--keep-train-in-pretrain', action='store_true', default=False,
                        help="Include the training dataset in the pretraining corpus in addition "
                             "to the standard background datasets. When combined with "
                             "--keep-all-in-pretrain, all three datasets (hausser, hull, lisberger) "
                             "are used for unsupervised pretraining.")
    parser.add_argument('--pretrain-max-epochs', type=int, default=100)
    parser.add_argument('--finetune-max-epochs', type=int, default=20)
    parser.add_argument('--supervised-max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--supervised-batch-size', type=int, default=64)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--train-val-split', type=float, default=0.9)
    parser.add_argument('--finetune-split', type=float, default=0.1)
    parser.add_argument('--limit-train-batches', type=float, default=None)
    parser.add_argument('--limit-val-batches', type=float, default=None)

    # New arguments for multimodal approach
    parser.add_argument('--model-type', type=str, choices=['unimodal', 'multimodal'], default='multimodal',
                        help='Whether to use separate models for each modality or a joint model')
    parser.add_argument('--mod1-weight', type=float, default=1.0,
                        help='Weight for the waveform modality loss in multimodal model')
    parser.add_argument('--mod2-weight', type=float, default=1.0,
                        help='Weight for the ISI modality loss in multimodal model')

    parser.add_argument('--wave-weight', type=float, default=1.0,
                        help='Weight for the waveform modality loss')
    parser.add_argument('--isi-weight', type=float, default=1.0,
                        help='Weight for the ISI modality loss')
    parser.add_argument('--acg-weight', type=float, default=1.0,
                        help='Weight for the ACG modality loss')

    parser.add_argument('--config', type=str, default="class_decoder_source_bn_aug_reg",
                        choices=["baseline", "with_source", "with_class", "with_both_embeddings",
                                 "with_batch_norm", "no_augmentations", "no_fusion",
                                 "with_light_augmentations", "with_heavy_augmentations",
                                 "full_architecture", "conditional_decoder_only",
                                 "full_architecture_heavy_reg", "class_decoder_source",
                                 "class_decoder_source_bn", "class_decoder_source_bn_strong_aug",
                                 "class_decoder_source_bn_aug_reg"])

    # Class balancing argument.
    # NOTE: This flag exists as an opt-in but was NOT used in the published runs.
    # The cached predictions in results/ were produced without --use_balanced_sampling
    # (i.e., default uniform random sampling). Kept for users who want to experiment.
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use class-balanced sampling via WeightedRandomSampler during supervised training. "
                             "Off by default; was NOT used in the published runs.")

    # Region conditioning flag
    parser.add_argument("--use-region-conditioning", action="store_true",
                        help="Enable layer/region conditioning from layer_labels.csv or super_regions.csv. "
                             "When set, loads cerebellar/cortical layer labels for both train and predict "
                             "datasets and passes them as a third conditioning input alongside source and class.")
    parser.add_argument("--decoder-only-region-cond", action="store_true",
                        help="When --use-region-conditioning is set, restrict region labels to the decoder only. "
                             "The encoder never sees region labels, so KNN embeddings are region-free. "
                             "Prevents region→cell-type shortcuts in cross-species transfer.")

    # DataLoader workers
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes for data loading (default: 4)")

    # Label shuffling for control experiments
    parser.add_argument("--shuffle_labels", action="store_true",
                        help="Shuffle labels for control experiment (null distribution)")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Random seed for label shuffling")
    parser.add_argument("--shift_labels", action="store_true",
                        help="Shift training labels by 1 position (class 0->1, 1->2, ...) "
                             "for control experiments. Predict labels stay unchanged.")
    parser.add_argument("--exclude-classes", type=str, default="",
                        help="Comma-separated list of class labels to drop from BOTH "
                             "training and predict datasets before any processing "
                             "(e.g. 'GoC' to remove Golgi cells from Lisberger).")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip Phase 1 (pretraining) and Phase 2 (finetuning) entirely. "
                             "Phase 3 supervised training starts from random init. "
                             "Use this to establish a no-pretraining baseline.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (seeds python/numpy/torch/CUDA and DataLoader workers via pl.seed_everything).")
    args = parser.parse_args()

    # Resolve total z_dim from per-modality default unless an explicit override was passed.
    # cross_dataset_script always uses the 3 modalities {wave, isi, acg}, so we can resolve
    # z_dim immediately after parsing (before it's used in run_name / output_dir paths).
    # Defaults (z_dim_per_modality=5, beta=0.9) come from systematic architectural ablation
    # on cellexplorer_cell_type + lisberger_labeled_cell_type only.
    if args.z_dim is None:
        _NUM_MODALITIES = 3
        args.z_dim = args.z_dim_per_modality * _NUM_MODALITIES
        print(f"[z_dim auto] z_dim_per_modality={args.z_dim_per_modality} * "
              f"num_modalities={_NUM_MODALITIES} -> z_dim={args.z_dim}")

    # -------------------------------
    # Common setup
    # -------------------------------
    accelerator = "gpu"
    limit_train_batches = args.limit_train_batches
    limit_val_batches = args.limit_val_batches
    project = args.project
    FINETUNE_WITHOUT_LABELS = args.finetune_without_labels
    trainer_kwargs = {}

    pl.seed_everything(args.seed, workers=True)

    # Build run name with optional shuffle/shift suffix
    shuffle_suffix = "_shuffled" if args.shuffle_labels else ("_shifted" if args.shift_labels else "")
    run_name = f"{args.wandb_tag}-train_{args.training_dataset}-predict_{args.predict_dataset}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}"

    # wandb is opt-in: default to disabled unless WANDB_MODE is set explicitly.
    os.environ.setdefault("WANDB_MODE", "disabled")
    # Initialize single wandb run at start (no-op when WANDB_MODE=disabled)
    wandb_config = vars(args).copy()
    wandb_config["is_shuffled_control"] = args.shuffle_labels
    wandb_config["is_shifted_control"] = args.shift_labels
    if os.environ["WANDB_MODE"].lower() not in ("disabled", "dryrun"):
        wandb.login()
    wandb.init(
        project=project,
        name=run_name,
        config=wandb_config
    )

    # Log shuffle status
    if args.shuffle_labels:
        print("=" * 60)
        print("CONTROL EXPERIMENT: LABELS WILL BE SHUFFLED")
        print(f"Shuffle seed: {args.shuffle_seed}")
        print("Expected accuracy: ~chance level (1/num_classes)")
        print("=" * 60)
    if args.shift_labels:
        print("=" * 60)
        print("CONTROL EXPERIMENT: TRAINING LABELS WILL BE SHIFTED BY 1")
        print("Predict labels remain unchanged - expect shifted diagonal in confusion matrix")
        print("=" * 60)

    # Define output directory path (with shuffle suffix if applicable)
    output_dir = f"../results_hippie_cross_dataset/train_{args.training_dataset}_predict_{args.predict_dataset}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}"

    # -------------------------------
    # Dataset setup
    # -------------------------------
    dataset_files = {
        "hausser_cell_type": 1,
        "hull_cell_type": 2,
        "lisberger_labeled_cell_type": 3,
    }

    all_dataset_files = dataset_files.copy()
    num_sources = max(all_dataset_files.values()) + 1

    # Validate dataset arguments
    if args.training_dataset not in dataset_files:
        raise ValueError(f"Training dataset '{args.training_dataset}' not found in available datasets: {list(dataset_files.keys())}")
    if args.predict_dataset not in dataset_files:
        raise ValueError(f"Predict dataset '{args.predict_dataset}' not found in available datasets: {list(dataset_files.keys())}")

    print(f"Training on: {args.training_dataset}")
    print(f"Predicting on: {args.predict_dataset}")
    
    # Remove training and predict datasets from pretraining by default.
    # --keep-all-in-pretrain: add predict dataset to pretraining corpus
    # --keep-train-in-pretrain: add training dataset to pretraining corpus
    # Both together: pretrain on all three (hausser + hull + lisberger).
    pretrain_dataset_files = dataset_files.copy()
    if not args.keep_train_in_pretrain:
        if args.training_dataset in pretrain_dataset_files:
            pretrain_dataset_files.pop(args.training_dataset)
    if not args.keep_all_in_pretrain:
        if args.predict_dataset in pretrain_dataset_files:
            pretrain_dataset_files.pop(args.predict_dataset)
    
    # Remove juxtacellular datasets if they're related to training/predict datasets
    if "juxtacellular" in args.training_dataset or "juxtacellular" in args.predict_dataset:
        pretrain_dataset_files.pop("juxtacellular_mouse_s1_area", None)
        pretrain_dataset_files.pop("juxtacellular_mouse_s1_cell_type", None)
    
    if "cellexplorer" in args.training_dataset or "cellexplorer" in args.predict_dataset:
        pretrain_dataset_files.pop("cellexplorer_cell_type", None)
        pretrain_dataset_files.pop("cellexplorer_area", None)
    
    print(f"Pretraining on: {list(pretrain_dataset_files.keys())}")

    # -------------------------------
    # PHASE 1: PRETRAINING
    # -------------------------------
    print("=" * 50)
    print("PHASE 1: PRETRAINING")
    print("=" * 50)

    time_phase1_start = time.time()

    # Define standard modality sizes - the MultiModalEphysDataset will handle padding/truncation
    modalities = {
        "wave": 50,   # Standard waveform size
        "isi": 100,   # Standard ISI size
        "acg": 200    # Standard ACG size
    }
    assert args.z_dim is not None, "args.z_dim should have been resolved right after argparse"

    if args.no_pretrain:
        print("[--no-pretrain] Skipping Phase 1 and Phase 2. Supervised training will start from random init.")
        joint_model = None
        joint_path = None
        EXPERIMENT_CONFIGS = {
            "baseline": ExperimentConfigs.baseline(),
            "with_source": ExperimentConfigs.with_source(),
            "with_class": ExperimentConfigs.with_class(),
            "with_both_embeddings": ExperimentConfigs.with_both_embeddings(),
            "with_batch_norm": ExperimentConfigs.with_batch_norm(),
            "no_augmentations": ExperimentConfigs.no_augmentations(),
            "no_fusion": ExperimentConfigs.no_fusion(),
            "with_light_augmentations": ExperimentConfigs.with_light_augmentations(),
            "with_heavy_augmentations": ExperimentConfigs.with_heavy_augmentations(),
            "full_architecture": ExperimentConfigs.full_architecture(),
            "conditional_decoder_only": ExperimentConfigs.conditional_decoder_only(),
            "full_architecture_heavy_reg": ExperimentConfigs.full_architecture_heavy_reg(),
            "class_decoder_source": ExperimentConfigs.class_decoder_source(),
            "class_decoder_source_bn": ExperimentConfigs.class_decoder_source_bn(),
            "class_decoder_source_bn_strong_aug": ExperimentConfigs.class_decoder_source_bn_strong_aug(),
            "class_decoder_source_bn_aug_reg": ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        }
        config = EXPERIMENT_CONFIGS[args.config]
        config.beta = args.beta
        modality_weights = {
            "wave": args.wave_weight,
            "isi": args.isi_weight,
            "acg": args.acg_weight,
        }

    elif args.pretrain_checkpoint:
        # ---- Shared-checkpoint path: skip pretraining entirely ----
        print(f"\n[--pretrain-checkpoint] Loading pretrained weights from: {args.pretrain_checkpoint}")
        _ckpt_data = _load_checkpoint_from_path(args.pretrain_checkpoint)
        _sd = _ckpt_data["state_dict"]

        # Infer num_sources and modality sizes from checkpoint to avoid mismatches.
        # The checkpoint was trained with train_multimodal_transductive.py which uses
        # ACG=100 (not 200 as cross_dataset_script defaults).
        _ckpt_num_sources = _sd["model.source_embedding.weight"].shape[0]
        # Override num_sources for the entire script so the supervised model
        # matches the finetuned model's source_embedding dimension.
        num_sources = _ckpt_num_sources
        print(f"[--pretrain-checkpoint] Checkpoint num_sources={_ckpt_num_sources} (overriding script default)")

        # Infer ACG output size from decoder weight shape
        if "model.decoders.acg.linear_out.weight" in _sd:
            _ckpt_acg_size = _sd["model.decoders.acg.linear_out.weight"].shape[0]
            modalities["acg"] = _ckpt_acg_size
            print(f"[--pretrain-checkpoint] Overriding ACG size to {_ckpt_acg_size} (from checkpoint)")

        # Get config (needed for model construction)
        EXPERIMENT_CONFIGS = {
            "baseline": ExperimentConfigs.baseline(),
            "with_source": ExperimentConfigs.with_source(),
            "with_class": ExperimentConfigs.with_class(),
            "with_both_embeddings": ExperimentConfigs.with_both_embeddings(),
            "with_batch_norm": ExperimentConfigs.with_batch_norm(),
            "no_augmentations": ExperimentConfigs.no_augmentations(),
            "no_fusion": ExperimentConfigs.no_fusion(),
            "with_light_augmentations": ExperimentConfigs.with_light_augmentations(),
            "with_heavy_augmentations": ExperimentConfigs.with_heavy_augmentations(),
            "full_architecture": ExperimentConfigs.full_architecture(),
            "conditional_decoder_only": ExperimentConfigs.conditional_decoder_only(),
            "full_architecture_heavy_reg": ExperimentConfigs.full_architecture_heavy_reg(),
            "class_decoder_source": ExperimentConfigs.class_decoder_source(),
            "class_decoder_source_bn": ExperimentConfigs.class_decoder_source_bn(),
            "class_decoder_source_bn_strong_aug": ExperimentConfigs.class_decoder_source_bn_strong_aug(),
            "class_decoder_source_bn_aug_reg": ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        }
        config = EXPERIMENT_CONFIGS[args.config]
        config.beta = args.beta

        # Build model matching checkpoint dimensions
        joint_model = MultiModalCVAE(
            modalities=modalities,
            z_dim=args.z_dim,
            num_sources=_ckpt_num_sources,
            num_classes=5,
            config=config,
        )
        modality_weights = {
            "wave": args.wave_weight,
            "isi": args.isi_weight,
            "acg": args.acg_weight
        }
        joint_model = MultiModalCVAETrainModule(
            joint_model,
            modality_weights=modality_weights,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )

        # Pop keys that may have shape mismatches (num_classes, super_region)
        _popped = []
        for key in list(_sd.keys()):
            if "class_embedding" in key or "super_region_embedding" in key:
                _sd.pop(key)
                _popped.append(key)
        if _popped:
            print(f"[--pretrain-checkpoint] Popped mismatched keys: {_popped}")

        joint_model.load_state_dict(_sd, strict=False)

        # Save to temp file so Phase 2 can load it
        import tempfile as _tmpmod
        _tmp_ckpt = _tmpmod.NamedTemporaryFile(suffix=".ckpt", delete=False)
        torch.save({"state_dict": joint_model.state_dict()}, _tmp_ckpt.name)
        joint_path = _tmp_ckpt.name
        print(f"[--pretrain-checkpoint] Loaded. Temp checkpoint at {joint_path}. "
              "Proceeding to Phase 2 (finetuning).")

    else:
        # ---- Standard pretraining path ----
        # Get config
        EXPERIMENT_CONFIGS = {
            "baseline": ExperimentConfigs.baseline(),
            "with_source": ExperimentConfigs.with_source(),
            "with_class": ExperimentConfigs.with_class(),
            "with_both_embeddings": ExperimentConfigs.with_both_embeddings(),
            "with_batch_norm": ExperimentConfigs.with_batch_norm(),
            "no_augmentations": ExperimentConfigs.no_augmentations(),
            "no_fusion": ExperimentConfigs.no_fusion(),
            "with_light_augmentations": ExperimentConfigs.with_light_augmentations(),
            "with_heavy_augmentations": ExperimentConfigs.with_heavy_augmentations(),
            "full_architecture": ExperimentConfigs.full_architecture(),
            "conditional_decoder_only": ExperimentConfigs.conditional_decoder_only(),
            "full_architecture_heavy_reg": ExperimentConfigs.full_architecture_heavy_reg(),
            "class_decoder_source": ExperimentConfigs.class_decoder_source(),
            "class_decoder_source_bn": ExperimentConfigs.class_decoder_source_bn(),
            "class_decoder_source_bn_strong_aug": ExperimentConfigs.class_decoder_source_bn_strong_aug(),
            "class_decoder_source_bn_aug_reg": ExperimentConfigs.class_decoder_source_bn_aug_reg(),
        }
        config = EXPERIMENT_CONFIGS[args.config]
        # Apply --beta override so the CLI flag is not silently ignored
        config.beta = args.beta
        print(f"[CVAEConfig] beta={config.beta}")

        # Load data for pretraining from all datasets except training and predict
        all_waveforms = []
        all_isi = []
        all_acg = []
        labels = []
        datasets_multi = []

        for folder in pretrain_dataset_files:
            wf = pd.read_csv(find_dataset_file(folder, "waveforms.csv")).to_numpy()
            isi = pd.read_csv(find_dataset_file(folder, "isi_dist.csv")).to_numpy()
            # Load ACG data if it exists, otherwise use zeros
            try:
                acg_path = find_dataset_file(folder, "acg.csv")
                acg = pd.read_csv(acg_path).to_numpy()
            except FileNotFoundError:
                acg = np.zeros_like(isi)

            # Sanitize NaNs
            wf = _nan_sanitize(wf, "wave", folder)
            isi = _nan_sanitize(isi, "isi", folder)
            acg = _nan_sanitize(acg, "acg", folder)

            source = np.full((wf.shape[0]), pretrain_dataset_files[folder])
            print(f"Pretraining folder {folder} has shapes - waveform: {wf.shape}, isi: {isi.shape}, acg: {acg.shape}")

            all_waveforms.append(wf)
            all_isi.append(isi)
            all_acg.append(acg)
            labels.append(source)

            # Create multimodal dataset with all modalities
            data_dict = {
                "wave": wf,
                "isi": isi,
                "acg": acg
            }

            dataset_multi = MultiModalEphysDataset(data_dict, source, mode="multi", modality_sizes=modalities)

            # Wrap with augmentations if enabled
            if config.use_augmentations and config.augment_pretraining:
                dataset_multi = AugmentedMultiModalEphysDataset(dataset_multi, config, phase="pretraining")

            datasets_multi.append(dataset_multi)

        if not datasets_multi:
            print("Warning: No datasets available for pretraining. Skipping pretraining phase.")
            joint_model = None
            joint_path = None
        else:
            labels = np.concatenate(labels, axis=0)
            all_multi_dataset = torch.utils.data.ConcatDataset(datasets_multi)

            # Split datasets for pretraining
            prop = args.train_val_split
            indices = list(range(len(all_multi_dataset)))
            train_indices, test_indices = random_split(
                indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
            )

            # Create dataloaders for pretraining
            train_multi_dataset = torch.utils.data.Subset(all_multi_dataset, train_indices)
            test_multi_dataset = torch.utils.data.Subset(all_multi_dataset, test_indices)

            train_loader_multi = torch.utils.data.DataLoader(
                train_multi_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=none_safe_collate, num_workers=args.num_workers
            )
            test_loader_multi = torch.utils.data.DataLoader(
                test_multi_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=none_safe_collate, num_workers=args.num_workers
            )

            # Create multimodal model for pretraining
            joint_model = MultiModalCVAE(
                modalities=modalities,
                z_dim=args.z_dim,
                num_sources=num_sources,
                num_classes=5,  # Dummy value for pretraining
                config=config,
            )

            # Define modality weights
            modality_weights = {
                "wave": args.wave_weight,
                "isi": args.isi_weight,
                "acg": args.acg_weight
            }

            joint_model = MultiModalCVAETrainModule(
                joint_model,
                modality_weights=modality_weights,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                config=config,
            )

            # PRETRAIN: callbacks & trainer
            joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
            joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
            timer_pretrain = Timer(duration=None)
            resource_cb_pretrain = ResourceMonitor(log_every_n_steps=50)

            # Phase marker for pretrain start
            wandb.log({"phase": "pretrain_start"})

            joint_trainer = pl.Trainer(
                max_epochs=args.pretrain_max_epochs,
                accelerator=accelerator,
                logger=pl.loggers.WandbLogger(experiment=wandb.run),
                callbacks=[joint_checkpoint, joint_earlystop, timer_pretrain, resource_cb_pretrain],
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                gradient_clip_val=args.gradient_clip_val,
                **trainer_kwargs,
            )

            # Train joint model
            joint_trainer.fit(joint_model, train_loader_multi, test_loader_multi)
            # Log pretrain timing
            _log_timer(timer_pretrain, prefix="pretrain")

            joint_path = joint_checkpoint.best_model_path
            joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    time_phase1_end = time.time()
    time_phase1 = time_phase1_end - time_phase1_start
    print(f"\nPhase 1 (Pretraining) completed in {time_phase1:.2f} seconds ({time_phase1/60:.2f} minutes)")
    wandb.log({"time/phase1_pretrain_seconds": time_phase1, "time/phase1_pretrain_minutes": time_phase1/60})

    # -------------------------------
    # PHASE 2: FINETUNING
    # -------------------------------
    print("=" * 50)
    print("PHASE 2: FINETUNING")
    print("=" * 50)

    time_phase2_start = time.time()
    
    # Load training dataset for finetuning
    train_wf, train_isi, train_acg, train_labels, train_source_id = load_dataset_data(args.training_dataset, all_dataset_files)

    if train_labels is None:
        raise ValueError(f"Training dataset '{args.training_dataset}' must have labels for supervised training")

    # Drop explicitly excluded classes from training data (e.g. GoC for fair
    # cross-species comparison when one dataset has far more of a class than another).
    _excluded_classes = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    _keep_train = None  # will be set below if exclusions are applied
    if _excluded_classes:
        _keep_train = np.array([lbl not in _excluded_classes for lbl in train_labels])
        n_dropped = int((~_keep_train).sum())
        print(f"[exclude-classes] dropping {n_dropped} training rows for classes: {sorted(_excluded_classes)}")
        train_wf, train_isi, train_acg = train_wf[_keep_train], train_isi[_keep_train], train_acg[_keep_train]
        train_labels = train_labels[_keep_train]
        if isinstance(train_source_id, np.ndarray):
            train_source_id = train_source_id[_keep_train]

    # Load layer/region labels for train and predict datasets (if --use-region-conditioning).
    # CRITICAL: fit one LabelEncoder on the UNION of both datasets so they share encoding.
    # "unknown" is a valid class that gets its own integer index — neurons with unknown
    # layer are NOT discarded.
    train_layer_labels_raw = None
    predict_layer_labels_raw = None
    layer_le = None
    train_layer_labels = None
    predict_layer_labels = None
    num_layer_classes = None
    _region_conditioning_enabled = False

    if args.use_region_conditioning:
        def _load_layer_labels_aligned(dataset_name):
            """Load layer_labels.csv and apply the same row-filters used by load_dataset_data.

            Returns the filtered layer label array (strings), or None if no file is found.
            The function mirrors the two-stage filtering in load_dataset_data:
              1. Drop rows where cell-type label is "unlabeled" / empty / etc.
              2. Drop rows whose cell-type class has fewer than MIN_CELLS_PER_CLASS samples.
            This keeps layer_labels row-aligned with the waveform/ISI/ACG arrays.
            """
            # --- find and load the layer file ---
            layer_vals = None
            for fname in ("super_regions.csv", "layer_labels.csv"):
                try:
                    path = find_dataset_file(dataset_name, fname)
                    df = pd.read_csv(path)
                    layer_vals = df[df.columns[0]].values.astype(str)
                    print(f"[region-conditioning] Loaded {fname} for {dataset_name}: {len(layer_vals)} rows")
                    break
                except FileNotFoundError:
                    continue
            if layer_vals is None:
                return None

            # --- replicate the label filtering from load_dataset_data ---
            # Step 1: load the corresponding cell-type labels to get the keep_mask
            ct_labels = None
            for lname in ("labels.csv", "celltypes.csv"):
                try:
                    lpath = find_dataset_file(dataset_name, lname)
                    ct_labels = pd.read_csv(lpath)[pd.read_csv(lpath).columns[0]].values
                    break
                except FileNotFoundError:
                    continue
            if ct_labels is None or len(ct_labels) == 0:
                # No cell-type labels — no filtering needed
                return layer_vals

            # Step 1 filter: unlabeled rows
            keep_mask = np.array([not _is_unlabeled(lbl) for lbl in ct_labels])
            layer_vals = layer_vals[keep_mask]
            ct_labels = ct_labels[keep_mask]
            ct_labels = np.array([_normalize_label(lbl) for lbl in ct_labels])

            # Step 2 filter: rare classes (< MIN_CELLS_PER_CLASS)
            MIN_CELLS_PER_CLASS = 10
            uniq, counts = np.unique(ct_labels, return_counts=True)
            rare = set(uniq[counts < MIN_CELLS_PER_CLASS].tolist())
            if rare:
                keep_rare = np.array([lbl not in rare for lbl in ct_labels])
                layer_vals = layer_vals[keep_rare]

            return layer_vals

        train_layer_labels_raw = _load_layer_labels_aligned(args.training_dataset)
        predict_layer_labels_raw = _load_layer_labels_aligned(args.predict_dataset)

        if train_layer_labels_raw is not None:
            # Build union of all layer label strings for a single shared encoder.
            # "unknown" is intentionally kept — it gets its own embedding index.
            all_layer_raw = list(train_layer_labels_raw)
            if predict_layer_labels_raw is not None:
                all_layer_raw += list(predict_layer_labels_raw)
            layer_le = LabelEncoder().fit(all_layer_raw)
            train_layer_labels = layer_le.transform(train_layer_labels_raw)
            # Re-apply class exclusion mask so layer labels stay aligned with train data.
            if _keep_train is not None and len(train_layer_labels) == len(_keep_train):
                train_layer_labels = train_layer_labels[_keep_train]
            num_layer_classes = len(layer_le.classes_)
            _region_conditioning_enabled = True
            print(f"[region-conditioning] Layer encoder fitted on union of train+predict: "
                  f"{num_layer_classes} classes: {layer_le.classes_}")
            if predict_layer_labels_raw is not None:
                predict_layer_labels = layer_le.transform(predict_layer_labels_raw)
        else:
            print(f"[region-conditioning] No layer/region file found for {args.training_dataset} — disabling")

    # Enable super_region embedding in config when conditioning is active
    if _region_conditioning_enabled:
        config.use_super_region_embedding = True
        print(f"[region-conditioning] config.use_super_region_embedding = True")
        if args.decoder_only_region_cond:
            config.encoder_uses_region_embedding = False
            print(f"[region-conditioning] decoder-only mode: encoder will not see region labels")
    
    # Create finetuning dataset (without labels for unsupervised adaptation)
    finetune_data_dict = {
        "wave": train_wf,
        "isi": train_isi,
        "acg": train_acg
    }
    
    # Keep modalities consistent with pretraining - all datasets will be resized to these dimensions
    
    if joint_model is not None and FINETUNE_WITHOUT_LABELS:
        label_ft = np.full((train_wf.shape[0]), train_source_id)
        finetune_dataset_multi = MultiModalEphysDataset(finetune_data_dict, label_ft, mode="multi", modality_sizes=modalities)
        
        print(f"Finetuning modality sizes config: {modalities}")
        sample_data = finetune_dataset_multi[0][0]  # Get first sample
        for mod_name, tensor in sample_data.items():
            print(f"Finetune {mod_name} tensor shape: {tensor.shape}")
        print(f"Original training dataset shapes - waveform: {train_wf.shape}, isi: {train_isi.shape}, acg: {train_acg.shape}")
        
        # Split for finetuning
        prop = args.finetune_split
        indices = list(range(len(finetune_dataset_multi)))
        train_indices, test_indices = random_split(
            indices, [int(prop * len(indices)), len(indices) - int(prop * len(indices))]
        )
        
        # Create new model instance for fine-tuning with lower learning rate
        # Keep the original model architecture
        original_model = joint_model.model
        joint_model = MultiModalCVAETrainModule(
            original_model,
            modality_weights=modality_weights,
            learning_rate=(1/10)*args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )
        
        # Create dataloaders for fine-tuning
        # Apply augmentations only to training subset for fine-tuning
        if config.use_augmentations and config.augment_finetuning:
            augmented_finetune_dataset = AugmentedMultiModalEphysDataset(finetune_dataset_multi, config, phase="finetuning")
            train_finetune_dataset = torch.utils.data.Subset(augmented_finetune_dataset, train_indices)
        else:
            train_finetune_dataset = torch.utils.data.Subset(finetune_dataset_multi, train_indices)
        
        # Validation dataset is never augmented
        test_finetune_dataset = torch.utils.data.Subset(finetune_dataset_multi, test_indices)
        
        train_finetune_loader_multi = torch.utils.data.DataLoader(
            train_finetune_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        test_finetune_loader_multi = torch.utils.data.DataLoader(
            test_finetune_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=none_safe_collate, num_workers=args.num_workers
        )
        
        # FINE-TUNE: callbacks & trainer
        joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
        timer_finetune = Timer(duration=None)
        resource_cb_finetune = ResourceMonitor(log_every_n_steps=50)
        
        # Phase marker for finetune start
        wandb.log({"phase": "finetune_start"})
        
        joint_trainer = pl.Trainer(
            max_epochs=args.finetune_max_epochs,
            accelerator=accelerator,
            logger=pl.loggers.WandbLogger(experiment=wandb.run),
            callbacks=[joint_checkpoint, joint_earlystop, timer_finetune, resource_cb_finetune],
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            gradient_clip_val=args.gradient_clip_val,
            **trainer_kwargs,
        )
        
        joint_trainer.fit(joint_model, train_finetune_loader_multi, test_finetune_loader_multi)
        # Log finetune timing
        _log_timer(timer_finetune, prefix="finetune")
        
        joint_path = joint_checkpoint.best_model_path
        joint_model.load_state_dict(torch.load(joint_path)["state_dict"])

    time_phase2_end = time.time()
    time_phase2 = time_phase2_end - time_phase2_start
    print(f"\nPhase 2 (Finetuning) completed in {time_phase2:.2f} seconds ({time_phase2/60:.2f} minutes)")
    wandb.log({"time/phase2_finetune_seconds": time_phase2, "time/phase2_finetune_minutes": time_phase2/60})

    # -------------------------------
    # PHASE 3: SUPERVISED TRAINING
    # -------------------------------
    print("=" * 50)
    print("PHASE 3: SUPERVISED TRAINING")
    print("=" * 50)

    time_phase3_start = time.time()
    
    # Encode labels for supervised training
    le = LabelEncoder().fit(train_labels)
    train_labels_encoded = le.transform(train_labels)

    # Apply label shuffling if control experiment
    if args.shuffle_labels:
        print(f"Shuffling training labels with seed {args.shuffle_seed}...")
        np.random.seed(args.shuffle_seed)
        train_labels_shuffled = train_labels_encoded.copy()
        np.random.shuffle(train_labels_shuffled)
        train_labels_encoded = train_labels_shuffled
        wandb.log({"shuffle_applied_to_training": True})

    # Apply label shifting if control experiment (training labels only)
    if args.shift_labels:
        print("Shifting training labels by 1 position...")
        n_train_classes = int(train_labels_encoded.max()) + 1
        train_labels_encoded = (train_labels_encoded + 1) % n_train_classes
        wandb.log({"shift_applied_to_training": True})

    # Create train/val split for supervised training
    indices = list(range(len(train_wf)))
    train_size = int(args.train_val_split * len(indices))
    train_indices, val_indices = random_split(indices, [train_size, len(indices) - train_size])

    wf_train = train_wf[train_indices]
    wf_val = train_wf[val_indices]
    isi_train = train_isi[train_indices]
    isi_val = train_isi[val_indices]
    acg_train = train_acg[train_indices]
    acg_val = train_acg[val_indices]
    label_train = train_labels_encoded[train_indices]
    label_val = train_labels_encoded[val_indices]

    # Split layer labels with the same indices (when region conditioning is active)
    layer_label_train = train_layer_labels[train_indices] if train_layer_labels is not None else None
    layer_label_val = train_layer_labels[val_indices] if train_layer_labels is not None else None

    num_class_labels = len(np.unique(label_train))

    # Keep modalities consistent across all phases - already defined in pretraining

    # Create supervised model (pass num_super_regions when layer conditioning is on)
    supervised_joint_model = MultiModalCVAE(
        modalities=modalities,
        z_dim=args.z_dim,
        num_sources=num_sources,
        num_classes=num_class_labels,
        num_super_regions=num_layer_classes if _region_conditioning_enabled else None,
        config=config,
    )

    # Load pretrained weights if available, but skip the class embedding layer
    if joint_model is not None and joint_path is not None:
        joint_seq = torch.load(joint_path)
        if "model.class_embedding.weight" in joint_seq["state_dict"]:
            joint_seq["state_dict"].pop("model.class_embedding.weight")

        # When super_region conditioning is enabled the pretrained model (which was
        # trained without it) won't have matching super_region_embedding weights, and
        # the fusion_encoder / decoder_fcs first-layer input sizes will differ.
        #
        # Instead of popping those layers entirely (which leaves them random and
        # destroys learned representations), warm-start: zero-pad the old weights
        # for the extra region-embedding dimension. This preserves all learned
        # weights while allowing the region embedding to be learned from scratch.
        if _region_conditioning_enabled:
            _sd_old = joint_seq["state_dict"]
            # Build the wrapped model first so we can compare state_dict shapes.
            # This is the final supervised model — we load into it below.
            supervised_joint_model = MultiModalCVAETrainModule(
                supervised_joint_model, modality_weights=modality_weights,
                learning_rate=(1/10)*args.learning_rate,
                weight_decay=args.weight_decay, config=config)
            _sd_new = supervised_joint_model.state_dict()
            _warmstarted = []

            for key in list(_sd_old.keys()):
                if key.startswith("model.super_region_embedding."):
                    _sd_old.pop(key)  # pop — will be freshly initialized
                    _warmstarted.append(f"popped {key}")
                elif key in _sd_new and _sd_old[key].shape != _sd_new[key].shape:
                    old_t = _sd_old[key]
                    new_t = _sd_new[key]
                    # Pad input dimension (dim=-1 for weights, or dim=0 for biases)
                    if old_t.dim() == 2 and new_t.dim() == 2 and old_t.shape[0] == new_t.shape[0]:
                        # Weight matrix: [out, in_old] → [out, in_new], pad cols with 0
                        pad_size = new_t.shape[1] - old_t.shape[1]
                        if pad_size > 0:
                            padded = torch.cat([old_t, torch.zeros(old_t.shape[0], pad_size, device=old_t.device)], dim=1)
                            _sd_old[key] = padded
                            _warmstarted.append(f"padded {key}: {old_t.shape} → {padded.shape}")
                        else:
                            _sd_old.pop(key)
                            _warmstarted.append(f"popped {key} (shape shrink)")
                    else:
                        _sd_old.pop(key)
                        _warmstarted.append(f"popped {key} (shape mismatch)")

            if _warmstarted:
                print(f"[region-conditioning] Warm-started checkpoint: {len(_warmstarted)} keys adjusted:")
                for msg in _warmstarted:
                    print(f"  {msg}")

            # Model already wrapped above; load adjusted weights
            supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)
            print("Loaded pretrained weights for supervised training (warm-started region conditioning)")
        else:
            supervised_joint_model = MultiModalCVAETrainModule(
                supervised_joint_model,
                modality_weights=modality_weights,
                learning_rate=(1/10)*args.learning_rate,
                weight_decay=args.weight_decay,
                config=config,
            )
            supervised_joint_model.load_state_dict(joint_seq["state_dict"], strict=False)
            print("Loaded pretrained weights for supervised training")
    else:
        # No pretraining, start from scratch
        supervised_joint_model = MultiModalCVAETrainModule(
            supervised_joint_model,
            modality_weights=modality_weights,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            config=config,
        )
        print("No pretrained model available, starting supervised training from scratch")

    # Create source labels for embedding
    label_train_for_embedding = train_source_id * np.ones_like(label_train)
    label_val_for_embedding = train_source_id * np.ones_like(label_val)

    # Create supervised training datasets
    train_data_dict = {
        "wave": wf_train,
        "isi": isi_train,
        "acg": acg_train
    }

    val_data_dict = {
        "wave": wf_val,
        "isi": isi_val,
        "acg": acg_val
    }

    # Build label arrays: 2-column (class, source) normally; 3-column (class, source, layer)
    # when region conditioning is enabled.
    if _region_conditioning_enabled and layer_label_train is not None:
        train_labels_stacked = np.vstack((label_train, label_train_for_embedding, layer_label_train)).T
        val_labels_stacked = np.vstack((label_val, label_val_for_embedding, layer_label_val)).T
    else:
        train_labels_stacked = np.vstack((label_train, label_train_for_embedding)).T
        val_labels_stacked = np.vstack((label_val, label_val_for_embedding)).T

    dataset_train_multi = MultiModalEphysDataset(
        train_data_dict,
        train_labels_stacked,
        mode="multi",
        modality_sizes=modalities
    )

    # Apply augmentations to supervised training dataset if enabled
    if config.use_augmentations and config.augment_supervised:
        dataset_train_multi = AugmentedMultiModalEphysDataset(dataset_train_multi, config, phase="supervised")

    dataset_val_multi = MultiModalEphysDataset(
        val_data_dict,
        val_labels_stacked,
        mode="multi",
        modality_sizes=modalities
    )

    # Create DataLoader with optional class-balanced sampling
    if args.use_balanced_sampling:
        # Create balanced sampler for training data
        train_sampler = create_balanced_sampler(dataset_train_multi, label_train)
        train_loader_multi = torch.utils.data.DataLoader(
            dataset_train_multi,
            batch_size=args.supervised_batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            collate_fn=none_safe_collate,
            num_workers=args.num_workers
        )
    else:
        train_loader_multi = torch.utils.data.DataLoader(
            dataset_train_multi,
            batch_size=args.supervised_batch_size,
            shuffle=True,
            collate_fn=none_safe_collate,
            num_workers=args.num_workers
        )

    test_loader_multi = torch.utils.data.DataLoader(
        dataset_val_multi, batch_size=args.supervised_batch_size,
        shuffle=False, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # SUPERVISED: callbacks & trainer
    joint_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    joint_earlystop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min")
    lr_monitor_joint = pl.callbacks.LearningRateMonitor(logging_interval="step")
    timer_supervised = Timer(duration=None)
    resource_cb_supervised = ResourceMonitor(log_every_n_steps=50)

    # Phase marker for supervised start
    wandb.log({"phase": "supervised_start"})

    joint_trainer = pl.Trainer(
        max_epochs=args.supervised_max_epochs,
        accelerator=accelerator,
        logger=pl.loggers.WandbLogger(experiment=wandb.run),
        callbacks=[joint_checkpoint, joint_earlystop, lr_monitor_joint, timer_supervised, resource_cb_supervised],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=args.gradient_clip_val,
        **trainer_kwargs,
    )

    joint_trainer.fit(supervised_joint_model, train_loader_multi, test_loader_multi)
    # Log supervised timing
    _log_timer(timer_supervised, prefix="supervised")

    # Load best model checkpoint
    joint_path = joint_checkpoint.best_model_path
    wandb.log({"best_epoch_joint": joint_path})

    joint_seq = torch.load(joint_path)
    supervised_joint_model.load_state_dict(joint_seq["state_dict"])
    supervised_joint_model.eval()

    time_phase3_end = time.time()
    time_phase3 = time_phase3_end - time_phase3_start
    print(f"\nPhase 3 (Supervised Training) completed in {time_phase3:.2f} seconds ({time_phase3/60:.2f} minutes)")
    wandb.log({"time/phase3_supervised_seconds": time_phase3, "time/phase3_supervised_minutes": time_phase3/60})

    # -------------------------------
    # GET EMBEDDINGS FOR TRAINING DATASET
    # -------------------------------
    time_embedding_start = time.time()

    # Create full training dataset
    train_full_data_dict = {
        "wave": train_wf,
        "isi": train_isi,
        "acg": train_acg
    }

    # Build full-dataset label array for embedding extraction
    if _region_conditioning_enabled and train_layer_labels is not None:
        train_full_labels_stacked = np.vstack(
            (train_labels_encoded,
             np.ones_like(train_labels_encoded) * train_source_id,
             train_layer_labels)
        ).T
    else:
        train_full_labels_stacked = np.vstack(
            (train_labels_encoded,
             np.ones_like(train_labels_encoded) * train_source_id)
        ).T

    train_full_dataset = MultiModalEphysDataset(
        train_full_data_dict,
        train_full_labels_stacked,
        mode="multi",
        modality_sizes=modalities
    )

    train_full_loader = torch.utils.data.DataLoader(
        train_full_dataset, batch_size=128, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # Get embeddings for training dataset
    train_embeddings, train_labels_final = get_embeddings_multimodal(train_full_loader, supervised_joint_model)

    # Save training embeddings
    os.makedirs(output_dir, exist_ok=True)

    train_embeddings_df = pd.DataFrame(train_embeddings)
    train_embeddings_df["label"] = le.inverse_transform(train_labels_final.astype(int))
    train_embeddings_df.to_csv(f"{output_dir}/train_embeddings.csv", index=False)

    # -------------------------------
    # LOAD AND PROCESS PREDICT DATASET
    # -------------------------------
    predict_wf, predict_isi, predict_acg, predict_labels, predict_source_id = load_dataset_data(args.predict_dataset, all_dataset_files)

    if predict_labels is None:
        print(f"Warning: Predict dataset '{args.predict_dataset}' has no labels. Creating dummy labels for processing.")
        predict_labels = np.zeros(len(predict_wf), dtype=str)
        predict_labels[:] = "unknown"

    # Apply same class exclusion to predict dataset for consistent evaluation.
    if _excluded_classes:
        _keep_pred = np.array([lbl not in _excluded_classes for lbl in predict_labels])
        n_dropped_pred = int((~_keep_pred).sum())
        if n_dropped_pred:
            print(f"[exclude-classes] dropping {n_dropped_pred} predict rows for classes: {sorted(_excluded_classes)}")
            predict_wf, predict_isi, predict_acg = predict_wf[_keep_pred], predict_isi[_keep_pred], predict_acg[_keep_pred]
            predict_labels = predict_labels[_keep_pred]
            if isinstance(predict_source_id, np.ndarray):
                predict_source_id = predict_source_id[_keep_pred]
            if predict_layer_labels is not None and len(predict_layer_labels) == len(_keep_pred):
                predict_layer_labels = predict_layer_labels[_keep_pred]

    # Store original predict labels for reference
    predict_labels_original = predict_labels.copy()

    # Apply label shuffling to predict labels if control experiment
    if args.shuffle_labels:
        print(f"Shuffling predict labels with seed {args.shuffle_seed + 1}...")
        np.random.seed(args.shuffle_seed + 1)  # Different seed from training
        predict_labels_shuffled = predict_labels.copy()
        np.random.shuffle(predict_labels_shuffled)
        predict_labels = predict_labels_shuffled
        wandb.log({"shuffle_applied_to_predict": True})

    # Keep a combined encoder for reporting/saving (not used by the model)
    # Convert all labels to strings to avoid type mixing issues
    train_labels_str = [str(label) for label in train_labels]
    predict_labels_str = [str(label) for label in predict_labels]
    all_unique_labels = np.unique(np.concatenate([train_labels_str, predict_labels_str]))
    le_combined = LabelEncoder().fit(all_unique_labels)

    # ---------- FIX PART 3 applied: map predict labels for MODEL space ----------
    predict_labels_for_model = map_labels_to_training_encoder(le, predict_labels, fallback=0)

    # Create predict dataset
    predict_data_dict = {
        "wave": predict_wf,
        "isi": predict_isi,
        "acg": predict_acg
    }

    # Build predict label array: add layer column when region conditioning is active
    if _region_conditioning_enabled and predict_layer_labels is not None:
        predict_labels_stacked = np.vstack(
            (predict_labels_for_model,
             np.ones_like(predict_labels_for_model) * predict_source_id,
             predict_layer_labels)
        ).T
    else:
        predict_labels_stacked = np.vstack(
            (predict_labels_for_model,
             np.ones_like(predict_labels_for_model) * predict_source_id)
        ).T

    predict_dataset = MultiModalEphysDataset(
        predict_data_dict,
        predict_labels_stacked,
        mode="multi",
        modality_sizes=modalities
    )

    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=128, collate_fn=none_safe_collate, num_workers=args.num_workers
    )

    # Get embeddings for predict dataset (mask_class_labels=True to prevent data leakage)
    predict_embeddings, predict_labels_final = get_embeddings_multimodal(predict_loader, supervised_joint_model, mask_class_labels=True)

    time_embedding_end = time.time()
    time_embedding = time_embedding_end - time_embedding_start
    print(f"\nEmbedding extraction completed in {time_embedding:.2f} seconds ({time_embedding/60:.2f} minutes)")
    wandb.log({"time/embedding_extraction_seconds": time_embedding, "time/embedding_extraction_minutes": time_embedding/60})

    # Save predict embeddings
    predict_embeddings_df = pd.DataFrame(predict_embeddings)
    # Labels in training-encoder space (may contain fallback)
    predict_embeddings_df["label_training_space"] = le.inverse_transform(np.clip(predict_labels_final.astype(int), 0, len(le.classes_) - 1))
    # Also store the original labels for reference
    predict_embeddings_df["label_original"] = predict_labels_original
    predict_embeddings_df.to_csv(f"{output_dir}/predict_embeddings.csv", index=False)

    # -------------------------------
    # TRAIN KNN ON TRAINING EMBEDDINGS AND PREDICT ON TEST EMBEDDINGS
    # -------------------------------
    time_knn_start = time.time()

    # We need to use the original training label encoder for KNN training
    # since we want to predict classes that exist in the training set
    train_knn_labels = train_labels_final.astype(int)

    # Evaluate using KNN with different neighbor counts (k = 1..20 to match
    # transductive/holdout scripts and the PhysMAP comparison method).
    neighbor_options = list(range(1, 21))

    best_accuracy = -1
    best_neighbors = neighbor_options[0]
    best_predictions = None

    print(f"Training KNN with {len(train_embeddings)} training samples and {len(np.unique(train_knn_labels))} classes")
    print(f"Testing different neighbor counts: {neighbor_options}")

    # For cross-validation on training set to select best k
    from sklearn.model_selection import cross_val_score

    train_embeddings_l2 = train_embeddings / np.maximum(
        np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12
    )
    predict_embeddings_l2 = predict_embeddings / np.maximum(
        np.linalg.norm(predict_embeddings, axis=1, keepdims=True), 1e-12
    )

    cv_scores = {}
    for neighbor in neighbor_options:
        knn = KNeighborsClassifier(n_neighbors=neighbor, metric='cosine')
        cv_score = cross_val_score(knn, train_embeddings_l2, train_knn_labels, cv=min(5, len(np.unique(train_knn_labels))), scoring='balanced_accuracy')
        cv_scores[neighbor] = np.mean(cv_score)
        print(f"KNN with {neighbor} neighbors: CV balanced accuracy = {np.mean(cv_score):.4f} ± {np.std(cv_score):.4f}")

    # Select best k based on cross-validation
    best_neighbors = max(cv_scores, key=cv_scores.get)
    print(f"Selected best k = {best_neighbors} with CV score = {cv_scores[best_neighbors]:.4f}")

    # Train final KNN model with best k
    final_knn = KNeighborsClassifier(n_neighbors=best_neighbors, metric='cosine')
    final_knn.fit(train_embeddings_l2, train_knn_labels)

    # Make predictions on predict dataset
    predictions = final_knn.predict(predict_embeddings_l2)
    prediction_probabilities = final_knn.predict_proba(predict_embeddings_l2)

    # Convert predictions back to original labels (training encoder space)
    predicted_labels = le.inverse_transform(predictions.astype(int))

    # Calculate accuracy if we have true labels for the predict dataset
    accuracy_calculated = False
    if not (predict_labels == "unknown").all():  # If we have real labels
        # Create a label encoder for the predict dataset to get numeric labels for evaluation
        le_predict = LabelEncoder().fit(predict_labels)
        predict_labels_encoded_for_eval = le_predict.transform(predict_labels)

        # Find overlapping classes between training and predict datasets
        train_classes = set(le.classes_)
        predict_classes = set(le_predict.classes_)
        overlapping_classes = train_classes.intersection(predict_classes)

        if overlapping_classes:
            print(f"Overlapping classes: {overlapping_classes}")

            # Create mapping for evaluation
            true_labels_for_eval = []
            pred_labels_for_eval = []

            for i, (true_label, pred_idx) in enumerate(zip(predict_labels, predictions)):
                if true_label in overlapping_classes:
                    # Map both true and predicted labels to the training label encoder
                    true_labels_for_eval.append(le.transform([true_label])[0])
                    pred_labels_for_eval.append(pred_idx)

            if len(true_labels_for_eval) > 0:
                accuracy = balanced_accuracy_score(true_labels_for_eval, pred_labels_for_eval)
                print(f"Cross-dataset balanced accuracy: {accuracy:.4f}")
                accuracy_calculated = True

                # Confusion matrix
                conf_matrix = confusion_matrix(true_labels_for_eval, pred_labels_for_eval)
                available_classes = [c for c in le.classes_ if c in overlapping_classes]

                # Log metrics
                wandb.log({
                    "cross_dataset_balanced_accuracy": accuracy,
                    "best_k_neighbors": best_neighbors,
                    "num_overlapping_classes": len(overlapping_classes),
                    "num_evaluated_samples": len(true_labels_for_eval)
                })

                # Make confusion matrix figure if we have the function
                try:
                    figure_multi = make_confmat(conf_matrix, available_classes, best_neighbors)
                    wandb.log({
                        f"cross_dataset_confusion_matrix": wandb.Image(figure_multi),
                    })
                except Exception as e:
                    print(f"Could not create confusion matrix figure: {e}")
            else:
                print("No samples with overlapping classes found for evaluation")
        else:
            print("No overlapping classes between training and predict datasets")

    if not accuracy_calculated:
        wandb.log({"best_k_neighbors": best_neighbors})

    time_knn_end = time.time()
    time_knn = time_knn_end - time_knn_start
    print(f"\nKNN training and evaluation completed in {time_knn:.2f} seconds ({time_knn/60:.2f} minutes)")
    wandb.log({"time/knn_training_seconds": time_knn, "time/knn_training_minutes": time_knn/60})

    # -------------------------------
    # SAVE PREDICTIONS
    # -------------------------------
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        "predicted_label": predicted_labels,        # in training encoder space
        "true_label": predict_labels,               # labels used for evaluation (shuffled if control experiment)
        "true_label_original": predict_labels_original,  # original labels before any shuffling
        "prediction_confidence": np.max(prediction_probabilities, axis=1)
    })

    # Add probability columns for each class
    for i, class_name in enumerate(le.classes_):
        predictions_df[f"prob_{class_name}"] = prediction_probabilities[:, i]

    # Save predictions
    predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    # -------------------------------
    # LOG ARTIFACTS TO WANDB
    # -------------------------------
    # Upload embeddings and predictions
    wandb.log_artifact(
        f"{output_dir}/train_embeddings.csv",
        name=f"train_embeddings_{args.training_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}",
        type="embeddings"
    )

    wandb.log_artifact(
        f"{output_dir}/predict_embeddings.csv",
        name=f"predict_embeddings_{args.predict_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}",
        type="embeddings"
    )

    wandb.log_artifact(
        f"{output_dir}/predictions.csv",
        name=f"predictions_train_{args.training_dataset}_predict_{args.predict_dataset}_config_{args.config}_zdim_{args.z_dim}_B_{args.beta}{shuffle_suffix}",
        type="predictions"
    )

    # Upload model if requested
    if args.upload_model:
        wandb.log_artifact(joint_path, name=f'cross_dataset_model_train_{args.training_dataset}_predict_{args.predict_dataset}_z{args.z_dim}_lr{args.learning_rate}.pt', type='model')

    # Hyperparameters already logged during wandb.init()
    # Final log
    time_total = time_phase1 + time_phase2 + time_phase3 + time_embedding + time_knn

    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    print(f"Phase 1 (Pretraining):     {time_phase1:8.2f}s ({time_phase1/60:6.2f} min) - {100*time_phase1/time_total:5.1f}%")
    print(f"Phase 2 (Finetuning):      {time_phase2:8.2f}s ({time_phase2/60:6.2f} min) - {100*time_phase2/time_total:5.1f}%")
    print(f"Phase 3 (Supervised):      {time_phase3:8.2f}s ({time_phase3/60:6.2f} min) - {100*time_phase3/time_total:5.1f}%")
    print(f"Embedding Extraction:      {time_embedding:8.2f}s ({time_embedding/60:6.2f} min) - {100*time_embedding/time_total:5.1f}%")
    print(f"KNN Training/Evaluation:   {time_knn:8.2f}s ({time_knn/60:6.2f} min) - {100*time_knn/time_total:5.1f}%")
    print("-" * 50)
    print(f"TOTAL TIME:                {time_total:8.2f}s ({time_total/60:6.2f} min)")
    print("="*50 + "\n")

    wandb.log({
        "time/total_seconds": time_total,
        "time/total_minutes": time_total/60,
        "time/phase1_percentage": 100*time_phase1/time_total,
        "time/phase2_percentage": 100*time_phase2/time_total,
        "time/phase3_percentage": 100*time_phase3/time_total,
        "time/embedding_percentage": 100*time_embedding/time_total,
        "time/knn_percentage": 100*time_knn/time_total,
        "phase": "complete"
    })
    wandb.finish()

    print("="*50)
    print("CROSS-DATASET CLASSIFICATION COMPLETE")
    print("="*50)
    print(f"Training dataset: {args.training_dataset}")
    print(f"Predict dataset: {args.predict_dataset}")
    print(f"Training classes: {list(le.classes_)}")
    print(f"Best k for KNN: {best_neighbors}")
    if args.shuffle_labels:
        print(f"Control experiment: Labels were SHUFFLED (seed={args.shuffle_seed})")
    print(f"Output directory: {output_dir}/")
    print("\nFiles generated:")
    print("- train_embeddings.csv: Embeddings for training dataset")
    print("- predict_embeddings.csv: Embeddings for predict dataset")
    print("- predictions.csv: KNN predictions on predict dataset")
    print("="*50)