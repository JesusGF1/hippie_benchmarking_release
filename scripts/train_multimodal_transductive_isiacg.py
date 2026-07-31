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

    local_roots = ("./datasets/", "datasets/", "./datasets_hippie/", "datasets_hippie/")
    for path in possible_paths:
        if os.path.exists(path):
            outside_checkout = not path.startswith(local_roots) and not (
                env_root and path.startswith(env_root.rstrip("/"))
            )
            if outside_checkout:
                # Resolving to a container mount rather than the checkout means the
                # run may not be reproducible from this repository, and any cache it
                # writes can silently disagree with datasets/. Make that loud.
                print(
                    f"WARNING: {filename} for '{dataset}' resolved to {path}, which is "
                    f"outside this repository. Results computed here may not match "
                    f"datasets/{dataset}/. Set HIPPIE_DATA_ROOT to pin the source.",
                    file=sys.stderr,
                )
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

    Pipeline:
      * StandardScaler(fit on train)  — matches NEMO
      * LogisticRegression(multinomial, max_iter=1000) — linear probe
      * MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, early_stopping=True,
                      validation_fraction=0.1) — same hidden widths as the old custom MLP

    Returns (linear_preds, linear_ba, mlp_preds, mlp_ba).
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

            # Extract class labels and source labels if 2D
            if batch_labels.dim() > 1 and batch_labels.shape[1] > 1:
                class_labels = batch_labels[:, 0]  # First column is class labels
                source_labels = batch_labels[:, 1]  # Second column is source labels

                # Check for invalid labels (-1) and handle them
                invalid_mask = class_labels == -1
                if invalid_mask.any():
                    print(f"Found {invalid_mask.sum().item()} invalid labels (-1), setting to 0")
                    class_labels = class_labels.clone()
                    class_labels[invalid_mask] = 0
            else:
                class_labels = batch_labels
                source_labels = None

            # Mask class labels if requested (for test set to prevent leakage)
            class_labels_for_encoding = None if mask_class_labels else class_labels

            # Get latent embeddings with proper labels
            h, mu, log_var = model.encode(data_dict, source_labels=source_labels, class_labels=class_labels_for_encoding)
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
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument("--z_dim", type=int, default=None,
                        help="Total latent dimension. If unset, computed as "
                             "z_dim_per_modality * num_modalities (e.g., 30 for trimodal, 20 for bimodal).")
    parser.add_argument("--z_dim_per_modality", type=int, default=10,
                        help="Latent dimensions per input modality. Locked default 10 (z=20 bimodal) "
                             "selected by Hausser hyperparameter sweep (42 configs; frozen before "
                             "any other benchmark dataset was evaluated).")
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0,
                        help="Weight for KL divergence loss. Locked default 1.0 selected by "
                             "Hausser hyperparameter sweep (42 configs; frozen before any other "
                             "benchmark dataset was evaluated).")
    parser.add_argument('--dataset', type=str, default="allen_s_n_a_subset_no_superregions")
    parser.add_argument('--skip-probes', action='store_true',
                        help='Skip sklearn LR + MLP probes. Embeddings are saved to disk '
                             'early so a separate CPU probe job can run them later.')
    parser.add_argument('--wandb-tag', type=str, default="Hippie_transductive")
    parser.add_argument('--project', type=str, default="HIPPIE")
    parser.add_argument('--pretrain-max-epochs', type=int, default=100)
    parser.add_argument('--finetune-max-epochs', type=int, default=20)
    parser.add_argument('--supervised-max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--supervised-batch-size', type=int, default=128)
    parser.add_argument('--early-stopping-patience', type=int, default=30)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--train-val-split', type=float, default=0.9)

    # New arguments for multimodal approach
    parser.add_argument('--isi-weight', type=float, default=1.0, help='Weight for the ISI modality loss')
    parser.add_argument('--acg-weight', type=float, default=1.0, help='Weight for the ACG modality loss')

    parser.add_argument('--config', type=str, default="class_decoder_source_bn_aug_reg",
                       choices=["baseline", "with_source", "with_class", "with_both_embeddings",
                               "with_batch_norm", "no_augmentations", "no_fusion", "with_light_augmentations",
                               "with_heavy_augmentations", "full_architecture",
                               "conditional_decoder_only", "full_architecture_heavy_reg",
                               "class_decoder_source", "class_decoder_source_bn",
                               "class_decoder_source_bn_strong_aug", "class_decoder_source_bn_aug_reg"])

    # Cross validation arguments
    parser.add_argument("--n_cv_folds", type=int, default=5, help="Number of cross validation folds")
    parser.add_argument("--cv_fold", type=int, default=0, help="Which CV fold to run (0 to n_cv_folds-1)")
    parser.add_argument("--shuffle_labels", action="store_true", help="Randomly shuffle labels for control experiments")
    parser.add_argument("--shift_labels", action="store_true", help="Shift training labels by 1 position (class 0->1, 1->2, etc.) for control experiments")

    # Pretraining pool control. Default is target-only single-dataset pretraining:
    # the encoder sees only the --dataset target during the pretrain phase, with no
    # cross-dataset data leakage. Multi-dataset pretraining must be requested explicitly.
    #
    # Accepted values:
    #   ""              (default)   -> target-only single-dataset pretraining
    #   "all"                       -> every dataset in ALL_DATASETS
    #   "d1,d2,d3,..."              -> whitelist; --dataset target is always auto-added to the pool
    #
    # Scaling-laws experiments should set this explicitly to "all" or to a whitelist;
    # the default is intentionally conservative to keep the per-dataset benchmark runs
    # free of cross-dataset leakage.
    parser.add_argument("--pretrain-datasets", type=str, default="",
                        help="Pretraining pool control. Empty (default) = target-only "
                             "single-dataset pretraining. 'all' = every dataset in "
                             "ALL_DATASETS. Comma-separated list = "
                             "whitelist with the --dataset target auto-added.")

    # Class balancing argument.
    # NOTE: This flag exists as an opt-in but was NOT used in the published runs.
    # The cached predictions in results/ were produced without --use_balanced_sampling
    # (i.e., default uniform random sampling). Kept for users who want to experiment.
    parser.add_argument("--use_balanced_sampling", action="store_true",
                        help="Use class-balanced sampling via WeightedRandomSampler during supervised training. "
                             "Off by default; was NOT used in the published runs.")

    # Output directory for predictions and embeddings
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save predictions and embeddings. If not specified, uses "
                             "../results_hippie_transductive/{dataset}/fold_{cv_fold}/config_...")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (seeds python/numpy/torch/CUDA and DataLoader workers via pl.seed_everything).")
    args = parser.parse_args()

    # Resolve total z_dim from per-modality default unless an explicit override was passed.
    # train_multimodal_transductive always uses the 3 modalities {wave, isi, acg}, so we
    # can resolve immediately after parsing (before z_dim is used in run_name / paths).
    # Defaults (z_dim_per_modality=5, beta=0.9) come from systematic architectural ablation
    # on cellexplorer_cell_type + lisberger_labeled_cell_type only.
    if args.z_dim is None:
        _NUM_MODALITIES = 2  # ISI + ACG only
        args.z_dim = args.z_dim_per_modality * _NUM_MODALITIES
        print(f"[z_dim auto] z_dim_per_modality={args.z_dim_per_modality} * "
              f"num_modalities={_NUM_MODALITIES} (ISI+ACG) -> z_dim={args.z_dim}")

    # ---- Compute parity timing setup ----
    # Every phase below is wrapped in `with _timed("<phase>", ...)` and writes
    # one row to results/compute_parity/timings.csv. A final "total" row is
    # emitted at the very end of main(). All rows from one launch share run_id.
    _cp_run_id = _cp.make_run_id(prefix="hippie-tx-")
    _cp_control = (
        "shuffled" if args.shuffle_labels
        else ("shifted" if args.shift_labels else "")
    )
    _cp_t0_total = time.perf_counter()
    _cp_total_started_at = __import__("datetime").datetime.utcnow().isoformat(
        timespec="seconds") + "Z"

    def _timed(phase_name, *, n_cells=0, n_classes=0, extra=None):
        return _cp.phase(
            method="hippie",
            dataset=args.dataset,
            phase=phase_name,
            fold=args.cv_fold,
            n_cells=n_cells,
            n_classes=n_classes,
            control=_cp_control,
            config=args.config,
            run_id=_cp_run_id,
            extra={
                "z_dim": args.z_dim,
                "z_dim_per_modality": args.z_dim_per_modality,
                "beta": args.beta,
                "script": "train_multimodal_transductive",
                **(extra or {}),
            },
        )

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
    # Load the actual data
    # ISI+ACG only — skip waveforms entirely
    supervised_isi = pd.read_csv(find_dataset_file(args.dataset, "isi_dist.csv")).to_numpy()
    supervised_acg = pd.read_csv(find_dataset_file(args.dataset, "acg.csv")).to_numpy()

    # Load labels
    try:
        labels_path = find_dataset_file(args.dataset, "labels.csv")
        labels = pd.read_csv(labels_path)
        supervised_labels = labels[labels.columns[0]].values
        
        # Drop rows whose label is missing/empty/'unlabeled'/etc BEFORE encoding.
        # Old filter only matched substrings on str(lbl).lower(), which let
        # b'' (empty bytestring) and pandas NaN through - hausser hippie/
        # labels.csv is 94% b'' so the encoder previously created a phantom
        # 6th class "no label" that dominated training. _is_unlabeled() also
        # decodes bytestrings before substring matching.
        n_before = len(supervised_labels)
        indices_to_keep = [i for i, lbl in enumerate(supervised_labels)
                           if not _is_unlabeled(lbl)]
        supervised_isi = supervised_isi[indices_to_keep]
        supervised_acg = supervised_acg[indices_to_keep]
        supervised_labels = supervised_labels[indices_to_keep]
        # Normalize bytestrings AND str(bytes) artifacts -> plain str so
        # LabelEncoder doesn't see mixed types or class labels like
        # "b'PkC_ss'" leaking through as a separate class from "PkC_ss".
        supervised_labels = np.array([_normalize_label(lbl) for lbl in supervised_labels])
        n_after = len(supervised_labels)
        print(f"[label filter] dropped {n_before - n_after}/{n_before} unlabeled rows "
              f"({100 * (n_before - n_after) / max(n_before, 1):.1f}%) for {args.dataset}")
        
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
        supervised_isi = supervised_isi[indices_to_keep]
        supervised_acg = supervised_acg[indices_to_keep]
        supervised_labels = supervised_labels[indices_to_keep]
        
        # Re-fit encoder after removing small classes to ensure sequential labels
        remaining_label_strings = le.inverse_transform(supervised_labels)
        le = LabelEncoder().fit(remaining_label_strings)
        supervised_labels = le.transform(remaining_label_strings)
        
    except FileNotFoundError:
        print(f"No labels.csv found for {args.dataset}")
        supervised_labels = np.zeros(len(supervised_isi))
        le = LabelEncoder().fit(supervised_labels)

    num_class_labels = len(np.unique(supervised_labels))
    print(f"Dataset: {args.dataset}")
    print(f"Total samples: {len(supervised_isi)}")
    print(f"Number of classes: {num_class_labels}")
    
    print(f"Samples after removing rare classes: {len(supervised_isi)}")
    num_class_labels = len(np.unique(supervised_labels))
    print(f"Classes after removing rare classes: {np.unique(supervised_labels)}")
    print(f"Number of classes after removing rare classes: {num_class_labels}")
    print(f"ISI shape: {supervised_isi.shape}")
    print(f"ACG shape: {supervised_acg.shape}")
    print(f"Labels shape: {supervised_labels.shape}")
    print(f"Label distribution: {np.unique(supervised_labels, return_counts=True)}")
    # Sanity check
    assert len(supervised_isi) == len(supervised_labels) == len(supervised_acg), "Data and labels must have the same number of samples"


    # -------------------------------
    # Cross-validation split
    # -------------------------------
    # Create stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=42)
    splits = list(skf.split(supervised_isi, supervised_labels))
    train_indices, test_indices = splits[args.cv_fold]

    # Apply the CV split
    isi_train = supervised_isi[train_indices]
    isi_test = supervised_isi[test_indices]
    acg_train = supervised_acg[train_indices]
    acg_test = supervised_acg[test_indices]
    label_train = supervised_labels[train_indices]
    label_test = supervised_labels[test_indices]

    print(f"Train set: {len(isi_train)} cells")
    print(f"Test set: {len(isi_test)} cells")
    print(f"Train classes: {np.unique(label_train)}")
    print(f"Test classes: {np.unique(label_test)}")

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
    # Apply --beta override so the CLI flag is not silently ignored
    config.beta = args.beta
    print(f"[CVAEConfig] beta={config.beta}")

    # Define modalities and weights (ISI + ACG ONLY — no waveforms)
    modalities = {
        "isi": 100,
        "acg": 100
    }

    assert args.z_dim is not None, "args.z_dim should have been resolved right after argparse"
    modality_weights = {
        "isi": args.isi_weight,
        "acg": args.acg_weight
    }

    # -------------------------------
    # PRETRAINING PHASE
    #
    # Default: single-dataset pretraining on --dataset (target only). No cross-dataset
    # leakage. Multi-dataset pretraining is opt-in via --pretrain-datasets (whitelist
    # or the "all" sentinel) — used by the scaling-benchmark runner.
    # -------------------------------

    # ISI+ACG dataset mapping.
    # - dandi_000473: source NWB waveform_means contains electrode metadata (no waveforms)
    # - dandi_000955: source NWB has no waveform data at all
    # E3 ablation datasets (trimodal datasets, waveforms dropped by this script):
    # hull, hausser, lisberger, dandi_000041, allen
    ALL_DATASETS = {
        "hausser_cell_type": 1,
        "hull_cell_type": 2,
        "lisberger_labeled_cell_type": 3,
        "dandi_000041_cell_type": 4,
        "allen_scope_neuropixel_area_subset": 5,
        "dandi_000473_cell_type": 6,
        "dandi_000955_cell_type": 7,
    }
    all_dataset_files = ALL_DATASETS

    num_sources = max(all_dataset_files.values()) + 1

    # Resolve which datasets to include in the pretraining pool.
    # - Empty --pretrain-datasets (default) -> target-only single-dataset pretraining.
    # - "all" -> every entry in ALL_DATASETS.
    # - Comma-separated whitelist -> the named datasets plus the target (auto-added).
    # The downstream load loop silently skips anything missing on disk, but the
    # pretrain_pool resolved here is the authoritative intent and is logged via
    # compute_parity extras.
    pretrain_arg = args.pretrain_datasets.strip() if args.pretrain_datasets else ""
    if pretrain_arg == "":
        # Safe default: target-only. Cross-dataset pretraining must be explicit.
        if args.dataset not in ALL_DATASETS:
            raise ValueError(
                f"--dataset '{args.dataset}' is not in ALL_DATASETS and --pretrain-datasets "
                f"is empty (the default is target-only single-dataset pretraining). "
                f"Options: (1) use one of the known trimodal datasets "
                f"{sorted(ALL_DATASETS.keys())}; (2) pass --pretrain-datasets all to load "
                f"every trimodal dataset; or (3) pass an explicit comma-separated whitelist."
            )
        pretrain_pool = {args.dataset: ALL_DATASETS[args.dataset]}
    elif pretrain_arg.lower() == "all":
        # Explicit multi-dataset opt-in — used by the scaling benchmark runner when
        # it wants the full pool, and kept available for the replication case.
        pretrain_pool = dict(ALL_DATASETS)
    else:
        requested = [d.strip() for d in pretrain_arg.split(",") if d.strip()]
        unknown = [d for d in requested if d not in ALL_DATASETS]
        if unknown:
            raise ValueError(
                f"--pretrain-datasets contains unknown dataset(s): {unknown}. "
                f"Allowed: {sorted(ALL_DATASETS.keys())}, or 'all' for every dataset, "
                f"or empty/omit for target-only."
            )
        # Always include the target so the encoder sees it during pretraining.
        if args.dataset in ALL_DATASETS and args.dataset not in requested:
            requested.append(args.dataset)
        pretrain_pool = {d: ALL_DATASETS[d] for d in requested}

    # Load datasets for pretraining (multi-dataset learning)
    print("\n" + "="*60)
    print("Loading datasets for multi-dataset pretraining...")
    print(f"Pretrain pool (requested): {list(pretrain_pool.keys())}")
    print("="*60)

    datasets_pretrain = []
    all_source_labels = []
    pretrain_datasets_loaded = []  # actual list after disk-skip + load failures
    n_pretrain_cells_per_dataset = {}

    for dataset_name, source_id in pretrain_pool.items():
        try:
            # Load dataset. ACG is required (we deliberately exclude bimodal
            # datasets from ALL_DATASETS, so a missing acg.csv is a real error).
            dataset_isi = pd.read_csv(find_dataset_file(dataset_name, "isi_dist.csv")).to_numpy()
            dataset_acg = pd.read_csv(find_dataset_file(dataset_name, "acg.csv")).to_numpy()

            # NOTE: pretraining is fully self-supervised (class labels are
            # zeroed out below), so we deliberately do NOT call _is_unlabeled()
            # here. Filtering by label would silently throw away ~94% of
            # hausser and ~42% of lisberger (the b'' / b'unlabeled' rows) for
            # no benefit -- those cells contribute valid waveform/ISI/ACG
            # signal to the self-supervised pretraining objective. The
            # unlabeled-row filter still runs in the supervised branch (above)
            # where class labels actually matter.

            # Create source labels (zeros for class labels in pretraining)
            source_labels = np.full(len(dataset_isi), source_id)
            zero_labels = np.zeros(len(dataset_isi))

            # Create multimodal dataset for this source
            data_dict = {
                "isi": dataset_isi,
                "acg": dataset_acg
            }

            dataset_multi = MultiModalEphysDataset(
                data_dict,
                np.vstack((zero_labels, source_labels)).T,  # No class labels for pretraining
                mode="multi"
            )
            datasets_pretrain.append(dataset_multi)
            all_source_labels.extend(source_labels.tolist())
            pretrain_datasets_loaded.append(dataset_name)
            n_pretrain_cells_per_dataset[dataset_name] = int(len(dataset_isi))

            print(f"  ✓ {dataset_name}: {len(dataset_isi)} samples (source_id={source_id})")

        except Exception as e:
            print(f"  ✗ {dataset_name}: Failed to load ({e})")
            continue

    if not datasets_pretrain:
        raise RuntimeError(
            "No pretraining datasets could be loaded. Check that the requested "
            "datasets are present under /data/datasets/<name>/. Requested: "
            f"{list(pretrain_pool.keys())}"
        )

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
    with _timed("pretrain",
                n_cells=int(sum(n_pretrain_cells_per_dataset.values())),
                n_classes=num_class_labels,
                extra={"epochs": args.pretrain_max_epochs,
                       "batch_size": args.batch_size,
                       "pretrain_datasets_loaded": pretrain_datasets_loaded,
                       "n_pretrain_datasets": len(pretrain_datasets_loaded),
                       "n_pretrain_cells": int(sum(n_pretrain_cells_per_dataset.values()))}):
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

    # Load target dataset for fine-tuning
    finetune_isi = supervised_isi
    finetune_acg = supervised_acg

    # Create source labels (no class labels for this phase)
    finetune_source_labels = all_dataset_files[args.dataset] * np.ones(len(finetune_isi))
    finetune_zero_labels = np.zeros(len(finetune_isi))

    finetune_data_dict = {
        "isi": finetune_isi,
        "acg": finetune_acg
    }

    finetune_dataset_multi = MultiModalEphysDataset(
        finetune_data_dict,
        np.vstack((finetune_zero_labels, finetune_source_labels)).T,  # No class labels
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
    with _timed("finetune",
                n_cells=len(train_indices),
                n_classes=num_class_labels,
                extra={"epochs": args.finetune_max_epochs,
                       "batch_size": args.batch_size}):
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
        config=config,
    )

    # Load pretrained weights but skip the class embedding layer
    joint_seq = torch.load(joint_path)
    if "model.class_embedding.weight" in joint_seq["state_dict"]:
        joint_seq["state_dict"].pop("model.class_embedding.weight")

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
    # The split inside the training fold is stratified on the class label.
    from sklearn.model_selection import train_test_split as _sk_train_test_split

    _source_id = all_dataset_files[args.dataset]
    train_source_labels = _source_id * np.ones(len(train_indices))
    test_source_labels = _source_id * np.ones(len(test_indices))

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
        "isi": np.vstack([isi_train[sup_tr_idx], isi_test]),
        "acg": np.vstack([acg_train[sup_tr_idx], acg_test]),
    }
    sup_train_class_labels = np.concatenate([
        label_train[sup_tr_idx],
        np.zeros_like(label_test),
    ])
    sup_train_source_labels = np.concatenate([
        train_source_labels[sup_tr_idx],
        test_source_labels,
    ])
    supervised_train_dataset = MultiModalEphysDataset(
        sup_train_data,
        np.vstack((sup_train_class_labels, sup_train_source_labels)).T,
        mode="multi",
    )

    # VAL dataset: training-fold subset ONLY (no test cells).
    sup_val_data = {
        "isi": isi_train[sup_val_idx],
        "acg": acg_train[sup_val_idx],
    }
    sup_val_class_labels = label_train[sup_val_idx]
    sup_val_source_labels = train_source_labels[sup_val_idx]
    supervised_val_dataset = MultiModalEphysDataset(
        sup_val_data,
        np.vstack((sup_val_class_labels, sup_val_source_labels)).T,
        mode="multi",
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
    with _timed("supervised",
                n_cells=len(train_indices),
                n_classes=num_class_labels,
                extra={"epochs": args.supervised_max_epochs,
                       "batch_size": args.supervised_batch_size}):
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
        "isi": isi_test,
        "acg": acg_test
    }

    source_labels_test = all_dataset_files[args.dataset] * np.ones_like(label_test)

    dataset_test_multi = MultiModalEphysDataset(
        test_data_dict,
        np.vstack((label_test, source_labels_test)).T,
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
        "isi": isi_train,
        "acg": acg_train
    }
    source_labels_train_only = all_dataset_files[args.dataset] * np.ones_like(label_train)
    dataset_train_only = MultiModalEphysDataset(
        train_data_only_dict,
        np.vstack((label_train, source_labels_train_only)).T,
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
    with _timed("embedding",
                n_cells=len(train_indices) + len(test_indices),
                n_classes=num_class_labels):
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
    if args.output_dir:
        results_dir = args.output_dir
    else:
        results_dir = f"../results_hippie_transductive/{args.dataset}/fold_{args.cv_fold}/config_{args.config}_zdim_{args.z_dim}_B_{args.beta}"
    os.makedirs(results_dir, exist_ok=True)

    # K-NN classification
    print("Performing K-NN classification...")
    best_accuracy = 0
    best_k = 1
    best_predictions = None

    train_embeddings_l2 = train_embeddings / np.maximum(
        np.linalg.norm(train_embeddings, axis=1, keepdims=True), 1e-12
    )
    test_embeddings_l2 = test_embeddings / np.maximum(
        np.linalg.norm(test_embeddings, axis=1, keepdims=True), 1e-12
    )

    # k is selected by stratified inner CV on the TRAINING fold only, then
    # evaluated once on the held-out test fold. See
    # hippie/utils.py:select_knn_k_by_train_cv for rationale.
    from hippie.utils import select_knn_k_by_train_cv

    with _timed("knn",
                n_cells=len(train_indices) + len(test_indices),
                n_classes=num_class_labels,
                extra={"k_grid": "1..20", "selection": "inner-cv-on-train"}) as _knn_info:
        knn_result = select_knn_k_by_train_cv(
            train_embeddings_l2,
            train_labels_extracted,
            test_embeddings_l2,
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
                "cv_fold": args.cv_fold,
                "n_train_cells": len(train_indices),
                "n_test_cells": len(test_indices),
            })
            print(f"Inner train-CV balanced accuracy (k={k}): {cv_score:.4f}")

        _knn_info["best_k"] = best_k
        _knn_info["best_train_cv_score"] = float(knn_result["best_train_cv_score"])
        _knn_info["best_balanced_accuracy"] = float(best_accuracy)

    print(f"Best K-NN performance: k={best_k}, accuracy={best_accuracy:.4f}")

    # Log best accuracy as summary metric
    wandb_logger.experiment.log({
        "best_transductive_knn_accuracy": best_accuracy,
        "best_k": best_k
    })

    label_names = le.classes_

    # Save embeddings early so they survive if the process is interrupted during probes.
    train_emb_df = pd.DataFrame(train_embeddings)
    train_emb_df['label'] = le.inverse_transform(train_labels_extracted.astype(int))
    train_emb_df.to_csv(f"{results_dir}/train_embeddings.csv", index=False)
    test_emb_df = pd.DataFrame(test_embeddings)
    test_emb_df['label'] = le.inverse_transform(test_labels_extracted.astype(int))
    test_emb_df.to_csv(f"{results_dir}/test_embeddings.csv", index=False)

    # -------------------------------
    # SIMPLE PROBES (linear + MLP, sklearn)
    # -------------------------------
    if args.skip_probes:
        lr_ba = mlp_ba = float("nan")
        print("\nSkipping sklearn probes (--skip-probes); embeddings saved for CPU probe job.")
        wandb_logger.experiment.log({"phase": "simple_probes_skipped"})
    else:
        print("\nRunning simple probes (sklearn LR + MLP) on embeddings...")
        wandb_logger.experiment.log({"phase": "simple_probes_start"})

        with _timed("probes",
                    n_cells=len(train_indices) + len(test_indices),
                    n_classes=num_class_labels) as _probe_info:
            lr_preds, lr_ba, mlp_preds, mlp_ba = _run_simple_probes(
                train_embeddings, train_labels_extracted,
                test_embeddings, test_labels_extracted,
            )
            _probe_info["linear_probe_balanced_accuracy"] = float(lr_ba)
            _probe_info["mlp_probe_balanced_accuracy"] = float(mlp_ba)

        print(f"Linear probe (LogisticRegression) balanced accuracy: {lr_ba:.4f}")
        print(f"MLP probe (sklearn MLPClassifier)   balanced accuracy: {mlp_ba:.4f}")

        wandb_logger.experiment.log({
            "linear_probe_transductive_accuracy": lr_ba,
            "mlp_transductive_accuracy": mlp_ba,
            "phase": "simple_probes_completed",
        })

        # Confusion matrices for both probes
        lr_conf_matrix = confusion_matrix(test_labels_extracted, lr_preds)
        mlp_conf_matrix = confusion_matrix(test_labels_extracted, mlp_preds)
        figure_lr = make_confmat(lr_conf_matrix, label_names, "LinearProbe")
        figure_mlp = make_confmat(mlp_conf_matrix, label_names, "MLP")
        wandb_logger.experiment.log({
            f"{args.dataset}_linear_probe_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_lr),
            f"{args.dataset}_mlp_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_mlp),
        })

        # Save MLP predictions (filename kept stable for pick_locked_configs.py)
        mlp_predictions_df = pd.DataFrame({
            "pred": le.inverse_transform(mlp_preds.astype(int)),
            "true": le.inverse_transform(test_labels_extracted.astype(int)),
        })
        mlp_predictions_df.to_csv(f"{results_dir}/mlp_predictions.csv", index=False)

        # Save linear-probe predictions alongside (new output; matches NEMO's layout).
        linear_predictions_df = pd.DataFrame({
            "pred": le.inverse_transform(lr_preds.astype(int)),
            "true": le.inverse_transform(test_labels_extracted.astype(int)),
        })
        linear_predictions_df.to_csv(f"{results_dir}/linear_probe_predictions.csv", index=False)

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

    # Embeddings already saved early; log as wandb artifacts.
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/train_embeddings.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-train_embeddings.csv",
        type="train_embeddings"
    )
    wandb_logger.experiment.log_artifact(
        f"{results_dir}/test_embeddings.csv",
        name=f"{args.dataset}-fold_{args.cv_fold}-config_{args.config}_zdim_{args.z_dim}_B_{args.beta}-test_embeddings.csv",
        type="test_embeddings"
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

    # ---- Compute parity: emit final "total" row for the whole launch ----
    _cp_total_wall = time.perf_counter() - _cp_t0_total
    try:
        _cp.append_row(
            method="hippie",
            dataset=args.dataset,
            fold=args.cv_fold,
            phase="total",
            wall_seconds=_cp_total_wall,
            peak_gpu_mb=_cp._peak_gpu_mb_and_reset(),
            peak_cpu_mb=_cp._peak_cpu_mb(),
            n_cells=len(train_indices) + len(test_indices),
            n_classes=num_class_labels,
            control=_cp_control,
            config=args.config,
            run_id=_cp_run_id,
            started_at=_cp_total_started_at,
            extra={
                "z_dim": args.z_dim,
                "z_dim_per_modality": args.z_dim_per_modality,
                "beta": args.beta,
                "best_knn_k": int(best_k),
                "best_knn_balanced_accuracy": float(best_accuracy),
                "mlp_balanced_accuracy": float(mlp_accuracy),
                # Scaling-benchmark fields. Aggregator pivots on these.
                "pretrain_datasets_requested": list(pretrain_pool.keys()),
                "pretrain_datasets_loaded": pretrain_datasets_loaded,
                "n_pretrain_datasets": len(pretrain_datasets_loaded),
                "n_pretrain_cells": int(sum(n_pretrain_cells_per_dataset.values())),
                "n_pretrain_cells_per_dataset": n_pretrain_cells_per_dataset,
                "script": "train_multimodal_transductive",
            },
        )
        print(f"[compute_parity] total wall_seconds={_cp_total_wall:.1f}")
    except Exception as e:
        print(f"[compute_parity] WARNING: failed to write total row: {e}")

    print("Transductive evaluation completed!")


if __name__ == "__main__":
    main()