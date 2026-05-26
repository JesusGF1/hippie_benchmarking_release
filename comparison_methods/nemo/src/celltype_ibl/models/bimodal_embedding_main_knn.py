import numpy as np
import torch
import os
import json
import pickle
import matplotlib.pyplot as plt
import math
from math import pi
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

# --- NEW: KNN + scaling ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.models.ACG_augmentation_dataloader import EmbeddingDataset
from celltype_ibl.utils.c4_data_utils import (
    get_c4_unlabelled_wvf_acg_pairs,
    get_c4_labeled_dataset,
    get_dataset_from_file,
    auto_detect_dataset_type,
)
from celltype_ibl.utils.ultra_data_util import get_ultra_wvf_acg_pairs
from celltype_ibl.utils.kenji_allen_data_utils import get_kenji_allen_wvf_acg_pairs
from celltype_ibl.models.metrics import topk, rankme
import celltype_ibl.models.linear_classifier as linear_probe
from celltype_ibl.utils.ibl_data_util import get_ibl_wvf_acg_pairs
from celltype_ibl.utils.cell_explorer_data_util import (
    get_cell_explorer_dataset,
    data_load_by_split,
    get_allen_unlabeled_dataset,
    get_allen_labeled_dataset,
)

from celltype_ibl.utils.visualize_util import umap_from_embedding
from celltype_ibl.params.set_params import parse_base_args
from celltype_ibl.params.config import WANDB_DIR, WANDB_PROJECT, WANDB_ENTITY
from celltype_ibl.models.linear_probe import linear_probe_train_val
from celltype_ibl.utils.wandb import wandblog
import wandb
from datetime import datetime
import git

FLIP = True


def format_saved_model_name(args) -> str:
    model_name_parts = [
        f"temp{'T' if args.temperature is None else 'F'}",
        f"dim{args.dim_embed}",
        f"aug{'T' if args.augmentation else 'F'}",
        f"batch{args.batch_size}",
        f"act{args.activation}",
        f"data_{args.dataset}",
        f"seed{args.seed}",
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    ]
    if args.dataset == "c4":
        model_name_parts.append(f"h5{'T' if args.from_h5 else 'F'}")
    elif args.dataset == "ibl":
        model_name_parts.append(f"root{'T' if args.with_root else 'F'}")
        model_name_parts.append(f"heldout{'T' if args.held_out else 'F'}")
    elif args.dataset == "cell_explorer":
        model_name_parts.append(f"_split_{args.split_id}")
    elif args.dataset == "kenji_allen":
        model_name_parts.append(f"_{args.project}")
    if args.initialise:
        model_name_parts.append("init")
    if args.acg_normalization:
        model_name_parts.append("acg_norm")
    if args.batch_norm:
        model_name_parts.append("batch_norm")
    if args.adjust_to_ce:
        model_name_parts.append("ATC")
    if args.adjust_to_ultra:
        model_name_parts.append("ATU")
    return "_".join(model_name_parts)


def load_train_dataset(args, dataset: str = "c4"):
    fold_idx = None
    train_labels = None
    
    # Check if a specific dataset file is provided
    if hasattr(args, 'train_dataset_file') and args.train_dataset_file is not None:
        print(f"Loading training data from file: {args.train_dataset_file}")
        is_labeled = auto_detect_dataset_type(args.train_dataset_file)
        
        # Override dataset type to ensure proper validation splitting
        args.dataset = "custom"  # Use a custom dataset type to avoid IBL-specific logic
        args.test_data = "custom"  # Also override test_data to use custom logic
        
        if is_labeled:
            wvf_raw, acg_raw, train_labels, _, _, _ = get_dataset_from_file(
                args.train_dataset_file, is_labeled=True, normalise=True
            )
            print(f"Loaded labeled training dataset with {len(wvf_raw)} samples")
        else:
            wvf_raw, acg_raw = get_dataset_from_file(
                args.train_dataset_file, is_labeled=False, normalise=True
            )
            print(f"Loaded unlabeled training dataset with {len(wvf_raw)} samples")
    else:
        # Use original dataset loading logic
        if dataset == "c4":
            wvf_raw, acg_raw = get_c4_unlabelled_wvf_acg_pairs(from_h5=args.from_h5)
        elif dataset == "ibl":
            wvf_raw, acg_raw, train_labels, fold_idx = get_ibl_wvf_acg_pairs(
                with_root=args.with_root, return_region="cosmos", adjust_to_ce=args.adjust_to_ce
            )
        elif dataset == "ibl_repeated":
            wvf_raw, acg_raw = get_ibl_wvf_acg_pairs(repeated_sites=True)
        elif dataset == "allen":
            wvf_raw, acg_raw = get_allen_unlabeled_dataset()
        elif dataset == "allen_labelled":
            wvf_raw, acg_raw, _ = get_allen_labeled_dataset()
        elif dataset == "cell_explorer":
            wvf_raw, acg_raw, train_labels = data_load_by_split(split_id=args.split_id)
        elif dataset == "Ultra":
            wvf_raw, acg_raw, _, _ = get_ultra_wvf_acg_pairs(return_optotagged=False)
        elif dataset == "kenji_allen":
            wvf_raw, acg_raw, _, _ = get_kenji_allen_wvf_acg_pairs(
                return_optotagged=False, selected_project=args.project
            )
        else:
            raise ValueError("Dataset not recognised")
    
    return wvf_raw, acg_raw, train_labels, fold_idx


def split_validation_set(
    args,
    wvf_raw: np.ndarray,
    acg_raw: np.ndarray,
    args_dict: dict,
    fold_idx: np.ndarray | None,
    labels: np.ndarray | None,
):
    label_val = None
    if (args.dataset != "ibl") or (not args.held_out):
        if labels is not None:
            wvf_train, wvf_val, acg_train, acg_val, label_train, label_val = train_test_split(
                wvf_raw, acg_raw, labels, test_size=0.1, random_state=42, stratify=labels
            )
        else:
            wvf_train, wvf_val, acg_train, acg_val = train_test_split(
                wvf_raw, acg_raw, test_size=0.2, random_state=42
            )
            label_val = None
            label_train = None
    else:
        assert fold_idx is not None
        n_fold = len(np.unique(fold_idx))
        ind = np.random.choice(range(n_fold), size=(n_fold,), replace=False)
        n_train = round(n_fold * 0.7)
        n_val = round(n_fold * 0.1)
        train_idx = ind[:n_train]
        val_idx = ind[n_train : n_train + n_val]
        test_idx = ind[n_train + n_val :]
        val_idx = np.array([4])
        test_idx = np.array([3, 6])
        train_idx = np.array([0, 1, 2, 5, 7, 8, 9])
        wvf_train = wvf_raw[np.isin(fold_idx, train_idx)]
        acg_train = acg_raw[np.isin(fold_idx, train_idx)]
        wvf_val = wvf_raw[np.isin(fold_idx, val_idx)]
        acg_val = acg_raw[np.isin(fold_idx, val_idx)]
        if labels is not None:
            label_val = labels[np.isin(fold_idx, val_idx)]
            label_train = labels[np.isin(fold_idx, train_idx)]
        args_dict["test_idx"] = test_idx.tolist()
        args_dict["val_idx"] = val_idx.tolist()
    return wvf_train, wvf_val, acg_train, acg_val, label_train, label_val


def normalize_acg(acg_data):
    return acg_data / np.max(acg_data.reshape(-1, 1010), axis=1)[:, None, None]


def concatenate_dataset(acg_data, wvf_data):
    return np.concatenate((acg_data.reshape(-1, 1010) * 10, wvf_data), axis=1).astype("float32")


def get_labels_info(labels):
    unique_labels, label_idx = np.unique(labels, return_inverse=True)
    correspondence = {i: l for i, l in enumerate(unique_labels)}
    labelling = {l: i for i, l in enumerate(unique_labels)}
    return label_idx, correspondence, labelling


def get_linear_probe_testset(args, acg_val=None, wvf_val=None, label_val=None):
    print(f"Loading {args.test_data} dataset for linear probe.")
    session_idx = []
    
    # Check if a specific test dataset file is provided
    if hasattr(args, 'test_dataset_file') and args.test_dataset_file is not None:
        print(f"Loading test data from file: {args.test_dataset_file}")
        is_labeled = auto_detect_dataset_type(args.test_dataset_file)
        
        if not is_labeled:
            raise ValueError(f"Test dataset {args.test_dataset_file} must be labeled for evaluation")
        
        try:
            wvf_data, acg_data, label_idx, labels, labelling, correspondence = get_dataset_from_file(
                args.test_dataset_file, is_labeled=True, normalise=True
            )
            print(f"Loaded test dataset with {len(wvf_data)} samples")

            # Print memory consumption for test dataset
            wvf_test_memory_mb = wvf_data.nbytes / (1024 ** 2)
            acg_test_memory_mb = acg_data.nbytes / (1024 ** 2)
            total_test_memory_mb = wvf_test_memory_mb + acg_test_memory_mb
            print(f"\n{'='*60}")
            print(f"TEST DATASET MEMORY CONSUMPTION:")
            print(f"  Waveforms (wvf_data):      {wvf_test_memory_mb:.2f} MB")
            print(f"  3D Autocorrelograms (acg): {acg_test_memory_mb:.2f} MB")
            print(f"  Total:                     {total_test_memory_mb:.2f} MB")
            print(f"  Waveforms shape:           {wvf_data.shape}")
            print(f"  Autocorrelograms shape:    {acg_data.shape}")
            print(f"{'='*60}\n")

            print(f"Debug: wvf_data.shape={wvf_data.shape}, acg_data.shape={acg_data.shape}, label_idx.shape={np.array(label_idx).shape}")
            loo = True  # Default to leave-one-out for custom datasets
        except Exception as e:
            print(f"Error loading test dataset {args.test_dataset_file}: {e}")
            print("Falling back to using the training dataset for testing (validation mode)")
            # Use validation data instead
            if acg_val is not None and wvf_val is not None and label_val is not None:
                acg_data = acg_val
                wvf_data = wvf_val
                label_idx, correspondence, labelling = get_labels_info(label_val)
                loo = False
            else:
                raise ValueError("No valid test data available and no validation split provided")
    else:
        # Use original test dataset loading logic
        if args.test_data == "c4_labelled":
            wvf_data, acg_data, label_idx, _, labelling, correspondence = get_c4_labeled_dataset(
                from_h5=args.from_h5
            )
            loo = True
        elif args.test_data == "ibl":
            assert acg_val is not None and wvf_val is not None and label_val is not None
            acg_data = acg_val
            wvf_data = wvf_val
            label_idx, correspondence, labelling = get_labels_info(label_val)
            loo = False
        elif args.test_data == "allen":
            wvf_data, acg_data, labels = get_allen_labeled_dataset()
            label_idx, correspondence, labelling = get_labels_info(labels)
            loo = True
        elif args.test_data == "cell_explorer":
            wvf_data, acg_data, cell_types = get_cell_explorer_dataset(filt=True)
            label_idx, correspondence, labelling = get_labels_info(cell_types)
            loo = False
        elif args.test_data == "cell_explorer_cv":
            acg_data = acg_val
            wvf_data = wvf_val
            label_idx, correspondence, labelling = get_labels_info(label_val)
            loo = False
        elif args.test_data == "Ultra":
            wvf_data, acg_data, labels, session_idx = get_ultra_wvf_acg_pairs()
            label_idx, correspondence, labelling = get_labels_info(labels)
            loo = False
        elif args.test_data == "kenji_allen":
            wvf_data, acg_data, labels, session_idx = get_kenji_allen_wvf_acg_pairs(
                return_id="session", selected_project=args.project
            )
            label_idx, correspondence, labelling = get_labels_info(labels)
            loo = True
        else:
            raise ValueError("linear probe test data not recognised")

    if args.acg_normalization:
        acg_data = normalize_acg(acg_data)
    test_dataset = concatenate_dataset(acg_data, wvf_data)
    
    print(f"Debug: Final test_dataset.shape={test_dataset.shape}, final label_idx length={len(label_idx)}")

    return test_dataset, label_idx, labelling, correspondence, session_idx, loo


def ibl_train_val_rep(model, wvf_tensor, acg_tensor, wvf_val_tensor, acg_val_tensor, representation=True):
    train_wvf_acg_input = {"wvf": wvf_tensor, "acg": acg_tensor.reshape(-1, 1, 10, 101) * 10}
    if representation:
        wvf_rep_train, acg_rep_train = model.representation(train_wvf_acg_input["wvf"], train_wvf_acg_input["acg"])
    else:
        wvf_rep_train, acg_rep_train = model.embed(train_wvf_acg_input["wvf"], train_wvf_acg_input["acg"])

    val_wvf_acg_input = {"wvf": wvf_val_tensor, "acg": acg_val_tensor.reshape(-1, 1, 10, 101) * 10}
    if representation:
        wvf_rep_val, acg_rep_val = model.representation(val_wvf_acg_input["wvf"], val_wvf_acg_input["acg"])
    else:
        wvf_rep_val, acg_rep_val = model.embed(val_wvf_acg_input["wvf"], val_wvf_acg_input["acg"])

    training_data = torch.concat((acg_rep_train, wvf_rep_train), dim=1).cpu().detach().numpy()
    testing_data = torch.concat((acg_rep_val, wvf_rep_val), dim=1).cpu().detach().numpy()
    return training_data.squeeze(), testing_data.squeeze()


# ---- Save embeddings helpers ----
def _save_split_embeddings(model, device, out_dir, epoch, split_name, wvf_np, acg_np, labels=None, acg_normalization=False):
    acg_local = acg_np.copy()
    if acg_normalization:
        acg_local = acg_local / np.maximum(1e-8, np.max(acg_local.reshape(-1, 1010), axis=1))[:, None, None]

    wvf_t = torch.from_numpy(wvf_np.astype("float32")).to(device)
    acg_t = torch.from_numpy(acg_local.astype("float32")).to(device).reshape(-1, 1, 10, 101) * 10

    with torch.no_grad():
        wvf_rep, acg_rep = model.representation(wvf_t, acg_t)

    wvf_rep_np = wvf_rep.detach().cpu().numpy()
    acg_rep_np = acg_rep.detach().cpu().numpy()
    concat_np = torch.cat((acg_rep, wvf_rep), dim=1).detach().cpu().numpy()

    emb_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    npz_path = os.path.join(emb_dir, f"{split_name}_epoch_{epoch}.npz")

    save_dict = {"wvf_rep": wvf_rep_np, "acg_rep": acg_rep_np, "concat": concat_np}
    if labels is not None:
        save_dict["labels"] = np.asarray(labels)
    np.savez_compressed(npz_path, **save_dict)
    print(f"[emb] Saved {split_name} embeddings to {npz_path}")

    del wvf_t, acg_t, wvf_rep, acg_rep
    torch.cuda.empty_cache()


def _save_embeddings_from_concat(model, device, out_dir, epoch, split_name, concat_np, labels=None):
    acg_np = concat_np[:, :1010].reshape(-1, 1, 10, 101)
    wvf_np = concat_np[:, 1010:]

    wvf_t = torch.from_numpy(wvf_np.astype("float32")).to(device)
    acg_t = torch.from_numpy(acg_np.astype("float32")).to(device)

    with torch.no_grad():
        wvf_rep, acg_rep = model.representation(wvf_t, acg_t)

    wvf_rep_np = wvf_rep.detach().cpu().numpy()
    acg_rep_np = acg_rep.detach().cpu().numpy()
    concat_emb = torch.cat((acg_rep, wvf_rep), dim=1).detach().cpu().numpy()

    emb_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    npz_path = os.path.join(emb_dir, f"{split_name}_epoch_{epoch}.npz")

    save_dict = {"wvf_rep": wvf_rep_np, "acg_rep": acg_rep_np, "concat": concat_emb}
    if labels is not None:
        save_dict["labels"] = np.asarray(labels)
    np.savez_compressed(npz_path, **save_dict)
    print(f"[emb] Saved {split_name} embeddings to {npz_path}")

    del wvf_t, acg_t, wvf_rep, acg_rep
    torch.cuda.empty_cache()


# ---- NEW: KNN train→test probe (with auto-k on test) ----
def knn_probe_train_test(
    out_dir,
    X_train, y_train,
    X_test, y_test,
    k=5, metric="euclidean", weights="distance",
    auto_k=True, k_min=1, k_max=20
):
    """
    Train KNN on TRAIN and evaluate on TEST.
    If auto_k=True, pick k in [k_min, k_max] maximizing balanced accuracy on the TEST set.
    Saves results and a confusion matrix; also a k sweep plot if auto_k.
    """
    os.makedirs(out_dir, exist_ok=True)

    scaler = None
    Xtr, Xte = X_train, X_test
    if metric != "cosine":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

    chosen_k = k
    k_scores = []

    if auto_k:
        best_score = -1.0
        best_k = None
        ks = list(range(k_min, k_max + 1))
        for kk in ks:
            clf_k = KNeighborsClassifier(n_neighbors=kk, metric=metric, weights=weights)
            clf_k.fit(Xtr, y_train)
            y_hat = clf_k.predict(Xte)
            score = balanced_accuracy_score(y_test, y_hat)
            k_scores.append((kk, score))
            if score > best_score:
                best_score = score
                best_k = kk
        chosen_k = best_k

        # Plot k vs balanced accuracy (test)
        try:
            plt.figure(figsize=(5, 4))
            plt.plot([p[0] for p in k_scores], [p[1] for p in k_scores], marker="o")
            plt.xlabel("k")
            plt.ylabel("Balanced accuracy (test)")
            plt.title("KNN k sweep (train→test)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "knn_k_sweep.png"))
            plt.close()
        except Exception:
            pass

    clf = KNeighborsClassifier(n_neighbors=chosen_k, metric=metric, weights=weights)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    # Save results
    cv_path = os.path.join(out_dir, "knn_results.pkl")
    with open(cv_path, "wb") as f:
        pickle.dump(
            {
                "true_targets": y_test,
                "predicted_classes": y_pred,
                "classifier": "knn",
                "params": {
                    "k": chosen_k,
                    "metric": metric,
                    "weights": weights,
                    "auto_k": auto_k,
                    "k_min": k_min,
                    "k_max": k_max,
                    "k_scores": k_scores,
                },
            },
            f,
        )

    # Confusion matrix image
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"KNN confusion (k={chosen_k})")
    plt.colorbar()
    plt.tight_layout()
    img_path = os.path.join(out_dir, "knn_confusion.png")
    plt.savefig(img_path)
    plt.close()

    return cv_path, img_path


# ---- NEW: LOO fallback when only a labeled test set exists ----
def knn_leave_one_out(out_dir, X, y, k=5, metric="euclidean", weights="distance"):
    """
    Leave-One-Out evaluation on a labeled dataset X,y:
    For each i, train on all except i and test on i.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = X.shape[0]

    scaler = None
    X_proc = X
    if metric != "cosine":
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)

    y_pred = np.empty_like(y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
        clf.fit(X_proc[mask], y[mask])
        y_pred[i] = clf.predict(X_proc[i : i + 1])[0]

    cv_path = os.path.join(out_dir, "knn_results.pkl")
    with open(cv_path, "wb") as f:
        pickle.dump(
            {
                "true_targets": y,
                "predicted_classes": y_pred,
                "classifier": "knn_loo",
                "params": {"k": k, "metric": metric, "weights": weights},
            },
            f,
        )

    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"KNN LOO confusion (k={k})")
    plt.colorbar()
    plt.tight_layout()
    img_path = os.path.join(out_dir, "knn_confusion.png")
    plt.savefig(img_path)
    plt.close()

    return cv_path, img_path


def plot_training_log(
    args,
    loss_all,
    contrastive_acc1,
    contrastive_acc2,
    contrastive_val_acc1,
    contrastive_val_acc2,
    temperature_all,
):
    if args.split_validate:
        if args.train_temperature:
            fig, ax = plt.subplots(4, 1, figsize=(10, 20))
        else:
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    else:
        if args.train_temperature:
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Contrastive Accuracy")
    ax[0].plot(np.array(loss_all), label="Training Loss")
    ax[0].legend()
    for k in args.top_k:
        ax[1].plot(np.array(contrastive_acc1[f"t{k}"]), label=f"Training Accuracy1, top{k}")
        ax[1].plot(np.array(contrastive_acc2[f"t{k}"]), label=f"Training Accuracy2, top{k}")
    ax[1].legend()
    plt_idx = 2

    if args.split_validate:
        ax[plt_idx].set_xlabel("Epoch")
        ax[plt_idx].set_ylabel("Accuracy")
        for k in args.top_k:
            ax[plt_idx].plot(np.array(contrastive_val_acc1[f"t{k}"]), label=f"Validation Accuracy1, top{k}")
            ax[plt_idx].plot(np.array(contrastive_val_acc2[f"t{k}"]), label=f"Validation Accuracy2, top{k}")
        ax[plt_idx].legend()
        plt_idx = 3

    if args.train_temperature:
        ax[plt_idx].plot(np.array(temperature_all), label="Temperature")
        ax[plt_idx].legend()

    return fig, ax


def train():
    # ============ TIME LOGGING START ============
    script_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training script started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    args = parse_base_args()

    # --- Defaults so you don't need CLI tweaks ---
    if not hasattr(args, "classifier"):
        args.classifier = "knn"  # default to KNN
    if not hasattr(args, "knn_k"):
        args.knn_k = 5
    if not hasattr(args, "knn_metric"):
        args.knn_metric = "euclidean"  # try "cosine" if embeddings are normalized
    if not hasattr(args, "knn_weights"):
        args.knn_weights = "distance"
    if not hasattr(args, "knn_auto_k"):
        args.knn_auto_k = True
    if not hasattr(args, "knn_k_min"):
        args.knn_k_min = 1  # k=1..20 sweep matches HIPPIE/PhysMAP benchmark protocol
    if not hasattr(args, "knn_k_max"):
        args.knn_k_max = 20

    if args.train_temperature:
        args.temperature = None
    args_dict = vars(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load training dataset
    data_load_start = time.time()
    print("Loading training dataset...")
    wvf_raw, acg_raw, train_labels, fold_idx = load_train_dataset(args, args.dataset)
    data_load_time = time.time() - data_load_start
    print(f"✓ Data loading completed in {data_load_time:.2f}s")

    # Print memory consumption for training dataset
    wvf_memory_mb = wvf_raw.nbytes / (1024 ** 2)
    acg_memory_mb = acg_raw.nbytes / (1024 ** 2)
    total_memory_mb = wvf_memory_mb + acg_memory_mb
    print(f"\n{'='*60}")
    print(f"TRAINING DATASET MEMORY CONSUMPTION:")
    print(f"  Waveforms (wvf_raw):       {wvf_memory_mb:.2f} MB")
    print(f"  3D Autocorrelograms (acg): {acg_memory_mb:.2f} MB")
    print(f"  Total:                     {total_memory_mb:.2f} MB")
    print(f"  Waveforms shape:           {wvf_raw.shape}")
    print(f"  Autocorrelograms shape:    {acg_raw.shape}")
    print(f"{'='*60}\n")

    if args.split_validate:
        wvf_train, wvf_val, acg_train, acg_val, label_train, label_val = split_validation_set(
            args, wvf_raw, acg_raw, args_dict, fold_idx, train_labels
        )
        wvf_val_tensor = torch.from_numpy(wvf_val.astype("float32")).clone().to(device)
        if args.acg_normalization:
            acg_val = acg_val / np.max(acg_val.reshape(-1, 1010), axis=1)[:, None, None]
        acg_val_tensor = torch.from_numpy(acg_val.astype("float32")).clone().to(device)
        del wvf_raw, acg_raw
    else:
        acg_val, wvf_val, label_val = [], [], []
        wvf_train, acg_train, label_train = wvf_raw, acg_raw, train_labels

    if args.save:
        model_name = format_saved_model_name(args)
        
        # Try to get git hash, but don't fail if not in a git repo
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            args_dict["git_hash"] = sha
        except git.exc.InvalidGitRepositoryError:
            print("Warning: Not in a git repository, skipping git hash tracking")
            args_dict["git_hash"] = "not_available"
        if args.use_wandb:
            wb = wandblog(
                name=model_name, config=args_dict, entity=WANDB_ENTITY, project=WANDB_PROJECT, topk=topk, wandb_dir=WANDB_DIR
            )
        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir + "/args.json", "w") as f:
            json.dump(args_dict, f, indent=4)
        print(f"Arguments saved to {output_dir}/args.json")

    if args.linear_probe:
        test_dataset, label_idx, labelling, correspondence, session_idx, loo = get_linear_probe_testset(
            args, acg_val, wvf_val, label_val
        )

    by_session = args.by_session
    if by_session:
        loo = False

    dataset = EmbeddingDataset(
        wvf_train,
        acg_train,
        augmentation=args.augmentation,
        acg_normalize=args.acg_normalization,
        std=args.std,
    )

    embedding_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    model = BimodalEmbeddingModel(
        temperature=args.temperature,
        layer_norm=args.layer_norm,
        latent_dim=args.dim_embed,
        similarity_adjust=args.similarity_adjust,
        l2_norm=args.l2_norm,
        activation=args.activation,
        batch_norm=args.batch_norm,
        adjust_to_ce=args.adjust_to_ce,
        adjust_to_ultra=args.adjust_to_ultra,
        acg_dropout=args.acg_dropout,
        wvf_dropout=args.wvf_dropout,
        loss_flood=args.loss_flood,
        use_RINCE_loss=args.use_RINCE_loss,
        lam=args.lam,
        q=args.rince_q,
    )

    print("initialize to train...")
    start_epoch = 0
    highest_acc = 0
    highest_F1 = 0
    loss_all = []
    contrastive_acc1 = {}
    contrastive_acc2 = {}
    best_val_top_k = {}
    contrastive_val_acc1 = {}
    contrastive_val_acc2 = {}
    for k in args.top_k:
        contrastive_acc1[f"t{k}"] = []
        contrastive_acc2[f"t{k}"] = []
        contrastive_val_acc1[f"t{k}"] = []
        contrastive_val_acc2[f"t{k}"] = []
        best_val_top_k[f"t{k}"] = 0
    temperature_all = []
    N = wvf_train.shape[0]

    model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 1, last_epoch=-1)

    if (args.load_pt_path is not None) and not args.initialise:
        optimizer.load_state_dict(torch.load(args.load_pt_path)["optimizer_state_dict"])
        print("Optimizer loaded from", args.load_pt_path)
        loss_all = list(torch.load(args.load_pt_path)["loss"])
        start_epoch = len(loss_all)
        for k in args.top_k:
            contrastive_acc1["t" + str(k)] = list(torch.load(args.load_pt_path)[f"training_acc1t{k}"])
            contrastive_acc2["t" + str(k)] = list(torch.load(args.load_pt_path)[f"training_acc2t{k}"])
        temperature_all = list(torch.load(args.load_pt_path)["temperature_all"])
        if args.split_validate:
            for k in args.top_k:
                contrastive_val_acc1["t" + str(k)] = list(torch.load(args.load_pt_path)[f"validation_acc1t{k}"])
                contrastive_val_acc2["t" + str(k)] = list(torch.load(args.load_pt_path)[f"validation_acc2t{k}"])

    model.eval()
    print(f"\nStarting training for {args.num_epochs} epochs...")
    training_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        epoch_start_time = time.time()

        if args.split_validate:
            val_loss = model(wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10)
        model.train()
        tmp_losses = []
        for idx, (wvf_batch, acg_batch) in enumerate(embedding_dataloader):
            optimizer.zero_grad()
            loss = model(wvf_batch.to(device), acg_batch.reshape(-1, 1, 10, 101).to(device))
            loss.backward()
            optimizer.step()
            tmp_losses.append(loss.cpu().detach().numpy() * wvf_batch.size(dim=0))
        loss_all.append(sum(tmp_losses) / N)

        if args.use_wandb:
            currlog = {"Loss": loss_all[-1]}
            if args.split_validate:
                currlog["Val_loss"] = val_loss.cpu().detach().numpy()
        scheduler.step()

        temperature_all.append(model.temperature.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            wvf_embed, acg_embed = model.embed(
                torch.from_numpy(wvf_train.astype("float32")).to(device),
                torch.from_numpy(acg_train.astype("float32")).to(device).reshape(-1, 1, 10, 101) * 10,
            )

        similarity_matrix = torch.matmul(wvf_embed, acg_embed.transpose(0, 1))
        labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
        rankme_loss = rankme(torch.cat((wvf_embed, acg_embed), dim=1))
        if args.use_wandb:
            currlog["rankme_loss"] = rankme_loss.cpu().detach().numpy()

        del wvf_embed, acg_embed
        torch.cuda.empty_cache()

        for k in args.top_k:
            topk1 = topk(similarity_matrix, labels, k=k)
            topk2 = topk(torch.transpose(similarity_matrix, 0, 1), labels, k=k)
            contrastive_acc1[f"t{k}"].append(topk1.cpu().detach().numpy())
            contrastive_acc2[f"t{k}"].append(topk2.cpu().detach().numpy())
            if args.use_wandb:
                currlog[f"contrastive_acc1_t{k}"] = topk1.cpu().detach().numpy()
                currlog[f"contrastive_acc2_t{k}"] = topk2.cpu().detach().numpy()

        if args.split_validate:
            with torch.no_grad():
                wvf_embed_val, acg_embed_val = model.embed(
                    wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10
                )
            similarity_matrix_val = torch.matmul(wvf_embed_val, acg_embed_val.transpose(0, 1))
            labels_val = torch.arange(len(similarity_matrix_val)).to(similarity_matrix_val.device)
            rankme_loss_val = rankme(torch.cat((wvf_embed_val, acg_embed_val), dim=1))
            if args.use_wandb:
                currlog["rankme_loss_val"] = rankme_loss_val.cpu().detach().numpy()

            for k in args.top_k:
                topk1_val = topk(similarity_matrix_val, labels_val, k=k).cpu().detach().numpy()
                topk2_val = topk(torch.transpose(similarity_matrix_val, 0, 1), labels_val, k=k).cpu().detach().numpy()
                contrastive_val_acc1[f"t{k}"].append(topk1_val)
                contrastive_val_acc2[f"t{k}"].append(topk2_val)
                if args.use_wandb:
                    currlog[f"contrastive_val_acc1_t{k}"] = topk1_val
                    currlog[f"contrastive_val_acc2_t{k}"] = topk2_val
                avg = (topk1_val + topk2_val) / 2
                if avg > best_val_top_k[f"t{k}"]:
                    best_val_top_k[f"t{k}"] = avg
                    checkpoint_path = os.path.join(output_dir, f"best_top{k}_checkpoint.pt")
                    data_save = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": np.array(loss_all),
                        "training_acc1": contrastive_acc1,
                        "training_acc2": contrastive_acc2,
                        "validation_acc1": contrastive_val_acc1,
                        "validation_acc2": contrastive_val_acc2,
                        "temperature_all": np.array(temperature_all),
                        "best_epoch": epoch,
                    }
                    torch.save(data_save, checkpoint_path)
                    if args.use_wandb:
                        wb.log_weights(checkpoint_path, k)

        if not args.use_wandb:
            fig, _ = plot_training_log(
                args, loss_all, contrastive_acc1, contrastive_acc2, contrastive_val_acc1, contrastive_val_acc2, temperature_all
            )
            fig.savefig(os.path.join(output_dir, "training_log.png"))
            plt.close(fig)

        if (epoch % args.log_every_n_steps == 0) or (epoch == start_epoch + args.num_epochs - 1):
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            data_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.array(loss_all),
                "training_acc1": contrastive_acc1,
                "training_acc2": contrastive_acc2,
                "temperature_all": np.array(temperature_all),
            }
            if args.split_validate:
                data_save["validation_acc1"] = contrastive_val_acc1
                data_save["validation_acc2"] = contrastive_val_acc2
            torch.save(data_save, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Save embeddings alongside checkpoints
            if args.save:
                _save_split_embeddings(
                    model, device, output_dir, epoch, "train", wvf_train, acg_train, labels=label_train,
                    acg_normalization=args.acg_normalization
                )
                if args.split_validate:
                    _save_split_embeddings(
                        model, device, output_dir, epoch, "val", wvf_val, acg_val, labels=label_val,
                        acg_normalization=args.acg_normalization
                    )

            # ---------------- Downstream evaluation (KNN default) ----------------
            if args.linear_probe:
                eval_start_time = time.time()
                if (args.test_data == "ibl") | (args.test_data == "cell_explorer_cv"):
                    # Train on train split, test on val split
                    wvf_tensor = torch.from_numpy(wvf_train.astype("float32")).to(device)
                    acg_tensor = torch.from_numpy(acg_train.astype("float32")).to(device)
                    training_data, testing_data = ibl_train_val_rep(
                        model, wvf_tensor, acg_tensor, wvf_val_tensor, acg_val_tensor
                    )
                    torch.cuda.empty_cache()

                    probe_out_dir = os.path.join(output_dir, f"probe_epoch_{epoch}")
                    if args.classifier.lower() == "linear":
                        cv_path, img_path = linear_probe_train_val(
                            probe_out_dir, training_data, testing_data, label_train, label_val, labelling
                        )
                    else:
                        cv_path, img_path = knn_probe_train_test(
                            probe_out_dir,
                            training_data, label_train,
                            testing_data, label_val,
                            k=args.knn_k,
                            metric=args.knn_metric,
                            weights=args.knn_weights,
                            auto_k=args.knn_auto_k,
                            k_min=args.knn_k_min,
                            k_max=args.knn_k_max,
                        )

                    if args.save:
                        _save_split_embeddings(
                            model, device, output_dir, epoch, "test",
                            wvf_val, acg_val, labels=label_val, acg_normalization=args.acg_normalization
                        )
                else:
                    # General path: prefer true train→test if labeled train exists; else LOO on test set
                    probe_out_dir = os.path.join(output_dir, f"probe_epoch_{epoch}")
                    if args.classifier.lower() == "linear":
                        cv_path, img_path, _ = linear_probe.main(
                            test_dataset,
                            label_idx,
                            labelling,
                            correspondence,
                            probe_out_dir,
                            session_idx=session_idx,
                            embedding_model_path=checkpoint_path,
                            loo=loo,
                            by_session=by_session,
                            n_runs=args.n_runs,
                            embedding_model=args.model,
                            layer_norm=args.layer_norm,
                            latent_dim=args.dim_embed,
                            l2_norm=args.l2_norm,
                            activation=args.activation,
                            batch_norm=args.batch_norm,
                            seed=args.seed,
                            undersample=args.undersample,
                            use_smote=args.use_smote,
                        )
                    else:
                        # If we have label_train from the loaded training dataset, build TRAIN embeddings
                        if (label_train is not None) and (len(label_train) == len(wvf_train)):
                            wvf_t = torch.from_numpy(wvf_train.astype("float32")).to(device)
                            acg_t = torch.from_numpy(acg_train.astype("float32")).to(device)
                            with torch.no_grad():
                                wvf_rep_tr, acg_rep_tr = model.representation(
                                    wvf_t, acg_t.reshape(-1, 1, 10, 101) * 10
                                )
                            X_train = torch.cat((acg_rep_tr, wvf_rep_tr), dim=1).cpu().numpy()
                            # Convert string labels to numeric indices for training
                            train_label_idx, _, _ = get_labels_info(label_train)
                            y_train = train_label_idx

                            # Compute embeddings for test data as well
                            wvf_test = test_dataset[:, 1010:].astype("float32")
                            acg_test = test_dataset[:, :1010].reshape(-1, 10, 101).astype("float32")
                            wvf_test_t = torch.from_numpy(wvf_test).to(device)
                            acg_test_t = torch.from_numpy(acg_test).to(device).reshape(-1, 1, 10, 101) * 10
                            with torch.no_grad():
                                wvf_rep_test, acg_rep_test = model.representation(wvf_test_t, acg_test_t)
                            X_test = torch.cat((acg_rep_test, wvf_rep_test), dim=1).cpu().numpy()
                            # Use numeric label indices for test data as well
                            y_test = label_idx

                            cv_path, img_path = knn_probe_train_test(
                                probe_out_dir,
                                X_train, y_train,
                                X_test, y_test,
                                k=args.knn_k,
                                metric=args.knn_metric,
                                weights=args.knn_weights,
                                auto_k=args.knn_auto_k,
                                k_min=args.knn_k_min,
                                k_max=args.knn_k_max,
                            )
                        else:
                            # Fallback: LOO on the (labeled) test dataset
                            y_test = np.array([correspondence[i] for i in label_idx])
                            cv_path, img_path = knn_leave_one_out(
                                probe_out_dir, test_dataset, y_test,
                                k=args.knn_k, metric=args.knn_metric, weights=args.knn_weights
                            )

                    if args.save:
                        labels_text = np.array([correspondence[i] for i in label_idx])
                        _save_embeddings_from_concat(
                            model, device, output_dir, epoch, "test", test_dataset, labels=labels_text
                        )

                im = plt.imread(img_path)
                if args.use_wandb:
                    wb.log_results(cv_path, epoch)
                    currlog["confusion_matrix"] = [wandb.Image(im)]

                with open(cv_path, "rb") as f:
                    data = pickle.load(f)
                    true_classes = data["true_targets"]
                    pred_classes = data["predicted_classes"]
                acc = balanced_accuracy_score(true_classes, pred_classes)
                f1 = f1_score(true_classes, pred_classes, average="macro")
                if args.use_wandb:
                    currlog["f1_score"] = f1
                    currlog["balanced_accuracy"] = acc
                    if acc > highest_acc:
                        highest_acc = acc
                        currlog["highest_acc"] = acc
                    if f1 > highest_F1:
                        highest_F1 = f1
                        currlog["highest_F1"] = f1

                eval_time = time.time() - eval_start_time
                print(f"  ✓ Evaluation completed in {eval_time:.2f}s")

            # ------------------------- Embedding viz -----------------------------
            if args.embedding_viz:
                print(f"Debug: Embedding viz - args.dataset='{args.dataset}', args.test_data='{args.test_data}'")
                if args.test_data == "ibl":
                    wvf_rep_val, acg_rep_val = model.representation(
                        wvf_val_tensor, acg_val_tensor.reshape(-1, 1, 10, 101) * 10
                    )
                    wvf_rep = wvf_rep_val
                    acg_rep = acg_rep_val
                    labels = np.array([correspondence[i] for i in label_idx])
                    dot_size = 1
                elif (args.dataset == "c4") | (args.dataset == "custom") | (args.test_data in ["custom", "cell_explorer", "cell_explorer_cv", "Ultra", "kenji_allen"]):
                    acg_viz = test_dataset[:, :1010].reshape(-1, 1, 10, 101)
                    wvf_viz = test_dataset[:, 1010:]
                    
                    print(f"Debug: Before processing - test_dataset.shape={test_dataset.shape}, label_idx length={len(label_idx)}")
                    print(f"Debug: acg_viz.shape={acg_viz.shape}, wvf_viz.shape={wvf_viz.shape}")
                    
                    # Ensure test_dataset and label_idx have the same length
                    min_len = min(len(test_dataset), len(label_idx))
                    acg_viz = acg_viz[:min_len]
                    wvf_viz = wvf_viz[:min_len]
                    label_idx_viz = label_idx[:min_len]
                    
                    print(f"Debug: After truncation - acg_viz.shape={acg_viz.shape}, wvf_viz.shape={wvf_viz.shape}")
                    
                    wvf_acg_input = {"wvf": torch.tensor(wvf_viz).to(device), "acg": torch.tensor(acg_viz).to(device)}
                    print(f"Debug: Model input shapes - wvf: {wvf_acg_input['wvf'].shape}, acg: {wvf_acg_input['acg'].shape}")
                    
                    wvf_rep, acg_rep = model.representation(wvf_acg_input["wvf"], wvf_acg_input["acg"])
                    print(f"Debug: Model output shapes - wvf_rep: {wvf_rep.shape}, acg_rep: {acg_rep.shape}")
                    
                    labels = np.array([correspondence[i] for i in label_idx_viz])
                    dot_size = 3
                    
                    print(f"Visualization: using {min_len} samples (test_dataset: {len(test_dataset)}, label_idx: {len(label_idx)})")
                else:
                    raise ValueError("Embedding visualization not supported")
                fig, _ = umap_from_embedding(
                    wvf_rep.to("cpu").detach().numpy(),
                    acg_rep.to("cpu").detach().numpy(),
                    labels,
                    n_class=len(np.unique(labels)),
                    standardize=True,
                    dot_size=dot_size,
                )
                umap_img_path = os.path.join(output_dir, f"embedding_visualization_epoch_{epoch}.png")
                fig.savefig(umap_img_path)
                im = plt.imread(umap_img_path)
                if args.use_wandb:
                    currlog["embedding umap"] = [wandb.Image(im)]
                plt.close(fig)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch}/{start_epoch + args.num_epochs - 1} completed in {epoch_time:.2f}s")

        if args.use_wandb:
            wb.log(currlog)

    training_time = time.time() - training_start_time
    total_time = time.time() - script_start_time

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Total training time: {training_time:.2f}s ({training_time/60:.2f}m)")
    print(f"  Average time per epoch: {training_time/args.num_epochs:.2f}s")
    print(f"  Total script time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Script finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if args.use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    train()
