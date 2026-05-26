import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from celltype_ibl.models.BiModalEmbedding import BimodalEmbeddingModel
from celltype_ibl.models.encoders import SingleWVFEncoder, ConvolutionalEncoder
from celltype_ibl.utils.c4_data_utils import get_dataset_from_file
from celltype_ibl.params.config import DATASETS_DIRECTORY


def make_confmat(conf_matrix, label_names, k):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix (k={k})')
    plt.tight_layout()
    return fig


def get_embeddings_nemo(wvf_data, acg_data, model, device, batch_size=512):
    model.eval()
    embeddings = []

    wvf_tensor = torch.from_numpy(wvf_data.astype("float32")).to(device)
    acg_tensor = torch.from_numpy(acg_data.astype("float32")).to(device)

    n_samples = len(wvf_data)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            wvf_batch = wvf_tensor[i:end_idx]
            acg_batch = acg_tensor[i:end_idx].reshape(-1, 1, 10, 101) * 10

            wvf_emb, acg_emb = model.embed(wvf_batch, acg_batch)
            joint_emb = (wvf_emb + acg_emb) / 2

            embeddings.append(joint_emb.cpu().numpy())

    return np.vstack(embeddings)


def load_dataset_h5(dataset_file, normalise=True, force_labeled=None, remove_unlabeled=True, use_cache=True):
    file_path = os.path.join(DATASETS_DIRECTORY, dataset_file)

    # Auto-detect if labeled, but allow override
    if force_labeled is not None:
        is_labeled = force_labeled
    else:
        is_labeled = 'labelled' in dataset_file.lower() or 'labeled' in dataset_file.lower()

    try:
        result = get_dataset_from_file(
            dataset_file, is_labeled=is_labeled, normalise=normalise, use_cache=use_cache
        )

        if is_labeled:
            wvf_data, acg_data, label_idx, label_names, labelling_dict, correspondence_dict = result
            print(f"Debug: wvf_data.shape={wvf_data.shape}, acg_data.shape={acg_data.shape}")
            print(f"Debug: Sample ACG min/max: {acg_data[0].min():.3f}, {acg_data[0].max():.3f}")

            # Filter out unlabeled cells if requested.
            # Task doc "Dataset Filtering Rules (MANDATORY for all benchmarks)":
            #   1. Exclude "unlabeled" / "unknown" / etc.
            #   2. Exclude classes with < MIN_CELLS_PER_CLASS=10 samples.
            # Kept aligned with scripts/train_multimodal_transductive.py and
            # comparison_methods/physmap/physmap_script.r.
            MIN_CELLS_PER_CLASS = 10
            if remove_unlabeled:
                label_names_array = np.array(label_names)

                unlabeled_names = ['unlabelled', 'unlabeled', 'unknown', 'Unknown',
                                   'juxt', 'Juxt', 'axo', 'Axo', 'vgat', 'VGAT']
                unlabeled_mask = np.isin(label_names_array, unlabeled_names)
                labeled_mask = ~unlabeled_mask

                n_unlabeled = np.sum(unlabeled_mask)
                if n_unlabeled > 0:
                    print(f"Removing {n_unlabeled} unlabeled cells from {dataset_file}")
                    wvf_data = wvf_data[labeled_mask]
                    acg_data = acg_data[labeled_mask]
                    label_names = label_names_array[labeled_mask]
                    print(f"After unlabeled filter: wvf_data.shape={wvf_data.shape}")

                # Drop classes below MIN_CELLS_PER_CLASS so the StratifiedKFold
                # in main() never sees rare classes. Mirrors HIPPIE's count<5 rule.
                label_names = np.asarray(label_names)
                uniq, counts = np.unique(label_names, return_counts=True)
                rare = set(uniq[counts < MIN_CELLS_PER_CLASS].tolist())
                if rare:
                    keep = np.array([n not in rare for n in label_names])
                    n_rare = int((~keep).sum())
                    print(f"Removing {n_rare} cells from rare classes "
                          f"(< {MIN_CELLS_PER_CLASS} samples): {sorted(rare)}")
                    wvf_data = wvf_data[keep]
                    acg_data = acg_data[keep]
                    label_names = label_names[keep]
                    print(f"After rare-class filter: wvf_data.shape={wvf_data.shape}")

            return wvf_data, acg_data, label_names
        else:
            wvf_data, acg_data = result
            print(f"Debug: wvf_data.shape={wvf_data.shape}, acg_data.shape={acg_data.shape}")
            return wvf_data, acg_data, None
    except (NotImplementedError, ValueError) as e:
        # If labeled loading fails, try unlabeled
        if is_labeled:
            print(f"  Warning: Failed to load as labeled dataset, trying unlabeled... ({e})")
            result = get_dataset_from_file(
                dataset_file, is_labeled=False, normalise=normalise, use_cache=use_cache
            )
            wvf_data, acg_data = result
            print(f"Debug: wvf_data.shape={wvf_data.shape}, acg_data.shape={acg_data.shape}")
            return wvf_data, acg_data, None
        else:
            raise


def train_nemo_model(wvf_train, acg_train, args, device):
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
    for epoch in range(args.pretrain_max_epochs):
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.pretrain_max_epochs}, Loss: {avg_loss:.4f}")

        if wandb.run is not None:
            wandb.log({"pretrain_loss": avg_loss, "epoch": epoch})

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--z_dim", type=int, default=256, help="Dimension of latent space. Default 256 matches the locked NEMO configuration reported in the paper.")
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--temperature', type=float, default=0.5, help="Temperature for contrastive loss")
    parser.add_argument('--dataset-file', type=str, required=True, help="Dataset file for testing (e.g., C4_database_hull_labelled.h5)")
    parser.add_argument('--pretrain-datasets', type=str, nargs='+', default=None,
                        help="List of dataset files to use for pretraining (e.g., C4_database_hull_unlabelled.h5 C4_database_hausser.h5)")
    parser.add_argument('--wandb-tag', type=str, default="NEMO_transductive")
    parser.add_argument('--project', type=str, default="NEMO_transductive")
    parser.add_argument('--pretrain-max-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--batch-norm', action='store_true')
    parser.add_argument('--similarity-adjust', action='store_true')
    parser.add_argument('--l2-norm', action='store_true', default=True)

    parser.add_argument("--n_cv_folds", type=int, default=5, help="Number of cross validation folds")
    parser.add_argument("--cv_fold", type=int, default=0, help="Which CV fold to run (0 to n_cv_folds-1)")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./outputs_nemo_transductive')
    parser.add_argument('--no-cache', action='store_true', help='Disable dataset caching (force recompute ACGs)')

    # Data-scaling checkpoint flags. Used by the shared-checkpoint scaling
    # experiment (scripts/data_scaling_nrp/) to decouple training from evaluation.
    parser.add_argument("--pretrain-only", action="store_true",
                        help="Run contrastive training only, save model to "
                             "--pretrain-save-path, exit before evaluation.")
    parser.add_argument("--pretrain-save-path", type=str, default=None,
                        help="Where to save the trained model state_dict "
                             "(local path or s3:// URI).")
    parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                        help="Path to a saved model state_dict (.pt). Skip "
                             "contrastive training, go straight to evaluation. "
                             "Supports local paths and s3:// URIs.")

    args = parser.parse_args()

    if args.pretrain_only and args.pretrain_checkpoint:
        raise ValueError(
            "--pretrain-only and --pretrain-checkpoint are mutually exclusive."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # wandb is opt-in: default to disabled unless WANDB_MODE is set explicitly.
    os.environ.setdefault("WANDB_MODE", "disabled")
    if os.environ["WANDB_MODE"].lower() not in ("disabled", "dryrun"):
        wandb.login()

    wandb_logger = wandb.init(
        project=args.project,
        name=f"{args.wandb_tag}-{args.dataset_file}-fold_{args.cv_fold}_zdim_{args.z_dim}",
        tags=[args.wandb_tag, f"transductive_cv_fold_{args.cv_fold}"],
        config=vars(args)
    )

    wandb.log({"phase": "transductive_start"})

    use_cache = not args.no_cache
    print(f"Loading dataset: {args.dataset_file} (cache {'disabled' if args.no_cache else 'enabled'})")
    wvf_data, acg_data, labels = load_dataset_h5(args.dataset_file, use_cache=use_cache)

    if labels is None:
        # Retry with force_labeled=True for datasets whose filenames don't
        # contain "labelled" (e.g., Allen_visual_coding_nemo.h5, DANDI H5s).
        print(f"Auto-detect found no labels, retrying with force_labeled=True...")
        wvf_data, acg_data, labels = load_dataset_h5(
            args.dataset_file, force_labeled=True, use_cache=False)

    if labels is None:
        raise ValueError(f"Dataset {args.dataset_file} has no labels. Please use a labeled dataset.")

    le = LabelEncoder().fit(labels)
    labels_encoded = le.transform(labels)

    num_classes = len(np.unique(labels_encoded))
    print(f"Total samples: {len(wvf_data)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {np.bincount(labels_encoded)}")

    skf = StratifiedKFold(n_splits=args.n_cv_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(wvf_data, labels_encoded))
    train_indices, test_indices = splits[args.cv_fold]

    wvf_train = wvf_data[train_indices]
    wvf_test = wvf_data[test_indices]
    acg_train = acg_data[train_indices]
    acg_test = acg_data[test_indices]
    label_train = labels_encoded[train_indices]
    label_test = labels_encoded[test_indices]

    print(f"\nCross-validation fold {args.cv_fold}/{args.n_cv_folds}")
    print(f"Train set: {len(wvf_train)} cells")
    print(f"Test set: {len(wvf_test)} cells")
    print(f"Train classes: {np.unique(label_train)}")
    print(f"Test classes: {np.unique(label_test)}")

    print("\n" + "="*50)
    print("PRETRAINING PHASE (Transductive: using all data)")
    print("="*50)

    wandb.log({"phase": "pretrain_start"})

    # Load additional pretraining datasets if specified
    if args.pretrain_datasets:
        pretrain_wvf_list = [wvf_train, wvf_test]
        pretrain_acg_list = [acg_train, acg_test]

        print(f"\nLoading additional pretraining datasets...")
        for pretrain_file in args.pretrain_datasets:
            print(f"Loading: {pretrain_file}")
            # Force labeled=True so DANDI/IBL H5 files (whose filenames lack
            # "labelled") load via the correct code path. Labels are discarded
            # anyway — we only need wvf + acg for contrastive pretraining.
            # Fall back to unlabeled if labeled loading fails.
            try:
                wvf_extra, acg_extra, _ = load_dataset_h5(
                    pretrain_file, normalise=True, remove_unlabeled=False,
                    force_labeled=True, use_cache=use_cache)
            except Exception as e:
                print(f"  Labeled load failed ({e}), trying unlabeled...")
                wvf_extra, acg_extra, _ = load_dataset_h5(
                    pretrain_file, normalise=True, remove_unlabeled=False,
                    force_labeled=False, use_cache=use_cache)
            if len(wvf_extra) == 0:
                print(f"  WARNING: 0 samples loaded from {pretrain_file}, skipping")
                continue
            pretrain_wvf_list.append(wvf_extra)
            pretrain_acg_list.append(acg_extra)
            print(f"  Added {len(wvf_extra)} samples")

        wvf_all = np.vstack(pretrain_wvf_list)
        acg_all = np.vstack(pretrain_acg_list)
    else:
        wvf_all = np.vstack([wvf_train, wvf_test])
        acg_all = np.vstack([acg_train, acg_test])

    if args.pretrain_checkpoint:
        # ---- Load pre-trained model from checkpoint ----
        print(f"\n[--pretrain-checkpoint] Loading model from: {args.pretrain_checkpoint}")
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

        _ckpt_path = args.pretrain_checkpoint
        if _ckpt_path.startswith("s3://"):
            import tempfile, boto3
            _without_scheme = _ckpt_path[5:]
            _bucket, _, _key = _without_scheme.partition("/")
            _s3_ep = os.environ.get("S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook")
            _s3 = boto3.client("s3", endpoint_url=_s3_ep,
                               aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                               aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
            _tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            _s3.download_file(_bucket, _key, _tmp.name)
            _ckpt_path = _tmp.name

        model.load_state_dict(torch.load(_ckpt_path, map_location=device))
        print("[--pretrain-checkpoint] Loaded. Skipping training.")
    else:
        # ---- Standard contrastive training ----
        print(f"\nPretraining on {len(wvf_all)} samples...")
        model = train_nemo_model(wvf_all, acg_all, args, device)
        print("Pretraining completed!")

        # ---- --pretrain-only: save model and exit ----
        if args.pretrain_only:
            if args.pretrain_save_path:
                _dst = args.pretrain_save_path
                _state = model.state_dict()
                if _dst.startswith("s3://"):
                    import tempfile, boto3
                    _tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
                    torch.save(_state, _tmp.name)
                    _without_scheme = _dst[5:]
                    _bucket, _, _key = _without_scheme.partition("/")
                    _s3_ep = os.environ.get("S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook")
                    _s3 = boto3.client("s3", endpoint_url=_s3_ep,
                                       aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                       aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
                    _s3.upload_file(_tmp.name, _bucket, _key)
                    os.unlink(_tmp.name)
                    print(f"[--pretrain-only] Uploaded model -> {_dst}")
                else:
                    os.makedirs(os.path.dirname(_dst) or ".", exist_ok=True)
                    torch.save(_state, _dst)
                    print(f"[--pretrain-only] Saved model -> {_dst}")
            wandb.log({"phase": "pretrain_only_completed"})
            wandb.finish()
            print("[--pretrain-only] Done. Exiting before evaluation.")
            return

    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)

    wandb.log({"phase": "evaluation_start"})

    print("Extracting embeddings...")
    train_embeddings = get_embeddings_nemo(wvf_train, acg_train, model, device, args.batch_size)
    test_embeddings = get_embeddings_nemo(wvf_test, acg_test, model, device, args.batch_size)

    # Use basename of dataset_file to avoid absolute path clobbering output_dir
    _ds_base = os.path.basename(args.dataset_file).replace('.h5', '')
    results_dir = os.path.join(
        args.output_dir,
        _ds_base,
        f'fold_{args.cv_fold}',
        f'zdim_{args.z_dim}'
    )
    os.makedirs(results_dir, exist_ok=True)

    # K-NN classification — k selected by stratified inner CV on the TRAINING
    # fold only, then evaluated once on the held-out test fold (matches the
    # train-CV protocol used by PhysMAP and the HIPPIE benchmark scripts).
    # See hippie/utils.py:select_knn_k_by_train_cv.
    from hippie.utils import select_knn_k_by_train_cv
    print("\nPerforming K-NN classification...")
    knn_result = select_knn_k_by_train_cv(
        train_embeddings,
        label_train,
        test_embeddings,
        label_test,
        k_grid=range(1, 21),
        inner_cv_splits=5,
        metric="cosine",
        random_state=getattr(args, "seed", 42),
    )
    best_k = knn_result["best_k"]
    best_accuracy = knn_result["test_balanced_accuracy"]
    best_predictions = knn_result["test_predictions"]
    for k, cv_score in knn_result["per_k_train_cv"].items():
        wandb.log({
            f"knn_train_cv_balanced_accuracy_k{k}": cv_score,
            "cv_fold": args.cv_fold,
            "n_train_cells": len(train_indices),
            "n_test_cells": len(test_indices),
        })
        print(f"Inner train-CV balanced accuracy (k={k}): {cv_score:.4f}")

    print(f"\nBest K-NN performance: k={best_k}, accuracy={best_accuracy:.4f}")

    wandb.log({
        "best_transductive_accuracy": best_accuracy,
        "best_k": best_k
    })

    conf_matrix = confusion_matrix(label_test, best_predictions)
    label_names = le.classes_
    figure_transductive = make_confmat(conf_matrix, label_names, best_k)

    wandb.log({
        f"transductive_confusion_matrix_fold_{args.cv_fold}": wandb.Image(figure_transductive),
    })

    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    train_df = pd.DataFrame(train_embeddings)
    train_df['label'] = le.inverse_transform(label_train.astype(int))
    train_df.to_csv(os.path.join(results_dir, 'train_embeddings.csv'), index=False)

    wandb.save(os.path.join(results_dir, 'train_embeddings.csv'))

    test_df = pd.DataFrame(test_embeddings)
    test_df['label'] = le.inverse_transform(label_test.astype(int))
    test_df.to_csv(os.path.join(results_dir, 'test_embeddings.csv'), index=False)

    wandb.save(os.path.join(results_dir, 'test_embeddings.csv'))

    predictions_df = pd.DataFrame({
        "pred": le.inverse_transform(best_predictions.astype(int)),
        "true": le.inverse_transform(label_test.astype(int)),
    })
    predictions_df.to_csv(os.path.join(results_dir, 'transductive_predictions.csv'), index=False)

    wandb.save(os.path.join(results_dir, 'transductive_predictions.csv'))

    split_info = {
        'train_indices': train_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'fold': args.cv_fold,
        'n_folds': args.n_cv_folds,
        'dataset_file': args.dataset_file
    }

    with open(os.path.join(results_dir, 'cv_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    wandb.log({"phase": "transductive_completed"})

    # Write a minimal timings.csv compatible with the data-scaling aggregator.
    # This lets plot_data_scaling.py parse NEMO results the same way as HIPPIE.
    _timings_path = os.environ.get("COMPUTE_PARITY_CSV",
                                   os.path.join(results_dir, "timings.csv"))
    os.makedirs(os.path.dirname(_timings_path) or ".", exist_ok=True)
    _timings_rows = [
        {
            "method": "nemo",
            "dataset": os.path.basename(args.dataset_file),
            "fold": args.cv_fold,
            "phase": "knn",
            "wall_seconds": 0,
            "peak_gpu_mb": 0,
            "peak_cpu_mb": 0,
            "n_cells": len(wvf_data),
            "n_classes": num_classes,
            "control": "",
            "config": f"zdim_{args.z_dim}",
            "run_id": "",
            "started_at": "",
            "extra": json.dumps({
                "best_balanced_accuracy": float(best_accuracy),
                "best_k": int(best_k),
                "z_dim": args.z_dim,
            }),
        },
    ]
    _timings_df = pd.DataFrame(_timings_rows)
    _timings_df.to_csv(_timings_path, index=False)
    print(f"Wrote timings.csv -> {_timings_path}")

    wandb.finish()
    print("\nTransductive evaluation completed!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()