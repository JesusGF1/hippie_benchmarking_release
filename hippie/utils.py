import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch


def make_confmat(cm, label_names, best_neighbors_waveform):
    # Handle division by zero when some classes have no samples
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_cm = cm / row_sums

    # Create annotations with both normalized values and raw counts
    annotations = np.empty_like(normalized_cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{normalized_cm[i, j]:.2f}\n({cm[i, j]})"

    # Create heatmap with blue color scheme
    ax = sns.heatmap(
        normalized_cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )

    # Explicitly set the tick labels
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names, rotation=0)

    # Set the title
    plt.title(f"{best_neighbors_waveform} neighbors")

    # Get the figure to return
    figure = ax.get_figure()
    plt.close(figure)  # Close the plot to avoid displaying it in some environments
    return figure


def generate_kfolds(dataset_path):
    # Load the cell explorer data
    cell_explorer_wf = pd.read_csv(f"datasets/{dataset_path}/waveforms.csv")
    cell_explorer_isi = pd.read_csv(f"datasets/{dataset_path}/isi_dist.csv")
    cell_explorer_labels = pd.read_csv(f"../datasets/{dataset_path}/celltypes.csv", index_col=0)

    # Turn into numpy arrays
    cell_explorer_wf = cell_explorer_wf.to_numpy()
    cell_explorer_isi = cell_explorer_isi.to_numpy()
    cell_explorer_labels = cell_explorer_labels.to_numpy()

    le = LabelEncoder()
    cell_explorer_labels = le.fit_transform(cell_explorer_labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Generate the folds
    folds = []
    for train_index, val_index in skf.split(cell_explorer_wf, cell_explorer_labels):
        wf_train = cell_explorer_wf[train_index]
        wf_val = cell_explorer_wf[val_index]
        isi_train = cell_explorer_isi[train_index]
        isi_val = cell_explorer_isi[val_index]
        label_train = cell_explorer_labels[train_index]
        label_val = cell_explorer_labels[val_index]

        folds.append((wf_train, wf_val, isi_train, isi_val, label_train, label_val, le))

    return folds


# Try K nearest neighbor
def get_embeddings(dataloader_wave, dataloader_time, wave_model, time_model):
    embedding_waveform = []
    embedding_isi = []
    for i, ((wave, label_wave), (time, label_time)) in enumerate(zip(dataloader_wave, dataloader_time)):
        assert (label_wave == label_time).all()
        w_out = wave_model((wave, label_wave))
        t_out = time_model((time, label_time))
        e_wave, d_wave = w_out[0], w_out[-1]  # w_out = enc, zmean, zlogvar, dec
        e_time, d_time = t_out[0], t_out[-1]  # t_out = enc, zmean, zlogvar, dec

        e_wave = (e_wave - e_wave.mean(dim=1)[:, None]) / e_wave.std(dim=1)[:, None]
        e_time = (e_time - e_time.mean(dim=1)[:, None]) / e_time.std(dim=1)[:, None]

        embedding_waveform.append(e_wave)
        embedding_isi.append(e_time)
    embedding_waveform = torch.cat(embedding_waveform, dim=0)
    embedding_isi = torch.cat(embedding_isi, dim=0)
    # Run Umap in the embeddings
    embedding_waveform = embedding_waveform.detach().numpy()
    embedding_isi = embedding_isi.detach().numpy()
    # labels = torch.cat(labels, dim=0).detach().numpy()
    joint_embeddings = np.concatenate([embedding_waveform, embedding_isi], axis=1)
    # normalize the embeddings

    return embedding_waveform, embedding_isi, joint_embeddings


def select_knn_k_by_train_cv(
    train_embeddings_l2,
    train_labels,
    test_embeddings_l2,
    test_labels,
    k_grid=None,
    inner_cv_splits=5,
    metric="cosine",
    random_state=42,
):
    """Select KNN k by stratified CV on the training fold, then evaluate once on test.

    The chosen k is the argmax of mean balanced accuracy across an inner
    stratified CV on the training fold; that single k is then refit on the
    full training fold and evaluated once on the held-out test fold. The
    same per-fold k-selection rule is used by PhysMAP (through caret) and
    by the NEMO baseline.

    Returns
    -------
    dict with keys:
        best_k                  : int — k selected by inner train-CV
        best_train_cv_score     : float — mean balanced accuracy of best_k on inner CV
        test_balanced_accuracy  : float — single evaluation on the test fold
        test_predictions        : np.ndarray — test-fold predictions from the final KNN
        per_k_train_cv          : dict[int, float] — inner-CV score for every evaluated k
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score

    if k_grid is None:
        k_grid = list(range(1, 21))

    train_labels = np.asarray(train_labels)
    n_train = len(train_labels)

    n_per_class = np.bincount(train_labels.astype(int))
    nonzero = n_per_class[n_per_class > 0]
    smallest_class = int(nonzero.min()) if nonzero.size else 2
    n_splits = max(2, min(inner_cv_splits, smallest_class))
    inner_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    per_k_train_cv = {}
    best_k = None
    best_train_cv_score = -np.inf
    for k in k_grid:
        k = int(k)
        if k >= n_train:
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        try:
            scores = cross_val_score(
                knn,
                train_embeddings_l2,
                train_labels,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=1,
            )
        except ValueError:
            continue
        mean_score = float(np.mean(scores))
        per_k_train_cv[k] = mean_score
        if mean_score > best_train_cv_score:
            best_train_cv_score = mean_score
            best_k = k

    if best_k is None:
        best_k = 1
        best_train_cv_score = float("nan")

    final_knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    final_knn.fit(train_embeddings_l2, train_labels)
    test_predictions = final_knn.predict(test_embeddings_l2)
    test_balanced_accuracy = float(
        balanced_accuracy_score(test_labels, test_predictions)
    )

    return {
        "best_k": best_k,
        "best_train_cv_score": float(best_train_cv_score),
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_predictions": test_predictions,
        "per_k_train_cv": per_k_train_cv,
    }