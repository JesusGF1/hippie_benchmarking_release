"""Figure 3 — HIPPIE-bimodal and PhysMAP confusion matrices on A1 and S1.

Concatenates predictions across all 5 folds (HIPPIE) or across folds stored in
one CSV (PhysMAP), then plots row-normalized confusion matrices.

Outputs:
    figures/25_figure_3_confusion_matrices/<dataset>_<method>_cm.{svg,png}
    figures/25_figure_3_confusion_matrices/<dataset>_cm_pair.{svg,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "benchmark" / "celltype_cache"
OUT_DIR = REPO_ROOT / "figures" / "figure_3"

DATASETS = {
    "a1data_remove_undef": {
        "display": "A1 (Lakunina)",
        "class_order": ["EXC", "PV", "SOM"],
    },
    "juxtacellular_mouse_s1_area": {
        "display": "S1 (Yu)",
        "class_order": ["E_L4", "E_L5", "FS_L4", "FS_L5", "SOM"],
    },
}

METHODS = [
    ("hippie_waveisi", "HIPPIE-bimodal", "Blues"),
    ("physmap",        "PhysMAP (WNN)", "Blues"),
]


def collect_hippie_preds(dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    trues, preds = [], []
    for fold in range(5):
        p = CACHE / "hippie-waveisi" / dataset / f"fold_{fold}" / "predictions" / "transductive_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        trues.extend(df["true"].astype(str).tolist())
        preds.extend(df["pred"].astype(str).tolist())
    if not trues:
        return None
    return np.array(trues), np.array(preds)


def collect_physmap_preds(dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = CACHE / "physmap" / dataset / "fold_0" / "predictions" / "physmap_CV_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    return df["true"].astype(str).values, df["pred"].astype(str).values


def plot_cm(ax, y_true, y_pred, labels: list[str], title: str, cmap: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0).astype(float)

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)

    cutoff = 0.55
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            text_color = "white" if v >= cutoff else "#222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color)

    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    return im


def pair_for_dataset(dataset: str) -> None:
    info = DATASETS[dataset]
    classes = info["class_order"]

    hippie = collect_hippie_preds(dataset)
    physmap = collect_physmap_preds(dataset)
    if hippie is None:
        raise SystemExit(f"no HIPPIE preds for {dataset}")
    if physmap is None:
        raise SystemExit(f"no PhysMAP preds for {dataset}")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), facecolor="white")
    axes[0].set_facecolor("white")
    axes[1].set_facecolor("white")

    plot_cm(axes[0], *hippie, labels=classes,
            title=f"HIPPIE-bimodal — {info['display']}", cmap="Blues")
    plot_cm(axes[1], *physmap, labels=classes,
            title=f"PhysMAP (WNN) — {info['display']}", cmap="Blues")

    fig.tight_layout()
    stem = OUT_DIR / f"{dataset}_cm_pair"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")

    # individual CMs too
    for method_key, method_label, cmap in METHODS:
        y_true, y_pred = hippie if method_key == "hippie_waveisi" else physmap
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.3), facecolor="white")
        ax.set_facecolor("white")
        plot_cm(ax, y_true, y_pred, labels=classes,
                title=f"{method_label}\n{info['display']}", cmap=cmap)
        fig.tight_layout()
        stem = OUT_DIR / f"{dataset}_{method_key}_cm"
        fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[svg] {stem}.svg")
        print(f"[png] {stem}.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    for ds in DATASETS:
        pair_for_dataset(ds)


if __name__ == "__main__":
    main()
