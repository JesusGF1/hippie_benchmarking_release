"""Figure 5 panel C — confusion matrices for all methods, Lisberger→Hull.

Row-normalised confusion matrices for HIPPIE, HIPPIE+Pretraining, NEMO,
PhysMAP, VAE, and VAE (no pretrain).

Outputs:
    figures/29_figure_5_cross_species/panel_c_confusion.{svg,png}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "benchmark" / "cross_dataset_cache"
OUT_DIR = REPO_ROOT / "figures" / "figure_5"

DIRECTION = "cross_lisberger_labeled_cell_type_to_hull_cell_type"
SHARED = ["MFB", "MLI", "PkC_cs", "PkC_ss"]

PANELS = [
    ("PhysMAP",
     "physmap-cross-dataset",
     "predictions/physmap_cross_dataset_results.csv",
     "label", "prediction"),
    ("NEMO",
     "nemo-cross-dataset-allpretrain",
     "predictions/cross_dataset/cross_dataset_predictions.csv",
     "true", "pred"),
    ("VAE",
     "vae-cross-dataset",
     "predictions/predictions.csv",
     "true_label", "predicted_label"),
    ("HIPPIE",
     "hippie-cross-dataset-allpretrain",
     "predictions/predictions.csv",
     "true_label", "predicted_label"),
]


def _load(prefix: str, pred_rel: str, true_col: str, pred_col: str):
    p = CACHE / prefix / DIRECTION / pred_rel
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    tc, pc = true_col.lower(), pred_col.lower()
    if tc not in df.columns or pc not in df.columns:
        return None, None
    df = df[df[tc].isin(SHARED)]
    return df[tc].values, df[pc].values


def _plot_cm(ax, y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=SHARED).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    ba = balanced_accuracy_score(y_true, y_pred)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    for i in range(len(SHARED)):
        for j in range(len(SHARED)):
            v = cm_norm[i, j]
            color = "white" if v > 0.55 else "#222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    ax.set_xticks(range(len(SHARED)))
    ax.set_yticks(range(len(SHARED)))
    ax.set_xticklabels(SHARED, rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels(SHARED, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"{title}\n(BA = {ba:.2f})", fontsize=8.5, pad=4)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")


def _save(fig, stem: Path) -> None:
    for ext in ("svg", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"[{ext}] {p}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), facecolor="white")
    for ax, (method_label, prefix, pred_rel, true_col, pred_col) in zip(axes, PANELS):
        y_true, y_pred = _load(prefix, pred_rel, true_col, pred_col)
        if y_true is None:
            ax.text(0.5, 0.5, f"No data:\n{prefix}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.set_title(method_label)
            continue
        _plot_cm(ax, y_true, y_pred, method_label)

    fig.suptitle("Lisberger (macaque) → Hull (mouse) — confusion matrices",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, OUT_DIR / "panel_c_confusion")
    plt.close(fig)


if __name__ == "__main__":
    main()
