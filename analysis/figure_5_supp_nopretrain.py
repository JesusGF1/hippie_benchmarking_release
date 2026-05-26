"""Supplemental Figure 5 — full cross-species method comparison (Lisberger → Hull).

Bar chart of KNN balanced accuracy for Lisberger (macaque) → Hull (mouse)
transfer, all methods including pretraining variants.  Mirrors the main
Figure 5 panel B and adds HIPPIE WF+3D ACG with and without pretraining.

Evaluation restricted to the four shared cell types: PkC_ss, PkC_cs, MLI, MFB
(chance = 0.25).  Results are loaded directly from cached prediction CSVs.

Output:
    figures/33_figure_5_supp_nopretrain/supp_nopretrain_bars.{svg,png}

Usage:
    python analysis/figure_5_supp_nopretrain.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE     = REPO_ROOT / "results" / "benchmark" / "cross_dataset_cache"
OUT_DIR   = REPO_ROOT / "figures" / "figure_5" / "supplement"
DIRECTION = "cross_lisberger_labeled_cell_type_to_hull_cell_type"

SHARED    = {"PkC_ss", "PkC_cs", "MLI", "MFB"}
CHANCE    = 1.0 / len(SHARED)

_BLUE = "#1f77b4"

# (display label, cache prefix, pred rel path, true col, pred col)
METHODS = [
    ("HIPPIE",                  "hippie-cross-dataset",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
    ("HIPPIE\n+Pretraining",    "hippie-cross-dataset-allpretrain",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
    ("NEMO",                    "nemo-cross-dataset",
     "predictions/cross_dataset/cross_dataset_predictions.csv",        "true", "pred"),
    ("NEMO\n+Pretraining",      "nemo-cross-dataset-allpretrain",
     "predictions/cross_dataset/cross_dataset_predictions.csv",        "true", "pred"),
    ("HIPPIE\nWF+3D ACG",       "hippie-wf3dacg-cross-dataset",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
    ("HIPPIE\nWF+3D ACG\n+Pretraining", "hippie-wf3dacg-cross-dataset-allpretrain",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
    ("PhysMAP\n(WNN)",          "physmap-cross-dataset",
     "predictions/physmap_cross_dataset_results.csv",                  "label", "prediction"),
    ("VAE\n+Pretraining",       "vae-cross-dataset-allpretrain",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
    ("VAE",                     "vae-cross-dataset",
     "predictions/predictions.csv",                                    "true_label", "predicted_label"),
]


def _load_ba(prefix: str, pred_rel: str, true_col: str, pred_col: str) -> float | None:
    p = CACHE / prefix / DIRECTION / pred_rel
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    tc, pc = true_col.lower(), pred_col.lower()
    if tc not in df.columns or pc not in df.columns:
        return None
    df = df[df[tc].isin(SHARED)]
    if df.empty:
        return None
    return balanced_accuracy_score(df[tc], df[pc])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor="white")

    x = np.arange(len(METHODS))

    for xi, (label, prefix, pred_rel, tc, pc) in enumerate(METHODS):
        ba = _load_ba(prefix, pred_rel, tc, pc)
        if ba is not None:
            ax.bar(xi, ba, width=0.65, color=_BLUE, edgecolor="white", linewidth=0.8, zorder=2)
            ax.text(xi, ba + 0.015, f"{ba:.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#333")
        else:
            ax.bar(xi, 0, width=0.65, color=_BLUE, edgecolor="white", linewidth=0.8,
                   zorder=2, alpha=0.3)
            ax.text(xi, 0.02, "N/A", ha="center", va="bottom", fontsize=7, color="#999")

    # chance line
    ax.axhline(CHANCE, color="#888", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(len(METHODS) - 0.5, CHANCE + 0.02, f"chance = {CHANCE:.2f}",
            ha="right", va="bottom", fontsize=7, color="#888")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in METHODS], fontsize=7)
    ax.set_ylabel("KNN balanced accuracy", fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_title("Lisberger (macaque) → Hull (mouse)", fontsize=9, pad=5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")

    fig.tight_layout()

    for ext in ("svg", "png"):
        out = OUT_DIR / f"supp_nopretrain_bars.{ext}"
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"[{ext}] {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
