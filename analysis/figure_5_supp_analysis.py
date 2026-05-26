"""Figure 5 supplementary diagnostics — cross-species transfer.

Per-class diagnostics for the Lisberger ↔ Hull transfer setting:
  1. Per-class recall heatmaps across all HIPPIE variants × both directions
  2. GoC out-of-class prediction rate (the four shared cell types do not
     include GoC; this panel summarises how often each variant emits GoC)
  3. PkC_ss prediction pattern under HIPPIE+Pretrain+LayerCond (Liss→Hull)
  4. Hull→Lisberger (reverse direction) — bar chart + confusion matrices

Outputs to figures/figure_5/supplement/
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "benchmark" / "cross_dataset_cache"
OUT_DIR = REPO_ROOT / "figures" / "figure_5" / "supplement"

SHARED = ["MFB", "MLI", "PkC_cs", "PkC_ss"]

METHODS = [
    ("HIPPIE",                    "hippie-cross-dataset",                     "true_label", "predicted_label"),
    ("HIPPIE+LayerCond",          "hippie-cross-dataset-regioncond",           "true_label", "predicted_label"),
    ("HIPPIE+Pretraining",        "hippie-cross-dataset-allpretrain",          "true_label", "predicted_label"),
    ("HIPPIE+Pretrain+LayerCond", "hippie-cross-dataset-allpretrain-regioncond","true_label","predicted_label"),
]

DIRECTIONS = [
    ("cross_hull_cell_type_to_lisberger_labeled_cell_type",  "Hull→Liss"),
    ("cross_lisberger_labeled_cell_type_to_hull_cell_type",  "Liss→Hull"),
]


def _load_df(prefix: str, direction: str, tc: str, pc: str) -> pd.DataFrame | None:
    p = CACHE / prefix / direction / "predictions/predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    if tc.lower() not in df.columns or pc.lower() not in df.columns:
        return None
    df = df.rename(columns={tc.lower(): "true", pc.lower(): "pred"})
    return df


# ── 1. Per-class recall heatmaps ──────────────────────────────────────────

def plot_per_class_recall() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="white")
    fig.suptitle("Per-class recall: all HIPPIE variants × both directions", fontsize=11, y=1.01)

    for col, (label, prefix, tc, pc) in enumerate(METHODS):
        for row, (dir_key, dir_label) in enumerate(DIRECTIONS):
            ax = axes[row, col]
            df = _load_df(prefix, dir_key, tc, pc)
            if df is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{label}\n{dir_label}")
                continue

            df_s = df[df["true"].isin(SHARED)]
            cm = confusion_matrix(df_s["true"], df_s["pred"], labels=SHARED).astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums
            ba = balanced_accuracy_score(df_s["true"], df_s["pred"])

            im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
            for i in range(len(SHARED)):
                for j in range(len(SHARED)):
                    v = cm_norm[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="white" if v > 0.55 else "#222")

            ax.set_xticks(range(len(SHARED)))
            ax.set_yticks(range(len(SHARED)))
            ax.set_xticklabels(SHARED, rotation=40, ha="right", fontsize=6)
            ax.set_yticklabels(SHARED, fontsize=6)
            if row == 1:
                ax.set_xlabel("Predicted", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{dir_label}\nTrue", fontsize=7)
            ax.set_title(f"{label}\nBA={ba:.3f}", fontsize=7.5, pad=3)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"per_class_recall.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("[done] per_class_recall")


# ── 2. GoC out-of-class prediction rate ──────────────────────────────────

def print_goc_audit() -> None:
    print("\n=== GoC out-of-class prediction rate (shared-class evaluation) ===")
    for label, prefix, tc, pc in METHODS:
        for dir_key, dir_label in DIRECTIONS:
            df = _load_df(prefix, dir_key, tc, pc)
            if df is None:
                continue
            goc_preds = (df["pred"] == "GoC").sum()
            df_s = df[df["true"].isin(SHARED)]
            ba = balanced_accuracy_score(df_s["true"], df_s["pred"])
            print(f"  {label:<30} {dir_label}  GoC_preds={goc_preds:3d}  BA={ba:.3f}")


# ── 3. PkC_ss prediction pattern: HIPPIE+Pretrain+LayerCond Liss→Hull ────

def plot_pkc_pattern() -> None:
    prefix = "hippie-cross-dataset-allpretrain-regioncond"
    dir_key = "cross_lisberger_labeled_cell_type_to_hull_cell_type"
    df = _load_df(prefix, dir_key, "true_label", "predicted_label")
    if df is None:
        print("PkC_ss diagnostic: no data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
    fig.suptitle("HIPPIE+Pretrain+LayerCond  Liss→Hull — PkC_ss prediction breakdown",
                 fontsize=10, y=1.02)

    # Panel 1: full confusion matrix (all predicted classes)
    all_classes = sorted(df["pred"].unique())
    all_true    = sorted(df["true"].unique())
    cm = confusion_matrix(df["true"], df["pred"], labels=all_true + [c for c in all_classes if c not in all_true])
    df_s = df[df["true"].isin(SHARED)]
    cm_s = confusion_matrix(df_s["true"], df_s["pred"],
                            labels=SHARED + [c for c in df_s["pred"].unique() if c not in SHARED])
    all_pred_labels = SHARED + [c for c in df_s["pred"].unique() if c not in SHARED]
    row_s = cm_s.astype(float)
    rs = row_s.sum(1, keepdims=True); rs[rs == 0] = 1
    cm_norm = row_s / rs

    ax = axes[0]
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for i in range(len(SHARED)):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.55 else "#222")
    ax.set_xticks(range(cm_norm.shape[1]))
    ax.set_yticks(range(len(SHARED)))
    ax.set_xticklabels(all_pred_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(SHARED, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title("Full CM (SHARED true, all preds)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: prob_pkc_ss distribution if available
    ax2 = axes[1]
    prob_col = "prob_pkc_ss"
    if prob_col in df.columns:
        for cls in SHARED:
            sub = df[df["true"] == cls][prob_col]
            ax2.hist(sub, bins=20, alpha=0.6, label=cls, density=True)
        ax2.set_xlabel("Model P(PkC_ss)", fontsize=8)
        ax2.set_ylabel("Density", fontsize=8)
        ax2.set_title("PkC_ss probability by true class", fontsize=8)
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "prob_pkc_ss\nnot in CSV", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9)
        ax2.set_title("PkC_ss probability (unavailable)", fontsize=8)

    # Panel 3: prediction distribution for PkC_ss true neurons
    ax3 = axes[2]
    pkc_preds = df[df["true"] == "PkC_ss"]["pred"].value_counts()
    ax3.bar(pkc_preds.index, pkc_preds.values, color="#1f77b4")
    ax3.set_xlabel("Predicted class", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.set_title(f"Predictions for PkC_ss neurons (n={int(pkc_preds.sum())})", fontsize=8)
    ax3.tick_params(axis="x", rotation=30, labelsize=7)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"pkc_ss_pattern.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("[done] pkc_ss_pattern")


# ── 4. Summary BA table ───────────────────────────────────────────────────

def print_summary_table() -> None:
    print("\n=== Balanced-accuracy summary (shared four-class evaluation) ===")
    header = f"{'Method':<30} {'Hull→Liss':>12} {'Liss→Hull':>12}"
    print(header)
    print("-" * len(header))
    for label, prefix, tc, pc in METHODS:
        row = f"{label:<30}"
        for dir_key, _ in DIRECTIONS:
            df = _load_df(prefix, dir_key, tc, pc)
            if df is None:
                row += f"{'N/A':>12}"
                continue
            df_s = df[df["true"].isin(SHARED)]
            ba = balanced_accuracy_score(df_s["true"], df_s["pred"])
            row += f"{ba:>12.3f}"
        print(row)


ALL_METHODS_HULL_LISS = [
    ("PhysMAP",          "physmap-cross-dataset",          "predictions/physmap_cross_dataset_results.csv", "label",      "prediction"),
    ("VAE",              "vae-cross-dataset",              "predictions/predictions.csv",                  "true_label",  "predicted_label"),
    ("VAE\n+Pretraining","vae-cross-dataset-allpretrain",  "predictions/predictions.csv",                  "true_label",  "predicted_label"),
    ("HIPPIE",           "hippie-cross-dataset",           "predictions/predictions.csv",                  "true_label",  "predicted_label"),
    ("HIPPIE\n+Pretraining","hippie-cross-dataset-allpretrain","predictions/predictions.csv",              "true_label",  "predicted_label"),
]

HULL_LISS_DIR = "cross_hull_cell_type_to_lisberger_labeled_cell_type"

METHOD_COLORS = {
    "PhysMAP":            "#6a4c93",
    "VAE":                "#f4a261",
    "VAE\n+Pretraining":  "#e76f51",
    "HIPPIE":             "#2a9d8f",
    "HIPPIE\n+Pretraining": "#264653",
}


def _load_df2(prefix: str, rel: str, tc: str, pc: str, direction: str) -> pd.DataFrame | None:
    p = CACHE / prefix / direction / rel
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    tc, pc = tc.lower(), pc.lower()
    if tc not in df.columns or pc not in df.columns:
        return None
    return df.rename(columns={tc: "true", pc: "pred"})


# ── 4a. Hull→Liss bar chart ───────────────────────────────────────────────

def plot_hull_liss_bars() -> None:
    labels, bas = [], []
    for label, prefix, rel, tc, pc in ALL_METHODS_HULL_LISS:
        df = _load_df2(prefix, rel, tc, pc, HULL_LISS_DIR)
        if df is None:
            continue
        df_s = df[df["true"].isin(SHARED)]
        ba = balanced_accuracy_score(df_s["true"], df_s["pred"])
        labels.append(label)
        bas.append(ba)

    fig, ax = plt.subplots(figsize=(len(labels) * 1.4 + 0.8, 3.4), facecolor="white")
    colors = [METHOD_COLORS.get(l, "#888") for l in labels]
    bars = ax.bar(range(len(labels)), bas, color=colors, width=0.6, zorder=3)
    for bar, ba in zip(bars, bas):
        ax.text(bar.get_x() + bar.get_width() / 2, ba + 0.01, f"{ba:.2f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Balanced Accuracy", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Hull (mouse) → Lisberger (macaque) — balanced accuracy", fontsize=9, pad=6)
    ax.axhline(0.25, ls="--", lw=0.8, color="#aaa", zorder=2)
    ax.text(len(labels) - 0.5, 0.26, "chance", fontsize=7, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, zorder=0)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"hull_liss_bars.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("[done] hull_liss_bars")


# ── 4b. Hull→Liss confusion matrices ─────────────────────────────────────

def plot_hull_liss_confusion() -> None:
    available = [(l, p, r, tc, pc) for l, p, r, tc, pc in ALL_METHODS_HULL_LISS
                 if _load_df2(p, r, tc, pc, HULL_LISS_DIR) is not None]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, (label, prefix, rel, tc, pc) in zip(axes, available):
        df = _load_df2(prefix, rel, tc, pc, HULL_LISS_DIR)
        df_s = df[df["true"].isin(SHARED)]
        ba = balanced_accuracy_score(df_s["true"], df_s["pred"])
        cm = confusion_matrix(df_s["true"], df_s["pred"], labels=SHARED).astype(float)
        rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        cm_norm = cm / rs

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for i in range(len(SHARED)):
            for j in range(len(SHARED)):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.55 else "#222")
        ax.set_xticks(range(len(SHARED))); ax.set_yticks(range(len(SHARED)))
        ax.set_xticklabels(SHARED, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(SHARED, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(f"{label.replace(chr(10), ' ')}\n(BA = {ba:.2f})", fontsize=8.5, pad=4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    fig.suptitle("Hull (mouse) → Lisberger (macaque) — confusion matrices",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT_DIR / f"hull_liss_confusion.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("[done] hull_liss_confusion")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})

    print_summary_table()
    print_goc_audit()
    plot_per_class_recall()
    plot_pkc_pattern()
    plot_hull_liss_bars()
    plot_hull_liss_confusion()
    print(f"\nOutputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
