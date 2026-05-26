"""Figure 4 — trimodal (WF+ISI+ACG) classification benchmark on Hull and Lisberger.

Reads cached prediction CSVs that ship with the repo, computes per-fold balanced
accuracy and macro F1, and generates bar charts and confusion matrices for:
    Hull mouse cerebellar cortex    (101 neurons, 4 cell types)
    Lisberger macaque cerebellar cortex (666 neurons, 5 cell types)

Methods compared:
    HIPPIE (trimodal WF+ISI+ACG), VAE (trimodal), NEMO, WF-RF (Buccino features),
    PhysMAP (WNN), PCA-WF, PCA-ISI, PCA-ACG

Data sources (already in the repo, no network access required):
    results/benchmark/trimodal_cache/<method>/<dataset>/fold_*/...
    results/negative_controls/<control>/hippie/<dataset>/run_*.csv

Outputs to figures/figure_4/:
    {dataset}_bars.{svg,png}      — bar chart (balanced acc + macro F1)
    {dataset}_cm_grid.{svg,png}   — confusion matrix grid (4 methods)
    trimodal_metrics.csv          — tidy fold-level metrics

Usage:
    python analysis/figure_4_trimodal_benchmark.py            # plot (default)
    python analysis/figure_4_trimodal_benchmark.py --plot     # same
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths — all data lives in the repo; no S3 access required.
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
CACHE_DIR   = REPO_ROOT / "results" / "benchmark" / "trimodal_cache"
NEGCTRL_DIR = REPO_ROOT / "results" / "negative_controls"
OUT_DIR     = REPO_ROOT / "figures" / "figure_4"

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASETS = {
    "hull_cell_type": {
        "display":        "Hull (mouse, cerebellar cortex)",
        "n_classes":      4,
        "chance":         0.25,
        "short":          "Hull",
        "species":        "Mouse",
        "brain_region":   "Cerebellar cortex",
        "recording_tech": "Neuropixel 1.0",
        "n_cells":        101,
    },
    "lisberger_labeled_cell_type": {
        "display":        "Lisberger (macaque, cerebellar cortex)",
        "n_classes":      5,
        "chance":         0.20,
        "short":          "Lisberger",
        "species":        "Macaque",
        "brain_region":   "Cerebellar cortex",
        "recording_tech": "Plexon S-Probes",
        "n_cells":        666,
    },
}

N_FOLDS = 5

# Methods available on both datasets
METHODS_BOTH = [
    # key, display name, cache_subdir
    ("hippie-fullcond", "HIPPIE",  "hippie-fullcond"),
    ("vae",           "VAE",                      "vae"),
    ("nemo",          "NEMO",                     "nemo"),
    ("wf-rf",       "WF-RF",                    "wf-rf"),
    ("physmap",       "PhysMAP",                  "physmap"),
    ("pca_wf",        "PCA-WF",                   "physmap"),
    ("pca_isi",       "PCA-ISI",                  "physmap"),
    ("pca_acg",       "PCA-ACG",                  "physmap"),
]

# Methods available on Hull only (none currently — NEMO now covers both)
METHODS_HULL_ONLY: list = []

# Confusion matrix methods (subset shown in CM panel)
CM_METHODS = ["physmap", "vae", "nemo", "hippie-fullcond"]

# Per-dataset class lists for confusion matrices
CM_CLASSES = {
    "hull_cell_type": ["MFB", "MLI", "PkC_cs", "PkC_ss"],           # GoC excluded (very rare, not in most folds)
    "lisberger_labeled_cell_type": ["GoC", "MFB", "MLI", "PkC_cs", "PkC_ss"],
}

# Plot colours
BAR_COLOR      = "#1f77b4"   # blue  — all real methods
NEGCTRL_COLOR  = "#ff7f0e"   # orange — shuffle / shift controls
CHANCE_COLOR   = "#888"      # gray  — chance line
DOT_COLOR      = "#1e3a5f"   # dark navy — individual fold dots

IS_NEGCTRL = {"shuffle", "shift"}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred) -> tuple[float, float]:
    return (
        balanced_accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def _collect_hippie_like(cache_subdir: str, dataset: str) -> list[dict]:
    rows = []
    for fold in range(N_FOLDS):
        p = CACHE_DIR / cache_subdir / dataset / f"fold_{fold}" / "transductive_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        acc, f1 = _metrics(df["true"], df["pred"])
        rows.append({"method": cache_subdir, "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1,
                     "y_true": list(df["true"]), "y_pred": list(df["pred"])})
    return rows


def _collect_nemo(dataset: str) -> list[dict]:
    rows = []
    # Only evaluate on classes shown in the CM — keeps metrics consistent with
    # the confusion matrix display. For Hull, GoC appears in NEMO's test splits
    # but not in other methods', so we restrict to the shared 4-class set.
    cm_classes = CM_CLASSES.get(dataset)
    pred_fname = "mlp_probe_predictions.csv"
    for fold in range(N_FOLDS):
        p = CACHE_DIR / "nemo" / dataset / f"fold_{fold}" / pred_fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        if cm_classes is not None:
            df = df[df["true"].isin(cm_classes)]
        acc, f1 = _metrics(df["true"], df["pred"])
        rows.append({"method": "nemo", "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1,
                     "y_true": list(df["true"]), "y_pred": list(df["pred"])})
    return rows


def _collect_physmap_file(method_key: str, fname: str, dataset: str) -> list[dict]:
    p = CACHE_DIR / "physmap" / dataset / "fold_0" / fname
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    if not {"pred", "true", "fold"}.issubset(df.columns):
        return []
    rows = []
    for fold_name, sub in df.groupby("fold"):
        fold_idx = int(str(fold_name).lower().lstrip("fold")) - 1
        acc, f1 = _metrics(sub["true"], sub["pred"])
        rows.append({"method": method_key, "dataset": dataset, "fold": fold_idx,
                     "balanced_accuracy": acc, "macro_f1": f1,
                     "y_true": list(sub["true"]), "y_pred": list(sub["pred"])})
    return rows


def _collect_negctrl(control: str, dataset: str) -> list[dict]:
    base = NEGCTRL_DIR / control / "hippie" / dataset
    rows = []
    for run_file in sorted(base.glob("run_*.csv")):
        df = pd.read_csv(run_file)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        acc, f1 = _metrics(df["true"], df["pred"])
        fold = int(run_file.stem.split("_")[1]) - 1
        rows.append({"method": control, "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1,
                     "y_true": list(df["true"]), "y_pred": list(df["pred"])})
    return rows


def collect_all() -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in DATASETS:
        rows += _collect_hippie_like("hippie-fullcond", dataset)
        rows += _collect_hippie_like("vae",             dataset)
        rows += _collect_hippie_like("wf-rf",        dataset)
        rows += _collect_physmap_file("physmap", "physmap_CV_results.csv",      dataset)
        rows += _collect_physmap_file("pca_wf",  "PCA_WF_CV_results.csv",       dataset)
        rows += _collect_physmap_file("pca_isi", "PCA_ISI_CV_results.csv",       dataset)
        rows += _collect_physmap_file("pca_acg", "PCA_AUTOCORR_CV_results.csv",  dataset)

    # NEMO: both Hull and Lisberger
    rows += _collect_nemo("hull_cell_type")
    rows += _collect_nemo("lisberger_labeled_cell_type")

    # Negative controls
    for dataset in DATASETS:
        rows += _collect_negctrl("shuffle", dataset)
        rows += _collect_negctrl("shift",   dataset)

    if not rows:
        raise SystemExit(
            f"No prediction files found under {CACHE_DIR}. "
            f"The repo ships these caches at results/benchmark/trimodal_cache/; "
            f"check that the directory exists and is populated."
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

# Method display order (best first is handled per-dataset at plot time)
METHOD_DISPLAY = {
    "hippie-fullcond": "HIPPIE",
    "vae":             "VAE",
    "nemo":            "NEMO",
    "wf-rf":         "WF-RF",
    "physmap":         "PhysMAP",
    "pca_wf":          "PCA-WF",
    "pca_isi":         "PCA-ISI",
    "pca_acg":         "PCA-ACG",
    "shuffle":         "Shuffle",
    "shift":           "Shift",
}


def _axis_style(ax: plt.Axes, ylabel: str, chance: float) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(chance, color=CHANCE_COLOR, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(
        0.995, chance + 0.012, f"chance = {chance:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color=CHANCE_COLOR,
    )


def _draw_info_table(ax: plt.Axes, info: dict) -> None:
    ax.axis("off")
    rows = [
        ["Species",    info["species"]],
        ["Region",     info["brain_region"]],
        ["Recording",  info["recording_tech"]],
        ["N cells",    str(info["n_cells"])],
        ["Cell types", str(info["n_classes"])],
    ]
    tbl = ax.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.42, 0.58],
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.06
        if col == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold", color="#333")
        else:
            cell.set_facecolor("white")


def plot_bars(df: pd.DataFrame, dataset: str) -> None:
    info = DATASETS[dataset]
    chance = info["chance"]

    # Determine which methods have data for this dataset
    ds_df = df[df["dataset"] == dataset]
    available = [m for m in METHOD_DISPLAY if not ds_df[ds_df["method"] == m].empty]

    # Sort: real methods best-first, negctrls appended last
    real_methods = [m for m in available if m not in IS_NEGCTRL]
    ctrl_methods = [m for m in available if m in IS_NEGCTRL]
    mean_acc = {m: ds_df[ds_df["method"] == m]["balanced_accuracy"].mean() for m in available}
    real_sorted = sorted(real_methods, key=lambda m: mean_acc.get(m, 0), reverse=True)
    available_sorted = real_sorted + ctrl_methods

    labels = [METHOD_DISPLAY[m] for m in available_sorted]
    n = len(available_sorted)
    rng = np.random.default_rng(7)

    bar_w = max(6, n * 0.9)
    fig = plt.figure(figsize=(bar_w + 2.6, 3.6), facecolor="white")
    fig.suptitle(info["display"], fontsize=10, fontweight="bold")

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[2.6 / (bar_w + 2.6), bar_w / 2 / (bar_w + 2.6), bar_w / 2 / (bar_w + 2.6)],
        wspace=0.42,
        left=0.02, right=0.97, top=0.86, bottom=0.22,
    )
    ax_tbl  = fig.add_subplot(gs[0, 0])
    ax_acc  = fig.add_subplot(gs[0, 1])
    ax_f1   = fig.add_subplot(gs[0, 2])
    for ax in (ax_tbl, ax_acc, ax_f1):
        ax.set_facecolor("white")

    _draw_info_table(ax_tbl, info)

    for ax, metric, ylabel in zip([ax_acc, ax_f1],
                                   ["balanced_accuracy", "macro_f1"],
                                   ["Balanced accuracy", "Macro F1"]):
        for xi, mkey in enumerate(available_sorted):
            sub = ds_df[ds_df["method"] == mkey][metric].values
            if len(sub) == 0:
                continue
            mean = float(np.mean(sub))
            sem  = float(np.std(sub, ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
            is_ctrl = mkey in IS_NEGCTRL
            color = NEGCTRL_COLOR if is_ctrl else BAR_COLOR
            hatch = "////" if is_ctrl else None
            ec = "#555" if is_ctrl else "white"
            ax.bar(xi, mean, yerr=sem, color=color, edgecolor=ec, hatch=hatch,
                   linewidth=0.8, capsize=3, width=0.65,
                   error_kw={"elinewidth": 0.8, "ecolor": "#333"}, zorder=2)
            ax.scatter(
                xi + rng.uniform(-0.08, 0.08, size=len(sub)),
                sub, s=14, color=DOT_COLOR, zorder=3, linewidth=0,
            )
            ax.text(xi, mean + max(sem, 0.01) + 0.015,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        _axis_style(ax, ylabel, chance)

    _save_fig(fig, OUT_DIR / f"{dataset}_bars")


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def _pool_predictions(df: pd.DataFrame, method: str, dataset: str) -> tuple[list, list] | None:
    sub = df[(df["method"] == method) & (df["dataset"] == dataset)]
    if sub.empty:
        return None
    y_true = [y for row in sub["y_true"] for y in row]
    y_pred = [y for row in sub["y_pred"] for y in row]
    return y_true, y_pred


def plot_confusion_matrices(df: pd.DataFrame, dataset: str) -> None:
    info = DATASETS[dataset]

    # Determine which CM methods have data
    available_cm = [m for m in CM_METHODS
                    if not df[(df["method"] == m) & (df["dataset"] == dataset)].empty]
    if not available_cm:
        print(f"  No CM data for {dataset}")
        return

    # Collect pooled class labels (used as fallback when dataset not in CM_CLASSES)
    all_labels: set[str] = set()
    for m in available_cm:
        sub = df[(df["method"] == m) & (df["dataset"] == dataset)]
        for row in sub["y_true"]:
            all_labels.update(row)
        for row in sub["y_pred"]:
            all_labels.update(row)
    classes = CM_CLASSES.get(dataset, sorted(all_labels))
    n_cls = len(classes)

    n_methods = len(available_cm)
    fig, axes = plt.subplots(1, n_methods,
                              figsize=(3.2 * n_methods, 3.2),
                              facecolor="white")
    if n_methods == 1:
        axes = [axes]

    for ax, mkey in zip(axes, available_cm):
        result = _pool_predictions(df, mkey, dataset)
        if result is None:
            ax.set_visible(False)
            continue
        y_true, y_pred = result
        cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(METHOD_DISPLAY.get(mkey, mkey), fontsize=9, fontweight="bold")

        for i in range(n_cls):
            for j in range(n_cls):
                val = cm[i, j]
                color = "white" if val > 0.55 else "#222"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(f"{info['short']} — confusion matrices (pooled folds)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, OUT_DIR / f"{dataset}_cm_grid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 — trimodal benchmark")
    parser.add_argument("--plot", action="store_true",
                        help="Generate figures (default action)")
    parser.parse_args()  # only --plot is supported; kept for backward compatibility

    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_all()

    # Drop y_true/y_pred from metrics CSV (only keep scalars)
    metrics_df = df.drop(columns=["y_true", "y_pred"])
    csv = OUT_DIR / "trimodal_metrics.csv"
    metrics_df.to_csv(csv, index=False)
    print(f"\n[csv] {csv}")

    summary = (
        metrics_df
        .groupby(["dataset", "method"])[["balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n", summary)

    for ds in DATASETS:
        plot_bars(df, ds)
        plot_confusion_matrices(df, ds)

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
