"""Figure 7 — Brain region classification benchmark (Allen + IBL).

Reads:
    results/benchmark/brain_region_benchmark.csv
    results/benchmark/benchmark_cache/hippie/{dataset}/fold_0/predictions/
        transductive_predictions.csv  (for region label counts)

Emits in figures/figure_7/:
    allen_region_profile.{svg,png}  — Allen region proportions bar chart
    ibl_region_profile.{svg,png}    — IBL region proportions bar chart
    allen_bars.{svg,png}            — Allen transductive + holdout (4 sub-panels)
    ibl_bars.{svg,png}              — IBL transductive + holdout (4 sub-panels)

Run as:
    python analysis/figure_7_brain_region_benchmark.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH  = REPO_ROOT / "results" / "benchmark" / "brain_region_benchmark.csv"
CACHE_DIR = REPO_ROOT / "results" / "benchmark" / "benchmark_cache"
OUT_DIR   = REPO_ROOT / "figures" / "figure_7"

ALLEN = "allen_scope_neuropixel_area_subset"
IBL   = "ibl_brainwide_good"

DATASET_DISPLAY = {
    ALLEN: "Allen (mouse, 19 regions, neuropixels)",
    IBL:   "IBL (mouse, 10 regions, neuropixels)",
}
DATASET_TITLE = {
    ALLEN: "Allen Institute Visual Coding dataset",
    IBL:   "IBL Brainwide Map",
}
DATASET_N_CLASSES = {ALLEN: 19, IBL: 10}
CHANCE = {ALLEN: 1 / 19, IBL: 1 / 10}

BAR_COLOR     = "#1f77b4"
PENDING_COLOR = "#d0d0d0"
BAR_GREY      = "#c8c8c8"
BAR_GREY_EDGE = "#888888"

# (key, display_label, classifier)
# PhysMAP/PCA methods use knn_best (only classifier available for transductive);
# hippie_regioncond/NEMO use knn across 5 folds.
# HIPPIE is region-conditioned ("hippie_regioncond") for Allen and unconditioned
# ("hippie") for IBL — resolved per dataset via HIPPIE_METHOD below.
# "vae" = pure VAE baseline.
# Methods shown in transductive panels: neural methods + PCA/PhysMAP baselines
METHODS_TRANSDUCTIVE: list[tuple[str, str, str]] = [
    ("hippie_regioncond", "HIPPIE (KNN)",   "knn"),
    ("hippie_regioncond", "HIPPIE (MLP)",   "mlp"),
    ("vae",               "VAE",            "knn"),
    ("wf-rf",           "WF-RF",        "knn"),
    ("nemo",              "NEMO (KNN)",     "knn"),
    ("nemo",              "NEMO (MLP)",     "mlp"),
    ("physmap",           "PhysMAP (WNN)",  "knn_cv_mean"),
    ("pca_wf",            "PCA-WF",         "knn_cv_mean"),
    ("pca_isi",           "PCA-ISI",        "knn_cv_mean"),
    ("pca_acg",           "PCA-ACG",        "knn_cv_mean"),
]

# Methods shown in holdout (inductive) panels.
METHODS_HOLDOUT: list[tuple[str, str, str]] = [
    ("hippie_regioncond", "HIPPIE (KNN)",  "knn"),
    ("hippie_regioncond", "HIPPIE (MLP)",  "mlp"),
    ("vae",               "VAE",           "knn"),
    ("wf-rf",           "WF-RF",       "knn"),
    ("nemo",              "NEMO (KNN)",    "knn"),
    ("nemo",              "NEMO (MLP)",    "mlp"),
]

AVAILABLE: dict[tuple[str, str], set[str]] = {
    (ALLEN, "transductive"): {"hippie_regioncond", "vae", "wf-rf", "nemo", "physmap", "pca_wf", "pca_isi", "pca_acg"},
    (ALLEN, "holdout"):      {"hippie_regioncond", "vae", "wf-rf", "nemo"},
    (IBL,   "transductive"): {"hippie_regioncond", "vae", "wf-rf", "nemo", "physmap", "pca_wf", "pca_isi", "pca_acg"},
    (IBL,   "holdout"):      {"hippie_regioncond", "vae", "wf-rf", "nemo"},
}

# HIPPIE is region-conditioned only for Allen (broad super-region is realistic
# probe-targeting metadata there). IBL uses the unconditioned model by design,
# so its HIPPIE bars and region counts come from the plain "hippie" results.
HIPPIE_METHOD = {ALLEN: "hippie_regioncond", IBL: "hippie"}
HIPPIE_CACHE  = {ALLEN: "hippie-regioncond", IBL: "hippie"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_region_counts(dataset: str) -> dict[str, int]:
    pred_path = (CACHE_DIR / HIPPIE_CACHE[dataset] / dataset / "fold_0"
                 / "predictions" / "transductive_predictions.csv")
    df = pd.read_csv(pred_path)
    return df["true"].value_counts().to_dict()


def _collect(df: pd.DataFrame, dataset: str, split_type: str,
             method_key: str, classifier: str, metric: str) -> np.ndarray:
    sub = df[
        (df["dataset"]    == dataset) &
        (df["split_type"] == split_type) &
        (df["method"]     == method_key) &
        (df["classifier"] == classifier)
    ][metric].values
    return sub.astype(float)


# ---------------------------------------------------------------------------
# Axis style helpers
# ---------------------------------------------------------------------------

def _axis_style(ax: plt.Axes, ylabel: str, chance: float) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(chance, color="#666", linestyle="--", linewidth=0.9, zorder=1)
    # Drawn above the bars on an opaque patch: the label sits at the height of
    # the chance line, where bars taller than chance would otherwise hide it.
    ax.text(0.995, chance + 0.02, f"chance = {chance:.2f}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7, color="#444", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="square,pad=0.25", alpha=0.9))


def _plot_bars_single(ax: plt.Axes, df: pd.DataFrame, dataset: str,
                      split_type: str, metric: str, ylabel: str,
                      methods: list) -> None:
    chance    = CHANCE[dataset]
    available = AVAILABLE.get((dataset, split_type), set())
    rng = np.random.default_rng(7)

    for xi, (mkey, mlabel, mclf) in enumerate(methods):
        if mkey not in available:
            ax.bar(xi, 0.025, color=PENDING_COLOR, edgecolor="#888",
                   linewidth=0.8, hatch="////", width=0.65, zorder=2)
            ax.text(xi, 0.030, "pending", ha="center", va="bottom",
                    fontsize=6, color="#888", style="italic")
            continue

        collect_key = HIPPIE_METHOD[dataset] if mkey == "hippie_regioncond" else mkey
        vals = _collect(df, dataset, split_type, collect_key, mclf, metric)
        if len(vals) == 0:
            ax.bar(xi, 0.025, color=PENDING_COLOR, edgecolor="#888",
                   linewidth=0.8, hatch="////", width=0.65, zorder=2)
            ax.text(xi, 0.030, "pending", ha="center", va="bottom",
                    fontsize=6, color="#888", style="italic")
            continue

        mean = float(np.mean(vals))
        sem  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(xi, mean, yerr=sem, color=BAR_COLOR, edgecolor="white",
               linewidth=0.8, capsize=3, width=0.65,
               error_kw={"elinewidth": 0.8, "ecolor": "#333"}, zorder=2)
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xi + jitter, vals, s=14, color="#1e3a5f", zorder=3, linewidth=0)

    _axis_style(ax, ylabel, chance)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([m[1] for m in methods], rotation=45, ha="right", fontsize=7)


# ---------------------------------------------------------------------------
# Panel: combined region proportions (Allen + IBL side-by-side, one SVG)
# ---------------------------------------------------------------------------

def _draw_region_bars(ax: plt.Axes, dataset: str) -> None:
    counts  = _load_region_counts(dataset)
    regions = sorted(counts.keys(), key=lambda r: -counts[r])
    values  = [counts[r] for r in regions]
    total   = sum(values)
    props   = [v / total for v in values]

    x    = np.arange(len(regions))
    bars = ax.bar(x, props, color=BAR_GREY, edgecolor=BAR_GREY_EDGE,
                  linewidth=0.8, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title(f"Brain region distribution\n{DATASET_DISPLAY[dataset]}",
                 fontsize=8.5, pad=4)
    ax.set_ylim(0, max(props) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def plot_combined_region_profiles() -> None:
    """Emit a single SVG with Allen (left, 19 bars) and IBL (right, 10 bars)
    side-by-side, column widths proportional to number of regions."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_allen = len(_load_region_counts(ALLEN))
    n_ibl   = len(_load_region_counts(IBL))

    fig = plt.figure(figsize=(11, 3.0), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[n_allen, n_ibl],
        wspace=0.38,
        left=0.05, right=0.99, top=0.82, bottom=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _draw_region_bars(ax_a, ALLEN)
    _draw_region_bars(ax_b, IBL)

    stem = OUT_DIR / "region_profiles_combined"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")


# ---------------------------------------------------------------------------
# Panel: region proportions (individual, kept for reference)
# ---------------------------------------------------------------------------

def plot_region_profile(dataset: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts  = _load_region_counts(dataset)
    regions = sorted(counts.keys(), key=lambda r: -counts[r])
    values  = [counts[r] for r in regions]
    total   = sum(values)
    props   = [v / total for v in values]

    # Width scaled to number of regions so bar widths look similar across datasets
    fig_w = 3.0 + len(regions) * 0.55
    fig, ax = plt.subplots(figsize=(fig_w, 3.2), facecolor="white")
    ax.set_facecolor("white")

    x    = np.arange(len(regions))
    bars = ax.bar(x, props, color=BAR_GREY, edgecolor=BAR_GREY_EDGE,
                  linewidth=0.8, width=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title(f"Brain region distribution: {DATASET_DISPLAY[dataset]}",
                 fontsize=9, pad=4)
    ax.set_ylim(0, max(props) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    stem_key = "allen" if dataset == ALLEN else "ibl"
    stem = OUT_DIR / f"{stem_key}_region_profile"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")


# ---------------------------------------------------------------------------
# Panel: benchmark bars (transductive + holdout side by side)
# ---------------------------------------------------------------------------

def plot_bars(df: pd.DataFrame, dataset: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transductive has 8 methods, holdout has 4 — use proportional column widths
    fig = plt.figure(figsize=(14, 3.9), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 5, figure=fig,
        width_ratios=[2.5, 2.5, 0.06, 1.25, 1.25],
        left=0.05, right=0.99, top=0.80, bottom=0.28,
        wspace=0.45,
    )

    ax_tb = fig.add_subplot(gs[0, 0])  # transductive balanced accuracy
    ax_tf = fig.add_subplot(gs[0, 1])  # transductive macro F1
    # gs[0, 2] is the spacer
    ax_hb = fig.add_subplot(gs[0, 3])  # holdout balanced accuracy
    ax_hf = fig.add_subplot(gs[0, 4])  # holdout macro F1

    for ax in (ax_tb, ax_tf, ax_hb, ax_hf):
        ax.set_facecolor("white")

    _plot_bars_single(ax_tb, df, dataset, "transductive", "balanced_accuracy", "Balanced accuracy", METHODS_TRANSDUCTIVE)
    _plot_bars_single(ax_tf, df, dataset, "transductive", "macro_f1",          "Macro F1",          METHODS_TRANSDUCTIVE)
    _plot_bars_single(ax_hb, df, dataset, "holdout",      "balanced_accuracy", "Balanced accuracy", METHODS_HOLDOUT)
    _plot_bars_single(ax_hf, df, dataset, "holdout",      "macro_f1",          "Macro F1",          METHODS_HOLDOUT)

    ax_tb.set_title("Transductive - Balanced Accuracy",   fontsize=8, pad=4)
    ax_tf.set_title("Transductive - Macro F1",            fontsize=8, pad=4)
    ax_hb.set_title("Holdout - Balanced Accuracy",        fontsize=8, pad=4)
    ax_hf.set_title("Holdout - Macro F1",                 fontsize=8, pad=4)

    fig.suptitle(DATASET_TITLE[dataset], fontsize=10, y=1.01)

    stem_key = "allen" if dataset == ALLEN else "ibl"
    stem = OUT_DIR / f"{stem_key}_bars"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update({
        "font.family":       "Arial",
        "font.size":         8,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    plot_combined_region_profiles()
    for dataset in [ALLEN, IBL]:
        plot_region_profile(dataset)
        plot_bars(df, dataset)


if __name__ == "__main__":
    main()
