"""Figure 2 panel B primitive — Hausser mouse cerebellar cortex dataset profile.

Reads the locally cached unified Hippie dataset (hausser_cell_type), the dataset
all tools (HIPPIE, PhysMAP, NEMO) locked their hyperparameters against. This
is a **different** dataset from Hull (hull_cell_type) — Hull is reserved for
the Fig 5 cross-species train set, so tuning never saw it. Filters to the
labeled subset with ≥10 cells per class (matching the stability tiebreaker
described in the Methods). Emits a 5-panel profile:

    - Mean waveform per cell type (± SEM)
    - Mean ISI distribution per cell type (± SEM)
    - Mean autocorrelogram per cell type (± SEM)
    - Cell-type distribution bar chart (grey bars)
    - Dataset summary table

Outputs:
    figures/01_dataset_characterization/hull_fig2_profile.svg
    figures/01_dataset_characterization/hull_fig2_profile.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "hausser_cell_type"
OUT_DIR = REPO_ROOT / "figures" / "figure_2"
OUT_STEM = OUT_DIR / "hausser_fig2_profile"

DISPLAY_NAME = "Hausser"
MIN_CELLS_PER_CLASS = 10

# Paper-consistent per-class colours (kept for waveform/ISI/ACG panels).
CLASS_COLORS = {
    "GoC":    "#1f77b4",
    "MFB":    "#ff7f0e",
    "MLI":    "#2ca02c",
    "PkC_cs": "#d62728",
    "PkC_ss": "#9467bd",
}

# Light-grey palette for the class-distribution bars.
BAR_GREY = "#c8c8c8"
BAR_GREY_EDGE = "#888888"


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, list[str]]:
    labels = pd.read_csv(DATA_DIR / "labels.csv").iloc[:, 0]
    waveforms = pd.read_csv(DATA_DIR / "waveforms.csv").values
    isi = pd.read_csv(DATA_DIR / "isi_dist.csv").values
    acg = pd.read_csv(DATA_DIR / "acg.csv").values

    # drop the unnamed index column if present (first col all ints in same order)
    for arr_name, arr in [("waveforms", waveforms), ("isi", isi), ("acg", acg)]:
        pass

    # filter to labeled subset with ≥ MIN_CELLS_PER_CLASS per class
    mask_labeled = labels != "unlabeled"
    labels_lab = labels[mask_labeled]
    counts = labels_lab.value_counts()
    keep_classes = sorted([c for c, n in counts.items() if n >= MIN_CELLS_PER_CLASS])
    mask_final = labels.isin(keep_classes).values

    waveforms = waveforms[mask_final]
    isi = isi[mask_final]
    acg = acg[mask_final]
    labels = labels[mask_final].reset_index(drop=True)

    # normalise each row for visualisation
    waveforms = (waveforms - waveforms.mean(axis=1, keepdims=True)) / (
        waveforms.std(axis=1, keepdims=True) + 1e-8
    )
    isi = isi / (isi.sum(axis=1, keepdims=True) + 1e-8)
    acg = acg / (acg.sum(axis=1, keepdims=True) + 1e-8)

    return waveforms, isi, acg, labels, keep_classes


def _plot_modality(ax, data: np.ndarray, labels: pd.Series, classes: list[str],
                    title: str, xlabel: str, ylabel: str) -> None:
    x = np.arange(data.shape[1])
    for c in classes:
        m = (labels == c).values
        if m.sum() == 0:
            continue
        mean = data[m].mean(axis=0)
        sem = data[m].std(axis=0) / np.sqrt(m.sum())
        col = CLASS_COLORS.get(c, "#666666")
        ax.plot(x, mean, color=col, label=c, linewidth=1.3)
        ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.22, linewidth=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=1)


def _plot_distribution(ax, labels: pd.Series, classes: list[str]) -> None:
    counts = [int((labels == c).sum()) for c in classes]
    proportions = np.array(counts) / len(labels)
    x = np.arange(len(classes))
    bars = ax.bar(
        x, proportions,
        color=BAR_GREY,
        edgecolor=BAR_GREY_EDGE,
        linewidth=0.8,
        width=0.7,
    )
    for bar, n in zip(bars, counts):
        ax.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color="#222",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title("Cell-type distribution", fontsize=9, pad=4)
    ax.set_ylim(0, max(proportions) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    waveforms, isi, acg, labels, classes = load()

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.55, wspace=0.35,
        left=0.08, right=0.97, top=0.9, bottom=0.12,
    )
    ax_wf   = fig.add_subplot(gs[0, 0])
    ax_isi  = fig.add_subplot(gs[0, 1])
    ax_acg  = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    for ax in (ax_wf, ax_isi, ax_acg, ax_dist):
        ax.set_facecolor("white")

    _plot_modality(ax_wf,  waveforms, labels, classes,
                   "Mean waveform by cell type", "Time (samples)", "Amplitude (norm.)")
    _plot_modality(ax_isi, isi,       labels, classes,
                   "Mean ISI distribution",      "ISI bin",        "Density (norm.)")
    _plot_modality(ax_acg, acg,       labels, classes,
                   "Mean autocorrelogram",       "Lag (bins)",     "Density (norm.)")
    _plot_distribution(ax_dist, labels, classes)

    fig.suptitle(f"Dataset profile: {DISPLAY_NAME}", fontsize=11, fontweight="bold", y=0.98)

    svg = OUT_STEM.with_suffix(".svg")
    png = OUT_STEM.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


if __name__ == "__main__":
    main()
