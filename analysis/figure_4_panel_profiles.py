"""Figure 4 — trimodal (WF+ISI+ACG) dataset profiles for Hull and Lisberger.

Two datasets:
    Hull (hull_cell_type): Mouse, cerebellar cortex, silicon probe, Hull et al. 2022
    Lisberger (lisberger_labeled_cell_type): Macaque, cerebellar cortex, Utah array

Both ship waveforms.csv, isi_dist.csv, acg.csv, labels.csv under
results/benchmark/cache_datasets/.

Layout per dataset: wide figure (14 × 5.5), 2 rows × 4 columns
    Row 0: waveform | ISI | ACG | cell-type distribution
    Row 1: summary table spanning all 4 cols

Outputs:
    figures/30_figure_4_profiles/{dataset_key}_fig4_profile.{svg,png}

Run as:
    python analysis/figure_4_panel_profiles.py            # both
    python analysis/figure_4_panel_profiles.py hull       # Hull only
    python analysis/figure_4_panel_profiles.py lisberger  # Lisberger only
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "results" / "benchmark" / "cache_datasets"
OUT_DIR   = REPO_ROOT / "figures" / "figure_4"

DATASETS = {
    "hull": {
        "key":            "hull_cell_type",
        "display":        "Hull (mouse, cerebellar cortex)",
        "species":        "Mouse",
        "brain_region":   "Cerebellar cortex",
        "recording_tech": "Neuropixel 1.0",
        "class_order":    ["GoC", "MFB", "MLI", "PkC_cs", "PkC_ss"],
    },
    "lisberger": {
        "key":            "lisberger_labeled_cell_type",
        "display":        "Lisberger (macaque, cerebellar cortex)",
        "species":        "Macaque",
        "brain_region":   "Cerebellar cortex",
        "recording_tech": "Plexon S-Probes",
        "class_order":    ["GoC", "MFB", "MLI", "PkC_cs", "PkC_ss"],
    },
}

LABEL_COLORS = {
    "PkC_ss": "#1f77b4",
    "PkC_cs": "#ff7f0e",
    "GoC":    "#2ca02c",
    "MLI":    "#d62728",
    "MFB":    "#9467bd",
}

BAR_GREY      = "#c8c8c8"
BAR_GREY_EDGE = "#888888"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_numeric(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0].lower() in {"unnamed: 0", ""} or df.columns[0] == "0":
        df = df.iloc[:, 1:]
    return df.values.astype(np.float32)


def _strip_byte_labels(s: pd.Series) -> pd.Series:
    """Strip byte-string encoding like b'GoC' → GoC."""
    return s.str.replace("b'", "", regex=False).str.replace("'", "", regex=False)


def load(dataset_key: str, class_order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, list[str]]:
    data_dir = DATA_ROOT / dataset_key
    labels = pd.read_csv(data_dir / "labels.csv").iloc[:, 0].astype(str)
    labels = _strip_byte_labels(labels)

    waveforms = _read_numeric(data_dir / "waveforms.csv")
    isi       = _read_numeric(data_dir / "isi_dist.csv")
    acg       = _read_numeric(data_dir / "acg.csv")

    # Filter to labeled classes only
    mask = labels.isin(class_order).values
    waveforms = waveforms[mask]
    isi       = isi[mask]
    acg       = acg[mask]
    labels    = labels[mask].reset_index(drop=True)

    # Normalise
    waveforms = (waveforms - waveforms.mean(axis=1, keepdims=True)) / (
        waveforms.std(axis=1, keepdims=True) + 1e-8
    )
    isi = isi / (isi.sum(axis=1, keepdims=True) + 1e-8)
    acg = acg / (acg.sum(axis=1, keepdims=True) + 1e-8)

    classes = [c for c in class_order if c in labels.values]
    return waveforms, isi, acg, labels, classes


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_modality(ax: plt.Axes, data: np.ndarray, labels: pd.Series,
                   classes: list[str], title: str, xlabel: str, ylabel: str) -> None:
    x = np.arange(data.shape[1])
    for c in classes:
        m = (labels == c).values
        if m.sum() == 0:
            continue
        mean = data[m].mean(axis=0)
        sem  = data[m].std(axis=0) / np.sqrt(m.sum())
        col  = LABEL_COLORS.get(c, "#666666")
        ax.plot(x, mean, color=col, label=c, linewidth=1.3)
        ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.22, linewidth=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=7, frameon=False, ncol=1)


def _plot_distribution(ax: plt.Axes, labels: pd.Series, classes: list[str]) -> None:
    counts      = [int((labels == c).sum()) for c in classes]
    proportions = np.array(counts) / len(labels)
    x           = np.arange(len(classes))
    bars = ax.bar(
        x, proportions,
        color=BAR_GREY, edgecolor=BAR_GREY_EDGE,
        linewidth=0.8, width=0.7,
    )
    for bar, n in zip(bars, counts):
        ax.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, color="#222",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title("Cell-type distribution", fontsize=9, pad=4)
    ax.set_ylim(0, max(proportions) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def _plot_summary_table(ax: plt.Axes, info: dict, n_cells: int,
                        classes: list[str], counts: dict[str, int]) -> None:
    ax.axis("off")
    ax.set_title("Dataset summary", fontsize=9, pad=4)

    cell_text = [
        ["Dataset",         f"{info['display']} ({info['key']})"],
        ["Species",         info["species"]],
        ["Brain region",    info["brain_region"]],
        ["Recording tech",  info["recording_tech"]],
        ["N labeled cells", f"{n_cells}"],
        ["N cell types",    f"{len(classes)}"],
        ["Cell types",      ", ".join(classes)],
        ["Per-class N",     ", ".join(f"{c}: {counts[c]}" for c in classes)],
    ]

    table = ax.table(
        cellText=cell_text,
        cellLoc="left",
        colWidths=[0.30, 0.70],
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#999999")
        cell.PAD = 0.06
        if col == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold", color="#222")
        else:
            cell.set_facecolor("white")


# ---------------------------------------------------------------------------
# Per-dataset profile
# ---------------------------------------------------------------------------

def make_profile(which: str) -> None:
    info = DATASETS[which]
    key  = info["key"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    waveforms, isi, acg, labels, classes = load(key, info["class_order"])
    n_cells = len(labels)
    counts  = {c: int((labels == c).sum()) for c in classes}

    fig = plt.figure(figsize=(14, 3.2), facecolor="white")
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        wspace=0.35,
        left=0.05, right=0.98, top=0.85, bottom=0.18,
    )
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_wf   = fig.add_subplot(gs[0, 1])
    ax_isi  = fig.add_subplot(gs[0, 2])
    ax_acg  = fig.add_subplot(gs[0, 3])
    for ax in (ax_dist, ax_wf, ax_isi, ax_acg):
        ax.set_facecolor("white")

    _plot_distribution(ax_dist, labels, classes)
    _plot_modality(ax_wf,  waveforms, labels, classes,
                   "Mean waveform", "Time (samples)", "Amplitude (z-score)")
    _plot_modality(ax_isi, isi,       labels, classes,
                   "Mean ISI distribution", "ISI bin", "Density (norm.)")
    _plot_modality(ax_acg, acg,       labels, classes,
                   "Mean ACG distribution", "Lag bin", "Density (norm.)")

    fig.suptitle(f"Dataset profile: {info['display']}",
                 fontsize=11, fontweight="bold", y=0.98)

    stem = OUT_DIR / f"{key}_fig4_profile"
    svg  = stem.with_suffix(".svg")
    png  = stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update({
        "font.family":        "Arial",
        "font.size":          8,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
    })

    args    = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = args if args else list(DATASETS.keys())
    for t in targets:
        if t not in DATASETS:
            raise SystemExit(f"unknown dataset: {t!r} (choose from {list(DATASETS)})")
        make_profile(t)


if __name__ == "__main__":
    main()
