"""Figure 3 panel A/B primitive — bimodal (WF+ISI) dataset profiles.

Two datasets: A1 (a1data_remove_undef, Lakunina 2020) and S1
(juxtacellular_mouse_s1_area, Yu 2019). Both ship waveforms.csv +
isi_dist.csv + labels.csv under results/benchmark/cache_datasets/.

Emits a compact 2-row panel per dataset:
    - Mean waveform by cell type (± SEM)
    - Mean ISI distribution by cell type (± SEM)
    - Cell-type distribution bar chart (light grey bars)
    - Dataset summary table

Outputs:
    figures/23_figure_3_profiles/<dataset>_fig3_profile.{svg,png}

Run as:
    python analysis/figure_3_panel_ab_profile.py            # both
    python analysis/figure_3_panel_ab_profile.py a1         # A1 only
    python analysis/figure_3_panel_ab_profile.py s1         # S1 only
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
OUT_DIR = REPO_ROOT / "figures" / "figure_3"

DATASETS = {
    "a1": {
        "key": "a1data_remove_undef",
        "display": "A1 (Lakunina)",
        "species": "Mouse",
        "brain_region": "Auditory cortex (A1)",
        "recording_tech": "Silicon probe",
        "class_order": ["EXC", "PV", "SOM"],
    },
    "s1": {
        "key": "juxtacellular_mouse_s1_area",
        "display": "S1 (Yu)",
        "species": "Mouse",
        "brain_region": "Somatosensory cortex (S1)",
        "recording_tech": "Juxtacellular",
        "class_order": ["E_L4", "E_L5", "FS_L4", "FS_L5", "SOM"],
    },
}

CLASS_COLORS = {
    # A1
    "EXC": "#1f77b4",
    "PV":  "#d62728",
    "SOM": "#2ca02c",
    # S1
    "E_L4": "#ff7f0e",
    "E_L5": "#2ca02c",
    "FS_L4": "#d62728",
    "FS_L5": "#9467bd",
}

BAR_GREY = "#c8c8c8"
BAR_GREY_EDGE = "#888888"


def _read_numeric(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0].lower() in {"unnamed: 0", ""} or df.columns[0] == "0":
        df = df.iloc[:, 1:]
    return df.values.astype(np.float32)


def load(dataset_key: str, class_order: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    data_dir = DATA_ROOT / dataset_key
    labels = pd.read_csv(data_dir / "labels.csv").iloc[:, 0].astype(str)
    waveforms = _read_numeric(data_dir / "waveforms.csv")
    isi = _read_numeric(data_dir / "isi_dist.csv")

    mask = labels.isin(class_order).values
    waveforms = waveforms[mask]
    isi = isi[mask]
    labels = labels[mask].reset_index(drop=True)

    waveforms = (waveforms - waveforms.mean(axis=1, keepdims=True)) / (
        waveforms.std(axis=1, keepdims=True) + 1e-8
    )
    isi = isi / (isi.sum(axis=1, keepdims=True) + 1e-8)

    classes = [c for c in class_order if c in labels.values]
    return waveforms, isi, labels, classes


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
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title("Cell-type distribution", fontsize=9, pad=4)
    ax.set_ylim(0, max(proportions) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def _plot_summary_table(ax, info: dict, n_cells: int, classes: list[str],
                        counts: dict[str, int]) -> None:
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
        colWidths=[0.32, 0.68],
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.06
        if col == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold", color="#222")
        else:
            cell.set_facecolor("white")


def make_profile(which: str) -> None:
    info = DATASETS[which]
    key = info["key"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    waveforms, isi, labels, classes = load(key, info["class_order"])
    n_cells = len(labels)
    counts = {c: int((labels == c).sum()) for c in classes}

    fig = plt.figure(figsize=(12, 5.2), facecolor="white")
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.55, wspace=0.32,
        left=0.06, right=0.98, top=0.9, bottom=0.12,
    )
    ax_wf   = fig.add_subplot(gs[0, 0])
    ax_isi  = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_meta = fig.add_subplot(gs[1, :])
    for ax in (ax_wf, ax_isi, ax_dist, ax_meta):
        ax.set_facecolor("white")

    _plot_modality(ax_wf, waveforms, labels, classes,
                   "Mean waveform by cell type", "Time (samples)", "Amplitude (norm.)")
    _plot_modality(ax_isi, isi, labels, classes,
                   "Mean ISI distribution", "ISI bin", "Density (norm.)")
    _plot_distribution(ax_dist, labels, classes)
    _plot_summary_table(ax_meta, info, n_cells, classes, counts)

    fig.suptitle(f"Dataset profile: {info['display']}",
                 fontsize=11, fontweight="bold", y=0.98)

    stem = OUT_DIR / f"{key}_fig3_profile"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def main() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = args if args else list(DATASETS.keys())
    for t in targets:
        if t not in DATASETS:
            raise SystemExit(f"unknown dataset: {t} (choose from {list(DATASETS)})")
        make_profile(t)


if __name__ == "__main__":
    main()
