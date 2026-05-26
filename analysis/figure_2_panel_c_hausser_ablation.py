"""Figure 2 panel C primitive — Hausser-only architectural ablation ladder.

Reads results/benchmark/figure2_final_ladder.csv (tidy fold-level
balanced-accuracy table), filters to the Hausser cerebellar cortex cell-type
dataset, and emits a clean single-panel ladder chart for the composite.

Outputs:
    figures/figure_2/hausser_only_ladder.svg
    figures/figure_2/hausser_only_ladder.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
LADDER_CSV = REPO_ROOT / "results" / "benchmark" / "figure2_final_ladder.csv"
NEGCTRL = REPO_ROOT / "results" / "negative_controls"
OUT_DIR = REPO_ROOT / "figures" / "figure_2"
OUT_STEM = OUT_DIR / "hausser_only_ladder"

DATASET_KEY = "hausser_cell_type"
N_CLASSES = 5  # Hull (hausser) has 5 cell types → chance = 0.20

LADDER: list[tuple[int, str, str]] = [
    (1, "baseline",                           "baseline\nVAE"),
    (2, "with_source",                        "+ source"),
    (3, "with_both_embeddings",               "+ class\n(encoder)"),
    (4, "class_decoder_source",               "+ class\n(decoder)"),
    (5, "class_decoder_source_bn",            "+ BN"),
    (6, "conditional_decoder_only",           "+ light\naug"),
    (7, "class_decoder_source_bn_strong_aug", "+ strong\naug"),
    (8, "class_decoder_source_bn_aug_reg",    "+ consistency\nreg"),
]

BAR_COLOR = "#1f77b4"
POINT_COLOR = "#1e3a5f"
CHANCE_COLOR = "#888888"

SHUFFLE_COLOR = "#ff7f0e"
SHIFT_COLOR   = "#ff7f0e"


def _negctrl_scores(control: str, dataset: str) -> list[float]:
    run_file = NEGCTRL / control / "hippie" / dataset / "run_1.csv"
    if not run_file.exists():
        return []
    df = pd.read_csv(run_file)
    if {"pred", "true"}.issubset(df.columns):
        return [balanced_accuracy_score(df["true"], df["pred"])]
    return []


def main() -> None:
    df = pd.read_csv(LADDER_CSV)
    df = df[df["dataset"] == DATASET_KEY].copy()
    if df.empty:
        raise SystemExit(f"No {DATASET_KEY} rows in {LADDER_CSV}")

    labels = [label for (_, _, label) in LADDER]
    x = np.arange(len(LADDER))
    means = []
    stds = []
    fold_points = []
    for (_, cname, _) in LADDER:
        rows = df[df["config_name"] == cname]
        means.append(rows["balanced_accuracy"].mean())
        n = len(rows)
        stds.append(rows["balanced_accuracy"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        fold_points.append(rows["balanced_accuracy"].values)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(
        x, means,
        width=0.68,
        yerr=stds,
        color=BAR_COLOR,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
    )
    # individual fold points
    rng = np.random.default_rng(7)
    for xi, pts in zip(x, fold_points):
        jitter = rng.uniform(-0.08, 0.08, size=len(pts))
        ax.scatter(
            xi + jitter, pts,
            s=14, color=POINT_COLOR,
            zorder=3, linewidth=0,
        )

    # Per-step gain annotations (delta mean balanced accuracy vs previous rung)
    for i in range(1, len(means)):
        delta = means[i] - means[i - 1]
        if np.isnan(delta):
            continue
        sign = "+" if delta >= 0 else "−"
        color = "#2b7a2b" if delta >= 0 else "#b00020"
        y_text = max(means[i] + (stds[i] if not np.isnan(stds[i]) else 0),
                     means[i - 1] + (stds[i - 1] if not np.isnan(stds[i - 1]) else 0)) + 0.11
        y_text = min(y_text, 0.98)
        ax.annotate(
            "",
            xy=(x[i] - 0.08, y_text - 0.015),
            xytext=(x[i - 1] + 0.08, y_text - 0.015),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9, shrinkA=0, shrinkB=0),
        )
        ax.text(
            (x[i - 1] + x[i]) / 2, y_text + 0.005,
            f"{sign}{abs(delta):.2f}",
            ha="center", va="bottom", fontsize=7,
            color=color, fontweight="bold",
        )

    chance = 1.0 / N_CLASSES
    # dashed chance line — stop short of the right edge so the label is clean
    ax.plot(
        [-0.5, len(LADDER) - 0.55], [chance, chance],
        color=CHANCE_COLOR, linestyle="--", linewidth=0.9, zorder=1,
    )
    ax.text(
        len(LADDER) - 0.45, chance,
        f"chance = {chance:.2f}",
        ha="left", va="center", fontsize=7, color=CHANCE_COLOR,
        clip_on=False,
    )

    # negative-control reference lines (shuffle: mean ± range band; shift: single line)
    x_span = [-0.5, len(LADDER) - 0.55]
    shuffle_scores = _negctrl_scores("shuffle", DATASET_KEY)
    if shuffle_scores:
        shuf_mean = float(np.mean(shuffle_scores))
        ax.plot(x_span, [shuf_mean, shuf_mean],
                color=SHUFFLE_COLOR, linestyle=(0, (4, 2)), linewidth=1.1, zorder=1,
                label=f"Shuffle (n={len(shuffle_scores)})")
        if len(shuffle_scores) > 1:
            ax.fill_between(x_span,
                            [min(shuffle_scores)] * 2, [max(shuffle_scores)] * 2,
                            color=SHUFFLE_COLOR, alpha=0.12, linewidth=0)
        ax.text(len(LADDER) - 0.45, shuf_mean, f"shuffle = {shuf_mean:.2f}",
                ha="left", va="center", fontsize=7, color=SHUFFLE_COLOR, clip_on=False)

    shift_scores = _negctrl_scores("shift", DATASET_KEY)
    if shift_scores:
        shift_mean = float(np.mean(shift_scores))
        ax.plot(x_span, [shift_mean, shift_mean],
                color=SHIFT_COLOR, linestyle=(0, (2, 2)), linewidth=1.1, zorder=1,
                label=f"Shift (n={len(shift_scores)})")
        ax.text(len(LADDER) - 0.45, shift_mean, f"shift = {shift_mean:.2f}",
                ha="left", va="center", fontsize=7, color=SHIFT_COLOR, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("KNN balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Architectural ablation ladder — Hausser", fontsize=9, pad=6)

    ax.set_xlim(-0.6, len(LADDER) + 0.8)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUT_STEM.with_suffix(".svg")
    png = OUT_STEM.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


if __name__ == "__main__":
    main()
