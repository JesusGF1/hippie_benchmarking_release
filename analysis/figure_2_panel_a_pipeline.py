"""Figure 2 panel A primitive — architectural ablation + hyperparameter tuning pipeline.

Draws a horizontal flowchart describing the methodology used to arrive at the
locked HIPPIE configuration for downstream benchmarks:

    1. baseline VAE on Hausser
    2. architectural ablation ladder (8 rungs of conditioning / normalization / augmentation)
    3. pick winning architecture
    4. β × z_dim sweep over the winning architecture
    5. lock (architecture, β, z_dim) for downstream experiments

Outputs:
    figures/20_figure_2_pipeline/pipeline_schematic.svg
    figures/20_figure_2_pipeline/pipeline_schematic.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "figure_2"
OUT_STEM = OUT_DIR / "pipeline_schematic"

STEPS = [
    ("Hausser\ndataset", "#f0f0f0"),
    ("Architectural\nablation ladder\n(8 rungs)", "#d6eaf8"),
    ("Pick best\narchitecture", "#fdebd0"),
    ("β × z-dim\nsweep\n(6×7 = 42)", "#d6eaf8"),
    ("Locked\nconfiguration", "#d5f5e3"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10.5, 1.55), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w = 1.7
    box_h = 0.95
    gap = 0.3
    x0 = (10.5 - (box_w * len(STEPS) + gap * (len(STEPS) - 1))) / 2
    y_center = 0.78

    centers = []
    for i, (label, color) in enumerate(STEPS):
        x = x0 + i * (box_w + gap)
        y = y_center - box_h / 2
        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="#444444",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2, y + box_h / 2,
            label,
            ha="center", va="center",
            fontsize=9.5, linespacing=1.2,
        )
        centers.append((x, y, x + box_w, y + box_h))

    # arrows between consecutive boxes
    for i in range(len(STEPS) - 1):
        x_end_prev = centers[i][2]
        x_start_next = centers[i + 1][0]
        ax.annotate(
            "",
            xy=(x_start_next - 0.02, y_center),
            xytext=(x_end_prev + 0.02, y_center),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#444",
                lw=1.2,
                shrinkA=0, shrinkB=0,
            ),
        )

    # panel labels under the boxes describing which figure panel covers each
    panel_labels = ["(B)", "(C)", "", "(D)", ""]
    for (x0b, y0b, x1b, y1b), tag in zip(centers, panel_labels):
        if not tag:
            continue
        ax.text(
            (x0b + x1b) / 2, y0b - 0.12,
            tag, ha="center", va="top",
            fontsize=9, color="#666", style="italic",
        )

    fig.tight_layout(pad=0.2)
    svg = OUT_STEM.with_suffix(".svg")
    png = OUT_STEM.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


if __name__ == "__main__":
    main()
