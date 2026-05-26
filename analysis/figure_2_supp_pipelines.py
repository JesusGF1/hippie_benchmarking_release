"""Supplemental pipeline schematics — PhysMAP and NEMO hyperparameter tuning flows.

Simplified version of figure_2_panel_a_pipeline.py (no architectural ablation
rung — PhysMAP and NEMO are fixed architectures, only hyperparameters vary).

Outputs:
    figures/20_figure_2_pipeline/pipeline_schematic_physmap.{svg,png}
    figures/20_figure_2_pipeline/pipeline_schematic_nemo.{svg,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "figure_2" / "supplement"

PIPELINES = {
    "physmap": [
        ("Hausser\ndataset",                  "#f0f0f0"),
        ("PhysMAP hyperparameter\nsweep\n(4 dimV × 3 metric × 3 n_feat)", "#d6eaf8"),
        ("Stability\ntiebreaker",             "#fdebd0"),
        ("Locked\nconfiguration",             "#d5f5e3"),
    ],
    "nemo": [
        ("Hausser\ndataset",                  "#f0f0f0"),
        ("NEMO hyperparameter\nsweep\n(42 configs)", "#d6eaf8"),
        ("Stability\ntiebreaker",             "#fdebd0"),
        ("Locked\nconfiguration",             "#d5f5e3"),
    ],
}


def _draw(steps, out_stem: Path) -> None:
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

    box_w = 2.0
    box_h = 1.0
    gap = 0.35
    x0 = (10.5 - (box_w * len(steps) + gap * (len(steps) - 1))) / 2
    y_center = 0.78

    centers = []
    for label, color in steps:
        x = x0
        y = y_center - box_h / 2
        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="#444", linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center",
                fontsize=9.5, linespacing=1.2)
        centers.append((x, y, x + box_w, y + box_h))
        x0 += box_w + gap

    for i in range(len(steps) - 1):
        x_end_prev = centers[i][2]
        x_start_next = centers[i + 1][0]
        ax.annotate(
            "",
            xy=(x_start_next - 0.02, y_center),
            xytext=(x_end_prev + 0.02, y_center),
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2, shrinkA=0, shrinkB=0),
        )

    fig.tight_layout(pad=0.2)
    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method, steps in PIPELINES.items():
        _draw(steps, OUT_DIR / f"pipeline_schematic_{method}")


if __name__ == "__main__":
    main()
