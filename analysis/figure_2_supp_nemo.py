"""Supplemental to Figure 2 — NEMO hyperparameter sweep on Hausser.

Grid (per preview_sweep.nemo_grid): 42 configs varying temperature, learning rate,
z_dim, batch_size, and (for 39-42) weight_decay. NEMO locks its config on the
**MLP** probe (its native head).

Locked config: config_7 → temp=0.5, lr=1e-4, z_dim=256, batch=1024.

Selected by the stability tiebreaker rule documented in paper Methods (§
hyperparameter parity). The rule is a 3-level sort within the mean-BA tie
bucket (within 1.96 × SEM of the top mean): (1) lowest std, (2) highest mean,
(3) lowest numeric config_id as a deterministic 3rd-level tiebreak. For NEMO,
the naive point-estimate top was config_21; config_7 wins because std + mean
tied at level (1)+(2), and the 3rd-level config-id tiebreak selected the
lower id. Reproduce with:  python scripts/pick_locked_configs.py --verify
(CSV-only port of the production selection script). The constant below is
hardcoded here so plot regeneration does not depend on the selection
pipeline.

Outputs:
    figures/22_nemo_sweep_hausser/nemo_sweep.csv
    figures/22_nemo_sweep_hausser/nemo_sweep_bars.{svg,png}    # C1: sorted config bars
    figures/22_nemo_sweep_hausser/nemo_knn_vs_mlp.{svg,png}    # C2: locked-config KNN vs MLP
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = (
    REPO_ROOT / "results" / "benchmark" / "tuning_preview" / "sweep_results_long.csv"
)
SWEEP_RUN_TAG = "sweep-20260411T064942Z"
OUT_DIR = REPO_ROOT / "figures" / "figure_2" / "supplement"

LOCKED_CONFIG_ID = "config_7"
LOCKED_HP = dict(temperature=0.5, lr=1e-4, z_dim=256, batch_size=1024)

RAW = [
    "1:0.1:1e-4:512:1024", "2:0.3:1e-4:512:1024", "3:0.5:1e-4:512:1024",
    "4:0.7:1e-4:512:1024", "5:1.0:1e-4:512:1024", "6:0.5:1e-4:128:1024",
    "7:0.5:1e-4:256:1024", "8:0.5:1e-5:512:1024", "9:0.5:1e-3:512:1024",
    "10:0.5:1e-4:512:256", "11:0.3:1e-4:256:1024", "12:0.7:1e-4:256:1024",
    "13:0.3:1e-4:128:1024", "14:0.7:1e-4:128:1024", "15:0.3:1e-3:512:1024",
    "16:0.7:1e-3:512:1024", "17:0.3:1e-5:512:1024", "18:0.7:1e-5:512:1024",
    "19:0.3:1e-4:512:256", "20:0.7:1e-4:512:256", "21:0.5:1e-4:256:256",
    "22:0.5:1e-4:128:256", "23:0.1:1e-5:128:256", "24:1.0:1e-3:512:256",
    "25:0.3:1e-3:128:1024", "26:0.5:1e-3:128:1024", "27:0.7:1e-3:128:1024",
    "28:0.3:1e-3:256:1024", "29:0.5:1e-3:256:1024", "30:0.7:1e-3:256:1024",
    "31:0.3:1e-5:256:1024", "32:0.7:1e-5:256:1024", "33:0.3:1e-5:128:1024",
    "34:0.7:1e-5:128:1024", "35:0.3:1e-3:512:256", "36:0.7:1e-3:512:256",
    "37:0.3:1e-5:512:256", "38:0.7:1e-5:512:256",
    "39:0.5:1e-4:512:1024:1e-5", "40:0.5:1e-4:512:1024:1e-4",
    "41:0.5:1e-3:512:1024:1e-5", "42:0.5:1e-3:512:1024:1e-4",
]

def build_grid() -> pd.DataFrame:
    rows = []
    for e in RAW:
        parts = e.split(":")
        rows.append({
            "config_id": f"config_{parts[0]}",
            "temperature": float(parts[1]),
            "lr": float(parts[2]),
            "z_dim": int(parts[3]),
            "batch_size": int(parts[4]),
            "weight_decay": float(parts[5]) if len(parts) > 5 else 0.0,
        })
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    """Read the pre-aggregated 5-fold sweep CSV used by the paper figure."""
    if not SWEEP_CSV.exists():
        raise SystemExit(f"Sweep CSV not found: {SWEEP_CSV}")
    raw = pd.read_csv(SWEEP_CSV)
    sub = raw[
        (raw["run_tag"] == SWEEP_RUN_TAG)
        & (raw["method"] == "nemo")
        & (raw["dataset"] == "hausser_cell_type")
        & (raw["classifier"].isin(["knn", "mlp"]))
    ].copy()
    sub = sub.rename(columns={"classifier": "probe"})
    # Fill missing hyperparameters from the grid (raw CSV may lack weight_decay for some rows)
    grid = build_grid().set_index("config_id")
    for col in ["temperature", "lr", "z_dim", "batch_size", "weight_decay"]:
        sub[col] = sub["config_id"].map(grid[col])
    out = sub[[
        "config_id", "temperature", "lr", "z_dim", "batch_size",
        "weight_decay", "fold", "probe", "balanced_accuracy",
    ]].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"No matching NEMO rows in {SWEEP_CSV}")
    return out


def plot_sweep_bars(long_df: pd.DataFrame, out_stem: Path) -> None:
    mlp = long_df[long_df["probe"] == "mlp"]
    agg = (
        mlp.groupby(["config_id", "temperature", "lr", "z_dim", "batch_size", "weight_decay"])
        ["balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11.5, 4.0), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(agg))
    colors = ["#ff4d4d" if cid == LOCKED_CONFIG_ID else "#1f77b4" for cid in agg["config_id"]]
    ax.bar(
        x, agg["mean"],
        yerr=agg["std"].fillna(0),
        color=colors, edgecolor="white", linewidth=0.5, capsize=2.5,
        error_kw={"elinewidth": 0.7, "ecolor": "#333"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [cid.replace("config_", "c") for cid in agg["config_id"]],
        rotation=90, fontsize=7,
    )
    ax.set_ylabel("NEMO MLP probe — balanced accuracy")
    ax.set_ylim(0, 1.18)   # headroom for top-3 annotation box
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"NEMO hyperparameter sweep — Hausser   (red = locked {LOCKED_CONFIG_ID}: "
        "T=0.5, lr=1e-4, z=256, b=1024)",
        fontsize=9, pad=6,
    )

    # top-3 hyperparameter summary, placed inside the axis (right-side, above bars)
    lines = []
    for i in range(min(3, len(agg))):
        r = agg.iloc[i]
        lines.append(
            f"#{i+1}  {r['config_id']:>9s} — T={r['temperature']:.1f}, "
            f"lr={r['lr']:.0e}, z={int(r['z_dim']):>3d}, b={int(r['batch_size']):>4d}  "
            f"({r['mean']:.2f} ± {r['std']:.2f})"
        )
    box_text = "Top 3 configs:\n" + "\n".join(lines)
    ax.text(
        0.995, 0.97, box_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                  edgecolor="#bbb", linewidth=0.6),
    )

    fig.tight_layout()
    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def plot_locked_probe_comparison(long_df: pd.DataFrame, out_stem: Path) -> None:
    sel = long_df[long_df["config_id"] == LOCKED_CONFIG_ID]
    if sel.empty:
        raise SystemExit(f"no data for {LOCKED_CONFIG_ID}")
    probes = ["knn", "mlp"]
    means = [sel[sel["probe"] == p]["balanced_accuracy"].mean() for p in probes]
    stds = [sel[sel["probe"] == p]["balanced_accuracy"].std(ddof=1) for p in probes]
    points = [sel[sel["probe"] == p]["balanced_accuracy"].values for p in probes]

    colors = ["#1f77b4", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(3.0, 3.4), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(probes))
    ax.bar(
        x, means, width=0.55, yerr=stds,
        color=colors, edgecolor="white", linewidth=0.8, capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "#333"},
    )
    rng = np.random.default_rng(21)
    for xi, pts in zip(x, points):
        ax.scatter(
            xi + rng.uniform(-0.07, 0.07, size=len(pts)),
            pts, s=14, color="#1e3a5f", zorder=3, linewidth=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in probes])
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Locked {LOCKED_CONFIG_ID}\n(T=0.5, lr=1e-4, z=256)", fontsize=9, pad=4)
    fig.tight_layout()

    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    df = collect()
    csv = OUT_DIR / "nemo_sweep.csv"
    df.to_csv(csv, index=False)
    print(f"[csv] {csv}")
    plot_sweep_bars(df, OUT_DIR / "nemo_sweep_bars")
    plot_locked_probe_comparison(df, OUT_DIR / "nemo_knn_vs_mlp")


if __name__ == "__main__":
    main()
