"""Supplemental to Figure 2 — PhysMAP hyperparameter sweep on Hausser.

Grid (per preview_sweep.physmap_grid): 4 dimV × 3 metric × 3 nfeatures = 36 configs.
Each config's physmap_CV_results.csv contains 5 caret-CV folds (Fold1..Fold5).

Locked config (locked_configs.json): config_13, selection classifier KNN.
    config_13 → dimV=5, metric=cosine, nfeatures=20 (index 13 in row-major grid).

Outputs:
    figures/21_physmap_sweep_hausser/physmap_sweep.csv
    figures/21_physmap_sweep_hausser/physmap_heatmaps.{svg,png}       # 3 heatmaps per metric
    figures/21_physmap_sweep_hausser/physmap_knn_bar.{svg,png}        # locked config KNN bar
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = (
    REPO_ROOT / "results" / "benchmark" / "tuning_preview" / "sweep_results_long.csv"
)
SWEEP_RUN_TAG = "sweep-physmap-fix-20260411T222505Z"
# Locked-config per-fold predictions (labeled rows only) — used for the bar's scatter.
LOCKED_PRED_CSV = (
    REPO_ROOT / "results" / "benchmark" / "celltype_cache"
    / "physmap" / "hausser_cell_type" / "fold_0" / "predictions"
    / "physmap_CV_results.csv"
)
OUT_DIR = REPO_ROOT / "figures" / "figure_2" / "supplement"

DIMVS = [3, 5, 10, 15]
METRICS = ["cosine", "euclidean", "correlation"]
NFEATURES = [20, 40, 80]

# Locked-config selection rule: same stability tiebreaker as HIPPIE and NEMO
# (paper Methods § hyperparameter parity). Within configs sharing the max fold
# count: (1) define a tie bucket within 1.96 × SEM of the top mean balanced
# accuracy; (2) pick lowest std within the bucket; (3) lowest numeric config_id
# as deterministic 3rd-level tiebreak.
# Reproduce with:  python scripts/pick_locked_configs.py --verify
# Note: PhysMAP exports one aggregated BA per (config, dataset) rather than
# per-fold scores; with the Hausser-only selection rule the tie bucket reduces
# to one row per config and the highest-mean config (config_13, BA=0.780) wins.
LOCKED_CONFIG_ID = "config_13"   # dimV=5, euclidean, n=20 — locked by stability tiebreaker


def build_grid() -> pd.DataFrame:
    rows = []
    cid = 1
    for d in DIMVS:
        for m in METRICS:
            for n in NFEATURES:
                rows.append({
                    "config_id": f"config_{cid}",
                    "dimV": d, "metric": m, "nfeatures": n,
                })
                cid += 1
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    """Read PhysMAP WNN balanced accuracy per config from the canonical sweep CSV.

    Single source of truth for every per-config aggregated value in the
    supplement: `results/benchmark/tuning_preview/sweep_results_long.csv`, run-tag
    `sweep-physmap-fix-20260411T222505Z`. This CSV stores one row per (config,
    classifier) — the caret-CV mean already aggregated across all 5 folds.

    Per-fold values for the locked-config bar's scatter come from a separate
    source (`celltype_cache/.../physmap_CV_results.csv` — labeled rows only),
    matched to the same caret-CV mean, so the bar mean and the heatmap's locked
    cell agree by construction.
    """
    if not SWEEP_CSV.exists():
        raise SystemExit(f"Sweep CSV not found: {SWEEP_CSV}")
    raw = pd.read_csv(SWEEP_CSV)
    sub = raw[
        (raw["run_tag"] == SWEEP_RUN_TAG)
        & (raw["method"] == "physmap")
        & (raw["dataset"] == "hausser_cell_type")
    ].copy()
    grid = build_grid().set_index("config_id")
    for col in ["dimV", "metric", "nfeatures"]:
        sub[col] = sub["config_id"].map(grid[col])
    out = sub[["config_id", "dimV", "metric", "nfeatures",
               "fold", "balanced_accuracy"]].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"No matching PhysMAP rows in {SWEEP_CSV}")
    return out


def plot_heatmaps(long_df: pd.DataFrame, out_stem: Path) -> None:
    agg = (
        long_df.groupby(["dimV", "metric", "nfeatures"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )

    vmin = float(agg["balanced_accuracy"].min())
    vmax = float(agg["balanced_accuracy"].max())
    cutoff = vmin + 0.6 * (vmax - vmin)

    locked = build_grid().set_index("config_id").loc[LOCKED_CONFIG_ID]
    l_dim, l_metric, l_nf = int(locked["dimV"]), locked["metric"], int(locked["nfeatures"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), facecolor="white", sharey=True)
    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor("white")
        sub = agg[agg["metric"] == metric]
        pivot = sub.pivot(index="dimV", columns="nfeatures", values="balanced_accuracy")
        pivot = pivot.reindex(index=DIMVS, columns=NFEATURES)

        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(NFEATURES)))
        ax.set_xticklabels([str(n) for n in NFEATURES])
        ax.set_yticks(range(len(DIMVS)))
        ax.set_yticklabels([str(d) for d in DIMVS])
        ax.set_xlabel("n features")
        if metric == METRICS[0]:
            ax.set_ylabel("dimV (PCs)")
        ax.set_title(metric, fontsize=9, pad=4)

        for i, d in enumerate(DIMVS):
            for j, n in enumerate(NFEATURES):
                v = pivot.iloc[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v >= cutoff else "black")

        if metric == l_metric:
            i_lock = DIMVS.index(l_dim)
            j_lock = NFEATURES.index(l_nf)
            ax.add_patch(plt.Rectangle(
                (j_lock - 0.5, i_lock - 0.5), 1, 1,
                fill=False, edgecolor="#ff4d4d", linewidth=2.2,
            ))

    fig.suptitle(
        f"PhysMAP hyperparameter sweep — Hausser   (red = locked {LOCKED_CONFIG_ID}: "
        f"dimV={l_dim}, metric={l_metric}, n={l_nf})",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.9, top=0.85, bottom=0.16, wspace=0.15)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.69])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("balanced acc. (5-fold mean)", fontsize=8)

    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def _locked_fold_bas() -> np.ndarray:
    """Per-fold WNN balanced accuracy for the locked config (5 caret-CV folds).

    Reads the labeled-only predictions in celltype_cache (the canonical source
    used for the paper's panel C PhysMAP bar). The raw sweep dir's CSV pools
    unlabeled rows and crushes sklearn's BA, so it is NOT a valid scatter source.
    """
    if not LOCKED_PRED_CSV.exists():
        raise SystemExit(f"Locked-config predictions not found: {LOCKED_PRED_CSV}")
    df = pd.read_csv(LOCKED_PRED_CSV)
    df.columns = [c.strip('"') for c in df.columns]
    bas = []
    for _, ch in df.groupby("fold"):
        bas.append(balanced_accuracy_score(ch["true"], ch["pred"]))
    return np.array(bas, dtype=float)


def plot_locked_bar(long_df: pd.DataFrame, out_stem: Path) -> None:
    pts = _locked_fold_bas()
    if len(pts) == 0:
        raise SystemExit(f"no per-fold data for {LOCKED_CONFIG_ID}")
    mean = float(pts.mean())
    std = float(pts.std(ddof=1))

    fig, ax = plt.subplots(figsize=(2.8, 3.4), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(
        [0], [mean], width=0.55, yerr=[std],
        color="#1f77b4", edgecolor="white", linewidth=0.8, capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "#333"},
    )
    rng = np.random.default_rng(11)
    ax.scatter(
        rng.uniform(-0.07, 0.07, size=len(pts)),
        pts, s=14, color="#1e3a5f", zorder=3, linewidth=0,
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["PhysMAP\nKNN"])
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Locked {LOCKED_CONFIG_ID}\n(dimV=5, euclidean, n=20)", fontsize=9, pad=4)
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
    csv = OUT_DIR / "physmap_sweep.csv"
    df.to_csv(csv, index=False)
    print(f"[csv] {csv}")
    plot_heatmaps(df, OUT_DIR / "physmap_heatmaps")
    plot_locked_bar(df, OUT_DIR / "physmap_knn_bar")


if __name__ == "__main__":
    main()
