"""Figure 2 panel D primitive — Hausser β × z_dim hyperparameter sweep.

Uses the locked 5-fold sweep over the 42-config grid (6 z_dim × 7 β) that
mirrors NEMO's 42-config sweep to keep the two tools at parity. Highlights
the production HIPPIE config (z=30, β=1.0) selected by the stability
tiebreaker described in the Methods.

The β∈{6,10} saturation spot-check (configs 43-54) is intentionally
excluded here: it was a later add-on to confirm the regularization
frontier was saturated, not part of the main selection grid.

Outputs:
    figures/19_hyperparam_sweep_hausser/hausser_sweep.csv              # tidy long table
    figures/19_hyperparam_sweep_hausser/hausser_heatmap.{svg,png}      # D1: KNN heatmap
    figures/19_hyperparam_sweep_hausser/hausser_knn_vs_mlp.{svg,png}   # D2: locked-config KNN vs MLP
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
OUT_DIR = REPO_ROOT / "figures" / "figure_2"

Z_DIMS = [3, 9, 15, 30, 45, 60]
BETAS = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0]

# Locked-config selection rule (matches paper Methods § hyperparameter parity).
# Among configs sharing the max fold count: (1) define a "tie bucket" as configs
# whose mean balanced accuracy is within 1.96 × SEM of the top mean; (2) within
# the bucket, pick the lowest standard deviation; (3) on remaining ties, pick
# the lowest numeric config_id (deterministic third-level tiebreak).
# Reproduce with:  python scripts/pick_locked_configs.py --verify
# (CSV-only port of the production selection script; reads
#  results/benchmark/tuning_preview/sweep_results_long.csv). The constant
# below is hardcoded here so plot regeneration does not depend on the
# selection pipeline.
LOCKED_CONFIG_ID = "config_25"   # z=30, β=1.0 — locked by stability tiebreaker
LOCKED_Z = 30
LOCKED_BETA = 1.0


def build_grid() -> pd.DataFrame:
    """Row-major z × β, 6 × 7 = 42 configs (mirrors NEMO's 42-config sweep)."""
    rows = []
    cid = 1
    for z in Z_DIMS:
        for b in BETAS:
            rows.append({"config_id": f"config_{cid}",
                         "z_dim": z, "beta": b})
            cid += 1
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    """Read the pre-aggregated 5-fold sweep CSV used by the paper figure."""
    if not SWEEP_CSV.exists():
        raise SystemExit(f"Sweep CSV not found: {SWEEP_CSV}")
    raw = pd.read_csv(SWEEP_CSV)
    sub = raw[
        (raw["run_tag"] == SWEEP_RUN_TAG)
        & (raw["method"] == "hippie")
        & (raw["dataset"] == "hausser_cell_type")
        & (raw["classifier"].isin(["knn", "mlp"]))
        & (raw["z_dim"].isin(Z_DIMS))
        & (raw["beta"].isin(BETAS))
    ].copy()
    sub["z_dim"] = sub["z_dim"].astype(int)
    sub["beta"] = sub["beta"].astype(float)
    grid = build_grid().set_index(["z_dim", "beta"])["config_id"]
    sub["config_id"] = sub.set_index(["z_dim", "beta"]).index.map(grid)
    sub = sub.rename(columns={"classifier": "probe"})
    out = sub[["config_id", "z_dim", "beta", "fold", "probe",
               "balanced_accuracy"]].reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"No matching rows in {SWEEP_CSV}")
    return out


def plot_heatmap(long_df: pd.DataFrame, out_stem: Path) -> None:
    knn = long_df[long_df["probe"] == "knn"]
    agg = (
        knn.groupby(["z_dim", "beta"])["balanced_accuracy"]
        .mean()
        .reset_index()
    )
    mat = agg.pivot(index="z_dim", columns="beta", values="balanced_accuracy")
    mat = mat.reindex(index=Z_DIMS, columns=BETAS)

    fig, ax = plt.subplots(figsize=(5.0, 3.4), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(
        mat.values,
        cmap="Blues",
        aspect="auto",
        vmin=float(np.nanmin(mat.values)),
        vmax=float(np.nanmax(mat.values)),
    )
    ax.set_xticks(np.arange(len(BETAS)))
    ax.set_xticklabels([str(b) for b in BETAS])
    ax.set_yticks(np.arange(len(Z_DIMS)))
    ax.set_yticklabels([str(z) for z in Z_DIMS])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("latent dim (z)")
    ax.set_title("KNN balanced accuracy — Hausser (5-fold mean)", fontsize=9, pad=18)

    # Blues: dark at high values → put black text on light cells, white on dark
    cutoff = np.nanmin(mat.values) + 0.6 * (np.nanmax(mat.values) - np.nanmin(mat.values))
    for i, z in enumerate(Z_DIMS):
        for j, b in enumerate(BETAS):
            val = mat.iloc[i, j]
            if pd.isna(val):
                continue
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if val >= cutoff else "black",
            )

    # outline locked config
    i_lock = Z_DIMS.index(LOCKED_Z)
    j_lock = BETAS.index(LOCKED_BETA)
    ax.add_patch(plt.Rectangle(
        (j_lock - 0.5, i_lock - 0.5), 1, 1,
        fill=False, edgecolor="#ff4d4d", linewidth=2.2,
    ))
    # subtitle-style annotation under the title, out of the grid
    ax.text(
        0.5, 1.02,
        f"red box = locked config (z={LOCKED_Z}, β={LOCKED_BETA})",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, color="#ff4d4d", fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("balanced acc.")
    fig.tight_layout()

    svg = out_stem.with_suffix(".svg")
    png = out_stem.with_suffix(".png")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[svg] {svg}")
    print(f"[png] {png}")


def plot_probe_comparison(long_df: pd.DataFrame, out_stem: Path) -> None:
    sel = long_df[(long_df["z_dim"] == LOCKED_Z) & (long_df["beta"] == LOCKED_BETA)]
    if sel.empty:
        raise SystemExit(f"no data for locked config z={LOCKED_Z}, beta={LOCKED_BETA}")
    probes = ["knn", "mlp"]
    means = [sel[sel["probe"] == p]["balanced_accuracy"].mean() for p in probes]
    stds = [sel[sel["probe"] == p]["balanced_accuracy"].std(ddof=1) for p in probes]
    points = [sel[sel["probe"] == p]["balanced_accuracy"].values for p in probes]

    colors = ["#1f77b4", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(3.0, 3.4), facecolor="white")
    ax.set_facecolor("white")
    x = np.arange(len(probes))
    ax.bar(
        x, means,
        width=0.55,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "#333"},
    )
    rng = np.random.default_rng(3)
    for xi, pts in zip(x, points):
        ax.scatter(
            xi + rng.uniform(-0.07, 0.07, size=len(pts)),
            pts, s=14, color="#1e3a5f", zorder=3, linewidth=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in probes])
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Locked config (z={LOCKED_Z}, β={LOCKED_BETA})", fontsize=9, pad=6)
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
    long_df = collect()
    csv = OUT_DIR / "hausser_sweep.csv"
    long_df.to_csv(csv, index=False)
    print(f"[csv] {csv}")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    plot_heatmap(long_df, OUT_DIR / "hausser_heatmap")
    plot_probe_comparison(long_df, OUT_DIR / "hausser_knn_vs_mlp")


if __name__ == "__main__":
    main()
