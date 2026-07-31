"""Figure 4 Supplemental — trimodal (WF+ISI+ACG) benchmark on additional datasets.

Datasets NOT shown in main Figure 4:
    dandi_000041_cell_type   Watson (Rat Neocortex) — 221 cells, 2 cell types,
                             frontal cortex, 64-site silicon probes, 9 subjects.

Methods: HIPPIE, NEMO (where available), PhysMAP (WNN), PCA-WF, PCA-ISI, PCA-ACG.

Predictions read from results/benchmark/celltype_cache/.

Outputs to figures/32_figure_4_supp_trimodal/:
    {dataset}_bars.{svg,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "results" / "benchmark" / "celltype_cache"
OUT_DIR   = REPO_ROOT / "figures" / "figure_4" / "supplement"
N_FOLDS   = 5

DATASETS = {
    "dandi_000041_cell_type": {
        "display":        "Watson (Rat Neocortex)",
        "short":          "Watson",
        "n_classes":      2,
        "chance":         0.50,
        "species":        "Rat",
        "brain_region":   "Frontal cortex",
        "recording_tech": "64-site silicon probe",
        "n_cells":        221,
    },
}

NEMO_DATASETS = {"dandi_000041_cell_type"}

BAR_COLOR   = "#1f77b4"
CHANCE_COLOR = "#888"
DOT_COLOR   = "#1e3a5f"

METHOD_DISPLAY = {
    "hippie":   "HIPPIE",
    "nemo":     "NEMO",
    "physmap":  "PhysMAP (WNN)",
    "pca_wf":   "PCA-WF",
    "pca_isi":  "PCA-ISI",
    "pca_acg":  "PCA-ACG",
}


def set_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  saved: {out}")
    plt.close(fig)


def _metrics(y_true, y_pred) -> tuple[float, float]:
    return (
        balanced_accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def _collect_hippie(dataset: str) -> list[dict]:
    rows = []
    for fold in range(N_FOLDS):
        p = CACHE_DIR / "hippie" / dataset / f"fold_{fold}" / "predictions" / "transductive_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        acc, f1 = _metrics(df["true"], df["pred"])
        rows.append({"method": "hippie", "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1})
    return rows


def _collect_nemo(dataset: str) -> list[dict]:
    """Collect NEMO predictions from the MLP probe head.

    Matches the convention used by figure_4_trimodal_benchmark.py — MLP is the
    NEMO classification head reported in every figure for consistency.
    """
    rows = []
    for fold in range(N_FOLDS):
        p = (CACHE_DIR / "nemo" / dataset / f"fold_{fold}"
             / "predictions" / f"fold_{fold}" / "mlp_probe_predictions.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        acc, f1 = _metrics(df["true"], df["pred"])
        rows.append({"method": "nemo", "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1})
    return rows


def _collect_physmap_file(method_key: str, fname: str, dataset: str) -> list[dict]:
    p = CACHE_DIR / "physmap" / dataset / "fold_0" / "predictions" / fname
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    if not {"pred", "true", "fold"}.issubset(df.columns):
        return []
    rows = []
    for fold_name, sub in df.groupby("fold"):
        fold_idx = int(str(fold_name).lower().lstrip("fold")) - 1
        acc, f1 = _metrics(sub["true"], sub["pred"])
        rows.append({"method": method_key, "dataset": dataset, "fold": fold_idx,
                     "balanced_accuracy": acc, "macro_f1": f1})
    return rows


def collect_all() -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in DATASETS:
        rows += _collect_hippie(dataset)
        if dataset in NEMO_DATASETS:
            rows += _collect_nemo(dataset)
        rows += _collect_physmap_file("physmap", "physmap_CV_results.csv",     dataset)
        rows += _collect_physmap_file("pca_wf",  "PCA_WF_CV_results.csv",      dataset)
        rows += _collect_physmap_file("pca_isi", "PCA_ISI_CV_results.csv",     dataset)
        rows += _collect_physmap_file("pca_acg", "PCA_AUTOCORR_CV_results.csv", dataset)
    if not rows:
        raise SystemExit("No prediction files found in celltype_cache/.")
    return pd.DataFrame(rows)


def _axis_style(ax: plt.Axes, ylabel: str, chance: float) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(chance, color=CHANCE_COLOR, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(
        0.995, chance + 0.012, f"chance = {chance:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color=CHANCE_COLOR,
    )


def _draw_info_table(ax: plt.Axes, info: dict) -> None:
    ax.axis("off")
    rows = [
        ["Species",    info["species"]],
        ["Region",     info["brain_region"]],
        ["Recording",  info["recording_tech"]],
        ["N cells",    str(info["n_cells"])],
        ["Cell types", str(info["n_classes"])],
    ]
    tbl = ax.table(
        cellText=rows,
        cellLoc="left",
        colWidths=[0.42, 0.58],
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#cccccc")
        cell.PAD = 0.06
        if col == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold", color="#333")
        else:
            cell.set_facecolor("white")


def plot_bars(df: pd.DataFrame, dataset: str) -> None:
    info = DATASETS[dataset]
    chance = info["chance"]

    ds_df = df[df["dataset"] == dataset]
    method_order = [m for m in METHOD_DISPLAY if not ds_df[ds_df["method"] == m].empty]
    mean_acc = {m: ds_df[ds_df["method"] == m]["balanced_accuracy"].mean() for m in method_order}
    sorted_methods = sorted(method_order, key=lambda m: mean_acc.get(m, 0), reverse=True)
    labels = [METHOD_DISPLAY[m] for m in sorted_methods]
    n = len(sorted_methods)

    bar_w = max(6, n * 0.9)
    fig = plt.figure(figsize=(bar_w + 2.6, 3.6), facecolor="white")
    fig.suptitle(info["display"], fontsize=10, fontweight="bold")

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[2.6 / (bar_w + 2.6), bar_w / 2 / (bar_w + 2.6), bar_w / 2 / (bar_w + 2.6)],
        wspace=0.42,
        left=0.02, right=0.97, top=0.86, bottom=0.22,
    )
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_f1  = fig.add_subplot(gs[0, 2])
    for ax in (ax_tbl, ax_acc, ax_f1):
        ax.set_facecolor("white")

    _draw_info_table(ax_tbl, info)

    rng = np.random.default_rng(7)
    for ax, metric, ylabel in zip(
        [ax_acc, ax_f1],
        ["balanced_accuracy", "macro_f1"],
        ["Balanced accuracy", "Macro F1"],
    ):
        for xi, mkey in enumerate(sorted_methods):
            sub = ds_df[ds_df["method"] == mkey][metric].values
            if len(sub) == 0:
                continue
            mean = float(np.mean(sub))
            sem  = float(np.std(sub, ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
            ax.bar(xi, mean, yerr=sem, color=BAR_COLOR, edgecolor="white",
                   linewidth=0.8, capsize=3, width=0.65,
                   error_kw={"elinewidth": 0.8, "ecolor": "#333"}, zorder=2)
            ax.scatter(
                xi + rng.uniform(-0.08, 0.08, size=len(sub)),
                sub, s=14, color=DOT_COLOR, zorder=3, linewidth=0,
            )
            ax.text(xi, mean + max(sem, 0.01) + 0.015,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        _axis_style(ax, ylabel, chance)

    _save_fig(fig, OUT_DIR / f"{dataset}_bars")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = collect_all()

    csv_path = OUT_DIR / "supp_trimodal_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] {csv_path}")

    summary = (
        df.groupby(["dataset", "method"])[["balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n", summary)

    for ds in DATASETS:
        plot_bars(df, ds)

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
