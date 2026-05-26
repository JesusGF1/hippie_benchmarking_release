"""Figure 3 — bimodal (WF+ISI) classification benchmark on A1 and S1.

Loads cached 5-fold predictions under
results/benchmark/celltype_cache/ and computes per-fold
balanced accuracy and macro F1 for:
    HIPPIE-bimodal, VAE-bimodal, PhysMAP (WNN), PCA-WF, PCA-ISI.

Emits one bar chart per dataset plus a tidy CSV:
    figures/figure_3/bimodal_metrics.csv
    figures/figure_3/<dataset>_bars.{svg,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "benchmark" / "celltype_cache"
NEGCTRL = REPO_ROOT / "results" / "negative_controls"
OUT_DIR = REPO_ROOT / "figures" / "figure_3"

DATASETS = {
    "a1data_remove_undef": {
        "display": "A1 (Lakunina)",
        "n_classes": 3,
        "chance": 1.0 / 3.0,  # balanced-accuracy chance for 3 classes
    },
    "juxtacellular_mouse_s1_area": {
        "display": "S1 (Yu)",
        "n_classes": 5,
        "chance": 1.0 / 5.0,
    },
}

# Method display names, colors (HIPPIE highlighted blue, others neutral)
# neg_ctrl flag marks shuffle/shift controls — rendered with hatching
METHODS = [
    ("hippie_waveisi", "HIPPIE-bimodal",   "#1f77b4", False),
    ("vae_waveisi",    "VAE-bimodal",      "#1f77b4", False),
    ("physmap",        "PhysMAP (WNN)",    "#1f77b4", False),
    ("pca_wf",         "PCA-WF",           "#1f77b4", False),
    ("pca_isi",        "PCA-ISI",          "#1f77b4", False),
    ("shuffle",        "HIPPIE-Shuffle",   "#ff7f0e", True),
    ("shift",          "HIPPIE-Shift",     "#ff7f0e", True),
]


def _metrics(y_true, y_pred) -> tuple[float, float]:
    return (
        balanced_accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def collect_hippie_like(method_key: str, cache_subdir: str, dataset: str) -> list[dict]:
    rows = []
    for fold in range(5):
        p = CACHE / cache_subdir / dataset / f"fold_{fold}" / "predictions" / "transductive_predictions.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not {"pred", "true"}.issubset(df.columns):
            continue
        acc, f1 = _metrics(df["true"], df["pred"])
        rows.append({"method": method_key, "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1})
    return rows


def collect_physmap_like(method_key: str, fname: str, dataset: str) -> list[dict]:
    p = CACHE / "physmap" / dataset / "fold_0" / "predictions" / fname
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df.columns = [c.strip('"').lower() for c in df.columns]
    if not {"pred", "true", "fold"}.issubset(df.columns):
        return []
    rows = []
    for fold_name, sub in df.groupby("fold"):
        # fold labels look like "Fold1"..."Fold5"
        fold = int(str(fold_name).lower().lstrip("fold")) - 1
        acc, f1 = _metrics(sub["true"], sub["pred"])
        rows.append({"method": method_key, "dataset": dataset, "fold": fold,
                     "balanced_accuracy": acc, "macro_f1": f1})
    return rows


def collect_negctrl(control: str, method_subdir: str, dataset: str) -> list[dict]:
    base = NEGCTRL / control / method_subdir / dataset
    run_file = base / "run_1.csv"
    if not run_file.exists():
        return []
    df = pd.read_csv(run_file)
    if not {"pred", "true"}.issubset(df.columns):
        return []
    acc, f1 = _metrics(df["true"], df["pred"])
    return [{"method": control, "dataset": dataset, "fold": 0,
             "balanced_accuracy": acc, "macro_f1": f1}]


def collect_all() -> pd.DataFrame:
    rows: list[dict] = []
    for ds in DATASETS:
        rows += collect_hippie_like("hippie_waveisi", "hippie-waveisi", ds)
        rows += collect_hippie_like("vae_waveisi",    "vae-waveisi",    ds)
        rows += collect_physmap_like("physmap",       "physmap_CV_results.csv", ds)
        rows += collect_physmap_like("pca_wf",        "PCA_WF_CV_results.csv",  ds)
        rows += collect_physmap_like("pca_isi",       "PCA_ISI_CV_results.csv", ds)
        rows += collect_negctrl("shuffle", "hippie-waveisi", ds)
        rows += collect_negctrl("shift",   "hippie-waveisi", ds)
    if not rows:
        raise SystemExit(f"no predictions found under {CACHE}")
    return pd.DataFrame(rows)


def _axis_style(ax, ylabel: str, title: str, chance: float) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(chance, color="#666", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(
        0.995, chance + 0.015, f"chance = {chance:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="#666",
    )


def _plot_one(ax, data: list[np.ndarray], means: list[float], sems: list[float],
              method_colors: list[str], is_negctrl: list[bool]) -> None:
    x = np.arange(len(data))
    rng = np.random.default_rng(7)
    for xi, mean, sem, col, ctrl in zip(x, means, sems, method_colors, is_negctrl):
        hatch = "////" if ctrl else None
        ec = "#555" if ctrl else "white"
        ax.bar(xi, mean, yerr=sem, color=col, edgecolor=ec, linewidth=0.8,
               capsize=3, width=0.65, hatch=hatch,
               error_kw={"elinewidth": 0.8, "ecolor": "#333"}, zorder=2)
    for xi, pts in zip(x, data):
        ax.scatter(
            xi + rng.uniform(-0.08, 0.08, size=len(pts)),
            pts, s=14, color="#1e3a5f", zorder=3, linewidth=0,
        )


def plot_dataset(df: pd.DataFrame, dataset: str) -> None:
    info = DATASETS[dataset]
    chance = info["chance"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), facecolor="white")
    method_keys   = [m[0] for m in METHODS]
    method_labels = [m[1] for m in METHODS]
    method_colors = [m[2] for m in METHODS]
    is_negctrl    = [m[3] for m in METHODS]

    for ax, metric, ylabel in zip(axes,
                                  ["balanced_accuracy", "macro_f1"],
                                  ["Balanced accuracy", "Macro F1"]):
        data, means, sems = [], [], []
        for mk in method_keys:
            sub = df[(df["dataset"] == dataset) & (df["method"] == mk)][metric].values
            data.append(sub)
            means.append(float(np.mean(sub)) if len(sub) else np.nan)
            sems.append(float(np.std(sub, ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0)

        _plot_one(ax, data, means, sems, method_colors, is_negctrl)
        ax.set_xticks(np.arange(len(method_keys)))
        ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=7)
        _axis_style(ax, ylabel, f"{info['display']} — {ylabel.lower()}", chance)

    fig.suptitle(f"Bimodal (WF+ISI) benchmark — {info['display']} ({info['n_classes']} classes)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    stem = OUT_DIR / f"{dataset}_bars"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    df = collect_all()
    csv = OUT_DIR / "bimodal_metrics.csv"
    df.to_csv(csv, index=False)
    print(f"[csv] {csv}")

    # quick summary print
    summary = (
        df.groupby(["dataset", "method"])[["balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"]).round(3)
    )
    print(summary)

    for ds in DATASETS:
        plot_dataset(df, ds)


if __name__ == "__main__":
    main()
