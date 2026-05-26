"""Figure 3 panel C — L4/L5 misclassification analysis on S1 (Yu).

Three groups of excitatory cells are compared:
    1. L4 correct             (true=E_L4, HIPPIE pred=E_L4)
    2. L5 correct             (true=E_L5, HIPPIE pred=E_L5)
    3. L5→L4 misclassified    (true=E_L5, HIPPIE pred=E_L4)

Panels:
    Left  — mean waveform ± SEM per group with cosine-similarity to
            the L4-correct prototype annotated, plus shading for time
            bins where L5-correct differs from L5-misclassified
            (Welch t-test, p ≤ 0.05)
    Right — mean ISI distribution ± SEM per group with the same
            similarity annotations

Cached predictions are aligned to raw rows via the same
StratifiedKFold(n_splits=5, shuffle=True, random_state=42) used by
the benchmark trainer; match verified at 1.000 across all folds.

Outputs:
    figures/26_figure_3_l4l5/l4l5_waveform_isi.{svg,png}
    figures/26_figure_3_l4l5/l4l5_counts.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "results" / "benchmark" / "cache_datasets" / "juxtacellular_mouse_s1_area"
CACHE = REPO_ROOT / "results" / "benchmark" / "celltype_cache" / "hippie-waveisi" / "juxtacellular_mouse_s1_area"
OUT_DIR = REPO_ROOT / "figures" / "figure_3"

GROUPS = [
    ("L4 correct",            "#ff7f0e"),
    ("L5 correct",            "#2ca02c"),
    ("L5→L4 misclassified",   "#d62728"),
]


def _read_numeric(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0].lower() in {"unnamed: 0", ""} or df.columns[0] == "0":
        df = df.iloc[:, 1:]
    return df.values.astype(np.float32)


def assign_groups() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = pd.read_csv(DATA / "labels.csv").iloc[:, 0].astype(str).values
    wf = _read_numeric(DATA / "waveforms.csv")
    isi = _read_numeric(DATA / "isi_dist.csv")

    # normalize per-cell: wf z-scored along time, isi density-normalized
    wf_z = (wf - wf.mean(axis=1, keepdims=True)) / (wf.std(axis=1, keepdims=True) + 1e-8)
    isi_d = isi / (isi.sum(axis=1, keepdims=True) + 1e-8)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds_full = np.empty(len(labels), dtype=object)
    for fold, (_, test_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        pred_file = CACHE / f"fold_{fold}" / "predictions" / "transductive_predictions.csv"
        preds = pd.read_csv(pred_file)
        if len(preds) != len(test_idx):
            raise RuntimeError(f"fold {fold}: preds len {len(preds)} vs test_idx len {len(test_idx)}")
        preds_full[test_idx] = preds["pred"].astype(str).values

    group = np.full(len(labels), "", dtype=object)
    group[(labels == "E_L4") & (preds_full == "E_L4")] = "L4 correct"
    group[(labels == "E_L5") & (preds_full == "E_L5")] = "L5 correct"
    group[(labels == "E_L5") & (preds_full == "E_L4")] = "L5→L4 misclassified"

    return wf_z, isi_d, group


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _sig_bin_fraction(data: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                      alpha: float = 0.05) -> float:
    if mask_a.sum() < 2 or mask_b.sum() < 2:
        return float("nan")
    _, p = ttest_ind(data[mask_a], data[mask_b], axis=0, equal_var=False, nan_policy="omit")
    return float((p <= alpha).mean())


def _plot_modality(ax, data: np.ndarray, group: np.ndarray,
                   title: str, xlabel: str, ylabel: str,
                   legend_loc: str = "upper left") -> dict:
    x = np.arange(data.shape[1])
    means = {}
    for name, color in GROUPS:
        m = (group == name)
        if m.sum() == 0:
            continue
        mean = data[m].mean(axis=0)
        sem = data[m].std(axis=0, ddof=1) / np.sqrt(m.sum())
        means[name] = mean
        ax.plot(x, mean, color=color, linewidth=1.5,
                label=f"{name} (n={int(m.sum())})")
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)

    # highlight bins where L5 correct vs L5→L4 misclassified differ (p ≤ 0.05)
    m_c5 = (group == "L5 correct")
    m_mc = (group == "L5→L4 misclassified")
    if m_c5.sum() >= 2 and m_mc.sum() >= 2:
        _, p = ttest_ind(data[m_c5], data[m_mc], axis=0, equal_var=False, nan_policy="omit")
        sig = p <= 0.05
        if sig.any():
            runs = _runs(sig)
            for a, b in runs:
                ax.axvspan(a - 0.5, b + 0.5, ymin=0.0, ymax=0.04,
                           color="#555", alpha=0.45, lw=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, fontsize=7, frameon=True,
              facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    return means


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _annotate_cosine(ax, means: dict, modality: str) -> None:
    if "L4 correct" not in means:
        return
    ref = means["L4 correct"]
    short = {"L5 correct": "L5c", "L5→L4 misclassified": "L5→L4"}
    lines = []
    for name in ("L5 correct", "L5→L4 misclassified"):
        if name in means:
            lines.append(f"cos({short[name]}, L4c) = {_cosine(means[name], ref):.3f}")
    if not lines:
        return
    ax.text(
        0.98, 0.97, "\n".join(lines),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7",
                  edgecolor="#bbb", linewidth=0.6),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    wf, isi, group = assign_groups()

    counts = {name: int((group == name).sum()) for name, _ in GROUPS}
    pd.DataFrame([{"group": k, "n": v} for k, v in counts.items()]).to_csv(
        OUT_DIR / "l4l5_counts.csv", index=False
    )
    print(f"Group counts: {counts}")

    # Significance fractions
    m_c5 = (group == "L5 correct")
    m_mc = (group == "L5→L4 misclassified")
    frac_wf = _sig_bin_fraction(wf, m_c5, m_mc)
    frac_isi = _sig_bin_fraction(isi, m_c5, m_mc)
    print(f"Sig bin fraction (p≤0.05) — WF: {frac_wf:.3f}   ISI: {frac_isi:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")

    m_wf = _plot_modality(axes[0], wf, group,
                          "Average waveform per predicted class",
                          "Time (samples)", "Amplitude (norm.)",
                          legend_loc="upper left")
    _annotate_cosine(axes[0], m_wf, "WF")
    axes[0].text(0.02, 0.97,
                 f"sig bins: {100*frac_wf:.1f}%",
                 transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=7, color="#555")

    m_isi = _plot_modality(axes[1], isi, group,
                           "Average ISI distribution per predicted class",
                           "ISI bin", "Density (norm.)",
                           legend_loc="center right")
    _annotate_cosine(axes[1], m_isi, "ISI")
    axes[1].text(0.02, 0.97,
                 f"sig bins: {100*frac_isi:.1f}%",
                 transform=axes[1].transAxes, ha="left", va="top",
                 fontsize=7, color="#555")

    fig.suptitle("S1 (Yu) — Excitatory L4/L5 misclassification analysis",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    stem = OUT_DIR / "l4l5_waveform_isi"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[svg] {stem}.svg")
    print(f"[png] {stem}.png")


if __name__ == "__main__":
    main()
