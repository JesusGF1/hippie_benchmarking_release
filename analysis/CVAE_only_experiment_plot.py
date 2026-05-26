#!/usr/bin/env python3
"""Render Figure 6 panels (CVAE-only generative experiments).

Mirrors the original paper plot (final_paper_figures/.../figure_6_paper.py).
Four panels, written as separate files into --out (default: figures/figure_6/):

  panel_a_cross_species.{svg,png}       — 3 cell types (GoC/MLI/PkC_ss) × 3
                                           modalities (WF/ISI/ACG); macaque mean
                                           (orange), HIPPIE-decoded under mouse
                                           (green dashed), mouse mean ±std
                                           (cell-type colour). All traces
                                           DataLoader-normalised to [-1, 1]
                                           (log first for ISI).
  panel_b_imputation.{svg,png}          — 3×3 cross-modal MSE heatmap (G1).
  panel_c_per_class_accuracy.{svg,png}  — horizontal F1 bars per cell type for
                                           mouse vs macaque (G0).
  panel_d_latent_flow.{svg,png}         — joint UMAP (left) + 5-step waveform
                                           waterfall along the 16-D macaque→mouse
                                           path for the focus cell type (GoC),
                                           with intermediate steps obtained by
                                           5-NN averaging in latent space.

Required input files (in --results-dir):
  G0_per_class_accuracy.csv
  G0_classification_accuracy.csv
  G1_cross_modal_imputation.csv
  G2_cross_species.npz
  latent_flow.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.font_manager as _fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "benchmark" / "cvae_only"
DEFAULT_FIGURES = REPO / "figures" / "figure_6"

_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_body_font = "Arial" if "Arial" in _available_fonts else "DejaVu Sans"

RC = {
    "font.family": _body_font, "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

LABEL_COLORS = {
    "PkC_ss": "#1f77b4", "PkC_cs": "#ff7f0e", "GoC": "#2ca02c",
    "MLI":    "#d62728", "MFB":    "#9467bd", "GrC": "#8c564b",
}
_MAC_COLOR  = "#B85C00"
_PRED_COLOR = "#1A7A40"
_GREY_COL   = "#BBBBBB"
# Macaque = burnt orange (unified across panels A, C, D); mouse = blue.
_HAUSSER_COLOR    = "#1f77b4"
_LISSBERGER_COLOR = _MAC_COLOR
_MAC_FLOW_COL = _MAC_COLOR
_MOU_FLOW_COL = "#1f77b4"

_XSP_CT  = ("GoC", "MLI", "PkC_ss")
_MODS    = (("wave", "Waveform"), ("isi", "ISI"), ("acg", "ACG"))
_CT_ORDER  = ("PkC_ss", "PkC_cs", "GoC", "MFB", "MLI")
_CT_LABELS = {"PkC_ss": "PkC-ss", "PkC_cs": "PkC-cs",
              "GoC": "GoC", "MFB": "MFB", "MLI": "MLI"}
_FLOW_CT_ORDER = ("GoC", "MLI", "MFB", "PkC_cs", "PkC_ss")
_FOCUS_CT = "GoC"
_N_STEPS = 5


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  -> {p}")
    plt.close(fig)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ct_color(ct: str) -> str:
    return LABEL_COLORS.get(ct, "#444444")


def _dl_norm(arr: np.ndarray, log_transform: bool = False,
             std: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray | None]:
    """DataLoader-equivalent min-max normalisation → [-1, 1].

    Matches the training pipeline so shapes are directly comparable across
    macaque mean / decoded / mouse mean traces. Optional log transform applied
    first (for ISI). Std is propagated through delta method (log) + minmax.
    """
    if log_transform:
        if std is not None:
            std = std / (np.abs(arr) + 1.0)
        arr = np.log(np.maximum(arr, 0) + 1.0)
    lo, hi = arr.min(), arr.max()
    rng = hi - lo + 1e-8
    arr_n = (arr - lo) / rng * 2.0 - 1.0
    std_n = (std / (rng / 2.0)) if std is not None else None
    return arr_n, std_n


# ── Panel A: cross-species qualitative (3 cell types × 3 modalities) ─────────

def plot_panel_a(npz: Path, out_dir: Path) -> None:
    if not npz.exists():
        print(f"[plot] skip A: {npz} missing"); return
    plt.rcParams.update(RC)
    d = np.load(str(npz), allow_pickle=True)
    available = list(d["cell_types"])

    fig, axs = plt.subplots(len(_XSP_CT), len(_MODS),
                            figsize=(7.0, 4.4), squeeze=False)

    for ri, ct in enumerate(_XSP_CT):
        ct_color = _ct_color(ct)
        n_mac = int(d[f"n_macaque_{ct}"]) if ct in available else 0
        n_mou = int(d[f"n_mouse_{ct}"])   if ct in available else 0
        for ci, (mod_key, mod_label) in enumerate(_MODS):
            ax = axs[ri][ci]
            if ct not in available:
                ax.axis("off"); continue
            mac_mean = d[f"{ct}_{mod_key}_macaque_mean"].astype(float)
            pred     = d[f"{ct}_{mod_key}_predicted_mouse"].astype(float).squeeze()
            mou_mean = d[f"{ct}_{mod_key}_mouse_mean"].astype(float)
            mou_std  = d[f"{ct}_{mod_key}_mouse_std"].astype(float)

            is_isi = (mod_key == "isi")
            mac_n, _      = _dl_norm(mac_mean, log_transform=is_isi)
            pred_n, _     = _dl_norm(pred,     log_transform=False)
            mou_n, mou_s  = _dl_norm(mou_mean, log_transform=is_isi, std=mou_std)

            x = np.arange(len(mac_n))
            ax.plot(x, mac_n, lw=1.6, color=_MAC_COLOR, alpha=0.9,
                    label=f"Macaque mean (n={n_mac})")
            ax.plot(x, pred_n, lw=1.8, color=_PRED_COLOR, ls="--", alpha=0.95,
                    label="Decoded under mouse")
            ribbon = 0.4 * mou_s if mou_s is not None else np.zeros_like(mou_n)
            ax.fill_between(x, mou_n - ribbon, mou_n + ribbon,
                            alpha=0.22, color=ct_color, linewidth=0)
            ax.plot(x, mou_n, lw=1.6, color=ct_color, alpha=0.85,
                    label=f"Mouse mean (n={n_mou})")

            ax.axhline(0, color="#aaa", lw=0.35, ls=":")
            ax.set_xticks([]); ax.set_yticks([])
            _despine(ax)
            if ri == 0:
                ax.set_title(mod_label, fontsize=8, pad=4, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ct, fontsize=8, fontweight="bold",
                              color=ct_color, rotation=0,
                              labelpad=22, va="center")
                ax.text(-0.32, 0.15,
                        f"mac n={n_mac}\nmouse n={n_mou}",
                        transform=ax.transAxes, fontsize=5.5,
                        color="#666", va="center", ha="right", style="italic")

    handles = [
        Line2D([0], [0], color=_MAC_COLOR,  lw=1.6, alpha=0.9,
               label="Macaque real (population mean)"),
        Line2D([0], [0], color=_PRED_COLOR, lw=1.8, ls="--",
               label="HIPPIE decoded under mouse conditioning"),
        mpatches.Patch(color="#777777", alpha=0.55,
                       label="Mouse real (mean ±0.4 std, cell-type colour)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=7,
               framealpha=0.85, edgecolor="#ccc")
    fig.suptitle(
        "Cross-species generation: encode Lisberger (src=3) → decode under Hausser (src=1)",
        fontsize=9, y=1.0, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_dir / "panel_a_cross_species")


# ── Panel B: cross-modal imputation MSE heatmap ──────────────────────────────

def plot_panel_b(csv: Path, out_dir: Path) -> None:
    if not csv.exists():
        print(f"[plot] skip B: {csv} missing"); return
    plt.rcParams.update(RC)
    df = pd.read_csv(csv)
    sub = df[df["method"] == "hippie"]
    mods = ["wave", "isi", "acg"]
    mat = np.full((3, 3), np.nan)
    for i, masked in enumerate(mods):
        row = sub[sub["masked"] == masked].iloc[0]
        for j, recon in enumerate(mods):
            mat[i, j] = float(row[f"recon_{recon}_mse"])

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(mat, cmap="Blues", aspect="auto",
                   vmin=0, vmax=mat.max() * 1.05)
    labels = ["Waveform", "ISI", "ACG"]
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_yticks(range(3), labels, fontsize=8)
    ax.set_xlabel("Reconstructed modality", fontsize=8)
    ax.set_ylabel("Masked modality", fontsize=8)
    ax.set_title("Cross-modal imputation MSE\n(diagonal = self-reconstruction)",
                 fontsize=9)
    mid = mat.max() / 2
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > mid else "black",
                    fontweight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MSE", fontsize=8)
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_dir / "panel_b_imputation")


# ── Panel C: per-class F1 bars — macaque vs mouse ────────────────────────────

def plot_panel_c(per_class_csv: Path, summary_csv: Path, out_dir: Path) -> None:
    if not per_class_csv.exists():
        print(f"[plot] skip C: {per_class_csv} missing"); return
    plt.rcParams.update(RC)
    df_pc = pd.read_csv(per_class_csv)
    df_su = pd.read_csv(summary_csv) if summary_csv.exists() else None

    hdf = df_pc[df_pc["dataset"] == "hausser_cell_type"].set_index("cell_type")
    ldf = df_pc[df_pc["dataset"] == "lisberger_labeled_cell_type"].set_index("cell_type")
    cts = [ct for ct in _CT_ORDER if ct in hdf.index or ct in ldf.index]
    y = np.arange(len(cts))
    bh = 0.36

    h_f1 = [hdf.loc[ct, "f1"] if ct in hdf.index else np.nan for ct in cts]
    l_f1 = [ldf.loc[ct, "f1"] if ct in ldf.index else np.nan for ct in cts]

    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(cts) + 1.0))
    ax.barh(y + bh / 2, h_f1, height=bh, color=_HAUSSER_COLOR,
            alpha=0.85, label="Mouse (Hausser)")
    for i, lv in enumerate(l_f1):
        if not np.isnan(lv):
            ax.barh(y[i] - bh / 2, lv, height=bh, color=_LISSBERGER_COLOR,
                    alpha=0.85, label="Macaque (Lisberger)" if i == 0 else "_")
    for i, (hv, lv) in enumerate(zip(h_f1, l_f1)):
        if not np.isnan(hv):
            ax.text(hv + 0.01, y[i] + bh / 2, f"{hv:.2f}",
                    va="center", ha="left", fontsize=6.5, color="#222")
        if not np.isnan(lv):
            ax.text(lv + 0.01, y[i] - bh / 2, f"{lv:.2f}",
                    va="center", ha="left", fontsize=6.5, color="#222")
    ax.set_yticks(y, [_CT_LABELS.get(c, c) for c in cts])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("F1 score", fontsize=8)
    ax.axvline(0.5, color="#aaa", lw=0.6, ls="--")
    if df_su is not None:
        hba = df_su.loc[df_su["dataset"] == "hausser_cell_type", "balanced_accuracy"].values[0]
        lba = df_su.loc[df_su["dataset"] == "lisberger_labeled_cell_type", "balanced_accuracy"].values[0]
        ax.set_title(
            "Cell-type decodability from latent codes\n"
            f"(linear probe, balanced acc: mouse {hba:.0%} / macaque {lba:.0%})",
            fontsize=9)
    else:
        ax.set_title("Cell-type decodability — per-class F1", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85, edgecolor="#ccc")
    _despine(ax)
    fig.tight_layout()
    _save(fig, out_dir / "panel_c_per_class_accuracy")


# ── Panel D: joint latent UMAP + macaque→mouse waveform waterfall ────────────

def plot_panel_d(npz: Path, out_dir: Path) -> None:
    if not npz.exists():
        print(f"[plot] skip D: {npz} missing"); return
    try:
        import umap as umap_lib
        from scipy.ndimage import gaussian_filter1d
    except ImportError as e:
        print(f"[plot] skip D: {e}"); return

    plt.rcParams.update(RC)
    d = np.load(str(npz), allow_pickle=True)
    cts = list(d["cell_types"])

    # Stack Z: background first, then focus
    bg_sizes, all_z = [], []
    for ct in _FLOW_CT_ORDER:
        if ct not in cts or ct == _FOCUS_CT:
            continue
        for key in (f"{ct}_z_mouse", f"{ct}_z_macaque"):
            v = d[key]; all_z.append(v); bg_sizes.append(len(v))

    z_focus_mou_all = d[f"{_FOCUS_CT}_z_mouse"].astype(float)
    z_focus_mac_all = d[f"{_FOCUS_CT}_z_macaque"].astype(float)
    all_z += [z_focus_mou_all, z_focus_mac_all]
    Z = np.vstack(all_z)

    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15,
                            min_dist=0.3, random_state=42, verbose=False)
    Z2 = reducer.fit_transform(Z)

    ptr = sum(bg_sizes)
    n_mou = len(z_focus_mou_all); n_mac = len(z_focus_mac_all)
    z2_bg      = Z2[:ptr]
    z2_pkc_mou = Z2[ptr:           ptr + n_mou]
    z2_pkc_mac = Z2[ptr + n_mou:   ptr + n_mou + n_mac]

    wave_mac = d[f"{_FOCUS_CT}_wave_macaque"].astype(float)
    wave_mou = d[f"{_FOCUS_CT}_wave_mouse"].astype(float)

    def _canonical(w: np.ndarray) -> np.ndarray:
        w = w.astype(float)
        lo, hi = w.min(), w.max()
        if abs(hi) > abs(lo):
            w = -w
        lo, hi = w.min(), w.max()
        return gaussian_filter1d((w - lo) / (hi - lo + 1e-8) * 2 - 1, sigma=1)

    wc_mac = np.stack([_canonical(w) for w in wave_mac])
    wc_mou = np.stack([_canonical(w) for w in wave_mou])

    # Choose focus pair: 16-D proximity AND maximum waveform-shape difference.
    diff16 = z_focus_mac_all[:, None, :] - z_focus_mou_all[None, :, :]
    d16_all = np.sqrt((diff16 ** 2).sum(2))
    diff_wf = wc_mac[:, None, :] - wc_mou[None, :, :]
    d_wf    = np.sqrt((diff_wf ** 2).sum(2))
    std_mac = wc_mac.std(axis=1); std_mou = wc_mou.std(axis=1)
    thresh16 = np.percentile(d16_all, 60)
    sel_mask = (d16_all <= thresh16) & (std_mac[:, None] > 0.25) & (std_mou[None, :] > 0.25)
    score = np.where(sel_mask, d_wf, -np.inf)
    i_mac, i_mou = (int(x) for x in np.unravel_index(score.argmax(), score.shape))

    z_focus_mac  = z_focus_mac_all[i_mac]
    z_focus_mou  = z_focus_mou_all[i_mou]
    z2_focus_mac = z2_pkc_mac[i_mac]
    z2_focus_mou = z2_pkc_mou[i_mou]

    # Figure layout: UMAP (left) | waveform waterfall (right)
    fig = plt.figure(figsize=(8.0, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.32)
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_wave = fig.add_subplot(gs[0, 1])

    # ── UMAP scatter ──────────────────────────────────────────────────────────
    mask_mac = np.arange(n_mac) != i_mac
    mask_mou = np.arange(n_mou) != i_mou
    ax_umap.scatter(*z2_bg.T,                s=6,  color=_GREY_COL,     alpha=0.25,
                    linewidths=0, zorder=1)
    ax_umap.scatter(*z2_pkc_mac[mask_mac].T, s=18, color=_MAC_FLOW_COL, marker="^",
                    alpha=0.35, linewidths=0, zorder=2)
    ax_umap.scatter(*z2_pkc_mou[mask_mou].T, s=18, color=_MOU_FLOW_COL, marker="D",
                    alpha=0.35, linewidths=0, zorder=2)
    ax_umap.scatter(*z2_focus_mac[None].T, s=140, color=_MAC_FLOW_COL, marker="^",
                    linewidths=1.2, edgecolors="white", zorder=5)
    ax_umap.scatter(*z2_focus_mou[None].T, s=160, color=_MOU_FLOW_COL, marker="D",
                    linewidths=1.2, edgecolors="white", zorder=5)
    ax_umap.annotate("", xy=z2_focus_mou, xytext=z2_focus_mac,
                     arrowprops=dict(arrowstyle="-|>", color="#111",
                                     lw=2.4, mutation_scale=18), zorder=6)
    ax_umap.set_xlabel("UMAP 1", fontsize=8)
    ax_umap.set_ylabel("UMAP 2", fontsize=8)
    ax_umap.set_title(f"{_FOCUS_CT} latent space\n(macaque n={n_mac}, mouse n={n_mou})",
                      fontsize=9)
    leg_umap = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=_MAC_FLOW_COL,
               markersize=8, label="Macaque (Lisberger)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=_MOU_FLOW_COL,
               markersize=8, label="Mouse (Hausser)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREY_COL,
               markersize=6, label="Other cell types"),
    ]
    ax_umap.legend(handles=leg_umap, fontsize=7, loc="lower right",
                   framealpha=0.9, edgecolor="#ccc")
    _despine(ax_umap)

    # ── Waveform waterfall ────────────────────────────────────────────────────
    z_all_pkc = np.vstack([z_focus_mac_all, z_focus_mou_all])
    wc_all_pkc = np.vstack([wc_mac, wc_mou])

    step_waves = []
    for k in range(_N_STEPS):
        t = k / (_N_STEPS - 1)
        if k == 0:
            step_waves.append(wc_mac[i_mac])
        elif k == _N_STEPS - 1:
            step_waves.append(wc_mou[i_mou])
        else:
            z_interp = z_focus_mac * (1 - t) + z_focus_mou * t
            d16 = np.linalg.norm(z_all_pkc - z_interp, axis=1)
            nearest = np.argsort(d16)[:5]
            step_waves.append(wc_all_pkc[nearest].mean(axis=0))

    def _blend(t: float) -> tuple[float, float, float]:
        r0, g0, b0 = mc.to_rgb(_MAC_FLOW_COL)
        r1, g1, b1 = mc.to_rgb(_MOU_FLOW_COL)
        return (r0 * (1 - t) + r1 * t,
                g0 * (1 - t) + g1 * t,
                b0 * (1 - t) + b1 * t)

    all_wn = np.stack(step_waves)
    row_means = all_wn.mean(axis=1, keepdims=True)
    dev = np.abs(all_wn - row_means).max(axis=0)
    active = np.where(dev > 0.05)[0]
    if len(active) == 0:
        active = np.where(np.abs(all_wn).max(axis=0) > 0.10)[0]
    sl = slice(max(0, active[0] - 4), min(all_wn.shape[1], active[-1] + 5))
    x_samp = np.arange(sl.stop - sl.start)

    row_height = 3.2
    trace_scale = row_height * 0.44

    for k in range(_N_STEPS):
        t = k / (_N_STEPS - 1)
        col = _blend(t)
        y0 = (_N_STEPS - 1 - k) * row_height
        wave = step_waves[k][sl]
        wc = wave - wave.mean()
        wamp = np.abs(wc).max()
        w_plot = (wc / (wamp + 1e-8)) * trace_scale + y0
        ax_wave.plot(x_samp, w_plot, lw=2.0, color=col, zorder=3)
        ax_wave.fill_between(x_samp, y0, w_plot, alpha=0.20, color=col, zorder=2)
        if k == 0:
            lbl, fw = "Macaque", "bold"
        elif k == _N_STEPS - 1:
            lbl, fw = "Mouse", "bold"
        else:
            lbl, fw = f"Step {k + 1}", "normal"
        ax_wave.text(-len(x_samp) * 0.04, y0, lbl, ha="right", va="center",
                     fontsize=7, color=col, fontweight=fw)

    n_disp = len(x_samp)
    y_top = (_N_STEPS - 1) * row_height + trace_scale + 0.3
    y_bot = -trace_scale - 0.3
    ax_wave.annotate("", xy=(n_disp * 1.10, y_bot), xytext=(n_disp * 1.10, y_top),
                     arrowprops=dict(arrowstyle="-|>", color="#555",
                                     lw=1.2, mutation_scale=10),
                     annotation_clip=False)
    ax_wave.text(n_disp * 1.13, (y_top + y_bot) / 2, "Macaque\n→\nMouse",
                 ha="left", va="center", fontsize=6, color="#555", clip_on=False)
    ax_wave.set_xlim(-n_disp * 0.24, n_disp * 1.04)
    ax_wave.set_ylim(y_bot - 0.4, y_top + 0.4)
    ax_wave.set_xticks([]); ax_wave.set_yticks([])
    ax_wave.set_title(f"{_FOCUS_CT} spike waveform along 16-D interpolation",
                      fontsize=9)
    ax_wave.spines["left"].set_visible(False)
    ax_wave.spines["bottom"].set_visible(False)
    _despine(ax_wave)

    fig.tight_layout()
    _save(fig, out_dir / "panel_d_latent_flow")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_FIGURES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = args.results_dir

    plot_panel_a(R / "G2_cross_species.npz",            args.out)
    plot_panel_b(R / "G1_cross_modal_imputation.csv",   args.out)
    plot_panel_c(R / "G0_per_class_accuracy.csv",
                 R / "G0_classification_accuracy.csv",  args.out)
    plot_panel_d(R / "latent_flow.npz",                 args.out)


if __name__ == "__main__":
    main()
