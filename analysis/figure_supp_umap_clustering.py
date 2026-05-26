"""Figure S_UMAP — Raw-feature UMAP comparison + HDBSCAN modality overlay.

Two sections per dataset (Lisberger primary, Hull secondary):

  Row A — UMAP colored by: True labels | HIPPIE | NEMO | PhysMAP predictions
           (raw-feature UMAP: WF + ISI + ACG concatenated, all labeled neurons)

  Row B — Same UMAP, HDBSCAN clusters, with average modality profiles
           overlaid at each cluster centroid (WF panel | ISI panel | ACG panel)

Data sources:
  - Raw features: local C4 database H5 files (no S3 download needed)
  - Predictions:  results/benchmark/trimodal_cache/ (pooled across 5 folds)

Usage:
    python analysis/figure_supp_umap_clustering.py [--dataset lisberger|hull|both]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "final_paper_figures" / "figs" / "figure_5"

LABEL_COLORS = {
    "PkC_ss": "#1f77b4",
    "PkC_cs": "#ff7f0e",
    "GoC":    "#2ca02c",
    "MLI":    "#d62728",
    "MFB":    "#9467bd",
}
HDBSCAN_CMAP = plt.get_cmap("tab10")

# ─── Dataset config ───────────────────────────────────────────────────────────

DATASETS = {
    "lisberger": {
        "h5":        REPO_ROOT / "data_wrangling/wrangling_C4_Database/C4_database_lisberger.h5",
        "h5_format": "flat",       # top-level groups = neurons
        "label_field": "expert_label",
        "classes":   ["GoC", "MFB", "MLI", "PkC_cs", "PkC_ss"],
        "sr":        40000,
        "cache_dir": REPO_ROOT / "results/benchmark/trimodal_cache",
        "cache_key": "lisberger_labeled_cell_type",
        "display":   "Lisberger (macaque cerebellum)",
        "wf_samples": 1,           # juxtacellular: single channel
    },
    "hull": {
        "h5":        REPO_ROOT / "data_wrangling/wrangling_C4_Database/C4_database_hull_labelled.h5",
        "h5_format": "nested",     # datasets → session → neuron_rel_id
        "label_field": "ground_truth_label",
        "classes":   ["MFB", "MLI", "PkC_cs", "PkC_ss"],   # GoC has only 2 neurons
        "sr":        30000,
        "cache_dir": REPO_ROOT / "results/benchmark/trimodal_cache",
        "cache_key": "hull_cell_type",
        "display":   "Hull (mouse cerebellum)",
        "wf_samples": 22,          # silicon probe: 22 channels
    },
}

N_FOLDS     = 5
ISI_BINS    = np.linspace(-0.3, 2.7, 31)   # log10(ms), 30 bins: 0.5–500 ms
ACG_WIN_MS  = 100.0
ACG_BIN_MS  = 1.0
ACG_NBINS   = int(ACG_WIN_MS / ACG_BIN_MS)  # 100 bins


# ─── Style ────────────────────────────────────────────────────────────────────

def _set_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ─── Feature extraction from C4 H5 ───────────────────────────────────────────

def _peak_waveform(wf: np.ndarray) -> np.ndarray:
    """Extract and L∞-normalise the peak-channel waveform → 90 samples."""
    if wf.ndim == 2:
        if wf.shape[1] == 1:
            wf = wf[:, 0]
        else:
            peak_ch = int(np.argmax(np.abs(wf).max(axis=1)))
            wf = wf[peak_ch]
    # Resample to 90 samples
    n = len(wf)
    idx = np.round(np.linspace(0, n - 1, 90)).astype(int)
    wf = wf[idx].astype(np.float64)
    mx = np.abs(wf).max()
    return wf / mx if mx > 0 else wf


def _isi_hist(spike_idx: np.ndarray, sr: int) -> np.ndarray:
    """Log10(ISI_ms) histogram, 30 bins, 0.5–500 ms."""
    if len(spike_idx) < 2:
        return np.zeros(len(ISI_BINS) - 1)
    isi_ms = np.diff(spike_idx.astype(np.float64)) / sr * 1000.0
    isi_ms = isi_ms[(isi_ms > 0)]
    if len(isi_ms) == 0:
        return np.zeros(len(ISI_BINS) - 1)
    log_isi = np.log10(np.clip(isi_ms, 10 ** ISI_BINS[0], 10 ** ISI_BINS[-1]))
    hist, _ = np.histogram(log_isi, bins=ISI_BINS)
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(float)


def _acg_1d(spike_idx: np.ndarray, sr: int) -> np.ndarray:
    """Fast 1D autocorrelogram using searchsorted. ACG_WIN_MS × 1ms bins."""
    n_bins = ACG_NBINS
    acg = np.zeros(n_bins)
    if len(spike_idx) < 2:
        return acg
    spikes = np.sort(spike_idx.astype(np.int64))
    win_samples = int(ACG_WIN_MS * sr / 1000)
    bin_samples = max(1, sr // 1000)   # samples per 1 ms bin

    # Subsample for very large spike trains to keep runtime <1s per neuron
    if len(spikes) > 20_000:
        rng = np.random.default_rng(0)
        spikes = spikes[rng.choice(len(spikes), 20_000, replace=False)]
        spikes.sort()

    for s in spikes:
        lo = np.searchsorted(spikes, s + 1)
        hi = np.searchsorted(spikes, s + win_samples + 1)
        diffs = spikes[lo:hi] - s
        bins = diffs // bin_samples
        bins = bins[bins < n_bins]
        acg[bins] += 1

    total = acg.sum()
    return acg / total if total > 0 else acg


def load_dataset(name: str) -> dict:
    """Load raw WF, ISI, ACG, labels from C4 H5. Returns dict of arrays."""
    cfg = DATASETS[name]
    h5_path = cfg["h5"]
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    classes = set(cfg["classes"])
    label_field = cfg["label_field"]
    sr = cfg["sr"]

    wfs, isis, acgs, labels, neuron_ids = [], [], [], [], []

    with h5py.File(h5_path, "r") as f:
        if cfg["h5_format"] == "flat":
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
            groups = [(k, f[k]) for k in keys]
        else:
            groups = []
            for sess_key in sorted(f["datasets"].keys()):
                sess = f["datasets"][sess_key]
                for nid_key in sorted(sess.keys(), key=lambda x: int(x)):
                    groups.append((f"{sess_key}/{nid_key}", sess[nid_key]))

        for gid, grp in groups:
            lbl_raw = grp[label_field][()]
            lbl = lbl_raw.decode().strip() if isinstance(lbl_raw, bytes) else str(lbl_raw).strip()
            if lbl not in classes:
                continue

            wf_raw = grp["mean_waveform_preprocessed"][:].astype(np.float64)
            if wf_raw.ndim == 2 and wf_raw.shape[1] == 1:
                wf_raw = wf_raw.T  # (1, 240) → peak_waveform handles

            spike_idx = grp["spike_indices"][:].astype(np.int64)
            sr_actual = int(grp["sampling_rate"][()])

            wf90 = _peak_waveform(wf_raw)
            isi  = _isi_hist(spike_idx, sr_actual)
            acg  = _acg_1d(spike_idx, sr_actual)

            wfs.append(wf90)
            isis.append(isi)
            acgs.append(acg)
            labels.append(lbl)
            neuron_ids.append(gid)

    return {
        "wf":   np.array(wfs),
        "isi":  np.array(isis),
        "acg":  np.array(acgs),
        "labels": np.array(labels),
        "ids":  neuron_ids,
        "n":    len(labels),
    }


# ─── Load pooled predictions from trimodal cache ──────────────────────────────

def _load_preds_method(cache_key: str, method: str, cache_dir: Path) -> pd.DataFrame | None:
    """Pool all 5 folds for a given method. Returns DataFrame with pred/true columns."""
    if method == "physmap":
        files = [
            cache_dir / method / cache_key / "fold_0" / "physmap_CV_results.csv",
        ]
    else:
        files = [
            cache_dir / method / cache_key / f"fold_{f}" / "transductive_predictions.csv"
            for f in range(N_FOLDS)
        ]

    dfs = []
    for p in files:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.strip().strip('"').lower() for c in df.columns]
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_predictions(dataset_name: str) -> dict[str, pd.DataFrame]:
    cfg = DATASETS[dataset_name]
    cache_dir = cfg["cache_dir"]
    cache_key  = cfg["cache_key"]
    out = {}
    for method in ("hippie", "nemo", "physmap"):
        df = _load_preds_method(cache_key, method, cache_dir)
        if df is not None and "true" in df.columns and "pred" in df.columns:
            out[method] = df[["true", "pred"]].reset_index(drop=True)
    return out


# ─── UMAP + HDBSCAN ───────────────────────────────────────────────────────────

def compute_umap(features: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn")
    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=20, min_dist=0.05, metric="euclidean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.fit_transform(features)


def compute_hdbscan(coords: np.ndarray, min_cluster_size: int = 8) -> np.ndarray:
    try:
        import hdbscan
    except ImportError:
        raise ImportError("pip install hdbscan")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=5, cluster_selection_method="eom")
    clusterer.fit(coords)
    return clusterer.labels_


# ─── Scatter helpers ──────────────────────────────────────────────────────────

def _scatter(ax, coords, color_arr, alpha=0.7, s=12, title="",
             legend_handles=None):
    ax.scatter(coords[:, 0], coords[:, 1], c=color_arr,
               s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=6, frameon=True,
                  framealpha=0.9, edgecolor="#ccc", loc="best",
                  markerscale=1.2, handlelength=1)


def _label_color_array(labels: np.ndarray, color_map: dict) -> list:
    return [color_map.get(l, "#999999") for l in labels]


def _plot_profile(ax, modality, mean_profile, color, bin_edges=None, label=None):
    """Draw average WF / ISI / ACG on ax."""
    xs = np.arange(len(mean_profile))
    if modality == "wf":
        ax.plot(xs, mean_profile, color=color, lw=1.2, label=label)
        ax.axhline(0, color="#ccc", lw=0.5, ls="--")
    elif modality == "isi":
        if bin_edges is not None:
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(centers, mean_profile, color=color, lw=1.2, label=label)
        else:
            ax.plot(xs, mean_profile, color=color, lw=1.2, label=label)
    elif modality == "acg":
        ax.plot(xs[:50], mean_profile[:50], color=color, lw=1.2, label=label)


# ─── Panel A: method comparison row ──────────────────────────────────────────

def plot_method_comparison(fig, gs_row, coords, labels, preds_dict, dataset_name):
    """4 subplots: true | HIPPIE | NEMO | PhysMAP, sharing same UMAP layout."""
    classes = DATASETS[dataset_name]["classes"]
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS.get(c, "#999"),
               markersize=5, label=c)
        for c in sorted(classes)
    ]

    panels = [
        ("True labels",  labels,                                None),
        ("HIPPIE",       preds_dict.get("hippie"),              "hippie"),
        ("NEMO",         preds_dict.get("nemo"),                "nemo"),
        ("PhysMAP (WNN)",preds_dict.get("physmap"),             "physmap"),
    ]

    n_neurons = len(labels)

    for col, (title, data, method) in enumerate(panels):
        ax = fig.add_subplot(gs_row[col])

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888")
            ax.set_title(title, fontsize=9)
            continue

        if isinstance(data, pd.DataFrame):
            # Method predictions: try to align by index
            pred_labels = data["pred"].values[:n_neurons]
            true_labels = data["true"].values[:n_neurons]
            n_used = min(len(pred_labels), n_neurons)
            color_arr = _label_color_array(pred_labels[:n_used], LABEL_COLORS)
            # Fill remaining with grey if mismatch
            if n_used < n_neurons:
                color_arr += ["#cccccc"] * (n_neurons - n_used)
            ax.set_title(f"{title}", fontsize=9)
        else:
            color_arr = _label_color_array(data, LABEL_COLORS)
            ax.set_title(title, fontsize=9)

        _scatter(ax, coords, color_arr, title="",
                 legend_handles=legend_handles if col == 0 else None)
        ax.set_title(title, fontsize=9, pad=3)


# ─── Panel B: HDBSCAN modality overlay row ───────────────────────────────────

def plot_hdbscan_overlay(fig, gs_row, coords, cluster_ids,
                         data_dict, dataset_name):
    """Layout per modality: UMAP scatter (left, 2/3 width) + profiles (right, 1/3)."""

    unique_clusters = sorted(set(cluster_ids))
    cluster_colors = {c: HDBSCAN_CMAP(i % 10) for i, c in enumerate(unique_clusters) if c >= 0}

    modalities = [
        ("wf",  "avg. waveform",  data_dict["wf"],  "WF (norm.)", "samples"),
        ("isi", "avg. ISI",       data_dict["isi"],  "Prob.",      "log10 ISI (ms)"),
        ("acg", "avg. ACG",       data_dict["acg"],  "Prob.",      "lag (ms)"),
    ]

    # Compute cluster info once
    cluster_info = {}
    for c in unique_clusters:
        if c < 0:
            continue
        mask = cluster_ids == c
        cluster_info[c] = {
            "centroid": coords[mask].mean(axis=0),
            "wf":  data_dict["wf"][mask].mean(axis=0),
            "isi": data_dict["isi"][mask].mean(axis=0),
            "acg": data_dict["acg"][mask].mean(axis=0),
            "n":   mask.sum(),
            "color": cluster_colors[c],
        }

    noise_mask  = cluster_ids < 0
    member_mask = cluster_ids >= 0

    for col, (mod, mod_label, mod_data, ylabel, xlabel) in enumerate(modalities):
        # Use nested gridspec: 2 cols inside each gs_row cell (3:2 ratio)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_row[col], width_ratios=[3, 2], wspace=0.05)

        ax_umap  = fig.add_subplot(gs_inner[0])
        ax_prof  = fig.add_subplot(gs_inner[1])

        # --- UMAP scatter ---
        if noise_mask.any():
            ax_umap.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                            c="#d0d0d0", s=6, alpha=0.4, linewidths=0, rasterized=True)
        mc = [cluster_colors.get(c, "#999") for c in cluster_ids[member_mask]]
        if member_mask.any():
            ax_umap.scatter(coords[member_mask, 0], coords[member_mask, 1],
                            c=mc, s=10, alpha=0.8, linewidths=0, rasterized=True)

        # Cluster centroid markers + labels
        for c, info in cluster_info.items():
            cx, cy = info["centroid"]
            ax_umap.scatter(cx, cy, s=60, marker="*", color=info["color"],
                            edgecolors="white", linewidths=0.6, zorder=5)
            ax_umap.text(cx, cy + 0.3, str(c), ha="center", va="bottom",
                         fontsize=6, fontweight="bold", color=info["color"])

        ax_umap.set_xticks([]); ax_umap.set_yticks([])
        for sp in ax_umap.spines.values():
            sp.set_visible(False)
        ax_umap.set_title(f"HDBSCAN clusters\n+ {mod_label}", fontsize=8, pad=3)

        # --- Profile panel ---
        for c, info in sorted(cluster_info.items()):
            _plot_profile(ax_prof, mod, info[mod], info["color"],
                          bin_edges=ISI_BINS if mod == "isi" else None,
                          label=f"C{c} (n={info['n']})")

        ax_prof.set_xlabel(xlabel, fontsize=7)
        ax_prof.set_ylabel(ylabel, fontsize=7)
        ax_prof.tick_params(labelsize=6)
        ax_prof.spines["top"].set_visible(False)
        ax_prof.spines["right"].set_visible(False)
        ax_prof.legend(fontsize=6, frameon=False, loc="upper right")


# ─── Feature normalisation ────────────────────────────────────────────────────

def _normalise_features(data: dict) -> np.ndarray:
    wf_n  = data["wf"]  / (np.abs(data["wf"]).max(axis=1, keepdims=True) + 1e-8)
    isi_n = data["isi"]
    acg_n = data["acg"] / (data["acg"].max(axis=1, keepdims=True) + 1e-8)
    return np.concatenate([wf_n, isi_n, acg_n], axis=1).astype(np.float32)


# ─── Joint figure (both species in same UMAP space) ──────────────────────────

DATASET_MARKERS = {"lisberger": "o", "hull": "^"}
DATASET_COLORS  = {"lisberger": "#4477aa", "hull": "#ee6677"}
DATASET_LABELS  = {"lisberger": "Lisberger (macaque)", "hull": "Hull (mouse)"}


def _joint_scatter_celltypes(ax, coords_l, labels_l, coords_h, labels_h):
    """UMAP scatter: color = cell type, marker = species."""
    from matplotlib.lines import Line2D

    for lbl, c in sorted(LABEL_COLORS.items()):
        mask_l = labels_l == lbl
        mask_h = labels_h == lbl
        if mask_l.any():
            ax.scatter(coords_l[mask_l, 0], coords_l[mask_l, 1],
                       c=c, marker="o", s=12, alpha=0.75, linewidths=0, rasterized=True)
        if mask_h.any():
            ax.scatter(coords_h[mask_h, 0], coords_h[mask_h, 1],
                       c=c, marker="^", s=14, alpha=0.75, linewidths=0, rasterized=True)

    # Legend: cell types
    type_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS.get(c, "#999"),
               markersize=5, label=c)
        for c in sorted(LABEL_COLORS)
    ]
    # Species marker legend
    species_handles = [
        Line2D([0], [0], marker="o", color="#666", markersize=5, linestyle="",
               label="Lisberger (macaque)"),
        Line2D([0], [0], marker="^", color="#666", markersize=5, linestyle="",
               label="Hull (mouse)"),
    ]
    leg1 = ax.legend(handles=type_handles, fontsize=6, frameon=True,
                     framealpha=0.9, edgecolor="#ccc", loc="upper left",
                     title="Cell type", title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=species_handles, fontsize=6, frameon=True,
              framealpha=0.9, edgecolor="#ccc", loc="lower right",
              title="Dataset", title_fontsize=6)


def _joint_scatter_species(ax, coords_l, coords_h):
    """UMAP scatter: color = species."""
    from matplotlib.lines import Line2D
    ax.scatter(coords_l[:, 0], coords_l[:, 1],
               c=DATASET_COLORS["lisberger"], marker="o", s=10, alpha=0.65,
               linewidths=0, rasterized=True, label=DATASET_LABELS["lisberger"])
    ax.scatter(coords_h[:, 0], coords_h[:, 1],
               c=DATASET_COLORS["hull"], marker="^", s=12, alpha=0.65,
               linewidths=0, rasterized=True, label=DATASET_LABELS["hull"])
    ax.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="#ccc",
              loc="best", markerscale=1.2)


def _ax_clean(ax, title=""):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def build_joint_figure() -> None:
    """Joint UMAP of Hull + Lisberger in the same embedding space."""
    print("[joint] Loading Lisberger …")
    data_l = load_dataset("lisberger")
    print(f"  {data_l['n']} neurons — {dict(zip(*np.unique(data_l['labels'], return_counts=True)))}")

    print("[joint] Loading Hull …")
    data_h = load_dataset("hull")
    print(f"  {data_h['n']} neurons — {dict(zip(*np.unique(data_h['labels'], return_counts=True)))}")

    feat_l = _normalise_features(data_l)
    feat_h = _normalise_features(data_h)
    features = np.vstack([feat_l, feat_h])
    n_l = data_l["n"]

    print(f"[joint] Computing UMAP (n={len(features)}, d={features.shape[1]}) …")
    coords = compute_umap(features)
    coords_l = coords[:n_l]
    coords_h = coords[n_l:]

    # Merge for HDBSCAN
    min_cs = max(10, len(features) // 30)
    print(f"[joint] Running HDBSCAN (min_cluster_size={min_cs}) …")
    cluster_ids = compute_hdbscan(coords, min_cluster_size=min_cs)
    print(f"  {len(set(cluster_ids)) - 1} clusters found, {(cluster_ids >= 0).sum()} neurons assigned")

    # Merge data dicts for profiles
    data_joint = {
        "wf":  np.vstack([data_l["wf"],  data_h["wf"]]),
        "isi": np.vstack([data_l["isi"], data_h["isi"]]),
        "acg": np.vstack([data_l["acg"], data_h["acg"]]),
        "n":   len(features),
    }

    # ── Layout ────────────────────────────────────────────────────────────────
    _set_style()
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle("Raw-feature UMAP — macaque (Lisberger) + mouse (Hull) cerebellum",
                 fontsize=11, fontweight="bold", y=0.99)

    # Row A: 2 overview panels (cell types | species) + 1 empty spacer (merged below)
    # Row B: 3 HDBSCAN+profile panels
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1, 1.2],
                           hspace=0.35, wspace=0.08,
                           left=0.03, right=0.99, top=0.95, bottom=0.05)

    # Row A
    ax_ct  = fig.add_subplot(gs[0, 0])   # colored by cell type
    ax_sp  = fig.add_subplot(gs[0, 1])   # colored by species
    # gs[0, 2] left empty (or could add a legend/text box)

    _joint_scatter_celltypes(ax_ct, coords_l, data_l["labels"], coords_h, data_h["labels"])
    _ax_clean(ax_ct, "Cell type  (circle=macaque, triangle=mouse)")

    _joint_scatter_species(ax_sp, coords_l, coords_h)
    _ax_clean(ax_sp, "Species")

    # Row B: HDBSCAN + profiles
    row_b = [gs[1, c] for c in range(3)]
    plot_hdbscan_overlay(fig, row_b, coords, cluster_ids, data_joint, "joint")

    # Panel labels
    positions = [(0,0),(0,1),(1,0),(1,1),(1,2)]
    for letter, (row, col) in zip("ABCDE", positions):
        pos = gs[row, col].get_position(fig)
        fig.text(pos.x0, pos.y1 + 0.01, letter,
                 fontsize=13, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = OUT_DIR / f"joint_umap_clustering.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  [{ext}] {out}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="both",
                   choices=["lisberger", "hull", "both"])
    args = p.parse_args(argv)

    if args.dataset == "both":
        build_joint_figure()
    else:
        # Single-dataset fallback (for debugging)
        data = load_dataset(args.dataset)
        print(f"  {data['n']} neurons loaded")
        feat = _normalise_features(data)
        coords = compute_umap(feat)
        min_cs = max(10, data["n"] // 30)
        cluster_ids = compute_hdbscan(coords, min_cluster_size=min_cs)
        print(f"  {len(set(cluster_ids))-1} clusters found")


if __name__ == "__main__":
    main()
