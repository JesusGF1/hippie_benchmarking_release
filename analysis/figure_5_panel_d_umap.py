"""Figure 5 panel D — joint UMAP of Lisberger and Hull embeddings per method.

For each method loads the train (Lisberger, macaque) and predict (Hull, mouse)
embeddings from the liss→hull cross-dataset run, concatenates them, runs UMAP,
and plots coloured by cell type with circle=macaque / triangle=mouse markers.

Outputs:
    figures/29_figure_5_cross_species/panel_d_umap.{svg,png}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "results" / "benchmark" / "cross_dataset_cache"
OUT_DIR = REPO_ROOT / "figures" / "figure_5"

SHARED = {"PkC_ss", "PkC_cs", "MLI", "MFB"}

CELL_TYPE_COLORS = {
    "PkC_ss": "#e63946",
    "PkC_cs": "#f4a261",
    "MLI":    "#2a9d8f",
    "MFB":    "#457b9d",
}

PAIR = "cross_lisberger_labeled_cell_type_to_hull_cell_type"

# (display label, s3_prefix, train_embed_rel, train_label_col,
#                             pred_embed_rel,  pred_label_col)
# train = Lisberger (macaque, circles); predict/test = Hull (mouse, triangles)
METHODS = [
    ("PhysMAP",
     "physmap-cross-dataset",
     "predictions/embeddings/train_combined_pca.csv", "CellType",
     "predictions/embeddings/predict_combined_pca.csv", "CellType"),
    ("NEMO",
     "nemo-cross-dataset-allpretrain",
     "predictions/cross_dataset/embeddings/train_embeddings.csv", "label",
     "predictions/cross_dataset/embeddings/test_embeddings.csv", "label"),
    ("VAE",
     "vae-cross-dataset",
     "predictions/train_embeddings.csv", "label",
     "predictions/predict_embeddings.csv", "label_original"),
    ("HIPPIE",
     "hippie-cross-dataset-allpretrain",
     "predictions/train_embeddings.csv", "label",
     "predictions/predict_embeddings.csv", "label_original"),
]


def _load_embeddings(prefix: str,
                     train_rel: str, train_lbl_col: str,
                     pred_rel: str, pred_lbl_col: str):
    tp = CACHE / prefix / PAIR / train_rel
    pp = CACHE / prefix / PAIR / pred_rel
    if not tp.exists() or not pp.exists():
        return None, None, None, None

    train_df = pd.read_csv(tp)
    pred_df  = pd.read_csv(pp)

    # Strip quote artifacts from column names (PhysMAP uses "WF_1" etc.)
    train_df.columns = [c.strip('"') for c in train_df.columns]
    pred_df.columns  = [c.strip('"') for c in pred_df.columns]

    exclude_train = {train_lbl_col, "label_training_space", "label_original"}
    exclude_pred  = {pred_lbl_col,  "label_training_space", "label_original"}
    feat_train = [c for c in train_df.columns if c not in exclude_train]
    feat_pred  = [c for c in pred_df.columns  if c not in exclude_pred]

    numeric_train = train_df[feat_train].select_dtypes(include=[np.number]).columns.tolist()
    numeric_pred  = pred_df[feat_pred].select_dtypes(include=[np.number]).columns.tolist()
    shared_feat   = [c for c in numeric_train if c in numeric_pred]

    if not shared_feat:
        X_train = train_df.select_dtypes(include=[np.number]).values
        X_pred  = pred_df.select_dtypes(include=[np.number]).values
        min_dim = min(X_train.shape[1], X_pred.shape[1])
        X_train, X_pred = X_train[:, :min_dim], X_pred[:, :min_dim]
    else:
        X_train = train_df[shared_feat].values
        X_pred  = pred_df[shared_feat].values

    y_train = train_df[train_lbl_col].astype(str).values
    y_pred  = pred_df[pred_lbl_col].astype(str).values

    return X_train, y_train, X_pred, y_pred


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    n_methods = len(METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(4.0 * n_methods, 3.4),
                             facecolor="white")

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                        random_state=42, metric="euclidean")

    for ax, (label, prefix, tr_rel, tr_lbl, pr_rel, pr_lbl) in zip(axes, METHODS):
        res = _load_embeddings(prefix, tr_rel, tr_lbl, pr_rel, pr_lbl)
        X_train, y_train, X_pred, y_pred = res

        if X_train is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="red")
            ax.set_title(label)
            continue

        # Keep only SHARED classes
        mask_tr = np.array([y in SHARED for y in y_train])
        mask_pr = np.array([y in SHARED for y in y_pred])
        X_train, y_train = X_train[mask_tr], y_train[mask_tr]
        X_pred,  y_pred  = X_pred[mask_pr],  y_pred[mask_pr]

        # Joint UMAP
        X_all = np.vstack([X_train, X_pred])
        X_all = StandardScaler().fit_transform(X_all)
        emb   = reducer.fit_transform(X_all)

        n_tr = len(X_train)
        emb_tr, emb_pr = emb[:n_tr], emb[n_tr:]

        # circles = Lisberger (macaque, train source)
        # triangles = Hull (mouse, predict target)
        for ct, col in CELL_TYPE_COLORS.items():
            idx_tr = y_train == ct
            idx_pr = y_pred  == ct
            if idx_tr.any():
                ax.scatter(emb_tr[idx_tr, 0], emb_tr[idx_tr, 1],
                           c=col, s=18, marker="o", alpha=0.75,
                           linewidths=0, zorder=3)
            if idx_pr.any():
                ax.scatter(emb_pr[idx_pr, 0], emb_pr[idx_pr, 1],
                           c=col, s=22, marker="^", alpha=0.75,
                           linewidths=0, zorder=3)

        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", "datalim")

    # Shared legend: cell types + species markers
    ct_handles = [mlines.Line2D([], [], color=col, marker="s", linestyle="",
                                markersize=7, label=ct)
                  for ct, col in CELL_TYPE_COLORS.items()]
    sp_handles = [
        mlines.Line2D([], [], color="#555", marker="o", linestyle="",
                      markersize=6, label="Macaque (Lisberger)"),
        mlines.Line2D([], [], color="#555", marker="^", linestyle="",
                      markersize=6, label="Mouse (Hull)"),
    ]
    fig.legend(handles=ct_handles + sp_handles,
               loc="lower center", ncol=len(ct_handles) + len(sp_handles),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Joint UMAP — Lisberger (macaque) → Hull (mouse) embeddings",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("svg", "png"):
        p = OUT_DIR / f"panel_d_umap.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"[{ext}] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
