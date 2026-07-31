#!/usr/bin/env python3
"""Polarity supplement regeneration — uses the correct per-figure prediction caches.

Produces updated Table 1 (between-dataset) and Table 2 (within-dataset)
for the Supplementary Note "Waveform polarity does not confound cell-type
classification accuracy."

Cache mapping (matches the main figures):
  Hull, Lisberger       -> trimodal_cache/hippie-fullcond  (Figure 4)
  Hausser, CellExplorer,
  DANDI-041              -> celltype_cache/hippie            (Figure 3 WF+ISI+ACG)
  A1, Juxtacellular      -> celltype_cache/hippie-waveisi   (Figure 3 bimodal)
  DANDI-473, DANDI-955   -> celltype_cache/hippie (isiacg/) (Figure 3 ISI+ACG)

Usage:
    cd hippie_benchmarking
    python analysis/figure_polarity_supplement.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "datasets"
RESULTS_DIR  = REPO / "results" / "benchmark"

OUT_DIR = REPO / "figures" / "supplement" / "polarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset → prediction cache mapping
# ---------------------------------------------------------------------------
# Each entry: (cache_root, pred_subpath_template)
# pred_subpath_template: path relative to cache_root/<dataset>/fold_N/
#   use {fold} as placeholder for the fold index.
CACHE_MAP = {
    "hull_cell_type": (
        RESULTS_DIR / "trimodal_cache" / "hippie-fullcond",
        "transductive_predictions.csv",
    ),
    "lisberger_labeled_cell_type": (
        RESULTS_DIR / "trimodal_cache" / "hippie-fullcond",
        "transductive_predictions.csv",
    ),
    "hausser_cell_type": (
        RESULTS_DIR / "celltype_cache" / "hippie",
        "predictions/transductive_predictions.csv",
    ),
    "cellexplorer_cell_type": (
        RESULTS_DIR / "celltype_cache" / "hippie",
        "predictions/transductive_predictions.csv",
    ),
    "dandi_000041_cell_type": (
        RESULTS_DIR / "celltype_cache" / "hippie",
        "predictions/transductive_predictions.csv",
    ),
    "a1data_remove_undef": (
        RESULTS_DIR / "celltype_cache" / "hippie-waveisi",
        "predictions/transductive_predictions.csv",
    ),
    "juxtacellular_mouse_s1_area": (
        RESULTS_DIR / "celltype_cache" / "hippie-waveisi",
        "predictions/transductive_predictions.csv",
    ),
    "dandi_000473_cell_type": (
        RESULTS_DIR / "celltype_cache" / "hippie",
        "isiacg/predictions/transductive_predictions.csv",
    ),
    "dandi_000955_cell_type": (
        RESULTS_DIR / "celltype_cache" / "hippie",
        "isiacg/predictions/transductive_predictions.csv",
    ),
}

# Display order and names for Table 1
DISPLAY_ORDER = [
    "hull_cell_type",
    "lisberger_labeled_cell_type",
    "dandi_000041_cell_type",
    "dandi_000473_cell_type",
    "dandi_000955_cell_type",
    "hausser_cell_type",
    "juxtacellular_mouse_s1_area",
    "a1data_remove_undef",
    "cellexplorer_cell_type",
]

DISPLAY_NAMES = {
    "hull_cell_type":                "Hull",
    "lisberger_labeled_cell_type":  "Lisberger",
    "dandi_000041_cell_type":        "Watson",
    "dandi_000473_cell_type":        "Calvigioni",
    "dandi_000955_cell_type":        "Ramachandran",
    "hausser_cell_type":             "Hausser",
    "juxtacellular_mouse_s1_area":   "Juxtacellular S1",
    "a1data_remove_undef":           "A1 auditory",
    "cellexplorer_cell_type":        "CellExplorer",
}

# These 4 datasets get the within-dataset polarity breakdown (Table 2).
WITHIN_DATASETS = [
    "hull_cell_type",
    "lisberger_labeled_cell_type",
    "hausser_cell_type",
    "cellexplorer_cell_type",
]

N_FOLDS = 5
MIN_CELLS_PER_CLASS = 10

# ---------------------------------------------------------------------------
# Helpers: label preprocessing (mirrors train_multimodal_transductive.py)
# ---------------------------------------------------------------------------
_BYTESTR_RE = re.compile(r"^b'(.*)'$")

def _normalize_label(lbl) -> str:
    s = str(lbl).strip()
    m = _BYTESTR_RE.match(s)
    if m:
        s = m.group(1)
    return s.strip()

_SYNONYM_MAP = {
    "PVALB": "PV", "parvalbumin": "PV",
    "SST": "SOM", "somatostatin": "SOM", "SOMnan": "SOM",
}

def _canonicalize_label(s: str) -> str:
    return _SYNONYM_MAP.get(s, s)

_UNLABELED_SUBSTRINGS = ("unlabeled", "undefined", "nan", "none", "n/a", "")

def _is_unlabeled(s: str) -> bool:
    return s.lower() in _UNLABELED_SUBSTRINGS or s == ""

def preprocess_labels(raw_labels) -> tuple[np.ndarray, np.ndarray]:
    """Return (filtered_indices, string_labels_after_filtering)."""
    normalized = np.array([_canonicalize_label(_normalize_label(l)) for l in raw_labels])
    keep1 = np.array([i for i, l in enumerate(normalized) if not _is_unlabeled(l)])
    normalized = normalized[keep1]

    # rare-class filter
    unique, counts = np.unique(normalized, return_counts=True)
    bad_classes = set(unique[counts < MIN_CELLS_PER_CLASS])
    keep2_mask = np.array([l not in bad_classes for l in normalized])
    keep2 = np.where(keep2_mask)[0]
    final_labels = normalized[keep2]
    # map keep2 back to keep1 indices in the original array
    original_indices = keep1[keep2]
    return original_indices, final_labels


# ---------------------------------------------------------------------------
# Load waveforms and labels for a dataset
# ---------------------------------------------------------------------------
def load_dataset(ds: str):
    wf_path  = DATASETS_DIR / ds / "waveforms.csv"
    lbl_path = DATASETS_DIR / ds / "labels.csv"
    wf  = pd.read_csv(wf_path, header=0).values.astype(float)
    lbl = pd.read_csv(lbl_path)["label"].values
    return wf, lbl


# ---------------------------------------------------------------------------
# Polarity classification
# ---------------------------------------------------------------------------
def classify_polarity(waveform: np.ndarray) -> str:
    """Classify a single waveform as negative, positive, or biphasic.

    ratio = max_positive_peak / abs(max_negative_trough)
    ratio > 2  -> positive
    ratio < 0.5 -> negative
    else        -> biphasic
    """
    peak   = float(waveform.max())
    trough = float(abs(waveform.min()))
    if trough == 0:
        return "positive" if peak > 0 else "biphasic"
    ratio = peak / trough
    if ratio > 2.0:
        return "positive"
    if ratio < 0.5:
        return "negative"
    return "biphasic"


# ---------------------------------------------------------------------------
# Load per-fold predictions from the correct cache
# ---------------------------------------------------------------------------
def load_fold_preds(ds: str, fold: int) -> pd.DataFrame | None:
    cache_root, pred_sub = CACHE_MAP[ds]
    p = cache_root / ds / f"fold_{fold}" / pred_sub
    if not p.exists():
        print(f"  [MISSING] {p}")
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def run():
    table1_rows = []  # between-dataset
    table2_rows = []  # within-dataset

    for ds in DISPLAY_ORDER:
        print(f"\n=== {DISPLAY_NAMES[ds]} ({ds}) ===")
        wf, raw_labels = load_dataset(ds)

        orig_idx, str_labels = preprocess_labels(raw_labels)
        wf_filt = wf[orig_idx]
        n_sup = len(str_labels)
        print(f"  After filtering: {n_sup} units")

        # Polarity classification
        polarities = np.array([classify_polarity(wf_filt[i]) for i in range(n_sup)])
        n_neg = (polarities == "negative").sum()
        n_bip = (polarities == "biphasic").sum()
        n_pos = (polarities == "positive").sum()
        pct_neg = 100.0 * n_neg / n_sup
        pct_bip = 100.0 * n_bip / n_sup
        pct_pos = 100.0 * n_pos / n_sup
        print(f"  Polarity: neg={n_neg}({pct_neg:.1f}%), bip={n_bip}({pct_bip:.1f}%), pos={n_pos}({pct_pos:.1f}%)")

        # Encode labels for stratified split
        le = LabelEncoder()
        enc_labels = le.fit_transform(str_labels)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_splits = list(skf.split(np.zeros(n_sup), enc_labels))

        # Per-fold balanced accuracy
        fold_bas = []
        # For within-dataset: accumulate per-polarity
        all_true_str = []
        all_pred_str = []
        all_pol      = []

        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            preds_df = load_fold_preds(ds, fold_idx)
            if preds_df is None:
                continue
            if len(preds_df) != len(test_idx):
                print(f"  WARNING fold {fold_idx}: pred rows {len(preds_df)} != test size {len(test_idx)}")

            # Overall fold BA
            ba = balanced_accuracy_score(preds_df["true"], preds_df["pred"])
            fold_bas.append(ba)
            print(f"  fold {fold_idx}: BA={ba:.4f}")

            if ds in WITHIN_DATASETS:
                # Store predictions with their polarity label
                test_polarities = polarities[test_idx]
                # Align prediction rows to test_idx order
                for j, (orig_j, pol) in enumerate(zip(test_idx, test_polarities)):
                    if j < len(preds_df):
                        all_true_str.append(preds_df["true"].iloc[j])
                        all_pred_str.append(preds_df["pred"].iloc[j])
                        all_pol.append(pol)

        if not fold_bas:
            print(f"  No predictions found for {ds}!")
            continue

        ba_mean = np.mean(fold_bas)
        ba_std  = np.std(fold_bas)
        print(f"  Overall: {ba_mean:.3f} ± {ba_std:.3f}")

        table1_rows.append({
            "dataset": DISPLAY_NAMES[ds],
            "N": n_sup,
            "pct_neg": pct_neg,
            "pct_bip": pct_bip,
            "pct_pos": pct_pos,
            "ba_mean": ba_mean,
            "ba_std":  ba_std,
        })

        # Within-dataset polarity breakdown
        if ds in WITHIN_DATASETS and all_true_str:
            all_true_arr = np.array(all_true_str)
            all_pred_arr = np.array(all_pred_str)
            all_pol_arr  = np.array(all_pol)

            n_test_total = len(all_true_arr)
            overall_ba = balanced_accuracy_score(all_true_arr, all_pred_arr)
            print(f"  Pooled BA (all test): {overall_ba:.3f}")
            table2_rows.append({
                "dataset": DISPLAY_NAMES[ds],
                "polarity": "Overall",
                "N_test": n_test_total,
                "ba": overall_ba,
            })

            # Per-polarity group — sorted by descending BA
            pol_results = []
            for pol in ["negative", "biphasic", "positive"]:
                mask = all_pol_arr == pol
                if mask.sum() == 0:
                    continue
                n_pol = int(mask.sum())
                t, p_ = all_true_arr[mask], all_pred_arr[mask]
                classes = np.unique(t)
                if len(classes) < 2:
                    print(f"    {pol}: only 1 class in test, skipping BA")
                    pol_results.append((pol, n_pol, float("nan")))
                    continue
                ba_pol = balanced_accuracy_score(t, p_)
                print(f"    {pol}: N={n_pol}, BA={ba_pol:.3f}")
                pol_results.append((pol, n_pol, ba_pol))

            pol_results.sort(key=lambda x: -x[2] if not np.isnan(x[2]) else -1)
            for pol, n_pol, ba_pol in pol_results:
                table2_rows.append({
                    "dataset": DISPLAY_NAMES[ds],
                    "polarity": pol,
                    "N_test": n_pol,
                    "ba": ba_pol,
                })

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------
    t1 = pd.DataFrame(table1_rows)
    t2 = pd.DataFrame(table2_rows)
    t1.to_csv(OUT_DIR / "table1_between_dataset.csv", index=False)
    t2.to_csv(OUT_DIR / "table2_within_dataset.csv", index=False)
    print(f"\n[ok] Saved CSVs to {OUT_DIR}")

    # -------------------------------------------------------------------------
    # Pearson correlations (Table 1 footer)
    # -------------------------------------------------------------------------
    print("\n--- Pearson r vs Bal. Acc ---")
    for col, label in [("pct_neg", "frac_neg"), ("pct_bip", "frac_bip"), ("pct_pos", "frac_pos")]:
        r, pval = pearsonr(t1[col], t1["ba_mean"])
        print(f"  {label}: r={r:+.2f}  p={pval:.3f}")

    # -------------------------------------------------------------------------
    # Generate LaTeX
    # -------------------------------------------------------------------------
    _write_latex(t1, t2, OUT_DIR / "polarity_tables.tex")
    print(f"[ok] LaTeX written to {OUT_DIR / 'polarity_tables.tex'}")


def _write_latex(t1: pd.DataFrame, t2: pd.DataFrame, out_path: Path):
    lines = []

    # -- Table 1 --
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Waveform polarity composition and HIPPIE classification accuracy "
        r"across nine cell-type datasets. "
        r"Bal.\ Acc is the mean balanced accuracy (KNN, transductive) $\pm$ standard "
        r"deviation across 5 folds, using the same prediction cache as the corresponding "
        r"main figure: Hull and Lisberger use the trimodal (WF+ISI+ACG) Figure~4 cache "
        r"(full conditioning); A1 and Juxtacellular S1 use the bimodal (WF+ISI) "
        r"Figure~3 cache; all other datasets use the trimodal Figure~3 cache.}"
    )
    lines.append(r"\label{tab:polarity_between}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Dataset} & \textbf{N} & \textbf{\% negative} & "
        r"\textbf{\% biphasic} & \textbf{\% positive} & \textbf{Bal. Acc} & \textbf{SD} \\"
    )
    lines.append(r"\hline")

    for _, row in t1.iterrows():
        name  = row["dataset"]
        N     = int(row["N"])
        neg   = row["pct_neg"]
        bip   = row["pct_bip"]
        pos   = row["pct_pos"]
        ba    = row["ba_mean"]
        sd    = row["ba_std"]
        lines.append(
            f"{name:<20} & {N:>4}  & {neg:>4.1f} & {bip:>5.1f} & {pos:>5.1f} "
            f"& {ba:.3f} & {sd:.3f} \\\\"
        )

    lines.append(r"\hline")
    # Pearson correlations
    pct_cols = ["pct_neg", "pct_bip", "pct_pos"]
    labels   = ["+", "-", "-"]
    rs = []
    for col in pct_cols:
        r, _ = pearsonr(t1[col], t1["ba_mean"])
        rs.append(r)
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{Pearson $r$ vs.\ Bal.\ Acc}} "
        rf"& ${rs[0]:+.2f}$ & ${rs[1]:+.2f}$ & ${rs[2]:+.2f}$ & & \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # -- Table 2 --
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Within-dataset balanced accuracy stratified by waveform polarity "
        r"group (HIPPIE transductive KNN, 5-fold CV, pooled across folds). "
        r"Overall is the aggregate balanced accuracy across all test neurons; "
        r"polarity-stratified rows show accuracy restricted to test neurons in each "
        r"polarity class. Hull and Lisberger use the trimodal (WF+ISI+ACG) Figure~4 "
        r"prediction cache; Hausser and CellExplorer use the Figure~3 cache. "
        r"Gaps $\leq 0.05$ indicate no meaningful polarity effect; larger gaps are "
        r"inconsistent in direction across datasets.}"
    )
    lines.append(r"\label{tab:polarity_within}")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Dataset} & \textbf{Polarity} & \textbf{N (test)} & \textbf{Bal. Acc} \\"
    )
    lines.append(r"\hline")

    cur_ds = None
    for _, row in t2.iterrows():
        ds_name = row["dataset"]
        pol     = row["polarity"]
        n_test  = int(row["N_test"])
        ba      = row["ba"]
        prefix  = ds_name if ds_name != cur_ds else ""
        cur_ds  = ds_name
        ba_str  = f"{ba:.3f}" if not np.isnan(ba) else "n.a."
        lines.append(f"{prefix:<12} & {pol:<10} & {n_test:>4} & {ba_str} \\\\")
        if pol == "Overall" and ds_name != cur_ds:
            pass
        # hline after last polarity group for each dataset
    # close final hline
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
