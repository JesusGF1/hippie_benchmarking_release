#!/usr/bin/env python3
"""
Reproduce the locked hyperparameter configs from the bundled sweep CSV.

This is the CSV-only port of `pick_locked_configs.py` from the production
benchmarking repo (which downloads predictions from S3). The selection
algorithm is identical; only the I/O layer differs. See the audit-trail in
`docs/REPRODUCING.md` for context.

Selection rule (paper Methods § hyperparameter parity):
  Within configs sharing the maximum fold count, apply a 3-level sort to
  the per-config (mean, std, fold-count) statistics:
    1. Define a "tie bucket" of configs whose mean balanced accuracy is
       within 1.96 × SEM of the global top mean. SEM uses the top config's
       std and n.
    2. Within the tie bucket, sort ascending by std (lowest std wins).
    3. Deterministic third-level tiebreak: lowest numeric config_id
       (config_7 beats config_21). Keeps byte-identical ties resolving the
       same way every run.

Per-method selection classifier (matches paper):
  - HIPPIE  : KNN (harmonized sklearn KNN on L2-normalized embeddings)
  - NEMO    : MLP (NEMO's native probe; KNN reported as secondary)
  - PhysMAP : KNN (only classifier; caret internal 5-fold CV)

Per-method selection dataset (matches paper):
  - All three methods : Hausser only.

USAGE
-----
  # Print the selection trace and exit
  python scripts/pick_locked_configs.py

  # Also verify against the hardcoded LOCKED_CONFIG_ID constants in the
  # analysis figure-2 supplement scripts. Non-zero exit code on mismatch.
  python scripts/pick_locked_configs.py --verify

  # Write the result to JSON for downstream tooling
  python scripts/pick_locked_configs.py --output results/locked_configs.json

NOTE on PhysMAP
---------------
PhysMAP's caret-driven sweep exports one aggregated balanced accuracy per
(config, dataset) rather than per-fold scores. Combined with the Hausser-only
selection rule, PhysMAP therefore has n=1 per config and std is undefined.
The 3-level sort still resolves cleanly because there is only one row per
config in the tie bucket; the highest-mean config (config_13, BA=0.780) wins.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = REPO_ROOT / "results" / "benchmark" / "tuning_preview" / "sweep_results_long.csv"

# Per-method selection classifier
METHOD_SELECTION_CLASSIFIER = {
    "physmap": "knn",
    "hippie":  "knn",
    "nemo":    "mlp",
}

# Per-method selection dataset (paper claim: hausser only for all three)
METHOD_SELECTION_DATASETS = {
    "physmap": ["hausser_cell_type"],
    "hippie":  ["hausser_cell_type"],
    "nemo":    ["hausser_cell_type"],
}

# Hardcoded constants from analysis/ — used by --verify
EXPECTED_LOCKED_CONFIGS = {
    "hippie":  "config_25",
    "nemo":    "config_7",
    "physmap": "config_13",   # see KNOWN LIMITATION above
}


def pick_with_stability_tiebreaker(stats: pd.DataFrame) -> tuple[str, float, float, int, list[str]]:
    """Apply the stability-based tiebreaker to a per-config statistics table.

    See module docstring for the full algorithm.

    Returns:
        (best_config_id, best_mean, best_std_or_nan, max_n_folds, tie_bucket_ids)
    """
    if stats.empty:
        raise ValueError("stats frame is empty")

    max_count = int(stats["count"].max())
    eligible = stats[stats["count"] == max_count].copy()

    top_idx = eligible["mean"].idxmax()
    top_row = eligible.loc[top_idx]
    top_mean = float(top_row["mean"])
    top_std = float(top_row["std"]) if pd.notna(top_row["std"]) else 0.0
    top_sem = top_std / np.sqrt(max_count) if max_count > 0 else 0.0
    tie_threshold = top_mean - 1.96 * top_sem

    tied = eligible[eligible["mean"] >= tie_threshold].copy()
    tied["_std_rank"] = tied["std"].fillna(np.inf)
    tied["_cfg_num"] = tied["config_id"].str.extract(r"config_(\d+)").astype(int)
    tied = tied.sort_values(
        ["_std_rank", "mean", "_cfg_num"],
        ascending=[True, False, True],
    )

    winner = tied.iloc[0]
    winner_id = str(winner["config_id"])
    winner_mean = float(winner["mean"])
    winner_std = float(winner["std"]) if pd.notna(winner["std"]) else float("nan")
    tie_bucket = [str(c) for c in tied["config_id"]]
    return winner_id, winner_mean, winner_std, max_count, tie_bucket


def select_best_configs(df: pd.DataFrame) -> dict:
    """Run the tiebreaker per method and return the locked-config dict."""
    locked = {}

    for method, classifier in METHOD_SELECTION_CLASSIFIER.items():
        method_df = df[df["method"] == method]
        if method_df.empty:
            print(f"\nWARNING: no rows for method={method}; skipping.")
            continue

        primary_df = method_df[method_df["classifier"] == classifier]
        if primary_df.empty:
            print(f"\nWARNING: no {classifier} rows for {method}; skipping.")
            continue

        sel_datasets = METHOD_SELECTION_DATASETS.get(method)
        if sel_datasets:
            primary_df = primary_df[primary_df["dataset"].isin(sel_datasets)]

        stats = (
            primary_df.groupby("config_id")["balanced_accuracy"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        counts = (
            primary_df.groupby("config_id")["fold"]
            .nunique()
            .reset_index(name="count")
        )
        stats = stats.merge(counts, on="config_id")

        winner_id, winner_mean, winner_std, max_count, tie_bucket = \
            pick_with_stability_tiebreaker(stats)

        eligible = stats[stats["count"] == max_count]
        naive_top = eligible.loc[eligible["mean"].idxmax()]
        naive_id = str(naive_top["config_id"])
        naive_mean = float(naive_top["mean"])
        naive_std = float(naive_top["std"]) if pd.notna(naive_top["std"]) else float("nan")

        locked[method] = {
            "config_id": winner_id,
            "selection_classifier": classifier,
            "selection_datasets": sel_datasets,
            "selection_rule": (
                "highest mean balanced accuracy with stability tiebreaker: "
                "within 1.96 × SEM tie bucket, lowest std (lowest config_id "
                "as 3rd-level deterministic tiebreak)"
            ),
            "mean_balanced_accuracy": winner_mean,
            "std_balanced_accuracy": winner_std,
            "n_folds": max_count,
            "tie_bucket_size": len(tie_bucket),
            "tie_bucket_config_ids": tie_bucket,
            "naive_top": {
                "config_id": naive_id,
                "mean": naive_mean,
                "std": naive_std,
            },
        }

    return locked


def print_trace(locked: dict) -> None:
    print("=" * 72)
    print("LOCKED-CONFIG SELECTION (stability tiebreaker)")
    print("=" * 72)
    for method, entry in locked.items():
        clf = entry["selection_classifier"]
        print(f"\n{method.upper()}  (classifier={clf}, datasets={entry['selection_datasets']}, folds={entry['n_folds']})")
        print(f"  Naive top:    {entry['naive_top']['config_id']:10s}  "
              f"mean={entry['naive_top']['mean']:.4f}  std={entry['naive_top']['std']:.4f}")
        print(f"  Tie bucket:   {entry['tie_bucket_size']} configs within 1.96 × SEM of top mean")
        print(f"  Locked:       {entry['config_id']:10s}  "
              f"mean={entry['mean_balanced_accuracy']:.4f}  std={entry['std_balanced_accuracy']:.4f}")


def verify(locked: dict) -> int:
    """Compare against the hardcoded LOCKED_CONFIG_IDs in analysis/.

    Returns the number of mismatches.
    """
    print("\n" + "=" * 72)
    print("VERIFY  (against analysis/figure_2_*.py LOCKED_CONFIG_ID constants)")
    print("=" * 72)
    mismatches = 0
    for method, expected in EXPECTED_LOCKED_CONFIGS.items():
        actual = locked.get(method, {}).get("config_id", "<missing>")
        if actual == expected:
            print(f"  OK    {method:8s}  {actual} == {expected}")
        else:
            print(f"  FAIL  {method:8s}  {actual} != {expected}")
            mismatches += 1
    return mismatches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-csv", type=Path, default=SWEEP_CSV,
                    help=f"Sweep results CSV (default: {SWEEP_CSV.relative_to(REPO_ROOT)})")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional path to write the locked-config JSON.")
    ap.add_argument("--verify", action="store_true",
                    help="Verify the reproduced locked configs against the hardcoded "
                         "LOCKED_CONFIG_ID constants in analysis/figure_2_*.py. "
                         "Exits non-zero on hard failure (HIPPIE or NEMO mismatch).")
    args = ap.parse_args()

    if not args.sweep_csv.exists():
        print(f"ERROR: sweep CSV not found: {args.sweep_csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.sweep_csv)
    print(f"Loaded {len(df)} rows from {args.sweep_csv.relative_to(REPO_ROOT)}")
    print(f"  methods={sorted(df['method'].unique())}  "
          f"classifiers={sorted(df['classifier'].unique())}  "
          f"datasets={sorted(df['dataset'].unique())}")

    locked = select_best_configs(df)
    print_trace(locked)

    if args.output:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"selection_rule_reference": "paper Methods § hyperparameter parity",
                       "configs": locked}, f, indent=2)
        try:
            display = out_path.relative_to(REPO_ROOT)
        except ValueError:
            display = out_path
        print(f"\nWrote {display}")

    if args.verify:
        return 1 if verify(locked) > 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
