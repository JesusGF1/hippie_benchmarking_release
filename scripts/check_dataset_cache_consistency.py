"""Check that the shipped datasets match the cached benchmark predictions.

Why this exists
---------------
The training scripts locate their inputs with ``find_dataset_file``, which
searches a fallback chain::

    $HIPPIE_DATA_ROOT/<dataset>/<file>
    ./datasets/<dataset>/<file>
    ./datasets_hippie/<dataset>/<file>
    /data/datasets/<dataset>/<file>
    /datasets_hippie/<dataset>/<file>
    /src/datasets_hippie/<dataset>/<file>
    /src/datasets/<dataset>/<file>

Runs performed inside the training container resolved to the mounted copies
under ``/data`` or ``/src``; a checkout of this repository resolves to
``./datasets``. Nothing recorded which copy a given cache came from, so the two
could drift without any error being raised, and a cached result could describe
units that are not in the shipped data.

This script makes that drift visible. For every dataset it compares the number
of labelled units in ``datasets/<name>/labels.csv`` against the number of
pooled predictions in each method's cache, and reports any cache that cannot be
explained by the exclusions documented in the Methods.

Exit status is 1 if an unexplained mismatch is found, so it can be used as a
pre-submission gate.

Usage:
    python scripts/check_dataset_cache_consistency.py [--verbose]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results" / "benchmark"

# Counts that are known and accepted, so the check stays a regression detector
# rather than a permanent failure. Each entry maps a dataset to the set of pooled
# counts its caches may legitimately hold, plus the reason.
DOCUMENTED = {
    "hausser_cell_type": (
        {104}, "labels.csv holds all 1,998 units; 104 are labelled and survive the "
               "n < 10 rare-class rule (granule cells, n = 9, dropped)"),
    "lisberger_labeled_cell_type": (
        {668}, "labels.csv holds 1,152 units; 484 are unlabelled"),
    "hull_cell_type": (
        {101, 103}, "103 labelled units; the n < 5 rule drops Golgi cells (n = 2) "
                    "to give the 101 used for the benchmark"),
    "cellexplorer_cell_type": (
        {359}, "430 units across 7 classes; axo-axonic (35), juxtacellular (23) "
               "and VGAT (13) not carried into the cell-type analyses"),
    "dandi_000041_cell_type": (
        {221, 236}, "the cache was computed against a larger copy of this dataset "
                    "than the one shipped; the manuscript reports the shipped "
                    "count, 221"),
}

CACHE_GLOBS = [
    "celltype_cache/*/{ds}/fold_*/predictions/transductive_predictions.csv",
    "celltype_cache/*/{ds}/fold_*/isiacg/predictions/transductive_predictions.csv",
    "trimodal_cache/*/{ds}/fold_*/transductive_predictions.csv",
]


def _pooled_counts(dataset: str) -> dict[str, int]:
    """Pooled prediction count per method cache, keyed 'cache:method'."""
    per: dict[str, int] = {}
    for pattern in CACHE_GLOBS:
        for f in glob.glob(str(RESULTS_DIR / pattern.format(ds=dataset))):
            m = re.search(r"(celltype_cache|trimodal_cache)/([^/]+)/", f)
            if not m:
                continue
            key = f"{m.group(1).split('_')[0]}:{m.group(2)}"
            per[key] = per.get(key, 0) + len(pd.read_csv(f))
    return per


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="list every dataset, not only the problems")
    args = parser.parse_args()

    problems: list[str] = []

    for dataset in sorted(os.listdir(DATASETS_DIR)):
        labels = DATASETS_DIR / dataset / "labels.csv"
        if not labels.exists():
            continue
        n_labels = len(pd.read_csv(labels))
        caches = _pooled_counts(dataset)
        if not caches:
            if args.verbose:
                print(f"  {dataset:34s} {n_labels:>7d}  (no cache)")
            continue

        accepted, reason = DOCUMENTED.get(dataset, ({n_labels}, None))
        bad = {k: v for k, v in caches.items() if v not in accepted}

        if bad:
            expected = "/".join(str(v) for v in sorted(accepted))
            for method, n in sorted(bad.items()):
                problems.append(
                    f"{dataset}: {method} holds {n} pooled predictions; expected "
                    f"{expected}" + (f" ({reason})" if reason else "")
                )
        if args.verbose or bad:
            status = "MISMATCH" if bad else "ok"
            detail = " ".join(f"{k}={v}" for k, v in sorted(caches.items()))
            print(f"  {dataset:34s} {n_labels:>7d}  {detail}   {status}")

    print()
    if problems:
        print(f"{len(problems)} unexplained mismatch(es):")
        for p in problems:
            print(f"  - {p}")
        print("\nA cache holding more units than the shipped dataset was computed "
              "against a different copy of that dataset — most likely a mount "
              "under /data or /src rather than ./datasets. Re-run that benchmark "
              "with HIPPIE_DATA_ROOT pointed at ./datasets, or ship the data the "
              "cache was built from.")
        return 1

    print("All caches are consistent with the shipped datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
