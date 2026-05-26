#!/usr/bin/env python3
"""Download a C4 database H5 file via the npyx helper.

The C4 database (https://www.c4-database.com) hosts the H5 files needed by
the NEMO and WF-RF baselines (and by HIPPIE_WF3DACG retraining). The download
URLs are XOR-encrypted inside npyx and require a password issued by the C4
authors — register at https://www.c4-database.com/apps/about to obtain it.

Usage:
    python scripts/download_c4_h5.py --dataset hull_labelled --out-dir ./datasets/c4_h5/

Available --dataset keys (from npyx.c4.DATASETS_URL):
    hull_labelled, hull_unlabelled, hausser, lisberger, medina

The script:
  1. Imports npyx and looks up the encrypted URL for the chosen dataset.
  2. Prompts for the C4 password and decrypts the URL (XOR).
  3. Downloads the H5 file to <out-dir>/C4_database_<dataset>.h5
  4. Verifies the file is readable as HDF5.

Requires: npyx (install via `pip install -e ".[nemo]"` or `pip install npyx`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (e.g. hull_labelled, hausser, lisberger).")
    parser.add_argument("--out-dir", default="./datasets/c4_h5/",
                        help="Output directory (default: ./datasets/c4_h5/).")
    parser.add_argument("--filename", default=None,
                        help="Override output filename. Default: C4_database_<dataset>.h5")
    args = parser.parse_args()

    try:
        import npyx  # type: ignore
    except ImportError:
        print("ERROR: npyx is not installed. Install with `pip install -e \".[nemo]\"`"
              " or `pip install npyx`.", file=sys.stderr)
        return 1

    try:
        encrypted_url = npyx.c4.DATASETS_URL[args.dataset]
    except (AttributeError, KeyError):
        keys = list(getattr(npyx.c4, "DATASETS_URL", {}).keys())
        print(f"ERROR: dataset '{args.dataset}' not in npyx.c4.DATASETS_URL.\n"
              f"Available keys: {keys}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.filename or f"C4_database_{args.dataset}.h5"
    out_path = out_dir / out_name

    if out_path.exists():
        print(f"File already exists: {out_path}. Delete it first if you want to re-download.")
        return 0

    print(f"Downloading {args.dataset} → {out_path}")
    print("This file requires the C4 password (request access at "
          "https://www.c4-database.com/apps/about).")

    # npyx.c4.download_file prompts interactively for the password.
    npyx.c4.download_file(
        encrypted_url,
        str(out_path),
        description=f"Downloading {args.dataset}",
        requires_password=True,
    )

    # Sanity check: file should be valid HDF5
    try:
        import h5py  # type: ignore
        with h5py.File(out_path, "r") as f:
            n_groups = len(list(f.keys()))
        print(f"✓ Downloaded and verified (HDF5, {n_groups} top-level groups).")
    except ImportError:
        print("Downloaded. (Install h5py to verify the file structure.)")
    except Exception as e:
        print(f"WARNING: file downloaded but could not be opened as HDF5: {e}")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
