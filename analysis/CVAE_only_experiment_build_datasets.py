#!/usr/bin/env python3
"""Build multi-dataset training spec and combined eval directory for CVAE-only experiments.

Run after downloading all datasets. Does three things:
  1. Writes datasets_train.json (training spec consumed by train_hippie.py).
  2. Writes source_id.csv and super_regions.csv into each individual dataset dir.
  3. Builds the combined-train directory by concatenating all training datasets.

Usage:
    python CVAE_only_experiment_build_datasets.py \\
        --sources '[{"data_dir":"./datasets/hull_cell_type","source_id":0,"super_region_id":0}, ...]' \\
        --datasets-json results/18_cvae_only_experiments/datasets_train.json \\
        --combined-dir results/18_cvae_only_experiments/combined_train
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd


def _resample(arr: np.ndarray, target_cols: int) -> np.ndarray:
    """Linearly interpolate each row to target_cols length."""
    if arr.shape[1] == target_cols:
        return arr
    x_old = np.linspace(0, 1, arr.shape[1])
    x_new = np.linspace(0, 1, target_cols)
    return np.stack([np.interp(x_new, x_old, row) for row in arr])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True,
                    help="JSON list of {data_dir, source_id, super_region_id} dicts")
    ap.add_argument("--datasets-json", required=True, type=pathlib.Path)
    ap.add_argument("--combined-dir", required=True, type=pathlib.Path)
    args = ap.parse_args()

    sources = json.loads(args.sources)

    # ── 1. Write training spec ────────────────────────────────────────────────
    spec = [{"data_dir": s["data_dir"],
             "source_id": s["source_id"],
             "super_region_id": s["super_region_id"]} for s in sources]
    args.datasets_json.parent.mkdir(parents=True, exist_ok=True)
    args.datasets_json.write_text(json.dumps(spec, indent=2))
    print(f"[ok] {args.datasets_json} written")

    # ── 2. Write source_id.csv + super_regions.csv into each dataset dir ──────
    sr_map = {0: "cerebellar", 1: "cortex"}
    for s in sources:
        p = pathlib.Path(s["data_dir"])
        n = len(pd.read_csv(p / "waveforms.csv"))
        pd.DataFrame({"source_id": [s["source_id"]] * n}).to_csv(
            p / "source_id.csv", index=False)
        sr_label = sr_map.get(s["super_region_id"], str(s["super_region_id"]))
        pd.DataFrame({"super_region": [sr_label] * n}).to_csv(
            p / "super_regions.csv", index=False)
        print(f"[ok] {p.name}: source_id={s['source_id']} super_region={sr_label} n={n}")

    # ── 3. Build combined eval dir ────────────────────────────────────────────
    # Resample all modalities to the first dataset's column count so vstack works.
    out = args.combined_dir
    out.mkdir(parents=True, exist_ok=True)
    waves, isis, acgs, src_ids, sr_labels, labels = [], [], [], [], [], []
    canonical: dict[str, int] = {}
    for s in sources:
        p = pathlib.Path(s["data_dir"])
        w = pd.read_csv(p / "waveforms.csv").to_numpy()
        i = pd.read_csv(p / "isi_dist.csv").to_numpy()
        a = pd.read_csv(p / "acg.csv").to_numpy()

        if not canonical:
            canonical = {"wave": w.shape[1], "isi": i.shape[1], "acg": a.shape[1]}
            print(f"[canonical] wave={canonical['wave']} isi={canonical['isi']} acg={canonical['acg']}")

        w = _resample(w, canonical["wave"])
        i = _resample(i, canonical["isi"])
        a = _resample(a, canonical["acg"])

        n = len(w)
        waves.append(w); isis.append(i); acgs.append(a)
        src_ids.extend([s["source_id"]] * n)
        sr_label = sr_map.get(s["super_region_id"], str(s["super_region_id"]))
        sr_labels.extend([sr_label] * n)
        lbl_path = p / "labels.csv"
        if lbl_path.exists():
            labels.extend(pd.read_csv(lbl_path).iloc[:, 0].tolist())
        else:
            labels.extend(["unlabeled"] * n)

    pd.DataFrame(np.vstack(waves)).to_csv(out / "waveforms.csv", index=False)
    pd.DataFrame(np.vstack(isis)).to_csv(out / "isi_dist.csv", index=False)
    pd.DataFrame(np.vstack(acgs)).to_csv(out / "acg.csv", index=False)
    pd.DataFrame({"source_id": src_ids}).to_csv(out / "source_id.csv", index=False)
    pd.DataFrame({"super_region": sr_labels}).to_csv(out / "super_regions.csv", index=False)
    pd.DataFrame({"label": labels}).to_csv(out / "labels.csv", index=False)
    print(f"[ok] {out}: {len(src_ids)} neurons, sources={sorted(set(src_ids))}")


if __name__ == "__main__":
    main()
