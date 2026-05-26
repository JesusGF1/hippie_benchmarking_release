#!/usr/bin/env python3
"""Encode A1, S1, and CellExplorer cells into the 16-D HIPPIE latent space for
cross-technology latent-space flow visualisation.

Label normalisation maps each dataset's native labels to canonical cell-type names:
  A1 (a1data_remove_undef):
      PV  → Parvalbumin,  SOM → Somatostatin,  EXC → Excitatory
  S1 (juxtacellular_mouse_s1_area):
      FS 4.0 / FS 5.0 → Parvalbumin,  SOMnan → Somatostatin,
      E  4.0 / E  5.0 → Excitatory
  CellExplorer (cellexplorer_cell_type):
      PV  → Parvalbumin,  SST → Somatostatin,  Pyra → Excitatory

Output NPZ keys:
    cell_types                       object array of shared canonical cell types
    tech_names                       object array of technology names
    src_source_id                    scalar: source_id of A1 (interpolation source)
    ref_source_id                    scalar: source_id of CellExplorer (interpolation target)

    For each canonical_ct ∈ cell_types and tech ∈ tech_names:
        {canonical_ct}_z_{tech}         (N, z_dim) float32 — latent μ vectors
        {canonical_ct}_wave_{tech}      (N, L_wave) float32
        {canonical_ct}_isi_{tech}       (N, L_isi)  float32
        {canonical_ct}_acg_{tech}       (N, L_acg)  float32

Usage:
    python analysis/CVAE_only_experiment_latent_flow_crosstech.py \\
        --ckpt /data/cvae_crosstech.ckpt \\
        --a1-dir /data/datasets/a1data_remove_undef \\
        --s1-dir /data/datasets/juxtacellular_mouse_s1_area \\
        --ce-dir /data/datasets/cellexplorer_cell_type \\
        --a1-source-id 0 \\
        --s1-source-id 1 \\
        --ce-source-id 2 \\
        --out /data/results/crosstech/latent_flow_crosstech.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hippie"))
sys.path.insert(0, str(REPO / "analysis"))

from CVAE_only_experiment_utils import load_dataset_csv, load_model
from CVAE_only_experiment_G2_cross_species import _encode_all_ct

# ── label normalisation ───────────────────────────────────────────────────────
# Maps raw label → canonical name; labels not listed are dropped (→ "unlabeled").
LABEL_MAPS: dict[str, dict[str, str]] = {
    "A1": {
        "PV":  "Parvalbumin",
        "SOM": "Somatostatin",
        "EXC": "Excitatory",
    },
    "S1": {
        "FS_L4": "Parvalbumin",
        "FS_L5": "Parvalbumin",
        "SOM":   "Somatostatin",
        "E_L4":  "Excitatory",
        "E_L5":  "Excitatory",
    },
    "CellExplorer": {
        "PV":   "Parvalbumin",
        "SST":  "Somatostatin",
        "Pyra": "Excitatory",
    },
}


def _normalise(raw_labels: np.ndarray, label_map: dict[str, str]) -> np.ndarray:
    return np.array([label_map.get(str(lbl).strip(), "unlabeled") for lbl in raw_labels])


def _encode_canonical(lm, data: dict, label_map: dict[str, str],
                      source_id: int, batch_size: int) -> dict[str, np.ndarray]:
    """Return {canonical_ct: (N, z_dim)} for all labelled cells in data."""
    raw = np.array(data.get("labels", []))
    norm = _normalise(raw, label_map)
    out: dict[str, np.ndarray] = {}
    for ct in sorted(set(norm) - {"unlabeled"}):
        idx = np.where(norm == ct)[0]
        if len(idx) == 0:
            continue
        zs = _encode_all_ct(lm, data, idx, source_id=source_id, batch_size=batch_size)
        out[ct] = zs.astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--a1-dir",       required=True, type=Path,
                    help="A1 dataset dir (a1data_remove_undef)")
    ap.add_argument("--s1-dir",       required=True, type=Path,
                    help="S1 dataset dir (juxtacellular_mouse_s1_area)")
    ap.add_argument("--ce-dir",       required=True, type=Path,
                    help="CellExplorer dataset dir (cellexplorer_cell_type)")
    ap.add_argument("--a1-source-id", type=int, default=0)
    ap.add_argument("--s1-source-id", type=int, default=1)
    ap.add_argument("--ce-source-id", type=int, default=2)
    ap.add_argument("--batch-size",   type=int, default=64)
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--out",          required=True, type=Path)
    args = ap.parse_args()

    print("[crosstech] loading model …")
    lm = load_model("hippie", str(args.ckpt), device=args.device)

    tech_cfg = [
        ("A1",           args.a1_dir, LABEL_MAPS["A1"],           args.a1_source_id),
        ("S1",           args.s1_dir, LABEL_MAPS["S1"],           args.s1_source_id),
        ("CellExplorer", args.ce_dir, LABEL_MAPS["CellExplorer"], args.ce_source_id),
    ]

    z_by_tech:     dict[str, dict[str, np.ndarray]] = {}
    data_by_tech:  dict[str, dict]                  = {}
    norm_by_tech:  dict[str, np.ndarray]            = {}

    for tech, data_dir, lmap, src_id in tech_cfg:
        print(f"[crosstech] loading {tech} from {data_dir} …")
        data = load_dataset_csv(data_dir, lm.modalities.keys(),
                                target_sizes=lm.modalities)
        data_by_tech[tech] = data
        raw = np.array(data.get("labels", []))
        norm_by_tech[tech] = _normalise(raw, lmap)

        print(f"[crosstech] encoding {tech} (src={src_id}) …")
        z_by_tech[tech] = _encode_canonical(lm, data, lmap, src_id, args.batch_size)

        # debug: show raw label samples and what they normalised to
        raw_sample = np.array(data.get("labels", []))
        norm_sample = _normalise(raw_sample, lmap)
        print(f"[crosstech] {tech} raw labels (first 5): {list(raw_sample[:5])}")
        print(f"[crosstech] {tech} norm labels (first 5): {list(norm_sample[:5])}")
        print(f"[crosstech] {tech} canonical types found: {sorted(z_by_tech[tech].keys())}")

    tech_names = [t for t, *_ in tech_cfg]

    shared = sorted(
        set(z_by_tech["A1"])
        & set(z_by_tech["S1"])
        & set(z_by_tech["CellExplorer"])
    )
    if not shared:
        print("[crosstech] per-tech canonical types:")
        for tech in tech_names:
            print(f"  {tech}: {sorted(z_by_tech[tech].keys())}")
        raise SystemExit("[crosstech] no canonical cell types shared across all 3 datasets")
    print(f"[crosstech] shared canonical cell types: {shared}")

    arrays: dict[str, np.ndarray] = {
        "cell_types":    np.array(shared, dtype=object),
        "tech_names":    np.array(tech_names, dtype=object),
        "src_source_id": np.array(args.a1_source_id),   # A1 is the interpolation source
        "ref_source_id": np.array(args.ce_source_id),   # CellExplorer is the target
    }

    for ct in shared:
        for tech in tech_names:
            arrays[f"{ct}_z_{tech}"] = z_by_tech[tech][ct]
            nl = norm_by_tech[tech]
            idx = np.where(nl == ct)[0]
            d = data_by_tech[tech]
            for mod in ("wave", "isi", "acg"):
                if mod not in d:
                    continue
                arr = d[mod][idx].astype(np.float32)
                # Skip modalities that are all zeros (synthetic fill-ins for missing data).
                if arr.size > 0 and not np.any(arr):
                    continue
                arrays[f"{ct}_{mod}_{tech}"] = arr
        counts = {tech: len(z_by_tech[tech][ct]) for tech in tech_names}
        print(f"  {ct}: " + ", ".join(f"{t} n={n}" for t, n in counts.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), **arrays)
    print(f"[crosstech] saved → {args.out}")


if __name__ == "__main__":
    main()
