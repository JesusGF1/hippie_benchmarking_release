#!/usr/bin/env python3
"""Encode all Hausser (mouse) and Lisberger (macaque) cells into 16-D latent
space, then save per-cell z vectors for latent-space flow visualisation.

Output NPZ keys:
    {ct}_z_mouse          (N_mouse, z_dim) float32
    {ct}_z_macaque        (N_mac,   z_dim) float32
    {ct}_wave_mouse       (N_mouse, L_wave) float32
    {ct}_wave_macaque     (N_mac,   L_wave) float32
    {ct}_isi_mouse        (N_mouse, L_isi)  float32
    {ct}_isi_macaque      (N_mac,   L_isi)  float32
    {ct}_acg_mouse        (N_mouse, L_acg)  float32
    {ct}_acg_macaque      (N_mac,   L_acg)  float32
    cell_types            object array of shared cell types
Usage:
    python analysis/CVAE_only_latent_flow.py \\
        --ckpt results/18_cvae_only_experiments/hippie_cvae.ckpt \\
        --hausser-dir datasets/hausser_cell_type \\
        --lisberger-dir datasets/lisberger_labeled_cell_type \\
        --out results/18_cvae_only_experiments/latent_flow.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hippie"))
sys.path.insert(0, str(REPO / "analysis"))

from CVAE_only_experiment_utils import (
    build_loader, encode_batch, load_dataset_csv,
)
from CVAE_only_experiment_G2_cross_species import _encode_all_ct


def load_model_local(ckpt_path: str, device: str = "cpu"):
    from CVAE_only_experiment_utils import load_model
    return load_model("hippie", ckpt_path, device=device)


def encode_by_ct(lm, data: dict, label_key: str = "labels",
                 source_id: int = 1, batch_size: int = 64,
                 cell_types: list[str] | None = None) -> dict[str, np.ndarray]:
    """Return {ct: (N, z_dim)} for all cells in data."""
    labels = np.array(data.get(label_key, []))
    cts = cell_types or [c for c in sorted(set(labels)) if c != "unlabeled"]
    out = {}
    for ct in cts:
        idx = np.where(labels == ct)[0]
        if len(idx) == 0:
            continue
        zs = _encode_all_ct(lm, data, idx, source_id=source_id, batch_size=batch_size)
        out[ct] = zs.astype(np.float32)
    return out


def main(ckpt_path: str, hausser_dir: str, lisberger_dir: str,
         out_path: str, device: str = "cpu") -> None:
    print("[flow] loading model …")
    lm = load_model_local(ckpt_path, device)

    print("[flow] loading Hausser …")
    hdf = load_dataset_csv(hausser_dir, lm.modalities)
    print("[flow] loading Lisberger …")
    ldf = load_dataset_csv(lisberger_dir, lm.modalities)

    print("[flow] encoding mouse (src=1) …")
    z_mouse = encode_by_ct(lm, hdf, source_id=1)
    print("[flow] encoding macaque (src=3) …")
    z_mac   = encode_by_ct(lm, ldf, source_id=3)

    shared_cts = sorted(set(z_mouse) & set(z_mac))
    print("[flow] shared cell types:", shared_cts)

    h_labels = np.array(hdf["labels"])
    l_labels = np.array(ldf["labels"])

    arrays: dict[str, np.ndarray] = {"cell_types": np.array(shared_cts)}
    for ct in shared_cts:
        arrays[f"{ct}_z_mouse"]    = z_mouse[ct]
        arrays[f"{ct}_z_macaque"]  = z_mac[ct]
        # Save raw signals for all modalities (waveform, ISI, ACG)
        h_idx = np.where(h_labels == ct)[0]
        l_idx = np.where(l_labels == ct)[0]
        for mod in ("wave", "isi", "acg"):
            if mod in hdf:
                arrays[f"{ct}_{mod}_mouse"]   = hdf[mod][h_idx].astype(np.float32)
            if mod in ldf:
                arrays[f"{ct}_{mod}_macaque"] = ldf[mod][l_idx].astype(np.float32)
        print(f"  {ct}: mouse n={len(z_mouse[ct])}, macaque n={len(z_mac[ct])}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"[flow] saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--hausser-dir",    required=True)
    ap.add_argument("--lisberger-dir", required=True)
    ap.add_argument("--out",            required=True)
    ap.add_argument("--device",         default="cpu")
    args = ap.parse_args()
    main(args.ckpt, args.hausser_dir, args.lisberger_dir, args.out, args.device)
