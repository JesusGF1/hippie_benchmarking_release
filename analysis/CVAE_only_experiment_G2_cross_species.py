#!/usr/bin/env python3
"""G2-cross-species — macaque → mouse counterfactual cell-type profiles.

For each overlapping cell type between a source (macaque, Lisberger src=3) and
a reference (mouse, Hausser src=1) dataset:

  1. Encode all macaque cells of that type using their true source conditioning
     (source_id = src_source_id).
  2. Compute the per-type mean latent z_c.
  3. Decode z_c under the macaque conditioning → macaque archetype.
  4. Decode z_c under the mouse conditioning   → predicted-mouse archetype.
  5. Compute empirical mean ± std from real mouse cells of the same type.
  6. Compute empirical mean ± std from real macaque cells of the same type.

Output: NPZ with per-(cell_type, modality) arrays:
    {ct}_{mod}_macaque_mean      real macaque mean  (n_feats,)
    {ct}_{mod}_macaque_std       real macaque std
    {ct}_{mod}_predicted_mouse   HIPPIE decoded under mouse conditioning
    {ct}_{mod}_mouse_mean        real mouse mean
    {ct}_{mod}_mouse_std         real mouse std
    cell_types                   object array of cell type strings
    modalities                   object array of modality strings
    n_macaque_{ct}               scalar: # macaque cells of this type
    n_mouse_{ct}                 scalar: # mouse cells of this type
    src_source_id                scalar
    ref_source_id                scalar

Usage:
    python analysis/CVAE_only_experiment_G2_cross_species.py \\
        --registry /tmp/cvae_registry.json \\
        --src-data-dir /data/datasets/lisberger_labeled_cell_type \\
        --ref-data-dir /data/datasets/hausser_cell_type \\
        --src-source-id 3 \\
        --ref-source-id 1 \\
        --out /data/results/cvae_only/G2_cross_species.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from CVAE_only_experiment_utils import (
    build_loader,
    decode_batch,
    encode_batch,
    iter_registry,
    load_dataset_csv,
    load_registry,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_all_ct(lm, data: dict, ct_idx: np.ndarray, source_id: int,
                   batch_size: int) -> np.ndarray:
    """Encode all cells at ct_idx under source_id; return mu array (N, z_dim)."""
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    loader = build_loader(data, mods, batch_size=batch_size, indices=ct_idx)
    all_mus = []
    for batch in loader:
        if batch is None:
            continue
        data_dict, _ = batch
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        bs = data_dict[mods[0]].shape[0]
        src_t = torch.full((bs,), source_id, dtype=torch.long, device=device)
        mu, _ = encode_batch(lm, data_dict, source_labels=src_t)
        all_mus.append(mu.cpu().numpy())
    return np.concatenate(all_mus, axis=0)


@torch.no_grad()
def _decode_z(lm, mean_z: np.ndarray, source_id: int) -> dict[str, np.ndarray]:
    """Decode a single z vector under source_id; return {mod: 1D array}."""
    device = next(lm.model.parameters()).device
    z_t = torch.as_tensor(mean_z, dtype=torch.float32, device=device).unsqueeze(0)
    src_t = torch.full((1,), source_id, dtype=torch.long, device=device)
    decoded = decode_batch(lm, z_t, source_labels=src_t)
    return {m: decoded[m].reshape(-1).cpu().numpy() for m in decoded}


def _real_stats(data: dict, mods: list[str],
                idx: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {mod: (mean, std)} for cells at idx."""
    stats = {}
    for m in mods:
        arr = data[m][idx].astype(np.float32)
        stats[m] = (arr.mean(axis=0), arr.std(axis=0))
    return stats


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    lm,
    src_data: dict,
    ref_data: dict,
    src_source_id: int,
    ref_source_id: int,
    cell_types: list[str] | None,
    batch_size: int,
) -> dict:
    """Return dict of arrays suitable for np.savez."""
    mods = list(lm.modalities.keys())

    src_labels = src_data.get("labels", np.array([]))
    ref_labels = ref_data.get("labels", np.array([]))

    src_types = set(src_labels[src_labels != "unlabeled"])
    ref_types = set(ref_labels[ref_labels != "unlabeled"])
    overlap = sorted(src_types & ref_types)

    if cell_types is not None:
        requested = set(cell_types)
        missing = requested - set(overlap)
        if missing:
            print(f"[G2-xsp] warning: requested cell types not in overlap: {missing}")
        overlap = [ct for ct in cell_types if ct in set(overlap)]

    if not overlap:
        print(f"[G2-xsp] no overlapping cell types found; src={sorted(src_types)}, "
              f"ref={sorted(ref_types)}")
        return {}

    print(f"[G2-xsp] processing {len(overlap)} cell types: {overlap}")
    print(f"[G2-xsp] source {src_source_id} → target {ref_source_id}")

    arrays: dict[str, np.ndarray] = {
        "cell_types": np.array(overlap, dtype=object),
        "modalities": np.array(mods, dtype=object),
        "src_source_id": np.array(src_source_id),
        "ref_source_id": np.array(ref_source_id),
    }

    for ct in overlap:
        src_idx = np.where(src_labels == ct)[0]
        ref_idx = np.where(ref_labels == ct)[0]

        print(f"  {ct}: {len(src_idx)} macaque cells, {len(ref_idx)} mouse cells")

        arrays[f"n_macaque_{ct}"] = np.array(len(src_idx))
        arrays[f"n_mouse_{ct}"]   = np.array(len(ref_idx))

        if len(src_idx) == 0 or len(ref_idx) == 0:
            print(f"  [skip] insufficient cells for {ct}")
            continue

        # Encode macaque cells → mean z
        mus = _encode_all_ct(lm, src_data, src_idx, src_source_id, batch_size)
        mean_z = mus.mean(axis=0)

        # Decode under macaque and mouse conditioning
        decoded_macaque = _decode_z(lm, mean_z, src_source_id)
        decoded_mouse   = _decode_z(lm, mean_z, ref_source_id)

        # Empirical stats for real cells
        src_stats = _real_stats(src_data, mods, src_idx)
        ref_stats = _real_stats(ref_data, mods, ref_idx)

        for m in mods:
            key = f"{ct}_{m}"
            arrays[f"{key}_macaque_archetype"]  = decoded_macaque[m]
            arrays[f"{key}_predicted_mouse"]    = decoded_mouse[m]
            arrays[f"{key}_macaque_mean"]       = src_stats[m][0]
            arrays[f"{key}_macaque_std"]        = src_stats[m][1]
            arrays[f"{key}_mouse_mean"]         = ref_stats[m][0]
            arrays[f"{key}_mouse_std"]          = ref_stats[m][1]

    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry",      required=True)
    ap.add_argument("--src-data-dir",  required=True, type=Path,
                    help="Macaque dataset dir (Lisberger)")
    ap.add_argument("--ref-data-dir",  required=True, type=Path,
                    help="Mouse dataset dir (Hausser)")
    ap.add_argument("--src-source-id", type=int, default=3,
                    help="Source ID used during training for the macaque dataset")
    ap.add_argument("--ref-source-id", type=int, default=1,
                    help="Source ID used during training for the mouse dataset")
    ap.add_argument("--cell-types",    nargs="+", default=None,
                    help="Cell types to process; defaults to all overlapping types")
    ap.add_argument("--batch-size",    type=int, default=256)
    ap.add_argument("--out",           required=True, type=Path)
    args = ap.parse_args()

    registry = load_registry(args.registry)
    for lm in iter_registry(registry):
        src_data = load_dataset_csv(
            args.src_data_dir, lm.modalities.keys(), target_sizes=lm.modalities
        )
        ref_data = load_dataset_csv(
            args.ref_data_dir, lm.modalities.keys(), target_sizes=lm.modalities
        )
        arrays = evaluate(
            lm, src_data, ref_data,
            src_source_id=args.src_source_id,
            ref_source_id=args.ref_source_id,
            cell_types=args.cell_types,
            batch_size=args.batch_size,
        )
        if arrays:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(args.out), **arrays)
            print(f"[ok] wrote {args.out}")
        else:
            print("[G2-xsp] no results to save")


if __name__ == "__main__":
    main()
