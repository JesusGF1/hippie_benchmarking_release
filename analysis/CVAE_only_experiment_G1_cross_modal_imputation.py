#!/usr/bin/env python3
"""G1 — cross-modal imputation.

For each checkpoint, iterate over its own modality set: mask one modality at
the encoder input (zero tensor), run forward, measure MSE + 1-D Wasserstein
against the real modality for every reconstructed output.

HIPPIE has three modalities (wave, isi, acg) → 3×3 cells.
HIPPIE-WF+3DACG has two (wave, acg_3d) → 2×2 cells. The `wave→wave` cell is
directly comparable between the two.

NEMO is intentionally absent — it has no decoder.

Writes `results/benchmark/cvae_only/G1_cross_modal_imputation.csv`.

Usage:
    python analysis/CVAE_only_experiment_G1_cross_modal_imputation.py \\
        --registry configs/cvae_registry.json \\
        --data-dir /path/to/dataset \\
        --dataset hausser_cell_type
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from CVAE_only_experiment_utils import (
    DEFAULT_RESULTS_DIR,
    build_loader,
    decode_batch,
    encode_batch,
    iter_registry,
    load_dataset_csv,
    load_registry,
    mask_modality,
    wasserstein_1d,
    write_results,
)

OUT = DEFAULT_RESULTS_DIR / "G1_cross_modal_imputation.csv"


def evaluate(lm, data: dict, dataset: str, batch_size: int) -> list[dict]:
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    loader = build_loader(data, mods, batch_size=batch_size)

    rows = []
    for masked in mods:
        sq = {m: 0.0 for m in mods}
        count_elems = {m: 0 for m in mods}
        flat_real = {m: [] for m in mods}
        flat_pred = {m: [] for m in mods}
        n = 0
        for batch in loader:
            if batch is None:
                continue
            data_dict, _ = batch
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            masked_in = mask_modality(data_dict, masked)
            mu, _ = encode_batch(lm, masked_in)
            decoded = decode_batch(lm, mu)
            for m in mods:
                pred = decoded[m]
                real = data_dict[m]
                if pred.shape != real.shape:
                    pred = pred.view_as(real)
                diff = (pred - real).flatten()
                sq[m] += float((diff ** 2).sum().cpu())
                count_elems[m] += diff.numel()
                flat_real[m].append(real.flatten().cpu().numpy())
                flat_pred[m].append(pred.flatten().cpu().numpy())
            n += data_dict[mods[0]].shape[0]

        row = dict(method=lm.method, dataset=dataset, masked=masked, n=n)
        for m in mods:
            row[f"recon_{m}_mse"] = sq[m] / max(count_elems[m], 1)
            row[f"recon_{m}_w1"] = wasserstein_1d(
                np.concatenate(flat_real[m]) if flat_real[m] else np.array([]),
                np.concatenate(flat_pred[m]) if flat_pred[m] else np.array([]),
            )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    registry = load_registry(args.registry)

    all_rows = []
    for lm in iter_registry(registry):
        data = load_dataset_csv(args.data_dir, lm.modalities.keys(), target_sizes=lm.modalities)
        all_rows.extend(evaluate(lm, data, args.dataset, args.batch_size))

    write_results(all_rows, args.out, keys=["method", "dataset", "masked"])


if __name__ == "__main__":
    main()
