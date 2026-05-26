#!/usr/bin/env python3
"""G3 — latent interpolation between cell types.

Pick a pair of cell types (`--type-a`, `--type-b`), compute the per-class mean
posterior means μ_A, μ_B, and decode along z_α = (1−α)·μ_A + α·μ_B for
α ∈ [0, 1]. For each α, report:

    - L2 distance of each decoded modality to the endpoint-decoded modalities
      (should grow linearly in α if the path is smooth).
    - If `--classifier` is provided (joblib pickle of a sklearn-compatible
      classifier trained on real encoder embeddings), the classifier's
      predicted P(class=B) along α — should increase monotonically.

Writes one row per (method, dataset, type-pair, alpha) to
`results/benchmark/cvae_only/G3_latent_interpolation.csv`.

Usage:
    python analysis/CVAE_only_experiment_G3_latent_interpolation.py \\
        --registry configs/cvae_registry.json \\
        --data-dir /path/to/dataset \\
        --dataset hausser_cell_type \\
        --type-a PV --type-b pyramidal \\
        --steps 11
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
    write_results,
)

OUT = DEFAULT_RESULTS_DIR / "G3_latent_interpolation.csv"


def _class_mean_latent(lm, data: dict, indices: np.ndarray, batch_size: int) -> torch.Tensor:
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    loader = build_loader(data, mods, batch_size=batch_size, indices=indices)
    accum = None
    count = 0
    for batch in loader:
        if batch is None:
            continue
        data_dict, _ = batch
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        mu, _ = encode_batch(lm, data_dict)
        summed = mu.sum(dim=0)
        accum = summed if accum is None else accum + summed
        count += mu.shape[0]
    if accum is None or count == 0:
        return torch.zeros(lm.z_dim, device=device)
    return accum / count


def evaluate(
    lm, data: dict, dataset: str, type_a: str, type_b: str, steps: int, classifier
) -> list[dict]:
    device = next(lm.model.parameters()).device
    labels = data["labels"].astype(str)
    idx_a = np.where(labels == type_a)[0]
    idx_b = np.where(labels == type_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        print(f"[G3] empty class in {dataset}: a={len(idx_a)}, b={len(idx_b)}")
        return []

    mu_a = _class_mean_latent(lm, data, idx_a, batch_size=256)
    mu_b = _class_mean_latent(lm, data, idx_b, batch_size=256)

    alphas = np.linspace(0.0, 1.0, steps)
    rows = []

    with torch.no_grad():
        rec_a = decode_batch(lm, mu_a.unsqueeze(0))
        rec_b = decode_batch(lm, mu_b.unsqueeze(0))

        for alpha in alphas:
            z = (1 - alpha) * mu_a + alpha * mu_b
            rec = decode_batch(lm, z.unsqueeze(0))
            row = dict(
                method=lm.method,
                dataset=dataset,
                type_a=type_a,
                type_b=type_b,
                alpha=float(alpha),
            )
            for m in lm.modalities:
                d_a = float(((rec[m] - rec_a[m]) ** 2).mean().cpu())
                d_b = float(((rec[m] - rec_b[m]) ** 2).mean().cpu())
                row[f"dist_a_{m}"] = d_a
                row[f"dist_b_{m}"] = d_b
            if classifier is not None:
                try:
                    z_np = z.detach().cpu().numpy().reshape(1, -1)
                    if hasattr(classifier, "predict_proba"):
                        proba = classifier.predict_proba(z_np)[0]
                        if hasattr(classifier, "classes_"):
                            classes = list(classifier.classes_)
                            if type_b in classes:
                                row["prob_b"] = float(proba[classes.index(type_b)])
                except Exception as exc:
                    print(f"[G3] classifier failure at alpha={alpha:.2f}: {exc}")
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--type-a", required=True)
    ap.add_argument("--type-b", required=True)
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--classifier", type=Path,
                    help="joblib pickle of a sklearn classifier on real embeddings")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    registry = load_registry(args.registry)
    classifier = None
    if args.classifier:
        import joblib
        classifier = joblib.load(args.classifier)

    all_rows = []
    for lm in iter_registry(registry):
        data = load_dataset_csv(args.data_dir, lm.modalities.keys())
        all_rows.extend(
            evaluate(lm, data, args.dataset, args.type_a, args.type_b, args.steps, classifier)
        )

    write_results(
        all_rows, args.out,
        keys=["method", "dataset", "type_a", "type_b", "alpha"],
    )


if __name__ == "__main__":
    main()
