#!/usr/bin/env python3
"""G2 — conditional counterfactual swap.

Encode a unit with its real conditioning, then decode with a swapped value
along `--swap-axis` (source or super_region). For every (c_true → c_swap)
pair, measure whether the decoded modalities move toward the population
statistics of the swapped bucket.

Each modality contributes one column:
    baseline_<m>_err           : ||decode(z, c_true) - mean_pop(c_true)_m||²
    counterfactual_<m>_err     : ||decode(z, c_swap) - mean_pop(c_swap)_m||²
    swap_gain_<m>              : baseline - counterfactual
                                 (positive = conditioning steered the decoder)

Modality set is per-model: HIPPIE gets wave/isi/acg columns, HIPPIE-WF+3DACG
gets wave/acg_3d columns (only `wave` is directly comparable).

Usage:
    python analysis/CVAE_only_experiment_G2_counterfactual_swap.py \\
        --registry configs/cvae_registry.json \\
        --data-dir /path/to/dataset \\
        --dataset hausser_cell_type \\
        --swap-axis source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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

OUT = DEFAULT_RESULTS_DIR / "G2_counterfactual_swap.csv"


def _population_means(data: dict, mods: list[str], bucket_ids: np.ndarray) -> dict:
    means = {}
    for b in np.unique(bucket_ids):
        idx = bucket_ids == b
        means[int(b)] = {m: data[m][idx].mean(axis=0) for m in mods}
    return means


def evaluate(
    lm,
    data: dict,
    dataset: str,
    swap_axis: str,
    bucket_ids: np.ndarray,
    batch_size: int,
) -> list[dict]:
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    unique = sorted(int(x) for x in np.unique(bucket_ids))
    if len(unique) < 2:
        print(f"[G2] {dataset}/{swap_axis}: only {len(unique)} bucket(s), skipping")
        return []

    pop_means = _population_means(data, mods, bucket_ids)
    rows = []
    for src in unique:
        src_idx = np.where(bucket_ids == src)[0]
        if len(src_idx) == 0:
            continue
        src_loader = build_loader(data, mods, batch_size=batch_size, indices=src_idx)
        for tgt in unique:
            if tgt == src:
                continue
            err_true = {m: 0.0 for m in mods}
            err_swap = {m: 0.0 for m in mods}
            n = 0
            for batch in src_loader:
                if batch is None:
                    continue
                data_dict, _ = batch
                data_dict = {k: v.to(device) for k, v in data_dict.items()}
                bs = data_dict[mods[0]].shape[0]

                kw_true, kw_swap = {}, {}
                if swap_axis == "source" and lm.num_sources > 0:
                    kw_true["source_labels"] = torch.full(
                        (bs,), src, dtype=torch.long, device=device
                    )
                    kw_swap["source_labels"] = torch.full(
                        (bs,), tgt, dtype=torch.long, device=device
                    )
                elif swap_axis == "super_region" and lm.num_super_regions:
                    kw_true["super_region_labels"] = torch.full(
                        (bs,), src, dtype=torch.long, device=device
                    )
                    kw_swap["super_region_labels"] = torch.full(
                        (bs,), tgt, dtype=torch.long, device=device
                    )
                else:
                    print(
                        f"[G2] {lm.method}: no embedding for axis '{swap_axis}', skipping"
                    )
                    return []

                mu, _ = encode_batch(lm, data_dict, **kw_true)
                rec_true = decode_batch(lm, mu, **kw_true)
                rec_swap = decode_batch(lm, mu, **kw_swap)

                for m in mods:
                    mean_true = torch.as_tensor(
                        pop_means[src][m], dtype=torch.float32, device=device
                    ).view(1, -1)
                    mean_swap = torch.as_tensor(
                        pop_means[tgt][m], dtype=torch.float32, device=device
                    ).view(1, -1)
                    pt = rec_true[m].view(bs, -1)
                    ps = rec_swap[m].view(bs, -1)
                    err_true[m] += float(((pt - mean_true) ** 2).mean(dim=1).sum().cpu())
                    err_swap[m] += float(((ps - mean_swap) ** 2).mean(dim=1).sum().cpu())
                n += bs

            if n == 0:
                continue
            row = dict(
                method=lm.method,
                dataset=dataset,
                swap_axis=swap_axis,
                src=src,
                tgt=tgt,
                n=n,
            )
            for m in mods:
                row[f"baseline_{m}_err"] = err_true[m] / n
                row[f"counterfactual_{m}_err"] = err_swap[m] / n
                row[f"swap_gain_{m}"] = (err_true[m] - err_swap[m]) / n
            rows.append(row)
    return rows


def _build_bucket_ids(data: dict, swap_axis: str) -> np.ndarray:
    if swap_axis == "source":
        if "source_id" not in data:
            raise SystemExit(
                "[G2] source_id.csv not found in --data-dir; cannot swap on source"
            )
        return data["source_id"]
    if "super_regions" not in data:
        raise SystemExit("[G2] no super_regions.csv; cannot swap on super_region")
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder().fit_transform(data["super_regions"]).astype(np.int64)


def _save_examples(
    lm, data: dict, bucket_ids: np.ndarray, swap_axis: str,
    src: int, tgt: int, n_examples: int, out_npz: Path,
) -> None:
    """Save n_examples before/after traces for one (src → tgt) swap to .npz."""
    device = next(lm.model.parameters()).device
    mods = list(lm.modalities.keys())
    src_idx = np.where(bucket_ids == src)[0][:n_examples]
    if len(src_idx) == 0:
        return

    loader = build_loader(data, mods, batch_size=len(src_idx), indices=src_idx)
    batch_raw, _ = next(iter(loader))
    batch_data = {k: v.to(device) for k, v in batch_raw.items()}

    kw_true, kw_swap = {}, {}
    bs = len(src_idx)
    if swap_axis == "source" and lm.num_sources > 0:
        kw_true["source_labels"] = torch.full((bs,), src, dtype=torch.long, device=device)
        kw_swap["source_labels"] = torch.full((bs,), tgt, dtype=torch.long, device=device)
    elif swap_axis == "super_region" and lm.num_super_regions:
        kw_true["super_region_labels"] = torch.full((bs,), src, dtype=torch.long, device=device)
        kw_swap["super_region_labels"] = torch.full((bs,), tgt, dtype=torch.long, device=device)
    else:
        return

    with torch.no_grad():
        mu, _ = encode_batch(lm, batch_data, **kw_true)
        rec_true = decode_batch(lm, mu, **kw_true)
        rec_swap = decode_batch(lm, mu, **kw_swap)

    arrays = {"src": np.array([src] * bs), "tgt": np.array([tgt] * bs)}
    labels = data.get("labels", np.array(["unlabeled"] * len(data[mods[0]])))
    arrays["cell_type"] = np.array(labels[src_idx].tolist())
    for m in mods:
        arrays[f"real_{m}"] = batch_data[m].cpu().numpy()
        arrays[f"true_{m}"] = rec_true[m].cpu().numpy()
        arrays[f"swap_{m}"] = rec_swap[m].cpu().numpy()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), **arrays)
    print(f"[G2 examples] saved {bs} examples ({src}→{tgt}) to {out_npz}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--swap-axis", choices=["source", "super_region"], default="source")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--save-examples-npz", type=Path, default=None,
                    help="If set, save before/after example traces for the most "
                         "interesting (highest swap-gain) source pair.")
    ap.add_argument("--n-examples", type=int, default=6)
    args = ap.parse_args()

    registry = load_registry(args.registry)

    all_rows = []
    for lm in iter_registry(registry):
        data = load_dataset_csv(args.data_dir, lm.modalities.keys(), target_sizes=lm.modalities)
        bucket_ids = _build_bucket_ids(data, args.swap_axis)
        rows = evaluate(
            lm, data, args.dataset,
            swap_axis=args.swap_axis,
            bucket_ids=bucket_ids,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)

        if args.save_examples_npz and rows:
            # Pick the (src, tgt) pair with highest mean swap gain across modalities
            df_tmp = pd.DataFrame(rows)
            gain_cols = [c for c in df_tmp.columns if c.startswith("swap_gain_")]
            if gain_cols:
                df_tmp["mean_gain"] = df_tmp[gain_cols].mean(axis=1)
                best = df_tmp.loc[df_tmp["mean_gain"].idxmax()]
                _save_examples(
                    lm, data, bucket_ids, args.swap_axis,
                    int(best["src"]), int(best["tgt"]),
                    args.n_examples, args.save_examples_npz,
                )

    write_results(
        all_rows, args.out, keys=["method", "dataset", "swap_axis", "src", "tgt"]
    )


if __name__ == "__main__":
    main()
