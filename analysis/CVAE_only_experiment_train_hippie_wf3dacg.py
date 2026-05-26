#!/usr/bin/env python3
"""Minimal HIPPIE-WF+3DACG trainer for the CVAE-only experiment harness.

Trains HippieWF3DACGCVAE on one or more C4 H5 files and saves a checkpoint
for the G1–G7 experiments. Multi-dataset training is required for source/
super_region/technology embeddings to be informative — single-dataset =
constant conditioning = posterior collapse.

Usage (multi-dataset, recommended):
    python analysis/CVAE_only_experiment_train_hippie_wf3dacg.py \\
        --datasets-json /tmp/cvae_only_cache/datasets_nemo.json \\
        --epochs 80 --batch-size 128 --z-dim 16 --beta-kl 0.5 \\
        --use-region-conditioning \\
        --out /tmp/cvae_only_cache/hippie_wf3dacg_local.ckpt

datasets_nemo.json format:
    [
      {"h5_path": "/tmp/.../hausser_hippie.h5",  "dataset_name": "hausser_cell_type",  "source_id": 3},
      {"h5_path": "/tmp/.../hull_hippie.h5",     "dataset_name": "hull_cell_type",     "source_id": 1},
      {"h5_path": "/tmp/.../lisberger_hippie.h5","dataset_name":"lisberger_labeled_cell_type","source_id": 4},
      {"h5_path": "/tmp/.../dandi041_hippie.h5", "dataset_name": "dandi_000041_cell_type","source_id": 5}
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "hippie_wf3dacg")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hippie_wf3dacg.dataloading import (  # noqa
    HippieWF3DACGDataset,
    LAYER_IDS,
    SUPER_REGION_IDS,
    TECHNOLOGY_IDS,
    none_safe_collate,
)
from hippie_wf3dacg.load_c4 import load_multi_dataset, load_c4_dataset  # noqa
from hippie_wf3dacg.model import HippieWF3DACGConfig, HippieWF3DACGLightning  # noqa


def _load_spec(spec: list) -> dict:
    """Concatenate multiple H5 datasets via the existing load_multi_dataset
    helper, which produces all conditioning ID arrays in one shot."""
    dataset_specs = {
        e["dataset_name"]: (e["h5_path"], int(e["source_id"]))
        for e in spec
    }
    return load_multi_dataset(dataset_specs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-json", type=Path)
    ap.add_argument("--h5-path", type=Path,
                    help="Single-dataset shortcut")
    ap.add_argument("--dataset-name")
    ap.add_argument("--source-id", type=int, default=0)
    ap.add_argument("--num-sources", type=int, default=11)
    ap.add_argument("--use-region-conditioning", action="store_true",
                    help="Enable super_region + layer embeddings")
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--beta-kl", type=float, default=0.5)
    ap.add_argument("--beta-acg", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--acg-cache-dir", type=Path,
                    default=Path("/tmp/cvae_only_cache/acg_cache"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--accelerator", default="auto")
    args = ap.parse_args()

    if args.datasets_json:
        spec = json.loads(args.datasets_json.read_text())
    elif args.h5_path and args.dataset_name:
        spec = [{"h5_path": str(args.h5_path),
                 "dataset_name": args.dataset_name,
                 "source_id": args.source_id}]
    else:
        raise SystemExit("must pass --datasets-json or (--h5-path + --dataset-name)")

    print("[load] datasets:")
    for e in spec:
        print(f"  {e['dataset_name']} <- {e['h5_path']}  source_id={e['source_id']}")

    if len(spec) == 1:
        e = spec[0]
        raw = load_c4_dataset(e["h5_path"], e["dataset_name"], source_id=int(e["source_id"]))
    else:
        raw = _load_spec(spec)
    n = len(raw["labels"])
    print(f"[load] total {n} cells")

    n_unique_src = len(np.unique(raw["source_ids"]))
    n_unique_sr = len(np.unique(raw["super_region_ids"]))
    n_unique_tech = len(np.unique(raw["technology_ids"]))
    n_unique_layer = len(np.unique(raw["layer_ids"]))
    print(f"[diag] uniques — src={n_unique_src} sr={n_unique_sr} tech={n_unique_tech} layer={n_unique_layer}")

    config = HippieWF3DACGConfig(
        z_dim=args.z_dim, beta_kl=args.beta_kl, beta_acg=args.beta_acg,
        warmup_epochs=args.warmup_epochs, use_augmentations=True,
        use_super_region_embedding=args.use_region_conditioning and n_unique_sr > 1,
        use_layer_embedding=args.use_region_conditioning and n_unique_layer > 1,
        # technology embedding is always on inside the model — not config-gated
    )

    ds = HippieWF3DACGDataset(
        waveforms=raw["waveforms"],
        spike_times=raw["spike_times"],
        labels=raw["labels"],
        source_ids=raw["source_ids"],
        wave_len=config.wave_len,
        acg_win_ms=config.acg_win_ms,
        acg_bin_ms=config.acg_bin_ms,
        acg_n_deciles=config.acg_n_deciles,
        acg_scale_factor=config.acg_scale_factor,
        super_region_ids=raw["super_region_ids"],
        technology_ids=raw["technology_ids"],
        layer_ids=raw["layer_ids"],
    )

    # ACG cache: one file per dataset name (shared across runs).
    args.acg_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "_".join(sorted(e["dataset_name"] for e in spec))
    cache_p = args.acg_cache_dir / f"acgs_{cache_key}_w{int(config.acg_win_ms)}_b{config.acg_bin_ms}_d{config.acg_n_deciles}.npy"
    if cache_p.exists():
        print(f"[acg] loading cache {cache_p}")
        ds.precomputed_acgs = np.load(cache_p)
    else:
        print("[acg] precomputing 3-D ACGs (one-shot)")
        t0 = time.time()
        ds.precompute_acgs()
        print(f"[acg] done in {time.time()-t0:.1f}s")
        np.save(cache_p, ds.precomputed_acgs)
        print(f"[acg] cached -> {cache_p}")

    loader = DataLoader(
        ds, batch_size=min(args.batch_size, n), shuffle=True,
        collate_fn=none_safe_collate, num_workers=0, drop_last=False,
    )

    model = HippieWF3DACGLightning(
        config=config,
        num_sources=args.num_sources,
        num_classes=len(raw["label_names"]),
        num_super_regions=len(SUPER_REGION_IDS) if config.use_super_region_embedding else None,
        num_technologies=len(TECHNOLOGY_IDS),
        num_layers=len(LAYER_IDS) if config.use_layer_embedding else None,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable params")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator, devices=1,
        enable_checkpointing=False, enable_model_summary=False,
        logger=False, log_every_n_steps=1,
    )
    trainer.fit(model, loader)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"[ok] saved checkpoint -> {args.out}")

    # Posterior diagnostic.
    model.eval()
    device = model.device
    mus, srcs = [], []
    with torch.no_grad():
        for batch in DataLoader(
            ds, batch_size=128, shuffle=False, collate_fn=none_safe_collate
        ):
            if batch is None:
                continue
            d, _ = batch
            wf = d["waveform"].to(device)
            acg = d["acg"].to(device)
            src = d["source_id"].to(device)
            kw = {}
            if "super_region_id" in d and config.use_super_region_embedding:
                kw["super_region_ids"] = d["super_region_id"].to(device)
            if "technology_id" in d:
                kw["technology_ids"] = d["technology_id"].to(device)
            if "layer_id" in d and config.use_layer_embedding:
                kw["layer_ids"] = d["layer_id"].to(device)
            _, mu, _ = model.model.encode(
                wf, acg, source_ids=src, class_ids=None, **kw,
            )
            mus.append(mu.cpu().numpy())
            srcs.append(src.cpu().numpy())
    mus = np.concatenate(mus)
    srcs = np.concatenate(srcs)
    print(f"[diag] mu  — mean={mus.mean():+.4f}  std-across-batch={mus.std(0).mean():+.4f}")
    for u in np.unique(srcs):
        ms = mus[srcs == u]
        print(f"[diag]   source={u}: n={len(ms)}  mu_std={ms.std(0).mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
