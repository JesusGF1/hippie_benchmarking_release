#!/usr/bin/env python3
"""Minimal HIPPIE trainer for the CVAE-only experiment harness.

Trains MultiModalCVAE on one or more datasets' CSVs and saves a checkpoint
for the G1–G7 experiments. Multi-dataset training is what makes conditioning
embeddings carry real information (single-source = degenerate conditioning =
posterior collapse).

Usage (multi-dataset, recommended for G2/G6 experiments):
    python analysis/CVAE_only_experiment_train_hippie.py \\
        --datasets-json /tmp/cvae_only_cache/datasets_hippie.json \\
        --epochs 80 --batch-size 256 --z-dim 16 --beta 0.1 --free-bits 0.5 \\
        --out /tmp/cvae_only_cache/hippie_local.ckpt

datasets_hippie.json format:
    [
      {"data_dir": "/tmp/cvae_only_cache/hausser",  "source_id": 3, "super_region_id": 0},
      {"data_dir": "/tmp/cvae_only_cache/hull",     "source_id": 1, "super_region_id": 0},
      {"data_dir": "/tmp/cvae_only_cache/lisberger","source_id": 4, "super_region_id": 0},
      {"data_dir": "/tmp/cvae_only_cache/dandi041", "source_id": 5, "super_region_id": 1}
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "hippie")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hippie.dataloading import MultiModalEphysDataset, none_safe_collate  # noqa
from hippie.multimodal_model import (  # noqa
    CVAEConfig,
    MultiModalCVAE,
    MultiModalCVAETrainModule,
)


_TARGETS = {"wave": 50, "isi": 100, "acg": 100}


class _CVAEFreeBitsTrainer(MultiModalCVAETrainModule):
    """MultiModalCVAETrainModule variant with free-bits KL + 2D conditioning labels.

    The base trainer expects scalar class labels; here the per-cell label vector
    packs [class_idx, source_id, super_region_id] so multi-dataset pretraining
    can carry both source and super-region conditioning. The free-bits floor on
    the KL term prevents posterior collapse when conditioning is informative.
    """

    def __init__(self, base, *, config, learning_rate, weight_decay,
                 free_bits, use_source, use_super_region):
        super().__init__(base, config=config, learning_rate=learning_rate,
                         weight_decay=weight_decay)
        self._free_bits = free_bits
        self._use_source = use_source
        self._use_super_region = use_super_region

    def training_step(self, batch, batch_idx):  # noqa: D401
        data_dict, label_tensor = self.process_batch(batch)
        # label_tensor: (B, 3) → [class, source, super_region]
        src_lbls = label_tensor[:, 1] if self._use_source else None
        sr_lbls = label_tensor[:, 2] if self._use_super_region else None
        _, mu, logvar, decoded = self.model(
            data_dict,
            source_labels=src_lbls,
            class_labels=None,
            super_region_labels=sr_lbls,
            apply_dropout=True,
        )
        recon = 0.0
        for m, w in self.modality_weights.items():
            recon = recon + w * self.mse_loss(decoded[m], data_dict[m])
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        if self._free_bits > 0.0:
            # Don't penalise dims that already use < free_bits nats.
            kl_above = torch.clamp(kl_per_dim - self._free_bits, min=0.0).sum(dim=-1).mean()
            kl_floor = self._free_bits * mu.shape[1]
            kl_loss = kl_above + kl_floor  # paid at least kl_floor regardless
            kl_for_log = kl_per_dim.sum(dim=-1).mean()  # true KL for monitoring
        else:
            kl_loss = kl_per_dim.sum(dim=-1).mean()
            kl_for_log = kl_loss
        loss = recon + self.beta * kl_loss
        self.log("train_recon", float(recon), prog_bar=True)
        self.log("train_kl_true", float(kl_for_log), prog_bar=True)
        self.log("train_loss", float(loss), prog_bar=True)
        return loss


def _resample_rowwise(arr: np.ndarray, target: int) -> np.ndarray:
    """Linear resample each row of (N, k) to (N, target)."""
    if arr.shape[1] == target:
        return arr.astype(np.float32)
    src = np.linspace(0.0, 1.0, arr.shape[1])
    dst = np.linspace(0.0, 1.0, target)
    out = np.empty((arr.shape[0], target), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.interp(dst, src, arr[i].astype(np.float64))
    return out


def _read_modality_csvs(data_dir: Path) -> dict:
    """Read CSVs and resample each modality to its canonical length so the
    arrays from different datasets can be concatenated."""
    out = {}
    for mod, fname in {"wave": "waveforms.csv", "isi": "isi_dist.csv", "acg": "acg.csv"}.items():
        a = pd.read_csv(data_dir / fname).to_numpy()
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        out[mod] = _resample_rowwise(a, _TARGETS[mod])
    return out


def _load_datasets(spec: list) -> tuple[dict, np.ndarray, np.ndarray]:
    """Concatenate multiple datasets. Returns (arrays_dict, source_ids, super_region_ids)."""
    parts, src, sr = [], [], []
    for entry in spec:
        d = Path(entry["data_dir"])
        arrays = _read_modality_csvs(d)
        n = len(arrays["wave"])
        parts.append(arrays)
        src.append(np.full(n, int(entry["source_id"]), dtype=np.int64))
        sr.append(np.full(n, int(entry["super_region_id"]), dtype=np.int64))
        print(f"  {d.name}: {n} cells, source_id={entry['source_id']}, super_region_id={entry['super_region_id']}")
    merged = {m: np.concatenate([p[m] for p in parts], axis=0) for m in ("wave", "isi", "acg")}
    return merged, np.concatenate(src), np.concatenate(sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-json", type=Path,
                    help="JSON list of {data_dir, source_id, super_region_id}")
    ap.add_argument("--data-dir", type=Path,
                    help="Single-dataset shortcut (use with --source-id, --super-region-id)")
    ap.add_argument("--source-id", type=int, default=0)
    ap.add_argument("--super-region-id", type=int, default=0)
    ap.add_argument("--num-sources", type=int, default=11)
    ap.add_argument("--num-super-regions", type=int, default=8)
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--free-bits", type=float, default=0.5,
                    help="Per-dim KL floor (anti-collapse)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--accelerator", default="auto")
    args = ap.parse_args()

    if args.datasets_json:
        spec = json.loads(args.datasets_json.read_text())
    elif args.data_dir:
        spec = [{"data_dir": str(args.data_dir),
                 "source_id": args.source_id,
                 "super_region_id": args.super_region_id}]
    else:
        raise SystemExit("must pass --datasets-json or --data-dir")

    print("[load] datasets:")
    arrays, src_ids, sr_ids = _load_datasets(spec)
    n = len(arrays["wave"])
    print(f"[load] total {n} cells across {len(spec)} dataset(s)")
    n_unique_src = len(np.unique(src_ids))
    n_unique_sr = len(np.unique(sr_ids))
    print(f"[load] unique source_ids: {n_unique_src} / unique super_regions: {n_unique_sr}")

    # Pack [class_idx (unused), source_id, super_region_id] into a 2D label tensor
    # so the existing dataloader can carry conditioning through to the training step.
    labels = np.stack([np.zeros(n, dtype=np.int64), src_ids, sr_ids], axis=1)
    ds = MultiModalEphysDataset(arrays, labels, mode="multi", normalize=True)
    loader = DataLoader(
        ds, batch_size=min(args.batch_size, n), shuffle=True,
        collate_fn=none_safe_collate, num_workers=0, drop_last=False,
    )

    config = CVAEConfig()
    config.use_source_embedding = n_unique_src > 1
    config.use_class_embedding = False
    config.use_super_region_embedding = n_unique_sr > 1
    config.beta = args.beta
    config.encoder_uses_class_embedding = False

    base = MultiModalCVAE(
        modalities={"wave": 50, "isi": 100, "acg": 100},
        z_dim=args.z_dim,
        config=config,
        num_sources=args.num_sources if config.use_source_embedding else None,
        num_classes=None,
        num_super_regions=args.num_super_regions if config.use_super_region_embedding else None,
    )

    use_src = config.use_source_embedding
    use_sr = config.use_super_region_embedding

    pl_module = _CVAEFreeBitsTrainer(
        base, config=config, learning_rate=args.lr, weight_decay=1e-4,
        free_bits=args.free_bits, use_source=use_src, use_super_region=use_sr,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator, devices=1,
        enable_checkpointing=False, enable_model_summary=False,
        logger=False, log_every_n_steps=1,
    )
    trainer.fit(pl_module, loader)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": pl_module.state_dict()}, args.out)
    print(f"[ok] saved checkpoint to {args.out}")

    # Posterior diagnostic — overall + per source bucket
    pl_module.eval()
    device = pl_module.device
    mus, logs, srcs = [], [], []
    with torch.no_grad():
        for batch in DataLoader(
            ds, batch_size=256, shuffle=False, collate_fn=none_safe_collate
        ):
            if batch is None:
                continue
            d, lbl = batch
            d = {k: v.to(device) for k, v in d.items()}
            lbl = lbl.to(device)
            src = lbl[:, 1] if use_src else None
            sr = lbl[:, 2] if use_sr else None
            _, mu, logvar = pl_module.model.encode(
                d, source_labels=src, class_labels=None, super_region_labels=sr
            )
            mus.append(mu.cpu().numpy())
            logs.append(logvar.cpu().numpy())
            if src is not None:
                srcs.append(src.cpu().numpy())
    mus = np.concatenate(mus)
    logs = np.concatenate(logs)
    print(f"[diag] mu  — mean={mus.mean():+.4f}  std-across-batch={mus.std(0).mean():+.4f}")
    print(f"[diag] sigma — mean={np.sqrt(np.exp(logs)).mean():+.4f}")
    if srcs:
        srcs = np.concatenate(srcs)
        for u in np.unique(srcs):
            ms = mus[srcs == u]
            print(f"[diag]   source={u}: n={len(ms)}  mu_std={ms.std(0).mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
