"""CVAE-only experiment shared helpers.

Inference-time utilities for the G1–G7 scripts (`CVAE_only_experiment_*`)
that run against HIPPIE and HIPPIE-WF+3DACG checkpoints.

HIPPIE operates on wave + ISI + ACG (all 1D). HIPPIE-WF+3DACG operates on
wave + 3D ACG (the 10×101 tensor). Because the two checkpoints carry
different modality sets, this module infers each model's modalities from
the checkpoint's state_dict rather than hard-coding them.

For 2D modalities (e.g. HIPPIE-WF+3DACG's `acg_3d`) the MultiModalCVAE class in
`hippie/multimodal_model.py` is 1D-only; loading a HIPPIE-WF+3DACG checkpoint
requires its own 2D-compatible class. Until that class lands in `hippie/`,
this loader raises a clear error on 2D modalities. Hook point marked below.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hippie.dataloading import MultiModalEphysDataset, none_safe_collate  # noqa: E402
from hippie.multimodal_model import (  # noqa: E402
    CVAEConfig,
    MultiModalCVAE,
)

HIPPIE_MODALITY_SIZES = {"wave": 50, "isi": 100, "acg": 100}
DEFAULT_RESULTS_DIR = REPO / "results" / "benchmark" / "cvae_only"
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    modalities: dict  # {name: int size} for 1D or {name: tuple shape} for 2D
    num_sources: int
    num_classes: int | None
    num_super_regions: int | None
    z_dim: int
    method: str
    modality_is_2d: dict = field(default_factory=dict)


def _load_state_dict(ckpt_path: str) -> dict:
    if ckpt_path.startswith("s3://"):
        import boto3
        without = ckpt_path[5:]
        bucket, _, key = without.partition("/")
        endpoint = os.environ.get(
            "S3_ENDPOINT", "https://s3.example.com"
        )
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            tmp_path = tmp.name
        s3.download_file(bucket, key, tmp_path)
        ckpt = torch.load(tmp_path, map_location="cpu")
        os.unlink(tmp_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def _strip_prefix(sd: dict) -> dict:
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}


def _infer_modalities(sd: dict) -> dict:
    """Infer modality name → output size from decoder keys.

    HIPPIE's ResNet18Dec has a `linear_out` final layer whose out_features is
    the flat modality size. We walk the state_dict for keys matching
    `decoders.<name>.linear_out.weight` and read shape[0].
    """
    mods = {}
    for key, tensor in sd.items():
        if key.startswith("decoders.") and key.endswith(".linear_out.weight"):
            name = key[len("decoders.") : -len(".linear_out.weight")]
            mods[name] = int(tensor.shape[0])
    return mods


def _infer_dims(sd: dict) -> tuple[int, int | None, int | None, int]:
    def _get(key: str):
        return sd.get(key)

    src = _get("source_embedding.weight")
    cls = _get("class_embedding.weight")
    sr = _get("super_region_embedding.weight")
    zmean = _get("z_mean.weight")
    if zmean is None:
        raise KeyError("state_dict missing z_mean.weight — not a HIPPIE-family checkpoint?")

    num_sources = int(src.shape[0]) if src is not None else 0
    num_classes = int(cls.shape[0]) if cls is not None else None
    num_super_regions = int(sr.shape[0]) if sr is not None else None
    z_dim = int(zmean.shape[0])
    return num_sources, num_classes, num_super_regions, z_dim


def load_model(method: str, ckpt_path: str, device: str = "cuda") -> LoadedModel:
    sd = _strip_prefix(_load_state_dict(ckpt_path))
    modalities = _infer_modalities(sd)
    if not modalities:
        raise RuntimeError(
            f"[{method}] could not infer modalities from state_dict "
            f"(no decoders.<name>.linear_out.weight keys found)"
        )

    # Detect modalities with 2D encoders (e.g. HIPPIE-WF+3DACG's acg_3d). Hook
    # point: if a model class for 2D modalities ships (expected for
    # HIPPIE-WF+3DACG's 3D-ACG input), branch here to construct it instead of
    # MultiModalCVAE. For now, fail loudly.
    two_d_modalities = {
        name for name in modalities
        if any(k.startswith(f"encoders.{name}.") and "conv2d" in k.lower() for k in sd)
    }
    if two_d_modalities:
        raise NotImplementedError(
            f"[{method}] checkpoint uses 2D modalities {sorted(two_d_modalities)}. "
            "The HIPPIE-WF+3DACG 2D-capable model class is not yet in hippie/. "
            "Add it and extend this loader once it lands."
        )

    num_sources, num_classes, num_super_regions, z_dim = _infer_dims(sd)

    config = CVAEConfig()
    config.use_source_embedding = num_sources > 0
    config.use_class_embedding = num_classes is not None
    config.use_super_region_embedding = num_super_regions is not None

    model = MultiModalCVAE(
        modalities=modalities,
        z_dim=z_dim,
        config=config,
        num_sources=num_sources if num_sources > 0 else None,
        num_classes=num_classes,
        num_super_regions=num_super_regions,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[{method}] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[{method}] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        modalities=modalities,
        num_sources=num_sources,
        num_classes=num_classes,
        num_super_regions=num_super_regions,
        z_dim=z_dim,
        method=method,
        modality_is_2d={m: False for m in modalities},
    )


_CSV_NAME_FOR_MODALITY = {
    "wave": "waveforms.csv",
    "isi": "isi_dist.csv",
    "acg": "acg.csv",
}


def _resample_rows(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[1] == target:
        return arr.astype(np.float32)
    src = np.linspace(0.0, 1.0, arr.shape[1])
    dst = np.linspace(0.0, 1.0, target)
    out = np.empty((arr.shape[0], target), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.interp(dst, src, arr[i].astype(np.float64))
    return out


def load_dataset_csv(
    data_dir: Path,
    modalities: Iterable[str],
    target_sizes: dict[str, int] | None = None,
) -> dict:
    """Load only the modalities a given model needs.

    For 1D modalities (HIPPIE's wave/isi/acg) reads the CSV. For unknown
    modality names (e.g. HIPPIE-WF+3DACG's `acg_3d`) reads `<name>.npy` if present;
    raises if missing.

    If `target_sizes` is provided (e.g. lm.modalities), each 1D modality is
    linearly resampled row-wise to its canonical length so checkpoints trained
    on canonical widths can score datasets shipped at raw recording widths.
    """
    data_dir = Path(data_dir)
    out = {}
    for mod in modalities:
        if mod in _CSV_NAME_FOR_MODALITY:
            arr = pd.read_csv(data_dir / _CSV_NAME_FOR_MODALITY[mod]).to_numpy()
        else:
            npy = data_dir / f"{mod}.npy"
            if not npy.exists():
                raise FileNotFoundError(
                    f"modality '{mod}' requires {npy} (not shipped as CSV)"
                )
            arr = np.load(npy)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if target_sizes is not None and mod in target_sizes and arr.ndim == 2:
            arr = _resample_rows(arr, int(target_sizes[mod]))
        out[mod] = arr

    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        labels_path = data_dir / "celltypes.csv"
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        out["labels"] = df[df.columns[0]].values
    else:
        any_arr = next(iter(out.values()))
        out["labels"] = np.array(["unlabeled"] * len(any_arr))

    for fname in ("super_regions.csv", "layer_labels.csv"):
        p = data_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            out["super_regions"] = df[df.columns[0]].values.astype(str)
            break

    src_path = data_dir / "source_id.csv"
    if src_path.exists():
        out["source_id"] = pd.read_csv(src_path).to_numpy().ravel().astype(np.int64)
    return out


def build_loader(
    data: dict,
    modalities: Iterable[str],
    batch_size: int = 256,
    indices: np.ndarray | None = None,
) -> DataLoader:
    mods = list(modalities)
    if indices is not None:
        subset = {m: data[m][indices] for m in mods}
        n = len(indices)
    else:
        subset = {m: data[m] for m in mods}
        n = len(subset[mods[0]])
    labels = np.arange(n)
    ds = MultiModalEphysDataset(subset, labels, mode="multi", normalize=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=none_safe_collate,
        num_workers=0,
    )


def mask_modality(data_dict: dict, modality: str) -> dict:
    out = dict(data_dict)
    out[modality] = torch.zeros_like(out[modality])
    return out


def _default_cond_labels(
    lm: LoadedModel, batch_size: int, device: torch.device,
    source_labels: torch.Tensor | None, super_region_labels: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Supply zero-label tensors when the model has conditioning embeddings but
    caller passed None. Without this, `_get_embeddings` falls back to
    batch_size=1 and the concat inside `encode`/`decode` fails."""
    if source_labels is None and lm.num_sources and lm.num_sources > 0:
        source_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    if super_region_labels is None and lm.num_super_regions:
        super_region_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    return source_labels, super_region_labels


@torch.no_grad()
def encode_batch(
    lm: LoadedModel,
    data_dict: dict,
    source_labels: torch.Tensor | None = None,
    super_region_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    any_tensor = next(iter(data_dict.values()))
    source_labels, super_region_labels = _default_cond_labels(
        lm, any_tensor.shape[0], any_tensor.device, source_labels, super_region_labels
    )
    _, mu, logvar = lm.model.encode(
        data_dict,
        source_labels=source_labels,
        class_labels=None,
        super_region_labels=super_region_labels,
    )
    return mu, logvar


@torch.no_grad()
def decode_batch(
    lm: LoadedModel,
    z: torch.Tensor,
    source_labels: torch.Tensor | None = None,
    super_region_labels: torch.Tensor | None = None,
) -> dict:
    source_labels, super_region_labels = _default_cond_labels(
        lm, z.shape[0], z.device, source_labels, super_region_labels
    )
    return lm.model.decode(
        z,
        source_labels=source_labels,
        class_labels=None,
        super_region_labels=super_region_labels,
    )


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a).ravel())
    b = np.sort(np.asarray(b).ravel())
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n)
    return float(
        np.mean(
            np.abs(
                np.interp(q, np.linspace(0, 1, len(a)), a)
                - np.interp(q, np.linspace(0, 1, len(b)), b)
            )
        )
    )


def load_registry(path: str | Path) -> dict[str, str]:
    with open(path) as fh:
        reg = json.load(fh)
    allowed = {"hippie", "hippie_wf3dacg"}
    unknown = set(reg) - allowed
    if unknown:
        print(
            f"[registry] ignoring non-cVAE entries: {sorted(unknown)} "
            f"(CVAE_only_experiments run on hippie + hippie_wf3dacg only)"
        )
    return {k: v for k, v in reg.items() if k in allowed}


def iter_registry(reg: dict[str, str], device: str = "cuda") -> Iterable[LoadedModel]:
    for method, path in reg.items():
        print(f"[load] {method} <- {path}")
        yield load_model(method, path, device=device)


def write_results(rows: list[dict], out_path: Path, keys: list[str]) -> None:
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        merged = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            keys, keep="last"
        )
    else:
        merged = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} ({len(merged)} rows)")
