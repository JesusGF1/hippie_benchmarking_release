"""Smoke-test the publicly released HIPPIE checkpoint on Hugging Face Hub.

Downloads `hippie_techcond_v1.ckpt` from huggingface.co/Jesusgf23/hippie
and verifies that it loads with the released `hippie` package and produces a
finite latent embedding from a synthetic input. Intended as a one-command
sanity check for reviewers and downstream users: if this script exits 0,
the published artifact and the published code are mutually compatible.

Usage
-----
From the repo root, after `pip install -e .`:

    python examples/smoke_test_hf_checkpoint.py

Exits with status 0 on success and prints a short report. Requires only
torch, numpy, and the `huggingface_hub` Python client (the latter is fetched
into a temp directory if absent).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


HF_REPO_ID = "Jesusgf23/hippie"
HF_FILENAME = "hippie_techcond_v1.ckpt"


def _ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[smoke] huggingface_hub not installed — installing into the current env.")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.22"]
        )


def _download_checkpoint(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"[smoke] downloading {HF_REPO_ID}/{HF_FILENAME} ...")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        cache_dir=str(cache_dir),
    )
    return Path(local_path)


def _load_with_hippie_package(ckpt_path: Path) -> dict:
    """Use the same loader the analysis scripts use, so the smoke test
    fails the same way analysis would if the checkpoint format drifts."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.CVAE_only_experiment_utils import load_model

    loaded = load_model(
        method="smoke_test",
        ckpt_path=str(ckpt_path),
        device="cpu",
    )
    return {
        "model": loaded.model,
        "modalities": loaded.modalities,
        "z_dim": loaded.z_dim,
        "num_sources": loaded.num_sources,
        "num_classes": loaded.num_classes,
        "num_super_regions": loaded.num_super_regions,
    }


def _build_synthetic_batch(modalities: dict, num_sources: int) -> dict:
    batch = {}
    for name, size in modalities.items():
        # ResNet18Enc expects (B, C=1, L).
        batch[name] = torch.zeros((2, 1, int(size)), dtype=torch.float32)
    source_labels = None
    if num_sources > 0:
        source_labels = torch.zeros(2, dtype=torch.long)
    return {"data": batch, "source_labels": source_labels}


def _encode(model: torch.nn.Module, batch: dict, num_sources: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        # MultiModalCVAE.encode returns (encoded, z_mean, z_log_var). We
        # mask class_labels at evaluation time (matches paper Methods § 490).
        encoded, z_mean, z_log_var = model.encode(
            batch["data"],
            source_labels=batch["source_labels"],
            class_labels=None,
        )
    return z_mean.detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache the downloaded checkpoint. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the downloaded checkpoint after the test (default: delete).",
    )
    args = parser.parse_args()

    _ensure_huggingface_hub()

    if args.cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="hippie_smoke_"))
        owns_cache = True
    else:
        cache_dir = args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        owns_cache = False

    try:
        ckpt_path = _download_checkpoint(cache_dir)
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"[smoke] downloaded ({size_mb:.1f} MB) -> {ckpt_path}")

        info = _load_with_hippie_package(ckpt_path)
        print(
            f"[smoke] loaded model: z_dim={info['z_dim']} "
            f"modalities={info['modalities']} "
            f"num_sources={info['num_sources']} "
            f"num_classes={info['num_classes']}"
        )

        batch = _build_synthetic_batch(info["modalities"], info["num_sources"])
        z = _encode(info["model"], batch, info["num_sources"])
        if not np.isfinite(z).all():
            raise RuntimeError(
                "Latent embedding contains non-finite values — checkpoint likely "
                "incompatible with the released hippie package."
            )
        print(
            f"[smoke] OK — latent embedding shape={z.shape} "
            f"mean={float(z.mean()):.4f} std={float(z.std()):.4f}"
        )
        return 0
    finally:
        if owns_cache and not args.keep:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
