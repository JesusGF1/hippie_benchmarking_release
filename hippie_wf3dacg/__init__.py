"""
hippie-wf3dacg
===========

Conditional VAE with NEMO's feature representations (90-sample waveform MLP +
3D ACG convolutional encoder) fused with HIPPIE's conditioning framework
(source, super_region, technology, layer, decoder-only class label).

Public API:
    HippieWF3DACGConfig       — hyperparameter dataclass
    HippieWF3DACGCVAE         — core nn.Module
    HippieWF3DACGLightning    — PyTorch Lightning training wrapper
    HippieWF3DACGDataset      — torch Dataset (waveform + 3D ACG)
    make_dataloader        — DataLoader factory
    load_c4_dataset      — load one C4 H5 → dict
    load_multi_dataset     — concatenate multiple C4 H5 datasets
    compute_3d_acg         — pure-numpy 3D ACG from raw spike times
    TECHNOLOGY_IDS, SUPER_REGION_IDS, LAYER_IDS  — shared vocabularies
"""

from .acg import compute_3d_acg
from .dataloading import (
    DATASET_SUPER_REGION,
    DATASET_TECHNOLOGY,
    LAYER_IDS,
    LAYER_NORMALISE,
    SUPER_REGION_IDS,
    TECHNOLOGY_IDS,
    HippieWF3DACGDataset,
    make_dataloader,
    none_safe_collate,
)
from .load_c4 import load_c4_dataset, load_multi_dataset
from .model import (
    HippieWF3DACGConfig,
    HippieWF3DACGCVAE,
    HippieWF3DACGLightning,
)

__all__ = [
    "HippieWF3DACGConfig",
    "HippieWF3DACGCVAE",
    "HippieWF3DACGLightning",
    "HippieWF3DACGDataset",
    "make_dataloader",
    "none_safe_collate",
    "load_c4_dataset",
    "load_multi_dataset",
    "compute_3d_acg",
    "TECHNOLOGY_IDS",
    "SUPER_REGION_IDS",
    "LAYER_IDS",
    "LAYER_NORMALISE",
    "DATASET_TECHNOLOGY",
    "DATASET_SUPER_REGION",
]
