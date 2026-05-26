"""
hippie-wf3dacg — C4 H5 Dataset Loader
=====================================

Reads C4-database H5 files (Beau et al., 2024) and returns the arrays needed
to construct a HippieWF3DACGDataset.

Per-dataset conditioning metadata (source / super-region / technology /
layer) follows the same vocabulary used by the rest of the HIPPIE pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .dataloading import (
    DATASET_SUPER_REGION,
    DATASET_TECHNOLOGY,
    LAYER_IDS,
    LAYER_NORMALISE,
    SUPER_REGION_IDS,
    TECHNOLOGY_IDS,
)


# ---------------------------------------------------------------------------
# Dataset-level config (mirrors C4 H5 loader exactly)
# ---------------------------------------------------------------------------

_LAYER_FIELD: Dict[str, Optional[str]] = {
    "hausser_cell_type":            "human_layer",
    "hull_cell_type":               "phyllum_layer",
    "lisberger_labeled_cell_type": "human_layer",
}

_LABEL_FIELD: Dict[str, str] = {
    "lisberger_labeled_cell_type": "expert_label",
}

_IBL_DATASETS   = {"ibl_brainwide", "ibl_brainwide_curated", "ibl_brainwide_good"}
_ALLEN_DATASETS = {"allen_scope_neuropixel_area_subset", "allen_scope_neuropixel"}

_ALLEN_LABEL_TO_SUPER_REGION: Dict[str, str] = {
    "CA1": "hippocampus", "CA3": "hippocampus", "DG":  "hippocampus",
    "ProS":"hippocampus", "SUB": "hippocampus",
    "VISal":"cortex", "VISam":"cortex", "VISl":"cortex",
    "VISp": "cortex", "VISpm":"cortex", "VISrl":"cortex",
    "LGd":"thalamus", "LP":"thalamus", "MGv":"thalamus",
    "PO": "thalamus", "SGN":"thalamus", "VPM":"thalamus",
    "MB":"midbrain",
    "Eth":"other",
}


# ---------------------------------------------------------------------------
# H5 reading helpers
# ---------------------------------------------------------------------------

def _read_bytes_field(grp: h5py.Group, field: str, default: str = "") -> str:
    if field not in grp:
        return default
    raw = grp[field][()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8").strip()
    if isinstance(raw, np.ndarray):
        item = raw.item() if raw.ndim == 0 else raw.flat[0]
        return item.decode("utf-8").strip() if isinstance(item, bytes) else str(item).strip()
    return str(raw).strip()


def _read_layer(grp: h5py.Group, layer_field: Optional[str]) -> str:
    if layer_field is None:
        return "unknown"
    raw = _read_bytes_field(grp, layer_field, default="")
    return LAYER_NORMALISE.get(raw, "unknown")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_c4_dataset(
    h5_path: str,
    dataset_name: str,
    source_id: int = 0,
) -> Dict:
    """Load one C4 H5 file into arrays for HippieWF3DACGDataset.

    Returns a dict with:
        waveforms        — (N, wave_samples) float32
        spike_times      — list of N np.ndarray (float64, seconds)
        labels           — (N,) int64
        str_labels       — list of N str
        label_names      — sorted list of unique labels
        source_ids       — (N,) int64  all equal to source_id
        super_region_ids — (N,) int64
        technology_ids   — (N,) int64
        layer_ids        — (N,) int64
        animal_ids       — list of N str  (empty string when absent)
    """
    h5_path   = str(h5_path)
    is_ibl    = dataset_name in _IBL_DATASETS
    is_allen  = dataset_name in _ALLEN_DATASETS
    layer_field = _LAYER_FIELD.get(dataset_name)
    label_field = _LABEL_FIELD.get(dataset_name, "ground_truth_label")

    tech_str = DATASET_TECHNOLOGY.get(dataset_name, "silicon_probe")
    tech_int = TECHNOLOGY_IDS.get(tech_str, 0)

    if not is_ibl and not is_allen:
        sr_str        = DATASET_SUPER_REGION.get(dataset_name, "other")
        sr_default_int = SUPER_REGION_IDS.get(sr_str, SUPER_REGION_IDS["other"])

    waveforms:       List[np.ndarray] = []
    spike_times:     List[np.ndarray] = []
    str_labels:      List[str]        = []
    super_region_list: List[int]      = []
    layer_list:      List[int]        = []
    animal_ids:      List[str]        = []

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            grp = f[key]

            if "mean_waveform_preprocessed" not in grp:
                continue

            # Waveform
            wave = grp["mean_waveform_preprocessed"][:].astype(np.float32)
            if wave.ndim == 2:
                peak_ch = int(np.argmax(np.abs(wave).max(axis=-1)))
                wave    = wave[peak_ch]
            waveforms.append(wave)

            # Spike times (seconds)
            if "spike_indices" in grp and "sampling_rate" in grp:
                rate = float(grp["sampling_rate"][()])
                st   = grp["spike_indices"][:].astype(np.float64) / rate
            else:
                st = np.empty(0, dtype=np.float64)
            spike_times.append(st)

            # Label
            lbl = _read_bytes_field(grp, label_field, default="unlabeled")
            str_labels.append(lbl)

            # Super-region (Allen only; IBL and all other datasets are
            # unconditioned by design and receive the neutral default).
            if is_allen:
                sr_str_n = _ALLEN_LABEL_TO_SUPER_REGION.get(lbl, "other")
                super_region_list.append(SUPER_REGION_IDS.get(sr_str_n, SUPER_REGION_IDS["other"]))
            else:
                super_region_list.append(sr_default_int)

            # Layer
            layer_str = _read_layer(grp, layer_field)
            layer_list.append(LAYER_IDS.get(layer_str, LAYER_IDS["unknown"]))

            # Animal ID
            animal_ids.append(_read_bytes_field(grp, "animal_id", default=""))

    N = len(waveforms)
    if N == 0:
        raise ValueError(f"No valid neurons found in {h5_path}")

    label_names = sorted(set(str_labels))
    label_map   = {l: i for i, l in enumerate(label_names)}
    labels      = np.array([label_map[l] for l in str_labels], dtype=np.int64)

    try:
        waveforms_arr = np.array(waveforms, dtype=np.float32)
    except ValueError:
        waveforms_arr = np.empty(N, dtype=object)
        for i, w in enumerate(waveforms):
            waveforms_arr[i] = w

    return {
        "waveforms":        waveforms_arr,
        "spike_times":      spike_times,
        "labels":           labels,
        "str_labels":       str_labels,
        "label_names":      label_names,
        "source_ids":       np.full(N, source_id, dtype=np.int64),
        "super_region_ids": np.array(super_region_list, dtype=np.int64),
        "technology_ids":   np.full(N, tech_int, dtype=np.int64),
        "layer_ids":        np.array(layer_list, dtype=np.int64),
        "animal_ids":       animal_ids,
    }


# ---------------------------------------------------------------------------
# Multi-dataset combiner (mirrors C4 H5 loader)
# ---------------------------------------------------------------------------

def load_multi_dataset(dataset_specs: Dict[str, Tuple[str, int]]) -> Dict:
    """Load and concatenate multiple C4 H5 datasets into one dict.

    Args:
        dataset_specs: Mapping of dataset_name → (h5_path, source_id).
    """
    all_parts: List[Dict] = []
    for ds_name, (h5_path, src_id) in dataset_specs.items():
        part = load_c4_dataset(h5_path, ds_name, source_id=src_id)
        prefixed = [f"{ds_name}::{l}" for l in part["str_labels"]]
        part["str_labels"]  = prefixed
        part["label_names"] = sorted(set(prefixed))
        all_parts.append(part)

    all_label_names  = sorted({l for p in all_parts for l in p["label_names"]})
    global_label_map = {l: i for i, l in enumerate(all_label_names)}

    def _vstack(arrays):
        all_rows = [row for arr in arrays for row in arr]
        lengths  = {len(r) for r in all_rows}
        if len(lengths) == 1:
            return np.vstack(all_rows)
        out = np.empty(len(all_rows), dtype=object)
        for i, r in enumerate(all_rows):
            out[i] = r
        return out

    return {
        "waveforms":        _vstack([p["waveforms"] for p in all_parts]),
        "spike_times":      [st for p in all_parts for st in p["spike_times"]],
        "labels":           np.concatenate([
            np.array([global_label_map[l] for l in p["str_labels"]], dtype=np.int64)
            for p in all_parts
        ]),
        "str_labels":       [l for p in all_parts for l in p["str_labels"]],
        "label_names":      all_label_names,
        "source_ids":       np.concatenate([p["source_ids"]       for p in all_parts]),
        "super_region_ids": np.concatenate([p["super_region_ids"] for p in all_parts]),
        "technology_ids":   np.concatenate([p["technology_ids"]   for p in all_parts]),
        "layer_ids":        np.concatenate([p["layer_ids"]        for p in all_parts]),
    }
