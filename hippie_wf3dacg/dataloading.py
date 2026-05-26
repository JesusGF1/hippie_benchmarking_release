"""
hippie-wf3dacg Dataset
===================

Loads paired (waveform, spike_times) data and converts spike_times → 3D ACG
on the fly (or from a precomputed cache), using NEMO's feature conventions:

  waveform  — 90-sample L∞-normalised peak-channel waveform (float32)
  acg       — (1, 10, 101) 3D autocorrelogram tensor, scaled by acg_scale_factor

Conditioning covariates :
  source_id, super_region_id, technology_id, layer_id

See load_nemo.py for the loader that populates all fields from C4 H5 files.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .acg import compute_3d_acg


# ---------------------------------------------------------------------------
# Shared vocabularies 
# ---------------------------------------------------------------------------

TECHNOLOGY_IDS: Dict[str, int] = {
    "neuropixels":   0,
    "silicon_probe": 1,
    "juxtacellular": 2,
    "tetrode":       3,
}

SUPER_REGION_IDS: Dict[str, int] = {
    "cerebellum":  0,
    "cortex":      1,
    "hippocampus": 2,
    "thalamus":    3,
    "striatum":    4,
    "midbrain":    5,
    "brainstem":   6,
    "other":       7,
}

LAYER_IDS: Dict[str, int] = {
    "unknown": 0,
    "L1":      1,
    "L2_3":    2,
    "L4":      3,
    "L5":      4,
    "L6":      5,
    "PCL":     6,
    "GCL":     7,
    "ML":      8,
}

LAYER_NORMALISE: Dict[str, str] = {
    "PCL": "PCL", "GCL": "GCL", "ML": "ML",
    "PkC_layer": "PCL", "GrC_layer": "GCL", "ML_layer": "ML",
    "L4": "L4", "L5": "L5",
}

DATASET_TECHNOLOGY: Dict[str, str] = {
    "ibl_brainwide":                      "neuropixels",
    "ibl_brainwide_curated":              "neuropixels",
    "ibl_brainwide_good":                 "neuropixels",
    "allen_scope_neuropixel":             "neuropixels",
    "allen_scope_neuropixel_area_subset": "neuropixels",
    "hausser_cell_type":                  "silicon_probe",
    "hull_cell_type":                     "silicon_probe",
    "dandi_000041_cell_type":             "silicon_probe",
    "dandi_000473_cell_type":             "silicon_probe",
    "cellexplorer_cell_type":             "silicon_probe",
    "lisberger_labeled_cell_type":       "juxtacellular",
    "juxtacellular_mouse_s1_area":        "juxtacellular",
}

DATASET_SUPER_REGION: Dict[str, str] = {
    "hausser_cell_type":                  "cerebellum",
    "hull_cell_type":                     "cerebellum",
    "lisberger_labeled_cell_type":       "cerebellum",
    "dandi_000041_cell_type":             "cortex",
    "dandi_000473_cell_type":             "cortex",
    "allen_scope_neuropixel":             "cortex",
    "allen_scope_neuropixel_area_subset": "cortex",
    "cellexplorer_cell_type":             "cortex",
    "juxtacellular_mouse_s1_area":        "cortex",
}



# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def none_safe_collate(batch):
    """Drop None entries and stack the rest."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    data_keys = batch[0][0].keys()
    data   = {k: torch.stack([item[0][k] for item in batch]) for k in data_keys}
    labels = torch.stack([item[1] for item in batch])
    return data, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HippieWF3DACGDataset(Dataset):
    """Paired waveform + spike-times dataset for hippie-wf3dacg.

    Args:
        waveforms:         (N, wave_samples) float array.
        spike_times:       List of N float arrays — spike timestamps in seconds.
        labels:            (N,) int array.  -1 = unknown / unlabelled.
        source_ids:        (N,) int array.
        wave_len:          Target waveform length after resampling (default 90).
        acg_win_ms:        ACG window in ms (default 100 → 101 bins at 1 ms).
        acg_bin_ms:        ACG bin width in ms (default 1).
        acg_n_deciles:     Number of IFR deciles (default 10).
        acg_scale_factor:  Multiply ACG before storing (NEMO convention, default 10).
        precomputed_acgs:  Optional (N, n_deciles, n_bins) to skip online computation.
        super_region_ids:  Optional (N,) int.
        technology_ids:    Optional (N,) int.
        layer_ids:         Optional (N,) int.
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        spike_times: List[np.ndarray],
        labels: np.ndarray,
        source_ids: np.ndarray,
        wave_len: int = 90,
        acg_win_ms: float = 100.0,
        acg_bin_ms: float = 1.0,
        acg_n_deciles: int = 10,
        acg_scale_factor: float = 10.0,
        precomputed_acgs: Optional[np.ndarray] = None,
        super_region_ids: Optional[np.ndarray] = None,
        technology_ids: Optional[np.ndarray] = None,
        layer_ids: Optional[np.ndarray] = None,
    ):
        N = len(waveforms)
        assert len(spike_times) == N
        assert len(labels) == N
        assert len(source_ids) == N

        if isinstance(waveforms, np.ndarray) and waveforms.dtype == object:
            self.waveforms = waveforms
        else:
            self.waveforms = np.asarray(waveforms, dtype=np.float32)

        self.spike_times      = spike_times
        self.labels           = np.asarray(labels, dtype=np.int64)
        self.source_ids       = np.asarray(source_ids, dtype=np.int64)
        self.wave_len         = wave_len
        self.acg_win_ms       = acg_win_ms
        self.acg_bin_ms       = acg_bin_ms
        self.acg_n_deciles    = acg_n_deciles
        self.acg_scale_factor = acg_scale_factor
        self.precomputed_acgs = precomputed_acgs
        self.super_region_ids = (
            np.asarray(super_region_ids, dtype=np.int64) if super_region_ids is not None else None
        )
        self.technology_ids = (
            np.asarray(technology_ids, dtype=np.int64) if technology_ids is not None else None
        )
        self.layer_ids = (
            np.asarray(layer_ids, dtype=np.int64) if layer_ids is not None else None
        )

    # ------------------------------------------------------------------
    # ACG precomputation
    # ------------------------------------------------------------------

    def precompute_acgs(self, n_jobs: int = -1) -> np.ndarray:
        """Compute and cache (N, n_deciles, n_bins) ACG array.

        Uses joblib.Parallel (loky backend) to distribute work across CPUs.
        Set n_jobs=-1 to use all available CPUs (default).
        n_jobs=1 falls back to sequential for debugging.
        """
        from joblib import Parallel, delayed
        N = len(self.spike_times)
        import os
        # Honour CPU quota in container: use min(n_jobs, cpu_count) processes.
        n_cpu = int(os.environ.get("NUM_ACG_WORKERS", os.cpu_count() or 1))
        effective_jobs = n_cpu if n_jobs == -1 else min(abs(n_jobs), n_cpu)
        print(f"  precompute_acgs: {N} neurons, {effective_jobs} parallel workers",
              flush=True)

        results = Parallel(n_jobs=effective_jobs, backend="loky")(
            delayed(compute_3d_acg)(
                st,
                win_ms=self.acg_win_ms,
                bin_ms=self.acg_bin_ms,
                n_deciles=self.acg_n_deciles,
            )
            for st in self.spike_times
        )
        self.precomputed_acgs = np.array(results, dtype=np.float32)
        print(f"  precompute_acgs: {N}/{N} done", flush=True)
        return self.precomputed_acgs

    # ------------------------------------------------------------------
    # Waveform preprocessing (NEMO convention: 90 samples, L∞-norm)
    # ------------------------------------------------------------------

    def _process_waveform(self, raw: np.ndarray) -> torch.Tensor:
        """Resize to wave_len samples, then normalise by max(abs)."""
        wave = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        wave = F.interpolate(wave, size=(self.wave_len,), mode="linear", align_corners=False)
        wave = wave.squeeze(0).squeeze(0)   # (wave_len,)
        mx   = wave.abs().max()
        if mx > 1e-8:
            wave = wave / mx
        return wave

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        try:
            wave = self._process_waveform(self.waveforms[idx])

            if self.precomputed_acgs is not None:
                acg_raw = self.precomputed_acgs[idx]   # (n_deciles, n_bins)
            else:
                acg_raw = compute_3d_acg(
                    self.spike_times[idx],
                    win_ms=self.acg_win_ms,
                    bin_ms=self.acg_bin_ms,
                    n_deciles=self.acg_n_deciles,
                )

            # Scale and add channel dim → (1, n_deciles, n_bins)
            acg = torch.as_tensor(acg_raw * self.acg_scale_factor, dtype=torch.float32)
            acg = acg.unsqueeze(0)

            data: Dict[str, torch.Tensor] = {
                "waveform":  wave,
                "acg":       acg,
                "source_id": torch.tensor(self.source_ids[idx], dtype=torch.long),
                "class_id":  torch.tensor(self.labels[idx],     dtype=torch.long),
            }

            if self.super_region_ids is not None:
                data["super_region_id"] = torch.tensor(
                    self.super_region_ids[idx], dtype=torch.long
                )
            if self.technology_ids is not None:
                data["technology_id"] = torch.tensor(
                    self.technology_ids[idx], dtype=torch.long
                )
            if self.layer_ids is not None:
                data["layer_id"] = torch.tensor(self.layer_ids[idx], dtype=torch.long)

            return data, torch.tensor(self.labels[idx], dtype=torch.long)

        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.waveforms)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: HippieWF3DACGDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=none_safe_collate,
        drop_last=drop_last,
    )
