"""
hippie_wf3dacg — ACG encoder and 3D ACG computation
=================================================

Provides:
  compute_3d_acg()  — pure-numpy 3D autocorrelogram from raw spike times
                      (NEMO-compatible format: n_deciles × n_bins)
  ACGEncoder        — NEMO's ConvolutionalEncoder (2D CNN on 10×101 ACG)
  ACGDecoder        — MLP decoder for ACG reconstruction

3D ACG format (matches NEMO):
  Rows   = 10 instantaneous-firing-rate deciles (lowest → highest IFR)
  Cols   = 101 time bins at 1 ms resolution (positive lag: 0 → 100 ms)
  Shape  = (10, 101)

  At training/load time this is multiplied by acg_scale_factor (default 10)
  to bring values into a numerically stable range — same convention as NEMO.

The ACGEncoder expects tensors of shape (B, 1, 10, 101).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D ACG computation (pure numpy, no npyx required)
# ---------------------------------------------------------------------------

def compute_3d_acg(
    spike_times_s: np.ndarray,
    win_ms: float = 100.0,
    bin_ms: float = 1.0,
    n_deciles: int = 10,
    min_spikes: int = 5,
) -> np.ndarray:
    """Compute a NEMO-style 3D autocorrelogram from raw spike times.

    Args:
        spike_times_s: Spike timestamps in **seconds**.  Any order is fine.
        win_ms:        One-sided window width in milliseconds (default 100 ms).
                       n_bins = int(win_ms / bin_ms) + 1.
        bin_ms:        Bin width in milliseconds (default 1 ms).
        n_deciles:     Number of instantaneous-firing-rate deciles (default 10).
        min_spikes:    Return zero-filled array when fewer spikes present.

    Returns:
        np.ndarray of shape (n_deciles, n_bins), dtype float32.
        Row d contains the (positive-lag) autocorrelogram of the subset of
        spikes whose instantaneous firing rate falls in the d-th decile
        (d=0 = lowest, d=n_deciles-1 = highest).
        Values are in units of spikes/spike (mean count per central spike per bin).
    """
    n_bins = int(round(win_ms / bin_ms)) + 1
    result = np.zeros((n_deciles, n_bins), dtype=np.float32)

    if len(spike_times_s) < min_spikes:
        return result

    st = np.sort(np.asarray(spike_times_s, dtype=np.float64))
    n = len(st)
    win_s = win_ms * 1e-3
    bin_s = bin_ms * 1e-3

    # ── Instantaneous firing rate (IFR) per spike ────────────────────────────
    # IFR[i] = harmonic mean of the previous and next ISI for interior spikes;
    # boundary spikes use only the one available ISI.
    isis = np.diff(st)  # (n-1,)
    ifr = np.empty(n, dtype=np.float64)
    if n > 2:
        ifr[1:-1] = 2.0 / (isis[:-1] + isis[1:] + 1e-12)
    if n > 1:
        ifr[0]  = 1.0 / (isis[0]  + 1e-12)
        ifr[-1] = 1.0 / (isis[-1] + 1e-12)
    else:
        ifr[:] = 0.0

    # ── Decile assignment ─────────────────────────────────────────────────────
    bounds = np.percentile(ifr, np.linspace(0.0, 100.0, n_deciles + 1))
    # searchsorted on interior bounds → 0 … n_deciles-1
    decile_ids = np.searchsorted(bounds[1:-1], ifr, side="right")
    # clip in case of numerical tie at the top boundary
    np.clip(decile_ids, 0, n_deciles - 1, out=decile_ids)

    # ── Per-decile ACG ────────────────────────────────────────────────────────
    for d in range(n_deciles):
        central_idx = np.where(decile_ids == d)[0]
        n_central = len(central_idx)
        if n_central == 0:
            continue

        counts = np.zeros(n_bins, dtype=np.float64)
        central_times = st[central_idx]

        for t_c in central_times:
            # spikes strictly after t_c up to t_c + win_s
            lo = np.searchsorted(st, t_c + 1e-12, side="left")
            hi = np.searchsorted(st, t_c + win_s + 1e-12, side="left")
            if lo >= hi:
                continue
            lags_bins = ((st[lo:hi] - t_c) / bin_s).astype(np.intp)
            # np.bincount is faster than add.at for dense small arrays
            bc = np.bincount(lags_bins, minlength=n_bins)
            counts += bc[:n_bins]

        result[d] = (counts / n_central).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# ACG encoder (NEMO's ConvolutionalEncoder, self-contained)
# ---------------------------------------------------------------------------

class ACGEncoder(nn.Module):
    """2D convolutional encoder for 10×101 3D ACGs.

    Architecture mirrors NEMO's ``ConvolutionalEncoder`` exactly:

        Input   : (B, 1, 10, 101)
        Conv2d(1→8,  (1,10)) + AvgPool2d(2,2) + BN + GELU
        Conv2d(8→16, (5,1))  + AvgPool2d(1,2) + BN + GELU
        Flatten → Linear(16×1×23 → dim_rep) + Dropout + GELU

    Output: (B, dim_rep).  ``dim_rep`` defaults to 200 (NEMO default).
    """

    def __init__(self, dim_rep: int = 200, dropout: float = 0.2):
        super().__init__()
        self.dim_rep = dim_rep

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 10))
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.bn2   = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()
        # After conv+pool: (B, 16, 1, 23) → 16×1×23 = 368
        self.fc1    = nn.Linear(16 * 1 * 23, dim_rep)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 10, 101) → (B, dim_rep)."""
        x = F.gelu(self.pool1(self.conv1(x)))
        x = self.bn1(x)
        x = F.gelu(self.pool2(self.conv2(x)))
        x = self.bn2(x)
        x = self.flatten(x)
        return self.drop(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# ACG decoder
# ---------------------------------------------------------------------------

class ACGDecoder(nn.Module):
    """MLP decoder: latent code → reconstructed 3D ACG (B, n_deciles × n_bins).

    Output is ReLU-activated (non-negative) to match the ACG target convention.
    The training target is the raw ACG multiplied by acg_scale_factor (values
    can exceed 1.0, so Sigmoid would be incorrect here).
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_deciles: int = 10, n_bins: int = 101):
        super().__init__()
        out_dim = n_deciles * n_bins
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )
        self.n_deciles = n_deciles
        self.n_bins    = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_deciles, n_bins)."""
        return self.net(x).view(x.shape[0], self.n_deciles, self.n_bins)
