"""
hippie-wf3dacg — Conditional VAE with waveform + 3D-ACG features
================================================================

Variant of HIPPIE that ingests the waveform + 3D-ACG featurization from
the C4 database (Beau et al., 2024) instead of HIPPIE's default
waveform + ISI + 1D-ACG. The unimodal encoders are the MLP / CNN
architectures used by NEMO (Yu et al., 2024) for these same features:

  Modality 1 — Waveform  : 90-sample peak-channel waveform encoded by
                            a 2-layer MLP (90→600→300).
  Modality 2 — 3D ACG    : 10-decile × 101-bin autocorrelogram encoded by
                            a ConvolutionalEncoder (CNN).

The CVAE framework (conditioning, reparameterisation, decoder, training loss)
is identical to standard HIPPIE:

  Encoder:
    wave  (B, 90)          → WaveformEncoder MLP → wave_feats (B, 300)
    acg   (B, 1, 10, 101)  → ACGEncoder      CNN → acg_feats  (B, 200)
    [wave_feats | acg_feats | source | super_region | tech | layer]
      → FusionEncoder (MLP) → z_mean, z_log_var  (z_dim each)

  Decoder:
    z + [source | class | super_region | tech | layer]
      → decoder_fc →
          wave_dec  (MLP)  → (B, 90)           waveform
          acg_dec   (MLP)  → (B, 10, 101)       3D ACG (sigmoid, scaled)

Key design decisions (same rationale as standard HIPPIE):
  1. Decoder-only class conditioning — keeps the encoder class-agnostic so its
     behavior is identical between training (label visible to decoder) and
     evaluation (label masked).
  2. Inference-time covariates — source, super_region, tech, layer always known.
  3. Reconstruction consistency loss — warmed up over warmup_epochs.
  4. Light waveform augmentations during training.

Training loss:
  L = MSE(wave_recon, wave)
    + β_acg  × MSE(acg_recon_scaled, acg_scaled)
    + β_kl   × KL(N(μ,σ²) ∥ N(0,1))
    + λ_cons × consistency_loss    [warmed up]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .acg import ACGDecoder, ACGEncoder


# ---------------------------------------------------------------------------
# NEMO waveform encoder (MLP: 90 → 600 → 300)
# ---------------------------------------------------------------------------

class WaveformEncoder(nn.Module):
    """NEMO-style 2-layer MLP waveform encoder.

    Input : (B, 90) — L∞-normalised peak-channel waveform.
    Output: (B, dim_rep) — representation vector. Default dim_rep = 300.
    """

    def __init__(
        self,
        in_features: int = 90,
        hidden_units: Tuple[int, int] = (600, 300),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_rep = hidden_units[-1]
        layers: List[nn.Module] = []
        prev = in_features
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Waveform decoder  (MLP: z_cond → 300 → 90)
# ---------------------------------------------------------------------------

class WaveformDecoder(nn.Module):
    """MLP decoder for 90-sample waveform reconstruction."""

    def __init__(self, in_dim: int, hidden_dim: int = 300, out_features: int = 90):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Fusion encoder
# ---------------------------------------------------------------------------

class FusionEncoder(nn.Module):
    """Fuses per-modality features + conditioning embeddings → (h, μ, log_var)."""

    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int, use_batch_norm: bool):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.LeakyReLU(0.2), nn.Linear(hidden_dim, z_dim * 2)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(z_dim * 2))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        self.z_mean    = nn.Linear(z_dim * 2, z_dim)
        self.z_log_var = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return h, self.z_mean(h), self.z_log_var(h)


def _build_decoder_fc(in_dim: int, out_dim: int, use_batch_norm: bool) -> nn.Sequential:
    """Small FC block used to project z+cond into decoder input space."""
    layers: List[nn.Module] = [
        nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2),
        nn.Linear(out_dim, out_dim),
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HippieWF3DACGConfig:
    """All hyperparameters for one hippie-wf3dacg run."""

    # ── Waveform ──────────────────────────────────────────────────────────────
    wave_len: int = 90                         # samples (NEMO's standard)
    wvf_hidden: Tuple[int, int] = (600, 300)   # NEMO's MLP hidden units
    wvf_dropout: float = 0.1

    # ── 3D ACG ───────────────────────────────────────────────────────────────
    acg_n_deciles: int = 10
    acg_n_bins: int = 101                      # 100 ms window at 1 ms resolution
    acg_encoder_dim: int = 200                 # NEMO ConvolutionalEncoder dim_rep
    acg_dropout: float = 0.2
    acg_scale_factor: float = 10.0             # multiply ACG before loss (NEMO convention)

    # ── ACG computation parameters (used by dataloading.py) ──────────────────
    acg_win_ms: float = 100.0
    acg_bin_ms: float = 1.0

    # ── Shared latent space ───────────────────────────────────────────────────
    z_dim: int = 16

    # ── Conditioning ─────────────────────────────────────────────────────────
    cond_dim: int = 5
    encoder_uses_class_embedding: bool = False
    use_super_region_embedding: bool = True
    use_technology_embedding: bool = True
    use_layer_embedding: bool = True
    class_emb_dropout: float = 0.3

    # ── Fusion encoder ────────────────────────────────────────────────────────
    fusion_hidden_dim: int = 256
    use_batch_norm: bool = True

    # ── Decoder ───────────────────────────────────────────────────────────────
    wave_dec_hidden: int = 300
    acg_dec_hidden: int = 512

    # ── Waveform augmentation ─────────────────────────────────────────────────
    use_augmentations: bool = True
    augment_prob: float = 0.3
    noise_std: float = 0.03
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)

    # ── ACG augmentation ─────────────────────────────────────────────────────
    acg_augment_prob: float = 0.3
    acg_noise_std: float = 0.05          # fraction of max ACG value
    acg_amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)

    # ── Loss ──────────────────────────────────────────────────────────────────
    beta_kl: float = 0.9
    # ACG reconstruction weight relative to wave.  Both losses are computed in
    # *normalised* space (ACG divided by acg_scale_factor before MSE) so they
    # are magnitude-comparable.  beta_acg=1.0 means equal weight.
    beta_acg: float = 1.0
    reconstruction_consistency_weight: float = 0.15

    # ── Cross-modal contrastive alignment (experimental, off by default) ──────
    # Adds a NEMO-style bimodal contrastive term between wave_feats and acg_feats
    # projections.  Disabled by default (0.0) — philosophically conflicts with the
    # CVAE objective, which already aligns modalities through the shared latent z.
    cross_modal_contrastive_weight: float = 0.0
    cross_modal_proj_dim: int = 64
    cross_modal_temperature: float = 0.5

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # ── Warmup ───────────────────────────────────────────────────────────────
    warmup_epochs: int = 5


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class HippieWF3DACGCVAE(nn.Module):
    """Conditional VAE with NEMO waveform + 3D ACG modalities.

    Args:
        config:            HippieWF3DACGConfig with all hyperparameters.
        num_sources:       Number of distinct recording datasets.
        num_classes:       Number of distinct labels (None → class conditioning off).
        num_super_regions: Number of coarse brain regions (None → disabled).
        num_technologies:  Number of recording technologies (None → disabled).
        num_layers:        Number of layer categories (None → disabled).
    """

    def __init__(
        self,
        config: HippieWF3DACGConfig,
        num_sources: int,
        num_classes: Optional[int] = None,
        num_super_regions: Optional[int] = None,
        num_technologies: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.z_dim  = config.z_dim

        # ── Encoders ──────────────────────────────────────────────────────────
        self.wave_enc = WaveformEncoder(
            in_features=config.wave_len,
            hidden_units=config.wvf_hidden,
            dropout=config.wvf_dropout,
        )
        self.acg_enc = ACGEncoder(
            dim_rep=config.acg_encoder_dim,
            dropout=config.acg_dropout,
        )

        # ── Conditioning embeddings ───────────────────────────────────────────
        self.source_embedding = nn.Embedding(num_sources, config.cond_dim)

        self.class_embedding: Optional[nn.Embedding] = (
            nn.Embedding(num_classes, config.cond_dim) if num_classes is not None else None
        )
        self.class_emb_dropout = nn.Dropout(config.class_emb_dropout)

        self.super_region_embedding: Optional[nn.Embedding] = (
            nn.Embedding(num_super_regions, config.cond_dim)
            if (config.use_super_region_embedding and num_super_regions is not None)
            else None
        )
        self.technology_embedding: Optional[nn.Embedding] = (
            nn.Embedding(num_technologies, config.cond_dim)
            if (config.use_technology_embedding and num_technologies is not None)
            else None
        )
        self.layer_embedding: Optional[nn.Embedding] = (
            nn.Embedding(num_layers, config.cond_dim)
            if (config.use_layer_embedding and num_layers is not None)
            else None
        )

        # ── Dimension accounting ──────────────────────────────────────────────
        always_dim = config.cond_dim  # source always present
        for emb in (self.super_region_embedding, self.technology_embedding, self.layer_embedding):
            if emb is not None:
                always_dim += config.cond_dim
        class_dim = config.cond_dim if self.class_embedding is not None else 0

        enc_emb_dim = always_dim + (class_dim if config.encoder_uses_class_embedding else 0)
        self._dec_emb_dim = always_dim + class_dim

        wave_rep = self.wave_enc.dim_rep   # 300
        acg_rep  = self.acg_enc.dim_rep    # 200

        # ── Fusion encoder ────────────────────────────────────────────────────
        fusion_in = wave_rep + acg_rep + enc_emb_dim
        self.fusion_enc = FusionEncoder(
            in_dim=fusion_in, hidden_dim=config.fusion_hidden_dim,
            z_dim=config.z_dim, use_batch_norm=config.use_batch_norm,
        )

        # ── Cross-modal contrastive projectors ───────────────────────────────
        if config.cross_modal_contrastive_weight > 0.0:
            self.wave_proj: Optional[nn.Linear] = nn.Linear(wave_rep, config.cross_modal_proj_dim)
            self.acg_proj:  Optional[nn.Linear] = nn.Linear(acg_rep,  config.cross_modal_proj_dim)
        else:
            self.wave_proj = None
            self.acg_proj  = None

        # ── Decoders ─────────────────────────────────────────────────────────
        dec_in = config.z_dim + self._dec_emb_dim

        self.wave_dec_fc = _build_decoder_fc(dec_in, wave_rep, config.use_batch_norm)
        self.wave_dec    = WaveformDecoder(
            in_dim=wave_rep, hidden_dim=config.wave_dec_hidden, out_features=config.wave_len
        )

        self.acg_dec_fc  = _build_decoder_fc(dec_in, acg_rep, config.use_batch_norm)
        self.acg_dec     = ACGDecoder(
            in_dim=acg_rep, hidden_dim=config.acg_dec_hidden,
            n_deciles=config.acg_n_deciles, n_bins=config.acg_n_bins,
        )

    # ── Conditioning helpers ──────────────────────────────────────────────────

    def _get_embeddings(
        self,
        source_ids: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        super_region_ids: Optional[torch.Tensor] = None,
        technology_ids: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        include_class: bool = True,
        apply_class_dropout: bool = False,
    ) -> torch.Tensor:
        B, dev = source_ids.shape[0], source_ids.device
        parts = [self.source_embedding(source_ids)]

        if self.class_embedding is not None and include_class:
            if class_ids is not None:
                cls_emb = self.class_embedding(class_ids)
                if apply_class_dropout and self.training:
                    cls_emb = self.class_emb_dropout(cls_emb)
            else:
                cls_emb = torch.zeros(B, self.config.cond_dim, device=dev)
            parts.append(cls_emb)

        if self.super_region_embedding is not None:
            parts.append(
                self.super_region_embedding(super_region_ids)
                if super_region_ids is not None
                else torch.zeros(B, self.config.cond_dim, device=dev)
            )
        if self.technology_embedding is not None:
            parts.append(
                self.technology_embedding(technology_ids)
                if technology_ids is not None
                else torch.zeros(B, self.config.cond_dim, device=dev)
            )
        if self.layer_embedding is not None:
            parts.append(
                self.layer_embedding(layer_ids)
                if layer_ids is not None
                else torch.zeros(B, self.config.cond_dim, device=dev)
            )

        return torch.cat(parts, dim=-1)

    def get_modality_features(
        self, waveform: torch.Tensor, acg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (wave_feats, acg_feats) before fusion — used for contrastive loss."""
        return self.wave_enc(waveform), self.acg_enc(acg)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    # ── Encode / decode / forward ─────────────────────────────────────────────

    def encode(
        self,
        waveform: torch.Tensor,
        acg: torch.Tensor,
        source_ids: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        super_region_ids: Optional[torch.Tensor] = None,
        technology_ids: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        apply_class_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h, mu, log_var)."""
        wave_feats = self.wave_enc(waveform)      # (B, 300)
        acg_feats  = self.acg_enc(acg)            # (B, 200)

        enc_emb = self._get_embeddings(
            source_ids,
            class_ids=class_ids if self.config.encoder_uses_class_embedding else None,
            super_region_ids=super_region_ids,
            technology_ids=technology_ids,
            layer_ids=layer_ids,
            include_class=self.config.encoder_uses_class_embedding,
            apply_class_dropout=apply_class_dropout if self.config.encoder_uses_class_embedding else False,
        )

        combined = torch.cat([wave_feats, acg_feats, enc_emb], dim=-1)
        return self.fusion_enc(combined)

    def decode(
        self,
        z: torch.Tensor,
        source_ids: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        super_region_ids: Optional[torch.Tensor] = None,
        technology_ids: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        apply_class_dropout: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Returns {'waveform': (B,90), 'acg': (B,10,101)}."""
        dec_emb = self._get_embeddings(
            source_ids, class_ids=class_ids,
            super_region_ids=super_region_ids, technology_ids=technology_ids,
            layer_ids=layer_ids, include_class=True,
            apply_class_dropout=apply_class_dropout,
        )
        z_cond = torch.cat([z, dec_emb], dim=-1)

        wave_recon = self.wave_dec(self.wave_dec_fc(z_cond))
        acg_recon  = self.acg_dec(self.acg_dec_fc(z_cond))
        return {"waveform": wave_recon, "acg": acg_recon}

    def forward(
        self,
        waveform: torch.Tensor,
        acg: torch.Tensor,
        source_ids: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
        super_region_ids: Optional[torch.Tensor] = None,
        technology_ids: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
        apply_class_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        _, mu, log_var = self.encode(
            waveform, acg, source_ids, class_ids,
            super_region_ids, technology_ids, layer_ids, apply_class_dropout,
        )
        z     = self.reparameterize(mu, log_var)
        recon = self.decode(
            z, source_ids, class_ids, super_region_ids, technology_ids, layer_ids,
            apply_class_dropout,
        )
        return mu, log_var, recon


# ---------------------------------------------------------------------------
# Cross-modal contrastive loss (NEMO-style bimodal SimCLR)
# ---------------------------------------------------------------------------

def _bimodal_contrastive(
    X: torch.Tensor, Y: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Symmetric cross-modal contrastive loss.

    X, Y: (B, dim) L2-normalised projections of the two modalities.
    Diagonal entries are positives (same neuron); off-diagonal are negatives.
    """
    sim = torch.matmul(X, Y.T) / temperature          # (B, B)
    labels = torch.arange(len(X), device=X.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class HippieWF3DACGLightning(pl.LightningModule):
    """PyTorch Lightning training wrapper for hippie-wf3dacg.

    Batch keys (from HippieWF3DACGDataset / none_safe_collate):
        'waveform'  — (B, 90)           L∞-normalised
        'acg'       — (B, 1, 10, 101)   scaled by acg_scale_factor
        'source_id', 'class_id',
        optionally 'super_region_id', 'technology_id', 'layer_id'
    """

    def __init__(
        self,
        config: HippieWF3DACGConfig,
        num_sources: int,
        num_classes: Optional[int] = None,
        num_super_regions: Optional[int] = None,
        num_technologies: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model  = HippieWF3DACGCVAE(
            config, num_sources, num_classes,
            num_super_regions, num_technologies, num_layers,
        )

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _augment_waveform(self, wave: torch.Tensor) -> torch.Tensor:
        """Stochastic augmentations on (B, 90) waveforms during training."""
        if not self.training or not self.config.use_augmentations:
            return wave
        B = wave.shape[0]
        mask = torch.rand(B, device=wave.device) < self.config.augment_prob

        # Gaussian noise
        wave = torch.where(
            mask.unsqueeze(1),
            wave + self.config.noise_std * torch.randn_like(wave),
            wave,
        )
        # Amplitude scaling
        lo, hi = self.config.amplitude_scale_range
        scale  = lo + (hi - lo) * torch.rand(B, 1, device=wave.device)
        wave   = torch.where(mask.unsqueeze(1), wave * scale, wave)

        # Re-normalise so max(abs) ≤ 1 (NEMO convention)
        max_abs = wave.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        wave    = wave / max_abs
        return wave

    def _augment_acg(self, acg: torch.Tensor) -> torch.Tensor:
        """Stochastic augmentations on (B, 1, 10, 101) ACGs during training."""
        if not self.training or not self.config.use_augmentations:
            return acg
        B = acg.shape[0]
        mask = torch.rand(B, device=acg.device) < self.config.acg_augment_prob

        # Gaussian noise (proportional to local max)
        acg_max = acg.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        noise   = self.config.acg_noise_std * acg_max * torch.randn_like(acg)
        acg     = torch.where(mask.view(B, 1, 1, 1), (acg + noise).clamp(min=0.0), acg)

        # Amplitude scaling
        lo, hi = self.config.acg_amplitude_scale_range
        scale  = lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=acg.device)
        acg    = torch.where(mask.view(B, 1, 1, 1), acg * scale, acg)
        return acg

    # ── Loss helpers ─────────────────────────────────────────────────────────

    def _kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

    def _warmup_factor(self) -> float:
        if self.config.warmup_epochs == 0:
            return 1.0
        return min(1.0, self.current_epoch / self.config.warmup_epochs)

    # ── Batch unpacking ───────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        data, _ = batch
        waveform = data["waveform"]                         # (B, 90)
        acg      = data["acg"]                              # (B, 1, 10, 101)
        source_ids        = data["source_id"]
        super_region_ids  = data.get("super_region_id")
        technology_ids    = data.get("technology_id")
        layer_ids         = data.get("layer_id")

        class_ids: Optional[torch.Tensor] = None
        if "class_id" in data and (data["class_id"] >= 0).any():
            class_ids = data["class_id"]

        return waveform, acg, source_ids, class_ids, super_region_ids, technology_ids, layer_ids

    # ── Shared train/val step ─────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        (waveform, acg, source_ids,
         class_ids, super_region_ids, technology_ids, layer_ids) = self._unpack_batch(batch)

        if stage == "train":
            waveform = self._augment_waveform(waveform)
            acg      = self._augment_acg(acg)

        mu, log_var, recon = self.model(
            waveform, acg, source_ids,
            class_ids=class_ids,
            super_region_ids=super_region_ids,
            technology_ids=technology_ids,
            layer_ids=layer_ids,
            apply_class_dropout=(stage == "train"),
        )

        wave_loss = F.mse_loss(recon["waveform"], waveform)
        # Compute ACG MSE in *unscaled* space so it is magnitude-comparable to
        # wave MSE (both ~0–1).  The model still processes scaled values internally
        # (numerical stability); we only un-scale when computing the loss.
        s = self.config.acg_scale_factor
        acg_loss  = F.mse_loss(recon["acg"] / s, acg.squeeze(1) / s)
        kl        = self._kl_loss(mu, log_var)
        warmup    = self._warmup_factor()

        loss = (
            wave_loss
            + self.config.beta_acg * acg_loss
            + warmup * self.config.beta_kl * kl
        )

        # Consistency loss (training only)
        consistency_loss = torch.tensor(0.0, device=waveform.device)
        if (
            stage == "train"
            and self.config.reconstruction_consistency_weight > 0.0
            and self.model.class_embedding is not None
        ):
            z = self.model.reparameterize(mu, log_var)
            recon_nc = self.model.decode(
                z, source_ids, class_ids=None,
                super_region_ids=super_region_ids,
                technology_ids=technology_ids,
                layer_ids=layer_ids,
                apply_class_dropout=False,
            )
            consistency_loss = (
                F.mse_loss(recon["waveform"], recon_nc["waveform"])
                + F.mse_loss(recon["acg"] / s, recon_nc["acg"] / s)
            )
            loss = loss + warmup * self.config.reconstruction_consistency_weight * consistency_loss

        # Cross-modal contrastive loss: align wave_feats ↔ acg_feats projections.
        # This restores NEMO's core bimodal alignment signal, which is lost when
        # the encoders are trained with only MSE reconstruction.
        contrastive_loss = torch.tensor(0.0, device=waveform.device)
        if (
            stage == "train"
            and self.config.cross_modal_contrastive_weight > 0.0
            and self.model.wave_proj is not None
        ):
            wave_feats, acg_feats = self.model.get_modality_features(waveform, acg)
            wave_emb = F.normalize(self.model.wave_proj(wave_feats), dim=-1)
            acg_emb  = F.normalize(self.model.acg_proj(acg_feats),  dim=-1)
            contrastive_loss = _bimodal_contrastive(
                wave_emb, acg_emb, temperature=self.config.cross_modal_temperature
            )
            loss = loss + self.config.cross_modal_contrastive_weight * contrastive_loss

        self.log(f"{stage}_loss",       loss,            prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_wave_loss",  wave_loss,        on_epoch=True, on_step=False)
        self.log(f"{stage}_acg_loss",   acg_loss,         on_epoch=True, on_step=False)
        self.log(f"{stage}_kl",         kl,               on_epoch=True, on_step=False)
        if stage == "train":
            self.log("train_beta_kl",          warmup * self.config.beta_kl, on_epoch=True, on_step=False)
            self.log("train_consistency_loss",  consistency_loss,              on_epoch=True, on_step=False)
            self.log("train_contrastive_loss",  contrastive_loss,              on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=100, eta_min=self.config.lr * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    # ── Embedding extraction (no label leakage) ───────────────────────────────

    @torch.no_grad()
    def get_embeddings(
        self,
        waveform: torch.Tensor,
        acg: torch.Tensor,
        source_ids: torch.Tensor,
        super_region_ids: Optional[torch.Tensor] = None,
        technology_ids: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return μ for downstream KNN / linear probing.  class_ids always masked."""
        self.model.eval()
        _, mu, _ = self.model.encode(
            waveform, acg, source_ids,
            class_ids=None,
            super_region_ids=super_region_ids,
            technology_ids=technology_ids,
            layer_ids=layer_ids,
        )
        return mu
