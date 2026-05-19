"""AION-style spectrum preprocessing for the codec (Tier A)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Defaults aligned with AION SpectrumCodec
CLIP_IVAR = 100.0
CLIP_FLUX: float | None = None
INPUT_SCALING = 0.2
DENORM_MIN = 0.1
NORM_MIN = 0.1


def aion_normalize(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    clip_ivar: float = CLIP_IVAR,
    clip_flux: float | None = CLIP_FLUX,
    input_scaling: float = INPUT_SCALING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware norm, affine scaling, arcsinh compression.

    flux, ivar: (B, L)
    mask: (B, L) True = bad pixel (DESI mask convention)
    Returns:
        x: (B, 2, L) arcsinh-normalized [flux, istd]
        denorm: (B,) per-spectrum scale for inverse transform
    """
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = torch.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_flux is not None:
        flux = flux.clamp(-clip_flux, clip_flux)
    ivar = ivar.clamp(0.0, clip_ivar)
    istd = torch.sqrt(ivar.clamp(min=1e-10))

    if mask is None:
        good = torch.ones_like(flux, dtype=torch.bool)
    else:
        good = ~mask

    denom = good.sum(dim=-1).float().clamp(min=1.0)
    norm = (flux * good.float()).sum(dim=-1) / denom
    norm = norm.clamp(min=NORM_MIN)
    norm_log = torch.log10(norm + 1.0)
    denorm = (10**norm_log - 1.0).clamp(min=DENORM_MIN)

    flux_n = (flux / denorm.unsqueeze(-1) - 1.0) * input_scaling
    istd_n = (istd / denorm.unsqueeze(-1)) * input_scaling
    x = torch.arcsinh(torch.stack([flux_n, istd_n], dim=1))
    return x, denorm


def aion_denormalize(
    x_arcsinh: torch.Tensor,
    denorm: torch.Tensor,
    *,
    input_scaling: float = INPUT_SCALING,
) -> torch.Tensor:
    """Inverse of aion_normalize. x_arcsinh: (B, 2, L) -> physical (B, 2, L)."""
    x = torch.sinh(x_arcsinh)
    flux = (x[:, 0] / input_scaling + 1.0) * denorm.unsqueeze(-1)
    istd = (x[:, 1] / input_scaling) * denorm.unsqueeze(-1)
    return torch.stack([flux, istd], dim=1)


def masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Reconstruction loss on flux channel (index 0) in arcsinh space."""
    diff = pred - target
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=huber_delta, reduction="none")
    if mask is None:
        return loss.mean()
    w = (~mask).float()
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def prepare_codec_batch(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build codec input from a collated batch dict."""
    flux, ivar = batch["flux"], batch["ivar"]
    mask = batch.get("mask")
    x, denorm = aion_normalize(flux, ivar, mask)
    return x, denorm, mask


# Backward-compatible alias
prepare_codec_input = aion_normalize
