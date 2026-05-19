"""Codec v4 losses and validation metrics (physical flux + codebook usage)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from desifm.training.codec_input import denormalize_spectrum_output, masked_recon_loss


def top_hat_smooth_flux(flux: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """5-pixel boxcar along wavelength (B, L). No-op if kernel_size < 2."""
    if kernel_size < 2:
        return flux
    pad = kernel_size // 2
    k = torch.ones(1, 1, kernel_size, device=flux.device, dtype=flux.dtype) / kernel_size
    x = flux.unsqueeze(1)
    return F.conv1d(x, k, padding=pad).squeeze(1)


def align_mask_to_length(mask: torch.Tensor | None, length: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape[-1] == length:
        return mask
    return (
        F.interpolate(mask.unsqueeze(1).float(), size=length, mode="nearest")
        .squeeze(1)
        .bool()
    )


def physical_flux_loss(
    recon_arcsinh: torch.Tensor,
    target_arcsinh: torch.Tensor,
    denorm: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Huber on flux channel after denormalizing recon and target from arcsinh space."""
    recon_phys = denormalize_spectrum_output(recon_arcsinh, denorm)[:, 0]
    target_phys = denormalize_spectrum_output(target_arcsinh, denorm)[:, 0]
    return masked_recon_loss(recon_phys, target_phys, mask, huber_delta=huber_delta)


def latent_index_entropy_penalty(indices: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """Penalty when latent indices are low-entropy (codebook collapse). Returns >= 0."""
    flat = indices.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((), device=indices.device)
    hist = torch.bincount(flat, minlength=n_bins).float()
    total = hist.sum()
    if total <= 0:
        return torch.ones((), device=indices.device)
    p = hist / total
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    max_ent = math.log(max(n_bins, 2))
    return (1.0 - entropy / max_ent).clamp(min=0.0, max=1.0)


def flux_rms(
    pred_flux: torch.Tensor,
    target_flux: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMS error on good pixels; pred/target (B, L)."""
    if mask is None:
        good = torch.ones_like(pred_flux, dtype=torch.bool)
    else:
        good = ~mask
    diff = (pred_flux - target_flux).float()
    w = good.float()
    denom = w.sum().clamp(min=1.0)
    return torch.sqrt(((diff * w) ** 2).sum() / denom)


def flux_std_ratio(
    pred_flux: torch.Tensor,
    target_flux: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """std(pred) / std(target) on good pixels — collapse detector."""
    if mask is None:
        good = torch.ones_like(pred_flux, dtype=torch.bool)
    else:
        good = ~mask

    def _std(t: torch.Tensor) -> torch.Tensor:
        g = t[good]
        if g.numel() < 2:
            return torch.zeros((), device=t.device)
        return g.float().std(unbiased=False)

    pred_s = _std(pred_flux)
    tgt_s = _std(target_flux)
    return pred_s / tgt_s.clamp(min=1e-8)
