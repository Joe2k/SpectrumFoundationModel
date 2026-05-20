"""Resample spectrum channels to a fixed length (AION / codec grid)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from desifm.constants import GRID_SIZE


def resample_batch_1d(
    tensor: torch.Tensor,
    length: int = GRID_SIZE,
    *,
    mode: str = "linear",
) -> torch.Tensor:
    """Linearly resample (B, L) -> (B, length)."""
    if tensor.shape[-1] == length:
        return tensor
    x = tensor.unsqueeze(1).float()
    out = F.interpolate(x, size=length, mode=mode, align_corners=False)
    return out.squeeze(1).to(dtype=tensor.dtype)


def resample_spectrum_batch(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor,
    wavelength: torch.Tensor,
    length: int = GRID_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Resample flux/ivar/mask/wavelength to ``length``. Returns (tensors, did_resample)."""
    did = int(flux.shape[-1]) != length
    if not did:
        return flux, ivar, mask, wavelength, False
    flux_r = resample_batch_1d(flux, length)
    ivar_r = resample_batch_1d(ivar, length)
    wave_r = resample_batch_1d(wavelength.float(), length)
    mask_f = mask.float().unsqueeze(1)
    mask_r = F.interpolate(mask_f, size=length, mode="nearest").squeeze(1).bool()
    return flux_r, ivar_r, mask_r, wave_r, True
