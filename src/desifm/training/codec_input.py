"""Input normalization for spectrum codec (stable scale across DESI spectra)."""

from __future__ import annotations

import torch


def prepare_codec_input(flux: torch.Tensor, ivar: torch.Tensor) -> torch.Tensor:
    """Median-normalize each spectrum so codec loss is comparable across targets.

    flux, ivar: (B, L) -> x: (B, 2, L) with channels [flux_norm, sqrt(ivar_norm)].
    """
    scale = flux.abs().median(dim=-1, keepdim=True).values.clamp(min=1e-12)
    flux_n = flux / scale
    ivar_n = ivar / (scale * scale)
    istd_n = torch.sqrt(ivar_n.clamp(min=1e-10))
    return torch.stack([flux_n, istd_n], dim=1)
