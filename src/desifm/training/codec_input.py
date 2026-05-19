"""Spectrum preprocessing and reconstruction loss for the codec."""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLIP_IVAR = 100.0
CLIP_FLUX: float | None = None
INPUT_SCALING = 0.2
DENORM_MIN = 0.1
NORM_MIN = 0.1

INPUT_STYLE_V3 = "mask_arcsinh_v3"
INPUT_STYLE_V4 = "mask_arcsinh_v4"
INPUT_STYLE_V5 = "mask_arcsinh_v5"


def normalize_spectrum_input(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    clip_ivar: float = CLIP_IVAR,
    clip_flux: float | None = CLIP_FLUX,
    input_scaling: float = INPUT_SCALING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare flux/ivar for the codec: clean, scale, compress.

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


def normalize_spectrum_median_v2(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    input_scaling: float = INPUT_SCALING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy codec_v2-style normalization (approximate).

    Per-spectrum **median** of flux on usable pixels as ``denorm`` (no log10 compression,
    no ivar clip). Same arcsinh affine and shared :func:`denormalize_spectrum_output` as v3.

    ``codec_v2`` checkpoints on W&B predate ``input_style`` metadata; this matches the
    documented "median-normalized flux" pipeline in ``RESEARCH_LOG.md`` for evaluation
    notebooks. If reconstructions look systematically off, revisit this helper against
    archived training code.
    """
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = torch.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    istd = torch.sqrt(ivar.clamp(min=1e-10))
    if mask is None:
        good = torch.ones_like(flux, dtype=torch.bool)
    else:
        good = ~mask
    denorm_list = []
    for b in range(flux.shape[0]):
        vals = flux[b, good[b]]
        if vals.numel() == 0:
            vals = flux[b]
        pos = vals[vals > 0]
        if pos.numel() > 0:
            med = pos.median()
        else:
            med = vals.median()
        if (not torch.isfinite(med)) or med <= 0:
            med = torch.tensor(1e-30, device=flux.device, dtype=flux.dtype)
        # Do not use DENORM_MIN=0.1 here — DESI coadd flux is ~1e-17; that clamp blows up physical decode.
        denorm_list.append(med.clamp(min=1e-30))
    denorm = torch.stack(denorm_list)
    flux_n = (flux / denorm.unsqueeze(-1) - 1.0) * input_scaling
    istd_n = (istd / denorm.unsqueeze(-1)) * input_scaling
    x = torch.arcsinh(torch.stack([flux_n, istd_n], dim=1))
    return x, denorm


def denormalize_spectrum_output(
    x_arcsinh: torch.Tensor,
    denorm: torch.Tensor,
    *,
    input_scaling: float = INPUT_SCALING,
) -> torch.Tensor:
    """Map codec output from arcsinh space back to physical flux and istd."""
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
    """Huber reconstruction loss on flux channel (index 0) in arcsinh space."""
    diff = pred - target
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=huber_delta, reduction="none")
    if mask is None:
        return loss.mean()
    w = (~mask).float()
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def prepare_codec_batch(
    batch: dict,
    *,
    top_hat_kernel: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build codec tensors from a collated batch dict."""
    flux, ivar = batch["flux"], batch["ivar"]
    mask = batch.get("mask")
    if top_hat_kernel >= 2:
        from desifm.training.codec_loss import top_hat_smooth_flux

        flux = top_hat_smooth_flux(flux, top_hat_kernel)
    x, denorm = normalize_spectrum_input(flux, ivar, mask)
    return x, denorm, mask


def prepare_codec_batch_v4(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """codec_v4: mask-aware arcsinh + 5-pixel top-hat flux smoothing."""
    return prepare_codec_batch(batch, top_hat_kernel=5)


def normalize_spectrum_positive_mean_log(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    clip_ivar: float = CLIP_IVAR,
    input_scaling: float = INPUT_SCALING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FoundationModel-style norm (mean of flux>0, log10 denorm). Try for legacy codec_v2 if median looks wrong."""
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = torch.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = ivar.clamp(0.0, clip_ivar)
    istd = torch.sqrt(ivar.clamp(min=1e-10))
    good = torch.ones_like(flux, dtype=torch.bool) if mask is None else ~mask
    pos = flux > 0
    use = good & pos
    denom = use.sum(dim=-1).float().clamp(min=1.0)
    norm = (flux * use.float()).sum(dim=-1) / denom
    norm = norm.clamp(min=1e-30)
    norm_log = torch.log10(norm + 1.0)
    denorm = (10**norm_log - 1.0).clamp(min=1e-30)
    flux_n = (flux / denorm.unsqueeze(-1) - 1.0) * input_scaling
    istd_n = (istd / denorm.unsqueeze(-1)) * input_scaling
    x = torch.arcsinh(torch.stack([flux_n, istd_n], dim=1))
    return x, denorm


def prepare_codec_batch_for_style(
    batch: dict,
    input_style: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Like :func:`prepare_codec_batch` but selects v2 vs v3 normalization."""
    flux, ivar = batch["flux"], batch["ivar"]
    mask = batch.get("mask")
    if input_style in (INPUT_STYLE_V3, INPUT_STYLE_V4, INPUT_STYLE_V5):
        if input_style in (INPUT_STYLE_V4, INPUT_STYLE_V5):
            x, denorm, mask = prepare_codec_batch_v4(batch)
            return x, denorm, mask
        x, denorm = normalize_spectrum_input(flux, ivar, mask)
    elif input_style == "median_v2":
        x, denorm = normalize_spectrum_median_v2(flux, ivar, mask)
    elif input_style == "positive_mean_log_v2":
        x, denorm = normalize_spectrum_positive_mean_log(flux, ivar, mask)
    else:
        raise ValueError(
            f"Unknown input_style {input_style!r} "
            "(expected mask_arcsinh_v3, median_v2, or positive_mean_log_v2)"
        )
    return x, denorm, mask


def prepare_codec_v2_linear(
    flux: torch.Tensor,
    ivar: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Input pipeline used to train **codec_v2** (commit before arcsinh Tier-A).

    Per-spectrum scale = median(|flux|); channels are flux/scale and sqrt(ivar/scale²).
    No arcsinh. ``mask`` is ignored (v2 training did not mask-weight the norm).
    """
    del mask  # unused; kept for API compatibility with other prepare_* helpers
    flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    ivar = torch.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
    scale = flux.abs().median(dim=-1, keepdim=True).values.clamp(min=1e-12)
    flux_n = flux / scale
    ivar_n = ivar / (scale * scale)
    istd_n = torch.sqrt(ivar_n.clamp(min=1e-10))
    x = torch.stack([flux_n, istd_n], dim=1)
    return x, scale.squeeze(-1)


# Alias used by older call sites
prepare_codec_input = normalize_spectrum_input
