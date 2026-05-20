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


def quant_temperature_at_step(
    step: int,
    *,
    start: float = 1.0,
    min_temp: float = 0.1,
    anneal_steps: int = 2000,
) -> float:
    """Linear anneal for LFQ ``sign(z / tau)`` (tau=start → min_temp over anneal_steps)."""
    if anneal_steps <= 0:
        return start
    frac = min(max(step / anneal_steps, 0.0), 1.0)
    return start + frac * (min_temp - start)


def latent_bit_balance_loss(z: torch.Tensor) -> torch.Tensor:
    """Differentiable LFQ diversity: penalize collapsed per-bit marginals (0 = balanced, 1 = collapsed)."""
    if z.ndim != 3:
        raise ValueError("z must have shape (B, dim, L)")
    probs = torch.sigmoid(z)
    p = probs.mean(dim=(0, 2)).clamp(1e-4, 1.0 - 1e-4)
    bit_ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).sum()
    max_ent = math.log(2) * z.shape[1]
    return (1.0 - bit_ent / max_ent).clamp(min=0.0, max=1.0)


def batch_codebook_entropy_loss(indices: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """FM-style penalty: low entropy of the code histogram over the whole batch (0 = uniform)."""
    flat = indices.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((), device=indices.device)
    hist = torch.bincount(flat, minlength=n_bins).float()
    total = hist.sum()
    if total <= 0:
        return torch.ones((), device=indices.device)
    p = (hist / total).clamp(min=1e-10)
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    max_ent = math.log(max(n_bins, 2))
    return ((max_ent - entropy) / max_ent).clamp(min=0.0, max=1.0)


def code_usage_gate_bins(n_codes: int) -> int:
    """Tier-1 gate uses 30% of 256 codes (8-bit LFQ), even when model has 10-bit (1024) indices."""
    return min(int(n_codes), 256)


def code_usage_passes_gate(n_unique: float, n_codes: int, min_fraction: float) -> bool:
    """True when at least ``min_fraction`` of the 256-code Tier-1 gate is used."""
    return float(n_unique) >= min_fraction * code_usage_gate_bins(n_codes)


def code_usage_stats(indices: torch.Tensor, n_codes: int = 256) -> dict[str, float | int | list[tuple[int, int]]]:
    """Summarize LFQ index usage for one forward pass (batch)."""
    flat = indices.reshape(-1)
    device = indices.device
    if flat.numel() == 0:
        return {
            "n_unique": 0,
            "n_codes": n_codes,
            "gate_bins": code_usage_gate_bins(n_codes),
            "usage_fraction": 0.0,
            "usage_fraction_gate": 0.0,
            "entropy_penalty": 1.0,
            "batch_entropy_penalty": 1.0,
            "top_codes": [],
            "per_row_n_unique": [],
        }
    hist = torch.bincount(flat, minlength=n_codes)
    used_mask = hist > 0
    n_unique = int(used_mask.sum().item())
    used_ids = torch.nonzero(used_mask, as_tuple=False).flatten()
    if used_ids.numel() == 0:
        top_codes: list[tuple[int, int]] = []
    else:
        vals = hist[used_ids]
        order = torch.argsort(vals, descending=True)
        top_codes = [
            (int(used_ids[i].item()), int(vals[i].item()))
            for i in order[:10]
        ]
    per_row: list[float] = []
    for b in range(indices.shape[0]):
        row = indices[b].reshape(-1)
        per_row.append(float(len(torch.unique(row))))
    gate_bins = code_usage_gate_bins(n_codes)
    return {
        "n_unique": n_unique,
        "n_codes": n_codes,
        "gate_bins": gate_bins,
        "usage_fraction": n_unique / max(n_codes, 1),
        "usage_fraction_gate": n_unique / max(gate_bins, 1),
        "entropy_penalty": float(latent_index_entropy_penalty(indices, n_bins=n_codes).item()),
        "batch_entropy_penalty": float(batch_codebook_entropy_loss(indices, n_bins=n_codes).item()),
        "top_codes": top_codes,
        "per_row_n_unique": per_row,
    }


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


def _align_flux_metrics_inputs(
    pred_flux: torch.Tensor,
    target_flux: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Match lengths for metrics (codec grid vs native padded batch)."""
    L = min(int(pred_flux.shape[-1]), int(target_flux.shape[-1]))
    pred_flux = pred_flux[..., :L]
    target_flux = target_flux[..., :L]
    mask = align_mask_to_length(mask, L)
    return pred_flux, target_flux, mask


def flux_rms(
    pred_flux: torch.Tensor,
    target_flux: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMS error on good pixels; pred/target (B, L)."""
    pred_flux, target_flux, mask = _align_flux_metrics_inputs(pred_flux, target_flux, mask)
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
    pred_flux, target_flux, mask = _align_flux_metrics_inputs(pred_flux, target_flux, mask)
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


def flux_std_ratio_per_sample(
    pred_flux: torch.Tensor,
    target_flux: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row std(pred)/std(target) on good pixels; shape (B,)."""
    pred_flux, target_flux, mask = _align_flux_metrics_inputs(pred_flux, target_flux, mask)
    ratios = [
        flux_std_ratio(pred_flux[b : b + 1], target_flux[b : b + 1], None if mask is None else mask[b : b + 1])
        for b in range(pred_flux.shape[0])
    ]
    return torch.stack(ratios).reshape(-1)
