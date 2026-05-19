"""Helpers for codec evaluation notebooks (forward pass, checkpoint metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from desifm.constants import GRID_SIZE
from desifm.training.codec_input import (
    INPUT_STYLE_V4,
    denormalize_spectrum_output,
    prepare_codec_batch_for_style,
    prepare_codec_v2_linear,
)
from desifm.tokenization.spectrum_codec import SpectrumCodec


def input_style_from_checkpoint(ckpt: dict[str, Any]) -> str:
    """Infer normalization / forward path from checkpoint metadata."""
    s = ckpt.get("input_style")
    if s in ("mask_arcsinh_v3", INPUT_STYLE_V4):
        return s
    # codec_v2 (nv7py9b1) was trained with linear median scaling, not arcsinh.
    return "codec_v2_linear"


def spec_to_batch(spec: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """One stitched spectrum (no padding) as a batch of size 1."""
    return {
        "flux": torch.from_numpy(np.asarray(spec["flux"], dtype=np.float32)).unsqueeze(0).to(device),
        "ivar": torch.from_numpy(np.asarray(spec["ivar"], dtype=np.float32)).unsqueeze(0).to(device),
        "mask": torch.from_numpy(np.asarray(spec["mask"], dtype=bool)).unsqueeze(0).to(device),
        "z": torch.tensor([float(spec["z"])], dtype=torch.float32, device=device),
    }


def resample_1d(flux_1d: torch.Tensor, length: int) -> torch.Tensor:
    """Linear resample (B=1) from codec grid or native L to ``length``."""
    if int(flux_1d.shape[-1]) == length:
        return flux_1d
    x = flux_1d.reshape(1, 1, -1).float()
    return F.interpolate(x, size=length, mode="linear", align_corners=False).reshape(length)


def load_spectrum_codec(ckpt_path: Path, device: torch.device) -> tuple[SpectrumCodec, str]:
    """Load weights and return ``(model, input_style)``."""
    ckpt = Path(ckpt_path)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    style = input_style_from_checkpoint(blob)
    model = SpectrumCodec().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, style


def forward_v2_legacy(
    model: SpectrumCodec,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Forward pass matching **codec_v2** training (linear norm + log10 median scale decode)."""
    x, _scale = prepare_codec_v2_linear(batch["flux"], batch["ivar"], batch.get("mask"))
    x = model._resize(x)
    log_norm = torch.log10(x[:, 0].abs().median(dim=-1).values.clamp(min=1e-12))

    with torch.no_grad():
        h = model.stem(x)
        for down, blocks in zip(model.downs, model.enc_blocks):
            h = down(h)
            h = blocks(h)
        h = model.to_latent(h)
        _z_q, q_loss, indices = model.quant(h)
        indices = model._pad_indices(indices)
        recon_norm = model._decode_latent(indices)
        scale_phys = (10**log_norm).view(-1, 1, 1)
        recon = recon_norm * scale_phys

    recon_loss = F.mse_loss(recon[:, 0], x[:, 0])
    return {
        "recon_loss": float(recon_loss.item()),
        "q_loss": float(q_loss.item()),
        "loss_total": float((recon_loss + q_loss).item()),
        "flux_true": batch["flux"].cpu(),
        "flux_recon": recon[:, 0].cpu(),
        "flux_coadd": batch["flux"].cpu(),
        "mask": batch["mask"].cpu() if batch.get("mask") is not None else None,
        "log_norm": log_norm.cpu(),
        "recon_norm": recon_norm[:, 0].cpu(),
        "target_norm": x[:, 0].cpu(),
    }


def forward_physical(
    model: SpectrumCodec,
    batch: dict[str, torch.Tensor],
    input_style: str,
) -> dict[str, Any]:
    """One forward pass; physical flux recon + loss terms."""
    if input_style == "codec_v2_linear":
        return forward_v2_legacy(model, batch)

    x, denorm, mask = prepare_codec_batch_for_style(batch, input_style)
    with torch.no_grad():
        out = model(x, denorm, mask=mask)
    recon_phys = denormalize_spectrum_output(out["recon"], denorm)
    target_phys = denormalize_spectrum_output(x, denorm)
    return {
        "recon_loss": float(out["recon_loss"].item()),
        "q_loss": float(out["q_loss"].item()),
        "loss_total": float(out["loss"].item()),
        "flux_true": target_phys[:, 0].cpu(),
        "flux_recon": recon_phys[:, 0].cpu(),
        "flux_coadd": batch["flux"].cpu(),
        "mask": mask.cpu() if mask is not None else None,
        "denorm": denorm.cpu(),
    }


def forward_physical_from_spec(
    model: SpectrumCodec,
    spec: dict,
    input_style: str,
    device: torch.device,
) -> dict[str, Any]:
    """Forward on a **single unpadded** stitched spectrum."""
    batch = spec_to_batch(spec, device)
    out = forward_physical(model, batch, input_style)
    L = int(batch["flux"].shape[-1])

    if input_style == "codec_v2_linear":
        out["length"] = L
        out["flux_recon_native"] = resample_1d(out["flux_recon"][0], L)
        out["flux_roundtrip_native"] = out["flux_coadd"][0].clone()
        out["target_norm_native"] = resample_1d(out["target_norm"][0], L)
        out["recon_norm_native"] = resample_1d(out["recon_norm"][0], L)
        return out

    out["length"] = L
    out["flux_recon_native"] = resample_1d(out["flux_recon"][0], L)
    out["flux_roundtrip_native"] = resample_1d(out["flux_true"][0], L)
    return out
