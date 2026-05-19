"""Spectrum codec: 1D ConvNeXt-style encoder/decoder + lookup-free quantization."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from desifm.constants import GRID_SIZE, N_LATENT_TOKENS
from desifm.training.codec_input import denormalize_spectrum_output, masked_recon_loss
from desifm.training.codec_loss import (
    align_mask_to_length,
    latent_index_entropy_penalty,
    physical_flux_loss,
)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None] + self.bias[:, None]


class ConvNeXtBlock1d(nn.Module):
    def __init__(self, dim: int, kernel: int = 7, expansion: int = 4):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        hidden = dim * expansion
        self.pw1 = nn.Conv1d(dim, hidden, 1)
        self.pw2 = nn.Conv1d(hidden, dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        return residual + x


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LFQuantizer(nn.Module):
    """Lookup-free quantizer: each dim in {-1, +1}, index = binary code."""

    def __init__(self, dim: int = 8, n_codes: int = 256, commitment_weight: float = 0.25):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.commitment_weight = commitment_weight

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.encode(z)
        z_q = self.decode(indices)
        commit = F.mse_loss(z_q.detach(), z)
        codebook = F.mse_loss(z_q, z.detach())
        return z_q, commit + self.commitment_weight * codebook, indices

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        bits = (z > 0).long()
        powers = (2 ** torch.arange(self.dim, device=z.device)).view(1, self.dim, 1)
        return (bits * powers).sum(dim=1).long()

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        bits = ((indices.unsqueeze(1) >> torch.arange(self.dim, device=indices.device).view(1, self.dim, 1)) & 1).float()
        return bits * 2 - 1


class SpectrumCodec(nn.Module):
    """Encode flux+ivar to discrete spectrum tokens; decode back to flux."""

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 8,
        widths: Tuple[int, ...] = (96, 192, 384, 512),
        *,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.n_tokens = N_LATENT_TOKENS
        self.latent_dim = latent_dim

        self.stem = nn.Conv1d(in_channels, widths[0], kernel_size=4, stride=4)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = widths[0]
        for w in widths[1:]:
            self.downs.append(Downsample(ch, w))
            self.enc_blocks.append(nn.Sequential(ConvNeXtBlock1d(w), ConvNeXtBlock1d(w), ConvNeXtBlock1d(w)))
            ch = w
        self.to_latent = nn.Conv1d(ch, latent_dim, 1)
        self.quant = LFQuantizer(latent_dim, n_codes=1024, commitment_weight=commitment_weight)

        self.from_latent = nn.Conv1d(latent_dim, ch, 1)
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for w in reversed(widths[1:]):
            self.ups.append(Upsample(ch, w))
            self.dec_blocks.append(nn.Sequential(ConvNeXtBlock1d(w), ConvNeXtBlock1d(w), ConvNeXtBlock1d(w)))
            ch = w
        self.head = nn.ConvTranspose1d(ch, in_channels, kernel_size=4, stride=4)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.grid_size:
            return x
        return F.interpolate(x, size=self.grid_size, mode="linear", align_corners=False)

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        h = self.stem(x)
        for down, blocks in zip(self.downs, self.enc_blocks):
            h = down(h)
            h = blocks(h)
        return self.to_latent(h)

    def _decode_latent(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quant.decode(indices)
        h = self.from_latent(z_q)
        for up, blocks in zip(self.ups, self.dec_blocks):
            h = up(h)
            h = blocks(h)
        out = self.head(h)
        return self._resize(out)

    def _pad_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] < self.n_tokens:
            indices = F.pad(indices, (0, self.n_tokens - indices.shape[-1]))
        elif indices.shape[-1] > self.n_tokens:
            indices = indices[:, : self.n_tokens]
        return indices

    def encode(self, x: torch.Tensor, denorm: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B, 2, L) arcsinh-normalized. Returns indices and optional denorm passthrough."""
        h = self._encode_latent(x)
        _, _, indices = self.quant(h)
        return self._pad_indices(indices), denorm

    def decode(
        self,
        indices: torch.Tensor,
        denorm: torch.Tensor,
        *,
        to_physical: bool = True,
    ) -> torch.Tensor:
        """Decode tokens to spectrum. Default: physical flux+istd (B, 2, L)."""
        recon = self._decode_latent(indices)
        if to_physical:
            return denormalize_spectrum_output(recon, denorm)
        return recon

    def forward(
        self,
        x: torch.Tensor,
        denorm: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        lambda_phys: float = 0.0,
        lambda_entropy: float = 0.0,
    ) -> dict:
        """x: arcsinh (B, 2, L); loss on flux channel (+ optional physical / entropy terms)."""
        x = self._resize(x)
        h = self._encode_latent(x)
        z_q, q_loss, indices = self.quant(h)
        indices = self._pad_indices(indices)
        recon = self._decode_latent(indices)
        loss_mask = align_mask_to_length(mask, x.shape[-1])
        recon_loss = masked_recon_loss(recon[:, 0], x[:, 0], loss_mask)

        phys_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if lambda_phys > 0:
            phys_loss = physical_flux_loss(recon, x, denorm, loss_mask)

        ent_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if lambda_entropy > 0:
            ent_loss = latent_index_entropy_penalty(indices)

        total = recon_loss + q_loss + lambda_phys * phys_loss + lambda_entropy * ent_loss
        recon_phys = denormalize_spectrum_output(recon, denorm)[:, 0]
        target_phys = denormalize_spectrum_output(x, denorm)[:, 0]
        return {
            "recon": recon,
            "loss": total,
            "indices": indices,
            "recon_loss": recon_loss,
            "q_loss": q_loss,
            "phys_loss": phys_loss,
            "entropy_loss": ent_loss,
            "recon_phys": recon_phys,
            "target_phys": target_phys,
        }
