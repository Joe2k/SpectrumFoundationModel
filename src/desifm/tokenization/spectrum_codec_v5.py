"""Spectrum codec v5: FM V2 ideas (U-Net skips, cross-attn, latent_dim=10) on desifm preprocessing."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from desifm.constants import GRID_SIZE, N_LATENT_TOKENS
from desifm.tokenization.spectrum_codec import ChannelLayerNorm
from desifm.training.codec_input import denormalize_spectrum_output, masked_recon_loss
from desifm.training.codec_loss import (
    align_mask_to_length,
    batch_codebook_entropy_loss,
    physical_flux_loss,
)


class ConvNeXtBlock1DV5(nn.Module):
    def __init__(self, dim: int, kernel: int = 7, expansion: int = 4, drop_path: float = 0.0):
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


class DownsampleBlockV5(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = ChannelLayerNorm(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class UpsampleBlockV5(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = ChannelLayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class CrossAttentionV5(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        b, c, lq = query.shape
        _, _, lk = key_value.shape
        q = self.q_proj(query).reshape(b, self.num_heads, self.head_dim, lq)
        k = self.k_proj(key_value).reshape(b, self.num_heads, self.head_dim, lk)
        v = self.v_proj(key_value).reshape(b, self.num_heads, self.head_dim, lk)
        attn = torch.einsum("bhdl,bhdk->bhlk", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhlk,bhdk->bhdl", attn, v).reshape(b, c, lq)
        return self.out_proj(out)


class LFQuantizerV5(nn.Module):
    """LFQ with project_in and FM-style batch entropy in the outer forward."""

    def __init__(self, dim: int = 10, n_codes: int | None = None, commitment_weight: float = 0.05):
        super().__init__()
        self.dim = dim
        self.n_codes = 2**dim if n_codes is None else n_codes
        self.commitment_weight = commitment_weight
        self.project_in = nn.Conv1d(dim, dim, kernel_size=1)

    def _indices(self, z_q: torch.Tensor) -> torch.Tensor:
        bits = ((z_q + 1) / 2).long().clamp(0, 1)
        powers = (2 ** torch.arange(self.dim, device=z_q.device, dtype=torch.long)).view(1, -1, 1)
        return (bits * powers).sum(dim=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.project_in(z)
        z_q = torch.sign(z)
        z_q = z + (z_q - z).detach()
        # Match FM V2 / desifm v4: commitment + β·codebook (not unweighted codebook).
        commit = F.mse_loss(z_q.detach(), z)
        codebook = F.mse_loss(z_q, z.detach())
        indices = self._indices(z_q)
        return z_q, commit + self.commitment_weight * codebook, indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        b, length = indices.shape
        z = torch.zeros(b, self.dim, length, device=indices.device, dtype=torch.float32)
        for i in range(self.dim):
            z[:, i, :] = ((indices // (2**i)) % 2).float() * 2 - 1
        return z


class SpectrumCodecV5(nn.Module):
    """v5 tokenizer: skips + cross-attention; physical flux loss primary when training."""

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 10,
        encoder_depths: Tuple[int, ...] = (3, 3, 9, 3),
        encoder_dims: Tuple[int, ...] = (96, 192, 384, 512),
        *,
        commitment_weight: float = 0.05,
        use_skip_connections: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.n_tokens = N_LATENT_TOKENS
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        self.use_cross_attention = use_cross_attention

        self.encoder_stem = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims[0], kernel_size=4, stride=4),
            ChannelLayerNorm(encoder_dims[0]),
        )
        self.encoder_stages = nn.ModuleList()
        for i, depth in enumerate(encoder_depths):
            stage = nn.ModuleList()
            if i > 0:
                stage.append(DownsampleBlockV5(encoder_dims[i - 1], encoder_dims[i]))
            for _ in range(depth):
                stage.append(ConvNeXtBlock1DV5(encoder_dims[i]))
            self.encoder_stages.append(stage)

        self.pre_quant_norm = ChannelLayerNorm(encoder_dims[-1])
        self.quant_conv = nn.Conv1d(encoder_dims[-1], latent_dim, kernel_size=1)
        self.quant = LFQuantizerV5(latent_dim, commitment_weight=commitment_weight)
        self.post_quant_conv = nn.Conv1d(latent_dim, encoder_dims[-1], kernel_size=1)

        decoder_dims = (encoder_dims[-1], encoder_dims[-2], encoder_dims[-3], encoder_dims[0])
        self.decoder_dims = decoder_dims
        if use_skip_connections:
            self.skip_proj = nn.ModuleList(
                [
                    nn.Conv1d(encoder_dims[3], decoder_dims[0], 1),
                    nn.Conv1d(encoder_dims[2], decoder_dims[1], 1),
                    nn.Conv1d(encoder_dims[1], decoder_dims[2], 1),
                    nn.Conv1d(encoder_dims[0], decoder_dims[3], 1),
                ]
            )
        else:
            self.skip_proj = nn.ModuleList()
        if use_cross_attention:
            self.cross_attn = nn.ModuleList([CrossAttentionV5(d) for d in decoder_dims])
        else:
            self.cross_attn = nn.ModuleList()

        self.decoder_stages = nn.ModuleList()
        for i, depth in enumerate(encoder_depths):
            stage = nn.ModuleList()
            for _ in range(depth):
                stage.append(ConvNeXtBlock1DV5(decoder_dims[i]))
            if i < len(decoder_dims) - 1:
                stage.append(UpsampleBlockV5(decoder_dims[i], decoder_dims[i + 1]))
            self.decoder_stages.append(stage)

        self.decoder_head = nn.Sequential(
            nn.ConvTranspose1d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            ChannelLayerNorm(decoder_dims[-1]),
            nn.Conv1d(decoder_dims[-1], in_channels, kernel_size=1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.grid_size:
            return x
        return F.interpolate(x, size=self.grid_size, mode="linear", align_corners=False)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._resize(x)
        h = self.encoder_stem(x)
        skips: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            for block in stage:
                h = block(h)
            skips.append(h)
        h = self.pre_quant_norm(h)
        h = self.quant_conv(h)
        return h, skips

    def _decode(self, indices: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        h = self.post_quant_conv(self.quant.decode(indices))
        for i, stage in enumerate(self.decoder_stages):
            if self.use_cross_attention and len(self.cross_attn) > i:
                skip = skips[-(i + 1)]
                if skip.shape[-1] != h.shape[-1]:
                    skip = F.interpolate(skip, size=h.shape[-1], mode="linear", align_corners=False)
                if self.use_skip_connections and len(self.skip_proj) > i:
                    h = h + self.skip_proj[i](skip)
                h = h + self.cross_attn[i](h, skip)
            elif self.use_skip_connections and len(self.skip_proj) > i:
                skip = skips[-(i + 1)]
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
                h = h + self.skip_proj[i](skip)
            for block in stage:
                h = block(h)
        return self._resize(self.decoder_head(h))

    def _pad_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] < self.n_tokens:
            return F.pad(indices, (0, self.n_tokens - indices.shape[-1]))
        if indices.shape[-1] > self.n_tokens:
            return indices[:, : self.n_tokens]
        return indices

    def forward(
        self,
        x: torch.Tensor,
        denorm: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        lambda_phys: float = 0.5,
        lambda_entropy: float = 0.5,
        use_batch_entropy: bool = True,
        lambda_arcsinh: float = 0.25,
    ) -> dict:
        x = self._resize(x)
        h, skips = self._encode(x)
        z_q, q_loss, indices = self.quant(h)
        recon = self._decode(self._pad_indices(indices), skips)
        loss_mask = align_mask_to_length(mask, recon.shape[-1])

        recon_phys = denormalize_spectrum_output(recon, denorm)
        target_phys = denormalize_spectrum_output(x, denorm)
        phys_loss = masked_recon_loss(recon_phys[:, 0], target_phys[:, 0], loss_mask)
        arcsinh_loss = masked_recon_loss(recon[:, 0], x[:, 0], loss_mask)

        # Entropy on real latent positions only (before index padding to N_LATENT_TOKENS).
        if use_batch_entropy and lambda_entropy > 0:
            ent_loss = batch_codebook_entropy_loss(indices, n_bins=self.quant.n_codes)
        elif lambda_entropy > 0:
            from desifm.training.codec_loss import latent_index_entropy_penalty

            ent_loss = latent_index_entropy_penalty(indices, n_bins=self.quant.n_codes)
        else:
            ent_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        recon_loss = phys_loss
        total = (
            lambda_phys * phys_loss
            + lambda_arcsinh * arcsinh_loss
            + q_loss
            + lambda_entropy * ent_loss
        )

        return {
            "recon": recon,
            "loss": total,
            "indices": self._pad_indices(indices),
            "recon_loss": recon_loss,
            "arcsinh_loss": arcsinh_loss,
            "q_loss": q_loss,
            "phys_loss": phys_loss,
            "entropy_loss": ent_loss,
            "recon_phys": recon_phys[:, 0],
            "target_phys": target_phys[:, 0],
        }
