"""Encoder-decoder transformer for discrete spectrum + redshift tokens."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from desifm.constants import (
    EOS,
    MASK,
    REDMASK,
    REDSHIFT_OFFSET,
    SOS,
    SPECTRUM_OFFSET,
    VOCAB_SIZE,
)
from desifm.model.blocks import DecoderLayer, EncoderLayer, RMSNorm


Approach = Literal["a", "b"]


class DesiFoundationModel(nn.Module):
    """
    Approach A: encoder sees redshift + spectrum; auxiliary z head on pooled encoder.
    Approach B: encoder sees spectrum only; decoder always predicts z from REDMASK slot.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        dropout: float = 0.1,
        n_redshift_classes: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_redshift_classes = n_redshift_classes

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.modality_embed = nn.Embedding(3, d_model)  # 0=special, 1=spectrum, 2=redshift

        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_enc_layers)])
        self.enc_norm = RMSNorm(d_model)

        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_dec_layers)])
        self.dec_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.z_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_redshift_classes),
        )

    def _modality_ids(self, ids: torch.Tensor) -> torch.Tensor:
        spec = (ids >= SPECTRUM_OFFSET) & (ids < REDSHIFT_OFFSET)
        rz = ids >= REDSHIFT_OFFSET
        out = torch.zeros_like(ids)
        out[spec] = 1
        out[rz] = 2
        return out

    def encode(self, enc_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(enc_ids) + self.modality_embed(self._modality_ids(enc_ids))
        for layer in self.encoder:
            x = layer(x)
        return self.enc_norm(x)

    def decode(self, dec_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(dec_ids) + self.modality_embed(self._modality_ids(dec_ids))
        for layer in self.decoder:
            x = layer(x, memory)
        return self.dec_norm(x)

    def forward(
        self,
        enc_ids: torch.Tensor,
        dec_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        z_weight: float = 20.0,
        aux_z_weight: float = 0.5,
        approach: Approach = "a",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory = self.encode(enc_ids)
        logits = self.lm_head(self.decode(dec_ids, memory))
        if targets is None:
            return logits, None

        per_tok = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(targets.shape)

        valid = targets != -100
        loss_z = (per_tok[:, 0] * valid[:, 0]).sum() / valid[:, 0].sum().clamp(min=1)
        loss_spec = (per_tok[:, 1:] * valid[:, 1:]).sum() / valid[:, 1:].sum().clamp(min=1)
        loss_seq = z_weight * loss_z + loss_spec

        loss = loss_seq
        if approach == "a":
            z_class = (targets[:, 0] - REDSHIFT_OFFSET).clamp(0, self.n_redshift_classes - 1)
            z_logits = self.z_head(memory.mean(dim=1))
            loss = loss + aux_z_weight * F.cross_entropy(z_logits, z_class)

        return logits, loss

    @torch.no_grad()
    def generate_ar(
        self,
        enc_ids: torch.Tensor,
        max_steps: int,
        start_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive decode from start_dec (includes SOS)."""
        memory = self.encode(enc_ids)
        generated = start_dec
        for _ in range(max_steps):
            logits = self.lm_head(self.decode(generated, memory))
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == EOS).all():
                break
        return generated
