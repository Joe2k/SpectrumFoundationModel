"""Build transformer sequences for Approach A and B."""

from __future__ import annotations

from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import torch

from desifm.constants import EOS, MASK, REDMASK, REDSHIFT_OFFSET, SOS, SPECTRUM_OFFSET
from desifm.tokenization.redshift_codec import RedshiftCodec
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.codec_input import prepare_codec_batch


Approach = Literal["a", "b"]


@runtime_checkable
class SpectrumTokenizer(Protocol):
    def encode_batch(self, batch: dict) -> tuple[torch.Tensor, dict]: ...


def tokenize_batch(
    batch: dict,
    spectrum_tok: SpectrumCodec | SpectrumTokenizer,
    z_codec: RedshiftCodec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        if hasattr(spectrum_tok, "encode_batch") and not isinstance(spectrum_tok, SpectrumCodec):
            spec_idx, _meta = spectrum_tok.encode_batch(batch)
        else:
            x, denorm, _ = prepare_codec_batch(batch)
            spec_idx, _ = spectrum_tok.encode(x, denorm)
    z_idx = torch.tensor([z_codec.encode(float(z)) for z in batch["z"]], device=device, dtype=torch.long)
    return spec_idx, z_idx


def build_sequences(
    spec_idx: torch.Tensor,
    z_idx: torch.Tensor,
    approach: Approach,
    encoder_mask_ratio: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    B, T = spec_idx.shape
    spec_tok = spec_idx + SPECTRUM_OFFSET
    rz_tok = z_idx + REDSHIFT_OFFSET

    sos = torch.full((B, 1), SOS, dtype=torch.long, device=spec_tok.device)
    eos = torch.full((B, 1), EOS, dtype=torch.long, device=spec_tok.device)
    rz = rz_tok.unsqueeze(1)

    spec_enc = spec_tok
    mask_pos = None
    if encoder_mask_ratio > 0:
        if rng is None:
            mask_pos = torch.rand(B, T, device=spec_tok.device) < encoder_mask_ratio
        else:
            mask_pos = torch.rand(B, T, device=spec_tok.device, generator=rng) < encoder_mask_ratio
        spec_enc = torch.where(mask_pos, torch.full_like(spec_tok, MASK), spec_tok)

    if approach == "a":
        enc = torch.cat([sos, rz, spec_enc, eos], dim=1)
        dec = torch.cat([sos, rz, spec_tok], dim=1)
    else:
        enc = torch.cat([sos, spec_enc, eos], dim=1)
        rz_slot = torch.full_like(rz, REDMASK)
        dec = torch.cat([sos, rz_slot, spec_tok], dim=1)

    tgt = torch.cat([rz, spec_tok, eos], dim=1)
    return enc, dec, tgt, mask_pos
