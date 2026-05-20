"""Official Polymathic AION DESI spectrum tokenizer bridge."""

from __future__ import annotations

from typing import Any

import torch

from desifm.constants import GRID_SIZE, N_LATENT_TOKENS, N_SPECTRUM_CODES
from desifm.tokenization.aion_grid import resample_spectrum_batch

AION_TOKEN_KEY = "tok_spectrum_desi"


class AionSpectrumTokenizer:
    """Frozen AION CodecManager for DESI spectra → 273 discrete codes per spectrum."""

    def __init__(self, device: torch.device | str, hf_repo: str = "polymathic-ai/aion-base"):
        self.device = torch.device(device)
        self.hf_repo = hf_repo
        self._codec_manager: Any = None

    def _manager(self):
        if self._codec_manager is None:
            from desifm.training.env import load_project_env

            load_project_env()
            try:
                from aion.codecs import CodecManager
            except ImportError as e:
                raise ImportError(
                    "polymathic-aion is required for AION tokenization. "
                    'Install with: pip install -e ".[aion]"'
                ) from e
            self._codec_manager = CodecManager(device=str(self.device))
        return self._codec_manager

    @torch.no_grad()
    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Encode collated batch; return (B, N_LATENT_TOKENS) code indices and metadata."""
        from aion.modalities import DESISpectrum

        flux = batch["flux"].to(self.device)
        ivar = batch["ivar"].to(self.device)
        mask = batch["mask"].to(self.device)
        wavelength = batch["wavelength"].to(self.device)

        flux, ivar, mask, wavelength, resampled = resample_spectrum_batch(
            flux, ivar, mask, wavelength, length=GRID_SIZE
        )

        spec = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)
        tokens = self._manager().encode(spec)
        if AION_TOKEN_KEY not in tokens:
            raise KeyError(f"expected {AION_TOKEN_KEY!r} in encode output, got {list(tokens.keys())}")

        spec_idx = tokens[AION_TOKEN_KEY].long()
        if spec_idx.ndim != 2 or spec_idx.shape[1] != N_LATENT_TOKENS:
            raise ValueError(f"expected ({spec_idx.shape[0]}, {N_LATENT_TOKENS}), got {tuple(spec_idx.shape)}")

        spec_idx = spec_idx.clamp(0, N_SPECTRUM_CODES - 1)
        meta = {"resampled_to_grid": resampled, "grid_size": GRID_SIZE}
        return spec_idx, meta

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        if self._codec_manager is not None:
            self._codec_manager = None
        return self
