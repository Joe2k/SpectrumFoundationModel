"""Official Polymathic AION DESI spectrum tokenizer bridge."""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch

log = logging.getLogger(__name__)

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
            log.info("creating CodecManager(device=%s)...", self.device)
            sys.stdout.flush()
            self._codec_manager = CodecManager(device=str(self.device))
            log.info("CodecManager ready (HF weights load on first encode)")
            sys.stdout.flush()
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
        log.info(
            "AION encode: loading DESI spectrum codec from HF (polymathic-ai/aion-base) "
            "if not cached — on NERSC run prefetch_aion_codec.py on a login node first"
        )
        sys.stdout.flush()
        tokens = self._manager().encode(spec)
        log.info("AION encode finished")
        sys.stdout.flush()
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
