"""Synthetic spectra for local smoke tests without FITS files."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticSpectrumDataset(Dataset):
    """Random smooth-ish spectra with random redshifts in [0, 1.2]."""

    def __init__(self, n_spectra: int = 512, length: int = 4096, seed: int = 0):
        self.n_spectra = n_spectra
        self.length = length
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_spectra

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(int(idx) + int(self.gen.initial_seed()))
        x = torch.linspace(0, 1, self.length)
        flux = torch.sin(x * 40) * 0.2 + torch.randn(self.length, generator=g) * 0.05 + 1.0
        ivar = torch.ones(self.length) * 10.0
        z = torch.rand(1, generator=g).item() * 1.2
        return {"flux": flux, "ivar": ivar, "z": torch.tensor(z, dtype=torch.float32)}
