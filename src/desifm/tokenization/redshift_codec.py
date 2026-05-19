"""Redshift scalar codec: empirical CDF -> Gaussian -> uniform binning."""

from __future__ import annotations

import torch

from desifm.constants import N_REDSHIFT_BINS


class RedshiftCodec:
    """Parameter-free discrete codec for redshift z."""

    def __init__(self, n_bins: int = N_REDSHIFT_BINS, gaussian_clip: float = 3.0):
        self.n_bins = n_bins
        self.gaussian_clip = gaussian_clip
        self._sorted_z: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self._sorted_z is not None

    def fit(self, z_values: torch.Tensor) -> None:
        z = z_values.detach().float().flatten()
        self._sorted_z = torch.sort(z)[0]

    def _cdf(self, z: torch.Tensor) -> torch.Tensor:
        assert self._sorted_z is not None
        idx = torch.searchsorted(self._sorted_z, z.flatten())
        p = idx.float() / len(self._sorted_z)
        return p.clamp(1e-6, 1 - 1e-6).reshape(z.shape)

    def _inv_cdf(self, p: torch.Tensor) -> torch.Tensor:
        assert self._sorted_z is not None
        p = p.clamp(0, 1)
        idx = (p * (len(self._sorted_z) - 1)).long().clamp(0, len(self._sorted_z) - 1)
        return self._sorted_z[idx].reshape(p.shape)

    @staticmethod
    def _to_gaussian(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.erfinv(2 * p - 1) * (2**0.5)

    @staticmethod
    def _from_gaussian(g: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.erf(g / (2**0.5)))

    def encode(self, z: float | torch.Tensor) -> int | torch.Tensor:
        if not self.fitted:
            raise RuntimeError("RedshiftCodec.fit() required before encode()")
        t = torch.as_tensor(z, dtype=torch.float32)
        g = self._to_gaussian(self._cdf(t))
        bin_idx = ((g + self.gaussian_clip) / (2 * self.gaussian_clip) * (self.n_bins - 1)).long()
        bin_idx = bin_idx.clamp(0, self.n_bins - 1)
        return int(bin_idx.item()) if t.numel() == 1 else bin_idx

    def decode(self, bin_idx: int | torch.Tensor) -> float | torch.Tensor:
        if not self.fitted:
            raise RuntimeError("RedshiftCodec.fit() required before decode()")
        t = torch.as_tensor(bin_idx, dtype=torch.float32)
        g = (t + 0.5) / self.n_bins * (2 * self.gaussian_clip) - self.gaussian_clip
        p = self._from_gaussian(g)
        z = self._inv_cdf(p)
        return float(z.item()) if t.numel() == 1 else z

    def state_dict(self) -> dict:
        return {"sorted_z": self._sorted_z, "n_bins": self.n_bins, "gaussian_clip": self.gaussian_clip}

    def load_state_dict(self, state: dict) -> None:
        self._sorted_z = state["sorted_z"]
        self.n_bins = state["n_bins"]
        self.gaussian_clip = state["gaussian_clip"]
