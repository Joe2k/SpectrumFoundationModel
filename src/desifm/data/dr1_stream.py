"""Stream spectra from a DR1 JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

from desifm.data.stitch import stitch_bands


def load_manifest(path: Path) -> list[dict]:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class DR1StreamDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        max_spectra: int | None = None,
        require_good_z: bool = True,
    ):
        self.records = records
        self.require_good_z = require_good_z
        self.index: list[tuple[int, int]] = []
        for ri, rec in enumerate(records):
            n = rec.get("n_rows")
            if n is None:
                with fits.open(rec["coadd"], memmap=True) as h:
                    n = int(h["FIBERMAP"].header["NAXIS2"])
                rec["n_rows"] = n
            for row in range(n):
                self.index.append((ri, row))
                if max_spectra and len(self.index) >= max_spectra:
                    break
            if max_spectra and len(self.index) >= max_spectra:
                break
        self._open: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _hdus(self, rec_idx: int):
        if rec_idx not in self._open:
            rec = self.records[rec_idx]
            self._open[rec_idx] = (fits.open(rec["coadd"], memmap=True), fits.open(rec["redrock"], memmap=True))
        return self._open[rec_idx]

    def __getitem__(self, idx: int) -> dict | None:
        rec_idx, row = self.index[idx]
        coadd, redrock = self._hdus(rec_idx)
        if self.require_good_z:
            if int(redrock["REDSHIFTS"].data["ZWARN"][row]) != 0:
                return None
            if int(coadd["FIBERMAP"].data["COADD_FIBERSTATUS"][row]) != 0:
                return None
        flux_sum = (
            np.abs(coadd["B_FLUX"].data[row]).sum()
            + np.abs(coadd["R_FLUX"].data[row]).sum()
            + np.abs(coadd["Z_FLUX"].data[row]).sum()
        )
        if flux_sum == 0:
            return None
        stitched = stitch_bands(
            [coadd["B_WAVELENGTH"].data, coadd["R_WAVELENGTH"].data, coadd["Z_WAVELENGTH"].data],
            [coadd["B_FLUX"].data[row], coadd["R_FLUX"].data[row], coadd["Z_FLUX"].data[row]],
            [coadd["B_IVAR"].data[row], coadd["R_IVAR"].data[row], coadd["Z_IVAR"].data[row]],
            [
                coadd["B_MASK"].data[row] != 0,
                coadd["R_MASK"].data[row] != 0,
                coadd["Z_MASK"].data[row] != 0,
            ],
        )
        z = float(redrock["REDSHIFTS"].data["Z"][row])
        return {
            "flux": torch.from_numpy(stitched["flux"]),
            "ivar": torch.from_numpy(stitched["ivar"]),
            "mask": torch.from_numpy(stitched["mask"]),
            "z": torch.tensor(z, dtype=torch.float32),
        }


def collate_spectra(batch: list) -> dict | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    L = max(b["flux"].shape[0] for b in batch)

    def pad(t, fill=0.0):
        if t.shape[0] == L:
            return t
        out = torch.full((L,), fill, dtype=t.dtype)
        out[: t.shape[0]] = t
        return out

    return {
        "flux": torch.stack([pad(b["flux"]) for b in batch]),
        "ivar": torch.stack([pad(b["ivar"]) for b in batch]),
        "mask": torch.stack([pad(b["mask"], fill=True) for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }


def healpix_split(records: list[dict], holdout: float, seed: int) -> tuple[list, list]:
    if not 0 < holdout < 1:
        raise ValueError("holdout must be in (0, 1)")
    rng = np.random.default_rng(seed)
    hp = sorted({r.get("healpix", i) for i, r in enumerate(records)})
    rng.shuffle(hp)
    n_val = max(1, int(len(hp) * holdout))
    val_hp = set(hp[:n_val])
    train, val = [], []
    for r in records:
        (val if r.get("healpix", -1) in val_hp else train).append(r)
    return train, val
