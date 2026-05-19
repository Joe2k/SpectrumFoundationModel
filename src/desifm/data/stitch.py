"""Stitch DESI B/R/Z bands into one spectrum."""

from __future__ import annotations

import numpy as np


def stitch_bands(
    waves: list[np.ndarray],
    fluxes: list[np.ndarray],
    ivars: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    ivar = np.concatenate(ivars)
    mask = np.concatenate(masks)
    order = np.argsort(wave)
    wave, flux, ivar, mask = wave[order], flux[order], ivar[order], mask[order]

    unique_w, unique_f, unique_i, unique_m = [], [], [], []
    i = 0
    while i < len(wave):
        w0 = wave[i]
        j = i
        while j < len(wave) and abs(wave[j] - w0) < 0.1:
            j += 1
        good = ~mask[i:j]
        if good.any():
            iv = ivar[i:j][good]
            fv = flux[i:j][good]
            wsum = iv.sum()
            f_avg = (fv * iv).sum() / wsum if wsum > 0 else fv.mean()
            i_avg = wsum
            m_avg = False
        else:
            f_avg = flux[i:j].mean()
            i_avg = 0.0
            m_avg = True
        unique_w.append(w0)
        unique_f.append(f_avg)
        unique_i.append(i_avg)
        unique_m.append(m_avg)
        i = j

    return {
        "wavelength": np.asarray(unique_w, dtype=np.float32),
        "flux": np.asarray(unique_f, dtype=np.float32),
        "ivar": np.asarray(unique_i, dtype=np.float32),
        "mask": np.asarray(unique_m, dtype=bool),
    }
