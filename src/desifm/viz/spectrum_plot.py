"""Plot DESI spectra with common rest-frame lines shifted by redshift."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from desifm.data.stitch import stitch_bands

# rest wavelength (Å), label, kind for legend color
REST_LINES: list[tuple[float, str, str]] = [
    (1215.67, "Lyα", "abs"),
    (3727.7, "O II", "emi"),
    (3933.66, "Ca II K", "abs"),
    (3968.59, "Ca II H", "abs"),
    (4861.36, "Hβ", "emi"),
    (4958.91, "O III", "emi"),
    (5006.84, "O III", "emi"),
    (5175.53, "Mg b", "abs"),
    (5892.95, "Na D", "abs"),
    (6562.85, "Hα", "emi"),
]


def load_stitched_spectrum(
    coadd_path: str | Path,
    redrock_path: str | Path,
    row: int,
    *,
    require_good_z: bool = True,
) -> dict | None:
    """Load one row: stitched wavelength, flux, ivar, mask, redshift, quality flags."""
    with fits.open(coadd_path, memmap=True) as coadd, fits.open(redrock_path, memmap=True) as redrock:
        zwarn = int(redrock["REDSHIFTS"].data["ZWARN"][row])
        fiberstatus = int(coadd["FIBERMAP"].data["COADD_FIBERSTATUS"][row])
        z = float(redrock["REDSHIFTS"].data["Z"][row])
        if require_good_z and (zwarn != 0 or fiberstatus != 0):
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
    return {
        "wavelength": stitched["wavelength"],
        "flux": stitched["flux"],
        "ivar": stitched["ivar"],
        "mask": stitched["mask"],
        "z": z,
        "zwarn": zwarn,
        "fiberstatus": fiberstatus,
    }


def observed_lines(z: float, lines: Iterable[tuple[float, str, str]] = REST_LINES) -> list[tuple[float, str, str]]:
    return [(w * (1.0 + z), label, kind) for w, label, kind in lines]


def _capped_sigma_plot(flux: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """Formal 1/√ivar with a per-spectrum cap so a few tiny-ivar pixels do not dominate the plot."""
    # np.where evaluates both branches; use masked divide to avoid ivar==0 warnings.
    sigma = np.divide(
        1.0,
        np.sqrt(ivar),
        out=np.full(np.shape(ivar), np.nan, dtype=np.float64),
        where=ivar > 0,
    )
    good_f = np.isfinite(flux)
    if good_f.any():
        spread = float(np.nanpercentile(flux[good_f], 99) - np.nanpercentile(flux[good_f], 1))
        spread = max(spread, float(np.nanstd(flux[good_f])), 1e-30)
        return np.minimum(sigma, 10.0 * spread)
    return sigma


def _good_pixel_mask(flux: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    good = np.isfinite(flux)
    if mask is not None:
        good = good & ~np.asarray(mask, dtype=bool)
    return good


def _adaptive_ylim(
    flux: np.ndarray,
    mask: np.ndarray | None,
    *,
    ivar: np.ndarray | None,
    k_std: float = 5.0,
) -> tuple[float, float] | None:
    """Median ± k·std on good pixels; widen to include capped ±1σ ribbon on good pixels."""
    good = _good_pixel_mask(flux, mask)
    if not np.any(good):
        return None
    fg = flux[good]
    y_med = float(np.median(fg))
    y_std = float(np.std(fg))
    if not np.isfinite(y_std) or y_std <= 0:
        q1, q99 = np.percentile(fg, [1.0, 99.0])
        y_std = max(float(q99 - q1) / 4.0, 1e-30)
    half = float(k_std) * y_std
    if half <= 0 or not np.isfinite(half):
        half = max(abs(y_med) * 0.05, 1e-30)
    lo, hi = y_med - half, y_med + half

    if ivar is not None:
        sp = _capped_sigma_plot(flux, ivar)
        lower = flux - sp
        upper = flux + sp
        finite_band = good & np.isfinite(sp)
        if np.any(finite_band):
            lo = min(lo, float(np.nanmin(lower[finite_band])))
            hi = max(hi, float(np.nanmax(upper[finite_band])))
    return lo, hi


def plot_spectrum_with_lines(
    ax,
    wavelength: np.ndarray,
    flux: np.ndarray,
    z: float,
    *,
    mask: np.ndarray | None = None,
    ivar: np.ndarray | None = None,
    title: str = "",
    show_lines: bool = True,
    flux_label: str = "flux",
    adaptive_ylim: bool = True,
    ylim_k_std: float = 5.0,
):
    """Plot flux vs wavelength; shade masked pixels; draw shifted rest lines.

    By default, y-axis limits follow good (unmasked) pixels: median ± ``ylim_k_std`` times the
    standard deviation of flux on those pixels, expanded to include the capped ±1σ band. This
    matches the spirit of FoundationModel's ``plot_spectrum`` and keeps DESI-sized flux visible.
    """
    ax.plot(wavelength, flux, "k-", lw=0.6, label=flux_label)
    if mask is not None and mask.any():
        ax.fill_between(
            wavelength,
            flux.min(),
            flux.max(),
            where=mask,
            color="lightgray",
            alpha=0.4,
            label="masked (DESI)",
        )
    if ivar is not None:
        sigma_plot = _capped_sigma_plot(flux, ivar)
        ax.fill_between(
            wavelength, flux - sigma_plot, flux + sigma_plot, color="steelblue", alpha=0.15, label="±1σ (capped)"
        )

    if adaptive_ylim:
        ylim = _adaptive_ylim(flux, mask, ivar=ivar, k_std=ylim_k_std)
        if ylim is not None:
            lo, hi = ylim
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.set_ylim(lo, hi)

    if show_lines:
        _, ymax = ax.get_ylim()
        colors = {"emi": "crimson", "abs": "navy"}
        for w_obs, label, kind in observed_lines(z):
            if wavelength.min() <= w_obs <= wavelength.max():
                ax.axvline(w_obs, color=colors.get(kind, "gray"), ls="--", lw=0.8, alpha=0.7)
                ax.text(
                    w_obs,
                    ymax,
                    label,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=7,
                    color=colors.get(kind, "gray"),
                )

    ax.set_xlabel("observed wavelength (Å)")
    # DESI iron coadd flux is typically ~1e-17 erg/s/cm²/Å; sci axis avoids every tick reading "0.00".
    ax.set_ylabel("flux (coadd native units)")
    med = float(np.nanmedian(np.abs(flux[np.isfinite(flux)])))
    if np.isfinite(med) and 0 < med < 1e-2:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def plot_reconstruction_fm_style(
    ax,
    wavelength: np.ndarray,
    flux: np.ndarray,
    recon: np.ndarray,
    z: float,
    *,
    mask: np.ndarray | None = None,
    k_std: float = 5.0,
) -> None:
    """Physical-flux overlay (black target, red recon) with FM-style median ± k·σ y-limits."""
    good = _good_pixel_mask(flux, mask)
    ax.plot(wavelength, flux, "k-", lw=0.7, label="target")
    ax.plot(wavelength, recon, color="crimson", lw=0.7, alpha=0.9, label="recon")
    ylim = _adaptive_ylim(flux, mask, ivar=None, k_std=k_std)
    if ylim is not None:
        lo, hi = ylim
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.set_ylim(lo, hi)
    ax.set_xlabel("observed wavelength (Å)")
    ax.set_ylabel("flux")
    ax.set_title(f"z={z:.4f}")
    ax.legend(loc="upper right", fontsize=8)
