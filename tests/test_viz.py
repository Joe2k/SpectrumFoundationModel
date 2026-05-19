"""Tests for spectrum visualization helpers."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from desifm.viz.spectrum_plot import (
    REST_LINES,
    _adaptive_ylim,
    _capped_sigma_plot,
    observed_lines,
    plot_spectrum_with_lines,
)


def test_observed_lines_redshift():
    z = 0.5
    lines = observed_lines(z)
    assert len(lines) == len(REST_LINES)
    w_rest, label, kind = REST_LINES[0]
    w_obs, label2, kind2 = lines[0]
    assert label == label2 and kind == kind2
    assert abs(w_obs - w_rest * 1.5) < 1e-6


def test_adaptive_ylim_uses_good_pixels_only():
    wave = np.linspace(3500, 9800, 500)
    flux = 2e-17 + 0.3e-17 * np.sin(wave / 500.0)
    mask = np.zeros_like(flux, dtype=bool)
    mask[100:120] = True
    flux[100:120] = 1e-15  # huge spike on masked pixels — should not set y window
    ivar = np.full_like(flux, 1e34)
    ylim = _adaptive_ylim(flux, mask, ivar=ivar, k_std=5.0)
    assert ylim is not None
    lo, hi = ylim
    fg = flux[~mask]
    med, std = float(np.median(fg)), float(np.std(fg))
    assert lo <= med <= hi
    assert hi - lo <= 20.0 * std + 1e-18


def test_capped_sigma_plot_zero_ivar_no_runtime_warning():
    flux = np.array([1e-17, 2e-17, 3e-17])
    ivar = np.array([1e32, 0.0, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sigma = _capped_sigma_plot(flux, ivar)
    assert np.isfinite(sigma[0])
    assert np.isnan(sigma[1])
    assert np.isnan(sigma[2])


def test_plot_spectrum_with_lines_sets_reasonable_ylim():
    wave = np.linspace(3500, 9800, 400)
    flux = 2e-17 + 0.2e-17 * np.sin(wave / 400.0) + np.random.default_rng(0).normal(0, 0.02e-17, size=wave.shape)
    ivar = np.full_like(flux, 1e32)
    ivar[50:60] = 1e-12  # absurd formal errors on a few pixels (not masked)
    fig, ax = plt.subplots()
    plot_spectrum_with_lines(ax, wave, flux, z=0.1, mask=None, ivar=ivar, show_lines=False)
    lo, hi = ax.get_ylim()
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
    # Without adaptive y-limits, a few tiny-ivar pixels can drive Matplotlib to ~O(1) span.
    assert hi - lo < 1e-15
