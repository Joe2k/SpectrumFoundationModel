import torch

from desifm.constants import GRID_SIZE
from desifm.tokenization.aion_grid import resample_batch_1d, resample_spectrum_batch


def test_resample_batch_1d_changes_length():
    x = torch.randn(2, 100)
    y = resample_batch_1d(x, GRID_SIZE)
    assert y.shape == (2, GRID_SIZE)


def test_resample_spectrum_batch_noop_at_grid():
    B, L = 2, GRID_SIZE
    flux = torch.rand(B, L)
    ivar = torch.ones(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    wave = torch.linspace(3600, 9800, L).unsqueeze(0).expand(B, -1)
    f, i, m, w, did = resample_spectrum_batch(flux, ivar, mask, wave, length=GRID_SIZE)
    assert not did
    assert f.shape == flux.shape


def test_resample_spectrum_batch_from_short():
    B, L = 2, 512
    flux = torch.rand(B, L)
    ivar = torch.ones(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    wave = torch.linspace(3600, 9800, L).unsqueeze(0).expand(B, -1)
    f, i, m, w, did = resample_spectrum_batch(flux, ivar, mask, wave, length=GRID_SIZE)
    assert did
    assert f.shape == (B, GRID_SIZE)
    assert w.shape == (B, GRID_SIZE)
