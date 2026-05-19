import torch
from desifm.training.codec_input import (
    denormalize_spectrum_output,
    masked_recon_loss,
    normalize_spectrum_input,
    prepare_codec_batch,
)


def test_normalize_roundtrip():
    flux = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    ivar = torch.ones_like(flux) * 5.0
    mask = torch.zeros_like(flux, dtype=torch.bool)
    x, denorm = normalize_spectrum_input(flux, ivar, mask)
    back = denormalize_spectrum_output(x, denorm)
    assert torch.allclose(back[:, 0], flux, rtol=0.05, atol=0.5)


def test_masked_pixels_excluded_from_norm():
    flux = torch.tensor([[100.0, 100.0, 1e6]])
    ivar = torch.ones_like(flux)
    mask = torch.tensor([[False, False, True]])
    x, denorm = normalize_spectrum_input(flux, ivar, mask)
    assert denorm.item() < 200.0


def test_prepare_codec_batch():
    batch = {
        "flux": torch.rand(2, 32),
        "ivar": torch.ones(2, 32),
        "mask": torch.zeros(2, 32, dtype=torch.bool),
    }
    x, denorm, mask = prepare_codec_batch(batch)
    assert x.shape == (2, 2, 32)
    assert denorm.shape == (2,)


def test_masked_recon_loss():
    pred = torch.zeros(2, 10)
    target = torch.ones(2, 10)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 0] = True
    loss = masked_recon_loss(pred, target, mask)
    assert loss.item() > 0
