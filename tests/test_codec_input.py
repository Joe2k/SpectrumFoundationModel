import torch
from desifm.training.codec_input import (
    denormalize_spectrum_output,
    masked_recon_loss,
    normalize_spectrum_input,
    normalize_spectrum_median_v2,
    prepare_codec_batch,
    prepare_codec_batch_for_style,
    prepare_codec_v2_linear,
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


def test_codec_v2_linear_roundtrip():
    flux = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    ivar = torch.ones_like(flux) * 5.0
    x, scale = prepare_codec_v2_linear(flux, ivar)
    back = x[:, 0] * scale.unsqueeze(-1)
    assert torch.allclose(back, flux, rtol=1e-5, atol=1e-5)


def test_median_v2_flux_roundtrip():
    torch.manual_seed(0)
    flux = torch.rand(2, 128) * 2.0 + 0.2
    ivar = torch.ones(2, 128) * 4.0
    mask = torch.zeros(2, 128, dtype=torch.bool)
    x, denorm = normalize_spectrum_median_v2(flux, ivar, mask)
    back = denormalize_spectrum_output(x, denorm)
    assert torch.allclose(back[:, 0], flux, rtol=1e-5, atol=1e-5)


def test_prepare_codec_batch_for_style():
    flux = torch.rand(2, 32) + 0.5
    ivar = torch.ones(2, 32)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    x3, d3, _ = prepare_codec_batch_for_style(batch, "mask_arcsinh_v3")
    x2, d2, _ = prepare_codec_batch_for_style(batch, "median_v2")
    assert x3.shape == (2, 2, 32) and d3.shape == (2,)
    assert x2.shape == (2, 2, 32) and d2.shape == (2,)


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
