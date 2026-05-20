import torch

from desifm.tokenization.spectrum_codec_v5 import LFQuantizerV5, SpectrumCodecV5
from desifm.training.codec_input import INPUT_STYLE_V4, prepare_codec_batch_for_style


def test_spectrum_codec_v5_forward_shapes():
    flux = torch.rand(2, 8704) + 0.5
    ivar = torch.ones(2, 8704)
    mask = torch.zeros(2, 8704, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    x, denorm, m = prepare_codec_batch_for_style(batch, INPUT_STYLE_V4)
    model = SpectrumCodecV5(commitment_weight=0.05)
    out = model(x, denorm, m, loss_profile="fm", lambda_phys=1.0, lambda_diversity=1.0)
    assert out["indices"].shape[0] == 2
    assert out["recon"].shape == x.shape
    assert out["recon_phys"].shape[0] == 2
    assert out["loss"].item() > 0
    assert "diversity_loss" in out
    assert model.quant.n_codes == 1024  # 2**10 latent_dim


def test_spectrum_codec_v5_native_padded_length():
    """Training batches are padded below GRID_SIZE; forward must resize for losses."""
    from desifm.constants import GRID_SIZE

    flux = torch.rand(2, 7781) + 0.5
    ivar = torch.ones(2, 7781)
    mask = torch.zeros(2, 7781, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    x, denorm, m = prepare_codec_batch_for_style(batch, INPUT_STYLE_V4)
    assert x.shape[-1] == 7781
    model = SpectrumCodecV5(commitment_weight=0.05)
    out = model(x, denorm, m, loss_profile="fm", lambda_phys=0.5, lambda_diversity=1.0)
    assert out["recon"].shape[-1] == GRID_SIZE
    assert out["recon_phys"].shape[-1] == GRID_SIZE
    assert out["target_phys"].shape[-1] == GRID_SIZE


def test_lfquantizer_v5_temperature_softens_quant():
    z = torch.randn(2, 10, 16, requires_grad=True)
    q = LFQuantizerV5(dim=10)
    _, loss_hot, _, _ = q(z, quant_temperature=2.0)
    _, loss_cold, _, _ = q(z, quant_temperature=0.1)
    loss_hot.backward()
    g_hot = z.grad.abs().mean().item()
    z.grad = None
    loss_cold.backward()
    g_cold = z.grad.abs().mean().item()
    assert g_hot > 0
    assert g_cold > 0


def test_spectrum_codec_v5_fm_backward():
    flux = torch.rand(2, 8704, requires_grad=False) + 0.5
    ivar = torch.ones(2, 8704)
    mask = torch.zeros(2, 8704, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    x, denorm, m = prepare_codec_batch_for_style(batch, INPUT_STYLE_V4)
    model = SpectrumCodecV5(commitment_weight=0.05)
    out = model(x, denorm, m, loss_profile="fm", lambda_phys=0.5, lambda_diversity=1.0)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)
