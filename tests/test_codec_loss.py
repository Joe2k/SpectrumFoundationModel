import torch
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.codec_input import prepare_codec_batch_v4
from desifm.training.codec_loss import (
    batch_codebook_entropy_loss,
    code_usage_stats,
    flux_rms,
    flux_std_ratio,
    flux_std_ratio_per_sample,
    latent_bit_balance_loss,
    latent_index_entropy_penalty,
    physical_flux_loss,
    top_hat_smooth_flux,
)


def test_top_hat_smooth_preserves_shape():
    flux = torch.randn(2, 64)
    out = top_hat_smooth_flux(flux, 5)
    assert out.shape == flux.shape


def test_entropy_penalty_collapsed_low():
    indices = torch.zeros(4, 32, dtype=torch.long)
    assert latent_index_entropy_penalty(indices).item() > 0.9


def test_code_usage_gate_bins_v5():
    from desifm.training.codec_loss import code_usage_gate_bins, code_usage_passes_gate

    assert code_usage_gate_bins(1024) == 256
    assert code_usage_passes_gate(77, 1024, 0.3)
    assert not code_usage_passes_gate(60, 1024, 0.3)


def test_code_usage_stats_empty():
    stats = code_usage_stats(torch.zeros(0, dtype=torch.long), n_codes=256)
    assert stats["n_unique"] == 0
    assert stats["usage_fraction"] == 0.0


def test_latent_bit_balance_collapsed_high():
    z = torch.full((4, 8, 32), 5.0)
    assert latent_bit_balance_loss(z).item() > 0.9


def test_latent_bit_balance_spread_lower():
    z = torch.randn(8, 10, 64)
    collapsed = latent_bit_balance_loss(torch.full((8, 10, 64), 4.0))
    spread = latent_bit_balance_loss(z)
    assert spread.item() < collapsed.item()


def test_latent_bit_balance_grad_flows():
    z = torch.randn(4, 10, 32, requires_grad=True)
    loss = latent_bit_balance_loss(z)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum().item() > 0


def test_batch_entropy_uniform_lower_than_collapsed():
    collapsed = batch_codebook_entropy_loss(torch.zeros(64, dtype=torch.long), n_bins=256)
    spread = batch_codebook_entropy_loss(torch.arange(256, dtype=torch.long).repeat(4), n_bins=256)
    assert spread.item() < collapsed.item()


def test_entropy_penalty_uniform_lower():
    idx = torch.arange(256, dtype=torch.long).repeat(4, 8)
    collapsed = latent_index_entropy_penalty(torch.zeros(4, 32, dtype=torch.long))
    spread = latent_index_entropy_penalty(idx)
    assert spread.item() < collapsed.item()


def test_physical_flux_loss_positive():
    flux = torch.rand(2, 128) + 0.5
    ivar = torch.ones(2, 128)
    mask = torch.zeros(2, 128, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    x, denorm, m = prepare_codec_batch_v4(batch)
    model = SpectrumCodec(commitment_weight=0.05)
    out = model(x, denorm, m, lambda_phys=0.5, lambda_entropy=0.1)
    assert out["phys_loss"].item() >= 0
    assert out["entropy_loss"].item() >= 0
    assert out["loss"].item() > out["recon_loss"].item()


def test_flux_metrics():
    target = torch.sin(torch.linspace(0, 10, 100)).unsqueeze(0)
    pred = target + 0.1 * torch.randn_like(target)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    assert flux_rms(pred, target, mask).item() > 0
    ratio = flux_std_ratio(pred, target, mask).item()
    assert 0.5 < ratio < 1.5


def test_flux_metrics_grid_vs_native_mask():
    """Val batches: physical flux on 8704 grid, mask at native L."""
    target = torch.randn(2, 8704)
    pred = target + 0.05 * torch.randn_like(target)
    mask = torch.zeros(2, 7781, dtype=torch.bool)
    assert flux_rms(pred, target, mask).item() >= 0
    assert flux_std_ratio(pred, target, mask).item() > 0


def test_flux_std_ratio_per_sample_vs_pooled():
    target = torch.stack(
        [
            torch.sin(torch.linspace(0, 8, 64)),
            torch.ones(64) * 3.0,
        ]
    )
    pred = target.clone()
    pred[1] = 1.0  # collapsed second spectrum
    mask = torch.zeros(2, 64, dtype=torch.bool)
    per = flux_std_ratio_per_sample(pred, target, mask)
    assert per.shape == (2,)
    assert per[0].item() > 0.9
    assert per[1].item() < 0.1
    pooled = flux_std_ratio(pred, target, mask).item()
    assert pooled > per[1].item()  # batch pooling masks per-spec collapse
