import torch

from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.codec_eval import audit_code_usage, input_style_from_checkpoint
from desifm.training.codec_input import INPUT_STYLE_V4, prepare_codec_batch_v4
from desifm.training.codec_loss import batch_codebook_entropy_loss, code_usage_stats


def test_input_style_from_checkpoint():
    assert input_style_from_checkpoint({"input_style": "mask_arcsinh_v3"}) == "mask_arcsinh_v3"
    assert input_style_from_checkpoint({}) == "codec_v2_linear"
    assert input_style_from_checkpoint({"input_style": None}) == "codec_v2_linear"
    assert input_style_from_checkpoint({"input_style": "mask_arcsinh_v5"}) == "mask_arcsinh_v5"


def test_code_usage_stats_keys_and_bounds():
    indices = torch.randint(0, 8, (4, 32))
    stats = code_usage_stats(indices, n_codes=256)
    assert set(stats.keys()) >= {
        "n_unique",
        "n_codes",
        "usage_fraction",
        "entropy_penalty",
        "batch_entropy_penalty",
        "top_codes",
        "per_row_n_unique",
    }
    assert 1 <= stats["n_unique"] <= 8
    assert stats["n_codes"] == 256
    assert 0.0 < stats["usage_fraction"] <= 1.0


def test_batch_entropy_collapsed_high():
    flat = torch.zeros(32, dtype=torch.long)
    assert batch_codebook_entropy_loss(flat, n_bins=256).item() > 0.9


def test_audit_code_usage_smoke():
    flux = torch.rand(2, 128) + 0.5
    ivar = torch.ones(2, 128)
    mask = torch.zeros(2, 128, dtype=torch.bool)
    batch = {"flux": flux, "ivar": ivar, "mask": mask}
    model = SpectrumCodec(commitment_weight=0.05)
    stats = audit_code_usage(
        model,
        batch,
        INPUT_STYLE_V4,
        torch.device("cpu"),
        lambda_entropy=0.1,
        use_batch_entropy=True,
    )
    assert stats["n_codes"] == model.quant.n_codes
    assert "usage_fraction" in stats
    assert "top_codes" in stats
    assert stats["recon_loss"] >= 0
