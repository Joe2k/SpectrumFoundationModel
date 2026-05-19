import torch
from desifm.training.codec_input import prepare_codec_input


def test_median_normalization_stabilizes_scale():
    flux = torch.tensor([[1e4, 2e4, 3e4], [1.0, 2.0, 3.0]])
    ivar = torch.ones_like(flux)
    x = prepare_codec_input(flux, ivar)
    medians = x[:, 0].abs().median(dim=-1).values
    assert torch.allclose(medians, torch.ones(2), atol=0.01)
