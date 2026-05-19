import torch
from desifm.tokenization.spectrum_codec import LFQuantizer, SpectrumCodec


def test_lfq_binary_codes():
    q = LFQuantizer(dim=4, n_codes=16)
    z = torch.randn(2, 4, 8)
    idx = q.encode(z)
    zq = q.decode(idx)
    assert set(zq.unique().tolist()).issubset({-1.0, 1.0})


def test_codec_forward():
    m = SpectrumCodec(widths=(32, 64, 64, 64))
    x = torch.randn(2, 2, 4000) * 0.1
    denorm = torch.ones(2) * 10.0
    out = m(x, denorm)
    assert out["recon"].shape[0] == 2
    assert out["loss"].ndim == 0
    assert out["indices"].shape[1] == m.n_tokens
