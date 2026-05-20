import torch
import pytest
from desifm.tokenization.redshift_codec import RedshiftCodec


def test_fit_encode_decode():
    c = RedshiftCodec(n_bins=64)
    c.fit(torch.linspace(0, 1.5, 200))
    z = 0.42
    idx = c.encode(z)
    z2 = c.decode(idx)
    assert abs(z2 - z) < 0.05


def test_not_fitted_raises():
    c = RedshiftCodec()
    with pytest.raises(RuntimeError):
        c.encode(0.1)


def test_encode_batch_cuda_input():
    c = RedshiftCodec(n_bins=64)
    c.fit(torch.linspace(0, 1.5, 200))
    z = torch.tensor([0.1, 0.42, 0.9], device="cuda" if torch.cuda.is_available() else "cpu")
    idx = c.encode_batch(z)
    assert idx.shape == z.shape
    assert idx.device.type == "cpu"
