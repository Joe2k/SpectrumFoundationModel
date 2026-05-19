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
