import torch
from desifm.model.blocks import EncoderLayer, MultiHeadAttention, RMSNorm


def test_rmsnorm_unit_scale():
    n = RMSNorm(32)
    x = torch.randn(2, 5, 32)
    y = n(x)
    rms = torch.sqrt((y**2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_causal_attention_differs_by_position():
    attn = MultiHeadAttention(32, 4, causal=True)
    x = torch.randn(1, 4, 32)
    y = attn(x)
    assert not torch.allclose(y[0, 0], y[0, 3])


def test_encoder_layer_shape():
    layer = EncoderLayer(64, 4, dropout=0.0)
    x = torch.randn(2, 8, 64)
    assert layer(x).shape == x.shape
