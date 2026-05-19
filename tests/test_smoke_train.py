"""End-to-end smoke: one training step on random spectra."""

import torch
from desifm.model.transformer import DesiFoundationModel
from desifm.tokenization.redshift_codec import RedshiftCodec
from desifm.training.batching import build_sequences, tokenize_batch


class TinyCodec:
    def encode(self, x):
        B, _, L = x.shape
        T = 8
        return torch.randint(0, 16, (B, T)), torch.zeros(B)


def test_one_step_approach_a():
    zc = RedshiftCodec(n_bins=32)
    zc.fit(torch.linspace(0, 1, 50))
    batch = {
        "flux": torch.rand(2, 64),
        "ivar": torch.ones(2, 64),
        "z": torch.tensor([0.1, 0.5]),
    }
    device = torch.device("cpu")
    spec, z = tokenize_batch(batch, TinyCodec(), zc, device)
    enc, dec, tgt, _ = build_sequences(spec, z, "a")
    model = DesiFoundationModel(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4)
    logits, loss = model(enc, dec, targets=tgt, approach="a")
    loss.backward()
    assert torch.isfinite(loss)
