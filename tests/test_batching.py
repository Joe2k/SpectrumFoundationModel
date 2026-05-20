import torch
from desifm.constants import REDMASK, REDSHIFT_OFFSET, SOS, SPECTRUM_OFFSET
from desifm.training.batching import build_sequences, tokenize_batch
from desifm.tokenization.redshift_codec import RedshiftCodec


class FakeCodec:
    def encode(self, x):
        B = x.shape[0]
        return torch.arange(B * 4).view(B, 4) % 16, torch.ones(B)


class FakeZ:
    def encode(self, z):
        return 3


def test_approach_b_redmask_decoder():
    spec = torch.tensor([[0, 1, 2, 3]]) + 8  # dummy, build uses indices+offset inside
    # use build_sequences properly
    spec_idx = torch.tensor([[0, 1, 2, 3]])
    z_idx = torch.tensor([3])
    enc, dec, tgt, _ = build_sequences(spec_idx, z_idx, "b")
    assert dec[0, 1].item() == REDMASK
    assert enc[0, 0].item() == SOS
    assert REDSHIFT_OFFSET not in enc[0].tolist()


def test_approach_a_has_redshift_in_encoder():
    spec_idx = torch.tensor([[0, 1, 2, 3]])
    z_idx = torch.tensor([3])
    enc, dec, tgt, _ = build_sequences(spec_idx, z_idx, "a")
    assert enc[0, 1].item() == REDSHIFT_OFFSET + 3


class MockEncodeBatch:
    def encode_batch(self, batch):
        B = batch["flux"].shape[0]
        T = 4
        return torch.arange(B * T).view(B, T) % 16, {}


def test_tokenize_batch_encode_batch_backend():
    batch = {
        "flux": torch.rand(2, 32),
        "ivar": torch.ones(2, 32),
        "mask": torch.zeros(2, 32, dtype=torch.bool),
        "wavelength": torch.linspace(3600, 9800, 32).unsqueeze(0).expand(2, -1),
        "z": torch.tensor([0.1, 0.5]),
    }
    zc = RedshiftCodec(n_bins=16)
    zc.fit(torch.linspace(0, 1, 20))
    spec_idx, z_idx = tokenize_batch(batch, MockEncodeBatch(), zc, torch.device("cpu"))
    _, _, tgt, _ = build_sequences(spec_idx, z_idx, "b")
    assert (tgt[:, 1:5] >= SPECTRUM_OFFSET).all()
