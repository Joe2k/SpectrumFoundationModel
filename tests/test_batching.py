import torch
from desifm.constants import REDMASK, REDSHIFT_OFFSET, SOS
from desifm.training.batching import build_sequences


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
