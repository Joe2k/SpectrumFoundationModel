import torch
from desifm.constants import REDMASK, REDSHIFT_OFFSET, SOS, SPECTRUM_OFFSET
from desifm.model.transformer import DesiFoundationModel


def test_forward_with_loss():
    m = DesiFoundationModel(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4)
    enc = torch.randint(0, 512, (2, 12))
    dec = torch.randint(0, 512, (2, 10))
    tgt = torch.randint(0, 512, (2, 10))
    tgt[:, 0] = REDSHIFT_OFFSET + 3
    logits, loss = m(enc, dec, targets=tgt, approach="a")
    assert logits.shape[:2] == (2, 10)
    assert loss is not None


def test_approach_a_aux_increases_loss():
    m = DesiFoundationModel(d_model=64, n_enc_layers=1, n_dec_layers=1, n_heads=4)
    enc = torch.randint(0, 512, (2, 12))
    dec = torch.randint(0, 512, (2, 8))
    tgt = torch.full((2, 8), REDSHIFT_OFFSET + 1, dtype=torch.long)
    _, l0 = m(enc, dec, targets=tgt, aux_z_weight=0.0, approach="a")
    _, l1 = m(enc, dec, targets=tgt, aux_z_weight=1.0, approach="a")
    assert l1.item() >= l0.item()
