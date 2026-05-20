import pytest
import torch

from desifm.constants import N_LATENT_TOKENS, N_SPECTRUM_CODES, SPECTRUM_OFFSET
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.data.dr1_stream import collate_spectra
from desifm.training.batching import build_sequences, tokenize_batch
from desifm.tokenization.redshift_codec import RedshiftCodec

aion = pytest.importorskip("aion")

from desifm.tokenization.aion_bridge import AionSpectrumTokenizer


@pytest.mark.aion
def test_encode_batch_shape_and_range():
    ds = SyntheticSpectrumDataset(n_spectra=4, length=4096, seed=0)
    batch = collate_spectra([ds[i] for i in range(2)])
    tok = AionSpectrumTokenizer("cpu")
    spec_idx, meta = tok.encode_batch(batch)
    assert spec_idx.shape == (2, N_LATENT_TOKENS)
    assert spec_idx.min() >= 0
    assert spec_idx.max() < N_SPECTRUM_CODES
    assert spec_idx.unique().numel() > 1
    assert "resampled_to_grid" in meta


@pytest.mark.aion
def test_tokenize_batch_with_aion_offset():
    ds = SyntheticSpectrumDataset(n_spectra=4, length=2048, seed=1)
    batch = collate_spectra([ds[0], ds[1]])
    tok = AionSpectrumTokenizer("cpu")
    zc = RedshiftCodec(n_bins=32)
    zc.fit(torch.linspace(0, 1, 50))
    spec_idx, z_idx = tokenize_batch(batch, tok, zc, torch.device("cpu"))
    enc, dec, tgt, _ = build_sequences(spec_idx, z_idx, "b")
    assert dec[0, 1].item() == 4  # REDMASK
    assert (tgt[:, 1 : 1 + N_LATENT_TOKENS] >= SPECTRUM_OFFSET).all()
