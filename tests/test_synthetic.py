from desifm.data.synthetic import SyntheticSpectrumDataset


def test_synthetic_len_and_keys():
    ds = SyntheticSpectrumDataset(n_spectra=10, length=128)
    assert len(ds) == 10
    item = ds[0]
    assert item["flux"].shape == (128,)
    assert "mask" in item
    assert "z" in item
