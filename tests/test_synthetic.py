from desifm.data.synthetic import SyntheticSpectrumDataset


def test_synthetic_len_and_keys():
    ds = SyntheticSpectrumDataset(n_spectra=10, length=128)
    assert len(ds) == 10
    item = ds[0]
    assert item["flux"].shape == (128,)
    assert "mask" in item
    assert "wavelength" in item
    assert item["wavelength"].shape == item["flux"].shape
    assert item["wavelength"][0] < item["wavelength"][-1]
    assert "z" in item
