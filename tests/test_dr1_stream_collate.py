import pytest
import torch

from desifm.data.dr1_stream import collate_spectra


def test_collate_includes_wavelength_padded():
    short = {
        "flux": torch.ones(10),
        "ivar": torch.ones(10),
        "mask": torch.zeros(10, dtype=torch.bool),
        "wavelength": torch.linspace(3600.0, 4000.0, 10),
        "z": torch.tensor(0.5),
    }
    long = {
        "flux": torch.ones(20),
        "ivar": torch.ones(20),
        "mask": torch.zeros(20, dtype=torch.bool),
        "wavelength": torch.linspace(3600.0, 5000.0, 20),
        "z": torch.tensor(0.3),
    }
    batch = collate_spectra([short, long])
    assert batch["wavelength"].shape == (2, 20)
    assert batch["flux"].shape == (2, 20)
    assert batch["wavelength"][0, 9].item() == pytest.approx(4000.0)
    assert batch["wavelength"][0, 19].item() == pytest.approx(4000.0)


def test_collate_wavelength_pad_repeat():
    short = {
        "flux": torch.ones(5),
        "ivar": torch.ones(5),
        "mask": torch.zeros(5, dtype=torch.bool),
        "wavelength": torch.tensor([3600.0, 3700.0, 3800.0, 3900.0, 4000.0]),
        "z": torch.tensor(0.1),
    }
    long = {
        "flux": torch.ones(8),
        "ivar": torch.ones(8),
        "mask": torch.zeros(8, dtype=torch.bool),
        "wavelength": torch.linspace(3600.0, 4300.0, 8),
        "z": torch.tensor(0.2),
    }
    batch = collate_spectra([short, long])
    assert batch["wavelength"][0, 7].item() == pytest.approx(4000.0)
