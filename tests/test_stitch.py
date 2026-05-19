import numpy as np
from desifm.data.stitch import stitch_bands


def test_stitch_no_overlap():
    w1 = np.array([3600.0, 3700.0])
    w2 = np.array([4000.0, 4100.0])
    r = stitch_bands(
        [w1, w2],
        [np.ones(2), np.ones(2) * 2],
        [np.ones(2), np.ones(2)],
        [np.zeros(2, bool), np.zeros(2, bool)],
    )
    assert len(r["wavelength"]) == 4
