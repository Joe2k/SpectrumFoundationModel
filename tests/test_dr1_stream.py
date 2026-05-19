from pathlib import Path

from desifm.data.dr1_stream import healpix_split, val_healpix_ids


def test_val_healpix_ids_matches_split(tmp_path: Path):
    records = [{"healpix": i} for i in range(20)]
    train, val = healpix_split(records, holdout=0.05, seed=42)
    val_hp = set(val_healpix_ids(records, holdout=0.05, seed=42))
    assert val_hp == {r["healpix"] for r in val}
    assert len(train) + len(val) == len(records)
