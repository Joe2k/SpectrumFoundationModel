import json
from desifm.data.dr1_stream import healpix_split, load_manifest


def test_healpix_split_disjoint(tmp_path):
    recs = [{"healpix": i, "coadd": f"/a/{i}.fits"} for i in range(20)]
    train, val = healpix_split(recs, holdout=0.25, seed=0)
    train_hp = {r["healpix"] for r in train}
    val_hp = {r["healpix"] for r in val}
    assert train_hp.isdisjoint(val_hp)
    assert len(train_hp) + len(val_hp) == 20


def test_load_manifest(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"healpix": 1, "coadd": "/x"}\n')
    assert load_manifest(p)[0]["healpix"] == 1
