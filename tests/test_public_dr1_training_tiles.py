"""Tests for training-order public DR1 tile discovery."""

from __future__ import annotations

from desifm.data import public_dr1 as pd


def test_healpix_group_candidates_low_hp():
    assert pd._healpix_group_candidates(0) == ["0"]
    assert pd._healpix_group_candidates(99) == ["0"]


def test_healpix_group_candidates_high_hp():
    assert "1" in pd._healpix_group_candidates(150)


def test_discover_public_training_tiles(monkeypatch):
    def fake_exists(survey, program, group, healpix, **kwargs):
        return survey == "main" and program == "bright" and group == "0" and healpix == 0

    monkeypatch.setattr(pd, "tile_exists_on_public", fake_exists)
    tiles = pd.discover_public_training_tiles(1)
    assert tiles == [("main", "bright", "0", 0)]


def test_group_from_manifest_record():
    rec = {
        "coadd": "/data/spectro/redux/iron/healpix/main/bright/0/42/coadd-main-bright-42.fits",
        "healpix": 42,
    }
    assert pd.group_from_manifest_record(rec) == "0"


def test_tiles_for_healpix_from_records():
    recs = [
        {
            "survey": "main",
            "program": "dark",
            "healpix": 7,
            "coadd": "/x/spectro/redux/iron/healpix/main/dark/0/7/coadd.fits",
        }
    ]
    assert pd.tiles_for_healpix([7], records=recs) == [("main", "dark", "0", 7)]


def test_tiles_for_healpix_portal_probe(monkeypatch):
    def fake_exists(survey, program, group, healpix, **kwargs):
        return survey == "main" and program == "dark" and group == "0" and healpix == 25

    monkeypatch.setattr(pd, "tile_exists_on_public", fake_exists)
    assert pd.tiles_for_healpix([25]) == [("main", "dark", "0", 25)]


def test_discover_respects_max_tiles(monkeypatch):
    def fake_exists(survey, program, group, healpix, **kwargs):
        return survey == "main" and healpix < 2

    monkeypatch.setattr(pd, "tile_exists_on_public", fake_exists)
    tiles = pd.discover_public_training_tiles(2, programs=("bright",))
    assert len(tiles) == 2
    assert tiles[0][1] == "bright" and tiles[0][3] == 0
    assert tiles[1][3] == 1
