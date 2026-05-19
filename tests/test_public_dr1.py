from desifm.data.public_dr1 import (
    IRON_TILE_CATALOG,
    PUBLIC_DR1_BASE_DEFAULT,
    iron_tile_rel_paths,
    public_url,
)


def test_iron_tile_rel_paths():
    c, r = iron_tile_rel_paths("main", "dark", "0", 0)
    assert c.endswith("coadd-main-dark-0.fits")
    assert r.endswith("redrock-main-dark-0.fits")
    assert "healpix/main/dark/0/0/" in c


def test_public_url():
    u = public_url("spectro/redux/iron/healpix/main/dark/0/0/coadd-main-dark-0.fits")
    assert u.startswith(PUBLIC_DR1_BASE_DEFAULT)
    assert u.endswith("coadd-main-dark-0.fits")


def test_iron_tile_catalog_nonempty():
    assert len(IRON_TILE_CATALOG) >= 1
