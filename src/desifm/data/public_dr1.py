"""Download DESI DR1 iron healpix coadds from the public web portal (local use only).

Base URL: https://data.desi.lbl.gov/public/dr1/

These helpers mirror the on-disk layout under the DR1 root
(``spectro/redux/iron/healpix/<survey>/<program>/<group>/<healpix>/``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

PUBLIC_DR1_BASE_DEFAULT = "https://data.desi.lbl.gov/public/dr1"

# Verified tiles on the public server (main survey, dark time, group directory "0").
# The notebook / CLI download **one tile by default** (minimal disk); use more slices from this list.
IRON_TILE_CATALOG: list[tuple[str, str, str, int]] = [
    ("main", "dark", "0", 0),
    ("main", "dark", "0", 1),
    ("main", "dark", "0", 2),
]


def iron_tile_rel_paths(survey: str, program: str, group: str, healpix: int) -> tuple[str, str]:
    """Relative paths under the DR1 root for coadd and redrock FITS."""
    hp = str(int(healpix))
    g = str(group)
    base = f"spectro/redux/iron/healpix/{survey}/{program}/{g}/{hp}"
    coadd = f"{base}/coadd-{survey}-{program}-{hp}.fits"
    redrock = f"{base}/redrock-{survey}-{program}-{hp}.fits"
    return coadd, redrock


def public_url(dr1_relative_path: str, *, public_base: str = PUBLIC_DR1_BASE_DEFAULT) -> str:
    rel = dr1_relative_path.lstrip("/")
    return f"{public_base.rstrip('/')}/{rel}"


def _download_file(url: str, dest: Path, *, timeout_seconds: float = 300.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "desifm-local-dr1/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading {url}") from e
    tmp.write_bytes(data)
    tmp.replace(dest)


def _n_rows_coadd(coadd_path: Path) -> int:
    from astropy.io import fits

    with fits.open(coadd_path, memmap=True) as h:
        return int(h["FIBERMAP"].header["NAXIS2"])


def ensure_dr1_tiles_local(
    data_root: Path,
    manifest_path: Path,
    tiles: Sequence[tuple[str, str, str, int]],
    *,
    public_base: str = PUBLIC_DR1_BASE_DEFAULT,
    timeout_seconds: float = 300.0,
) -> list[dict]:
    """Download coadd + redrock for each tile into ``data_root`` (DR1-relative tree).

    Writes ``manifest_path`` JSONL with absolute local paths. Returns the records.
    """
    data_root = data_root.resolve()
    manifest_path = manifest_path.resolve()
    records: list[dict] = []

    for survey, program, group, healpix in tiles:
        rel_coadd, rel_redrock = iron_tile_rel_paths(survey, program, group, healpix)
        dest_coadd = data_root / rel_coadd
        dest_redrock = data_root / rel_redrock
        url_coadd = public_url(rel_coadd, public_base=public_base)
        url_redrock = public_url(rel_redrock, public_base=public_base)

        if not dest_coadd.is_file():
            print(f"downloading {url_coadd}")
            _download_file(url_coadd, dest_coadd, timeout_seconds=timeout_seconds)
        else:
            print(f"skip (exists): {dest_coadd}")

        if not dest_redrock.is_file():
            print(f"downloading {url_redrock}")
            _download_file(url_redrock, dest_redrock, timeout_seconds=timeout_seconds)
        else:
            print(f"skip (exists): {dest_redrock}")

        n_rows = _n_rows_coadd(dest_coadd)
        records.append(
            {
                "coadd": str(dest_coadd),
                "redrock": str(dest_redrock),
                "survey": survey,
                "program": program,
                "healpix": int(healpix),
                "n_rows": n_rows,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records
