#!/usr/bin/env python3
"""Build JSONL manifest by walking public DR1 iron tree (index-only; FITS paths on CFS)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from astropy.io import fits
except ImportError:
    fits = None

IRON = "spectro/redux/iron/healpix"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/global/cfs/cdirs/desi/public/dr1"))
    p.add_argument("--surveys", nargs="+", default=["sv3", "main"])
    p.add_argument("--programs", nargs="+", default=["bright", "dark"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-healpix", type=int, default=None)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as fout:
        for survey in args.surveys:
            for program in args.programs:
                base = args.root / IRON / survey / program
                if not base.is_dir():
                    continue
                for grp in sorted(base.iterdir()):
                    if not grp.is_dir():
                        continue
                    for hpx in sorted(grp.iterdir()):
                        if not hpx.is_dir():
                            continue
                        coadd = hpx / f"coadd-{survey}-{program}-{hpx.name}.fits"
                        redrock = hpx / f"redrock-{survey}-{program}-{hpx.name}.fits"
                        if not coadd.is_file() or not redrock.is_file():
                            continue
                        n_rows = -1
                        if fits is not None:
                            try:
                                with fits.open(coadd, memmap=True) as h:
                                    n_rows = int(h["FIBERMAP"].header["NAXIS2"])
                            except Exception:
                                pass
                        rec = {
                            "coadd": str(coadd),
                            "redrock": str(redrock),
                            "survey": survey,
                            "program": program,
                            "healpix": int(hpx.name),
                            "n_rows": n_rows,
                        }
                        fout.write(json.dumps(rec) + "\n")
                        n += 1
                        if args.max_healpix and n >= args.max_healpix:
                            break
                    if args.max_healpix and n >= args.max_healpix:
                        break
                if args.max_healpix and n >= args.max_healpix:
                    break
    print(f"wrote {n} records -> {args.out}")


if __name__ == "__main__":
    main()
