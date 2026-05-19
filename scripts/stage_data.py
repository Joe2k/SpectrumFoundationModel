#!/usr/bin/env python3
"""Copy FITS from CFS manifest paths to SCRATCH; write *_scratch.jsonl."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_if_needed(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-manifest", type=Path, required=True)
    p.add_argument("--dst-root", type=Path, required=True)
    p.add_argument("--dst-manifest", type=Path, required=True)
    p.add_argument("--src-prefix", default="/global/cfs/cdirs/desi/public/dr1/")
    args = p.parse_args()

    records = [json.loads(l) for l in args.src_manifest.read_text().splitlines() if l.strip()]
    out = []
    copied = 0
    for rec in records:
        new = dict(rec)
        for key in ("coadd", "redrock"):
            src = Path(rec[key])
            rel = src.relative_to(args.src_prefix) if str(src).startswith(args.src_prefix) else Path(src.name)
            dst = args.dst_root / rel
            if copy_if_needed(src, dst):
                copied += 1
            new[key] = str(dst)
        out.append(new)
    args.dst_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.dst_manifest.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"staged {copied} new files, {len(out)} records -> {args.dst_manifest}")


if __name__ == "__main__":
    main()
