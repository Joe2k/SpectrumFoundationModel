#!/usr/bin/env python3
"""Download DR1 iron healpix tiles from https://data.desi.lbl.gov/public/dr1/ into ./data/dr1_public/ (default: 1 tile)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.public_dr1 import (
    IRON_TILE_CATALOG,
    discover_public_training_tiles,
    ensure_dr1_tiles_local,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--data-root", type=Path, default=None, help="Default: <repo>/data/dr1_public")
    p.add_argument("--manifest", type=Path, default=None, help="Default: <repo>/data/manifests/local_dr1.jsonl")
    p.add_argument("--max-tiles", type=int, default=1, help="Number of healpix tiles (each: coadd + redrock). Default 1 = minimal disk.")
    p.add_argument(
        "--training-order",
        action="store_true",
        help="Pick tiles in NERSC build_dr1_index walk order (first available on public portal).",
    )
    args = p.parse_args()

    data_root = args.data_root or (args.repo_root / "data" / "dr1_public")
    default_manifest = (
        "train_eval_dr1.jsonl" if args.training_order else "local_dr1.jsonl"
    )
    manifest = args.manifest or (args.repo_root / "data" / "manifests" / default_manifest)
    if args.training_order:
        tiles = discover_public_training_tiles(max(1, args.max_tiles))
    else:
        tiles = IRON_TILE_CATALOG[: max(1, args.max_tiles)]

    ensure_dr1_tiles_local(data_root, manifest, tiles)
    print(f"wrote {len(tiles)} tiles under {data_root}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
