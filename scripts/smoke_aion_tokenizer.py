#!/usr/bin/env python3
"""Smoke-test official AION DESI spectrum tokenizer (encode one batch)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.constants import N_LATENT_TOKENS, N_SPECTRUM_CODES
from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, load_manifest
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.tokenization.aion_bridge import AionSpectrumTokenizer
from desifm.training.env import load_project_env


def main():
    load_project_env()
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.synthetic:
        ds = SyntheticSpectrumDataset(n_spectra=max(4, args.batch_size), length=4096)
        items = [ds[i] for i in range(args.batch_size)]
        batch = collate_spectra(items)
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        records = load_manifest(args.manifest)
        ds = DR1StreamDataset(records, max_spectra=args.batch_size * 4)
        items = []
        for i in range(len(ds)):
            it = ds[i]
            if it is not None:
                items.append(it)
            if len(items) >= args.batch_size:
                break
        if len(items) < 1:
            raise SystemExit("no valid spectra from manifest")
        batch = collate_spectra(items)

    device = torch.device(args.device)
    tok = AionSpectrumTokenizer(device)
    spec_idx, meta = tok.encode_batch(batch)

    print(f"shape={tuple(spec_idx.shape)} (expected B,{N_LATENT_TOKENS})")
    print(f"min={int(spec_idx.min())} max={int(spec_idx.max())} (codes 0..{N_SPECTRUM_CODES - 1})")
    print(f"n_unique={int(spec_idx.unique().numel())}")
    print(f"meta={meta}")
    assert spec_idx.shape[1] == N_LATENT_TOKENS
    assert spec_idx.max() < N_SPECTRUM_CODES
    print("OK")


if __name__ == "__main__":
    main()
