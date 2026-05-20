#!/usr/bin/env python3
"""Download AION DESI spectrum codec weights to HF cache (run on NERSC **login node**)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.data.dr1_stream import collate_spectra
from desifm.tokenization.aion_bridge import AionSpectrumTokenizer
from desifm.training.env import load_project_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prefetch_aion")


def main() -> None:
    import os

    load_project_env()
    log.info("HF_HOME=%s", os.environ.get("HF_HOME", "(default)"))
    log.info("HUGGING_FACE_HUB_TOKEN set=%s", bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s — running one synthetic encode to pull DESI codec weights", device)

    ds = SyntheticSpectrumDataset(n_spectra=2, length=4096)
    batch = collate_spectra([ds[0], ds[1]])
    t0 = time.time()
    tok = AionSpectrumTokenizer(device)
    spec_idx, meta = tok.encode_batch(batch)
    log.info("done shape=%s n_unique=%d meta=%s elapsed=%.1fs", tuple(spec_idx.shape), spec_idx.unique().numel(), meta, time.time() - t0)
    log.info("weights should now be in HF_HOME; re-run smoke on GPU nodes")


if __name__ == "__main__":
    main()
