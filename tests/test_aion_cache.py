"""AION token cache dataset and collate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from desifm.constants import N_LATENT_TOKENS
from desifm.data.aion_cache import (
    CACHE_VERSION,
    CachedAionDataset,
    cache_is_complete,
    collate_cached,
    load_cache_meta,
    merge_valid_indices,
)
from desifm.training.batching import tokenize_batch
from desifm.tokenization.redshift_codec import RedshiftCodec


def _write_fake_cache(cache_dir: Path, n: int = 8) -> None:
    """Write raw memmap arrays (same layout as build_aion_token_cache, not np.save headers)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    codes = np.memmap(cache_dir / "codes.npy", dtype=np.uint16, mode="w+", shape=(n, N_LATENT_TOKENS))
    codes[:] = np.random.randint(0, 1024, size=(n, N_LATENT_TOKENS), dtype=np.uint16)
    codes.flush()
    z = np.memmap(cache_dir / "z.npy", dtype=np.float32, mode="w+", shape=(n,))
    z[:] = np.linspace(0.01, 0.5, n).astype(np.float32)
    z.flush()
    # healpix 1,2 for train; 3 for val
    hp = np.memmap(cache_dir / "healpix.npy", dtype=np.int32, mode="w+", shape=(n,))
    hp[:] = np.array([1, 1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    hp.flush()
    meta = {
        "version": CACHE_VERSION,
        "manifest": "/fake/manifest.jsonl",
        "n_spectra": n,
        "n_skipped": 0,
        "n_latent_tokens": N_LATENT_TOKENS,
        "n_spectrum_codes": 1024,
        "grid_size": 8704,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta))


def test_merge_valid_indices_and_strided_rows():
    parts = [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
    valid = merge_valid_indices(parts)
    assert valid == list(range(10))
    world_size = 4
    for rank in range(world_size):
        mine = valid[rank::world_size]
        for j, ds_idx in enumerate(mine):
            global_row = rank + j * world_size
            assert ds_idx == global_row


def test_cache_roundtrip_and_collate(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_fake_cache(cache_dir)
    assert cache_is_complete(cache_dir)
    meta = load_cache_meta(cache_dir)
    assert meta["n_spectra"] == 8

    records = [
        {"healpix": 1, "coadd": "a", "redrock": "b"},
        {"healpix": 2, "coadd": "a", "redrock": "b"},
        {"healpix": 3, "coadd": "a", "redrock": "b"},
    ]
    train_ds = CachedAionDataset(cache_dir, records, holdout=0.25, seed=0, train=True)
    assert len(train_ds) == 6
    batch = collate_cached([train_ds[0], train_ds[1]])
    assert batch is not None
    assert batch["spec_idx"].shape == (2, N_LATENT_TOKENS)

    zc = RedshiftCodec(n_bins=32)
    zc.fit(torch.linspace(0, 1, 50))
    spec_idx, z_idx = tokenize_batch(batch, None, zc, torch.device("cpu"))
    assert spec_idx.shape == (2, N_LATENT_TOKENS)
