#!/usr/bin/env python3
"""Precompute AION spectrum tokens for a manifest (run once on GPU; then train with aion-cached).

Supports multi-GPU via torchrun / ``python -m torch.distributed.run``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.aion_cache import build_aion_token_cache, cache_is_complete, load_cache_meta
from desifm.training.distributed import cleanup_distributed, is_main_process, setup_distributed
from desifm.training.env import load_project_env
from desifm.training.paths import require_scratch_manifest


def setup_logging(rank: int, world_size: int) -> logging.Logger:
    fmt = logging.Formatter(
        f"%(asctime)s | rank {rank}/{world_size} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if rank == 0:
        for name in ("cache_aion_tokens", "desifm.tokenization.aion_bridge"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.handlers.clear()
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(fmt)
            lg.addHandler(h)
    return logging.getLogger("cache_aion_tokens")


def main():
    load_project_env()
    p = argparse.ArgumentParser(
        description="Build AION token cache for train_model.py --spectrum-tokenizer aion-cached"
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--token-cache",
        type=Path,
        required=True,
        help="Output directory (e.g. $SCRATCH/token_cache/dr1_10k)",
    )
    p.add_argument("--batch-size", type=int, default=16, help="Encode batch size per GPU rank")
    p.add_argument("--device", default="cuda", help="Ignored under torchrun (uses LOCAL_RANK)")
    p.add_argument("--aion-hf-repo", default="polymathic-ai/aion-base")
    p.add_argument("--max-spectra", type=int, default=None, help="Limit for smoke tests")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if cache exists")
    p.add_argument(
        "--log-every",
        type=int,
        default=256,
        help="Log progress every N spectra encoded per rank (0=only start/end)",
    )
    p.add_argument(
        "--reuse-indices",
        action="store_true",
        help="Skip index scan if valid_indices.npy exists in --token-cache",
    )
    args = p.parse_args()

    try:
        import aion  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            'Missing polymathic-aion. Install with: pip install -e ".[aion]"'
        ) from e

    import torch
    import torch.distributed as dist

    rank, world_size, _local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)
    log = setup_logging(rank, world_size)

    manifest = require_scratch_manifest(args.manifest)
    cache_dir = Path(args.token_cache)

    skip = False
    if main_proc and cache_is_complete(cache_dir) and not args.overwrite:
        meta = load_cache_meta(cache_dir)
        log.info("cache already exists: %s (n=%s)", cache_dir, meta.get("n_spectra"))
        skip = True

    if world_size > 1 and dist.is_initialized():
        flag = torch.tensor([1 if skip else 0], device=device)
        dist.broadcast(flag, src=0)
        skip = bool(int(flag.item()))

    if skip:
        if main_proc:
            print(f"OK existing cache -> {cache_dir}", flush=True)
        cleanup_distributed()
        return

    if main_proc and cache_dir.exists() and args.overwrite:
        log.info("overwriting cache at %s", cache_dir)

    if main_proc:
        log.info("manifest=%s", manifest)
        log.info("cache_dir=%s batch_size=%d world_size=%d", cache_dir, args.batch_size, world_size)
        if world_size > 1:
            log.info("DDP: one log stream from rank 0; ranks 1-%d run quietly", world_size - 1)
    sys.stdout.flush()

    t0 = time.perf_counter()
    meta = build_aion_token_cache(
        manifest,
        cache_dir,
        batch_size=args.batch_size,
        device=device,
        max_spectra=args.max_spectra,
        aion_hf_repo=args.aion_hf_repo,
        rank=rank,
        world_size=world_size,
        log_every=args.log_every,
        reuse_indices=args.reuse_indices,
        log=log if main_proc else None,
    )
    elapsed = time.perf_counter() - t0
    if main_proc:
        print(
            f"OK cached {meta['n_spectra']} spectra -> {cache_dir} "
            f"({elapsed:.1f}s, world_size={world_size})",
            flush=True,
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
