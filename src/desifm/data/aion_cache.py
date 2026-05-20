"""Precomputed AION spectrum token cache for fast transformer training."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from desifm.constants import GRID_SIZE, N_LATENT_TOKENS, N_SPECTRUM_CODES
from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, healpix_split, load_manifest

CACHE_VERSION = 1
CODES_FILE = "codes.npy"
Z_FILE = "z.npy"
HEALPIX_FILE = "healpix.npy"
META_FILE = "meta.json"
VALID_INDICES_FILE = "valid_indices.npy"


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    root = Path(cache_dir)
    return {
        "root": root,
        "codes": root / CODES_FILE,
        "z": root / Z_FILE,
        "healpix": root / HEALPIX_FILE,
        "meta": root / META_FILE,
    }


def load_cache_meta(cache_dir: Path) -> dict[str, Any]:
    meta_path = cache_paths(cache_dir)["meta"]
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing cache metadata: {meta_path}")
    with meta_path.open() as f:
        meta = json.load(f)
    if meta.get("version") != CACHE_VERSION:
        raise ValueError(f"unsupported cache version {meta.get('version')}")
    return meta


def cache_is_complete(cache_dir: Path) -> bool:
    paths = cache_paths(cache_dir)
    return all(paths[k].is_file() for k in ("codes", "z", "healpix", "meta"))


def collate_cached(batch: list[dict]) -> dict | None:
    """Collate CachedAionDataset items (spec_idx + z only)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "spec_idx": torch.stack([b["spec_idx"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }


class CachedAionDataset(Dataset):
    """Rows from a pre-built AION token cache, filtered by manifest healpix split."""

    def __init__(
        self,
        cache_dir: Path,
        records: list[dict],
        *,
        holdout: float | None = None,
        seed: int = 42,
        train: bool = True,
    ):
        meta = load_cache_meta(cache_dir)
        paths = cache_paths(cache_dir)
        n = int(meta["n_spectra"])
        shape = (n, int(meta["n_latent_tokens"]))
        self.codes = np.memmap(paths["codes"], dtype=np.uint16, mode="r", shape=shape)
        self.z = np.memmap(paths["z"], dtype=np.float32, mode="r", shape=(n,))
        self.healpix = np.memmap(paths["healpix"], dtype=np.int32, mode="r", shape=(n,))

        if holdout is not None:
            train_rec, val_rec = healpix_split(records, holdout=holdout, seed=seed)
            allowed = {int(r["healpix"]) for r in (train_rec if train else val_rec)}
        else:
            allowed = {int(r["healpix"]) for r in records}

        self.indices = np.array(
            [i for i in range(n) if int(self.healpix[i]) in allowed],
            dtype=np.int64,
        )
        if len(self.indices) == 0:
            raise ValueError(f"no cached spectra for healpix filter (allowed={len(allowed)} tiles)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        j = int(self.indices[idx])
        return {
            "spec_idx": torch.from_numpy(self.codes[j].astype(np.int64)),
            "z": torch.tensor(float(self.z[j]), dtype=torch.float32),
        }


def merge_valid_indices(local_parts: list[list[int]]) -> list[int]:
    """Merge per-rank strided scans into global dataset order."""
    return sorted(idx for part in local_parts for idx in part)


def _dist_barrier(device: torch.device) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def _write_valid_indices(indices_path: Path, valid: list[int]) -> None:
    arr = np.array(valid, dtype=np.int64)
    mmap = np.memmap(indices_path, dtype=np.int64, mode="w+", shape=arr.shape)
    mmap[:] = arr
    mmap.flush()


def collect_valid_indices(
    ds: DR1StreamDataset,
    indices_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    device: torch.device,
    log: Any = None,
    reuse_existing: bool = False,
) -> int:
    """Scan dataset; write sorted valid row indices to ``indices_path``; return count."""
    import torch.distributed as dist

    if reuse_existing and indices_path.is_file():
        n_existing = int(np.memmap(indices_path, dtype=np.int64, mode="r").shape[0])
        if world_size > 1 and dist.is_initialized():
            n_t = torch.tensor([n_existing], dtype=torch.long, device=device)
            dist.broadcast(n_t, src=0)
            n_existing = int(n_t.item())
        if log is not None and rank == 0:
            log.info("reusing existing index file %s (n=%d)", indices_path, n_existing)
        return n_existing

    n = len(ds)
    local: list[int] = []
    t0 = time.perf_counter()
    for i in range(rank, n, world_size):
        if ds.is_valid(i):
            local.append(i)
        if log is not None and rank == 0 and world_size > 1 and (i - rank) % max(1, (n // world_size) // 20) == 0:
            log.info(
                "index scan shard 0/%d at row %d / %d (~%d rows/rank, local valid=%d)",
                world_size,
                i,
                n,
                (n + world_size - 1) // world_size,
                len(local),
            )

    if world_size > 1 and dist.is_initialized():
        gathered: list[list[int]] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(gathered, local)
        n_valid = 0
        if rank == 0:
            valid = merge_valid_indices(gathered)
            n_valid = len(valid)
            _write_valid_indices(indices_path, valid)
            if log is not None:
                log.info(
                    "index scan complete: %d valid / %d rows in %.1fs -> %s",
                    n_valid,
                    n,
                    time.perf_counter() - t0,
                    indices_path,
                )
        _dist_barrier(device)
        n_t = torch.tensor([0], dtype=torch.long, device=device)
        if rank == 0:
            n_t.fill_(n_valid)
        dist.broadcast(n_t, src=0)
        return int(n_t.item())

    _write_valid_indices(indices_path, local)
    if log is not None:
        log.info("index scan complete: %d valid / %d rows in %.1fs", len(local), n, time.perf_counter() - t0)
    return len(local)


def _warmup_aion(
    tok: Any,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    log: Any = None,
) -> None:
    """Load AION CodecManager one rank at a time (avoids 4× HF hammer on compute nodes)."""
    import torch.distributed as dist

    for r in range(world_size):
        if rank == r:
            if log is not None:
                log.info(
                    "rank %d loading AION CodecManager on %s (HF weights — can take several minutes; "
                    "run scripts/prefetch_aion_codec.py on login node if this stalls)",
                    rank,
                    device,
                )
            sys.stdout.flush()
            tok._manager()
            if log is not None:
                log.info("rank %d AION CodecManager ready", rank)
            sys.stdout.flush()
        if world_size > 1 and dist.is_initialized():
            _dist_barrier(device)


def _init_cache_files(paths: dict[str, Path], n_rows: int) -> None:
    np.memmap(paths["codes"], dtype=np.uint16, mode="w+", shape=(n_rows, N_LATENT_TOKENS))
    np.memmap(paths["z"], dtype=np.float32, mode="w+", shape=(n_rows,))
    np.memmap(paths["healpix"], dtype=np.int32, mode="w+", shape=(n_rows,))


def _log_cache_progress(
    log: Any,
    *,
    rank: int,
    world_size: int,
    local_done: int,
    local_total: int,
    global_done: int,
    global_total: int,
    t0: float,
    force: bool = False,
) -> None:
    if log is None:
        return
    if not force and local_total > 0 and local_done % max(1, local_total // 50) != 0 and local_done != local_total:
        return
    elapsed = max(time.perf_counter() - t0, 1e-6)
    local_rate = local_done / elapsed
    global_rate = global_done / elapsed
    local_eta = (local_total - local_done) / local_rate if local_rate > 0 else 0.0
    global_eta = (global_total - global_done) / global_rate if global_rate > 0 else 0.0
    log.info(
        "cached %d / %d (%.2f spec/s, ETA %.0fs)",
        global_done,
        global_total,
        global_rate,
        global_eta,
    )


def build_aion_token_cache(
    manifest: Path,
    cache_dir: Path,
    *,
    batch_size: int = 16,
    device: torch.device | str = "cuda",
    max_spectra: int | None = None,
    aion_hf_repo: str = "polymathic-ai/aion-base",
    rank: int = 0,
    world_size: int = 1,
    log_every: int = 256,
    log_heartbeat_sec: float = 30.0,
    reuse_indices: bool = False,
    log: Any = None,
) -> dict[str, Any]:
    """Encode all valid DR1 spectra once and write memmap arrays under ``cache_dir``.

    When ``world_size > 1``, launch with ``torchrun``; ranks shard work by striding
    the sorted valid-index list so global row order matches single-GPU builds.
    """
    import torch.distributed as dist

    from desifm.tokenization.aion_bridge import AionSpectrumTokenizer

    manifest = Path(manifest).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    device = torch.device(device)

    records = load_manifest(manifest)
    ds = DR1StreamDataset(records, max_spectra=max_spectra)

    if log is not None and rank == 0:
        if world_size > 1:
            log.info(
                "CPU index scan (%d rows, %d-way shard) — GPUs idle until encode starts...",
                len(ds),
                world_size,
            )
        else:
            log.info("CPU index scan (%d rows) — GPU idle until encode starts...", len(ds))

    indices_path = paths["root"] / VALID_INDICES_FILE
    n_valid = collect_valid_indices(
        ds,
        indices_path,
        rank=rank,
        world_size=world_size,
        device=device,
        log=log,
        reuse_existing=reuse_indices,
    )
    n_skipped = len(ds) - n_valid
    if n_valid == 0:
        raise RuntimeError("no valid spectra in manifest — check FITS paths")

    if rank == 0:
        codes_mb = n_valid * N_LATENT_TOKENS * 2 / (1024**3)
        if log is not None:
            log.info(
                "allocating cache for %d spectra (skipped %d, codes ~%.2f GiB on scratch)...",
                n_valid,
                n_skipped,
                codes_mb,
            )
        _init_cache_files(paths, n_valid)
        if log is not None:
            log.info("cache memmap files created")

    _dist_barrier(device)

    codes_mmap = np.memmap(paths["codes"], dtype=np.uint16, mode="r+", shape=(n_valid, N_LATENT_TOKENS))
    z_mmap = np.memmap(paths["z"], dtype=np.float32, mode="r+", shape=(n_valid,))
    hp_mmap = np.memmap(paths["healpix"], dtype=np.int32, mode="r+", shape=(n_valid,))
    idx_mmap = np.memmap(indices_path, dtype=np.int64, mode="r", shape=(n_valid,))
    my_indices = idx_mmap[rank::world_size]
    n_assigned = int(my_indices.shape[0])

    if log is not None and rank == 0:
        per_rank = (n_valid + world_size - 1) // world_size if world_size > 0 else n_valid
        log.info(
            "encoding %d spectra (%d per GPU, batch_size=%d, device=%s)",
            n_valid,
            per_rank,
            batch_size,
            device,
        )

    tok = AionSpectrumTokenizer(device, hf_repo=aion_hf_repo)
    _warmup_aion(tok, rank=rank, world_size=world_size, device=device, log=log)

    pending: list[dict] = []
    pending_hp: list[int] = []
    pending_rows: list[int] = []
    local_written = 0
    t0 = time.perf_counter()
    last_heartbeat = t0
    first_encode = True

    def flush_pending() -> None:
        nonlocal local_written, last_heartbeat, first_encode
        if not pending:
            return
        batch = collate_spectra(pending)
        if batch is None:
            pending.clear()
            pending_hp.clear()
            pending_rows.clear()
            return
        if log is not None and rank == 0 and first_encode:
            log.info("GPU encode: first batch (%d spectra) — loading HF codec if needed...", len(pending))
            sys.stdout.flush()
        codes, _meta = tok.encode_batch(batch)
        if log is not None and rank == 0 and first_encode:
            log.info("GPU encode: first batch done")
            sys.stdout.flush()
            first_encode = False
        b = codes.shape[0]
        rows = np.array(pending_rows, dtype=np.int64)
        codes_mmap[rows] = codes.cpu().numpy().astype(np.uint16)
        z_mmap[rows] = np.array([float(x["z"]) for x in pending], dtype=np.float32)
        hp_mmap[rows] = np.array(pending_hp, dtype=np.int32)
        local_written += b
        pending.clear()
        pending_hp.clear()
        pending_rows.clear()

        global_done = local_written
        if world_size > 1 and dist.is_initialized():
            t = torch.tensor([local_written], dtype=torch.long, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_done = int(t.item())

        if log is not None and (
            local_written == n_assigned
            or (log_every > 0 and local_written % log_every == 0)
        ):
            _log_cache_progress(
                log,
                rank=rank,
                world_size=world_size,
                local_done=local_written,
                local_total=n_assigned,
                global_done=global_done,
                global_total=n_valid,
                t0=t0,
                force=local_written == n_assigned,
            )

    for j in range(n_assigned):
        ds_idx = int(my_indices[j])
        now = time.perf_counter()
        if log is not None and rank == 0 and (j == 0 or now - last_heartbeat >= log_heartbeat_sec):
            log.info(
                "encode prep: read %d / %d rows on rank 0 (pending batch %d)",
                j + 1,
                n_assigned,
                len(pending),
            )
            sys.stdout.flush()
            last_heartbeat = now
        item = ds[ds_idx]
        if item is None:
            continue
        rec_idx, _row = ds.index[ds_idx]
        global_row = rank + j * world_size
        pending.append(item)
        pending_hp.append(int(records[rec_idx]["healpix"]))
        pending_rows.append(global_row)
        if len(pending) >= batch_size:
            flush_pending()

    flush_pending()
    codes_mmap.flush()
    z_mmap.flush()
    hp_mmap.flush()

    if world_size > 1 and dist.is_initialized():
        _dist_barrier(device)
        total_written = torch.tensor([local_written], dtype=torch.long, device=device)
        dist.all_reduce(total_written, op=dist.ReduceOp.SUM)
        if int(total_written.item()) != n_valid:
            raise RuntimeError(
                f"cache write mismatch on rank {rank}: global wrote {int(total_written.item())}, expected {n_valid}"
            )
    elif local_written != n_assigned:
        raise RuntimeError(f"rank {rank} wrote {local_written}, expected {n_assigned}")

    meta = {
        "version": CACHE_VERSION,
        "manifest": str(manifest),
        "n_spectra": n_valid,
        "n_skipped": n_skipped,
        "n_latent_tokens": N_LATENT_TOKENS,
        "n_spectrum_codes": N_SPECTRUM_CODES,
        "grid_size": GRID_SIZE,
        "aion_hf_repo": aion_hf_repo,
        "world_size": world_size,
    }
    if rank == 0:
        with paths["meta"].open("w") as f:
            json.dump(meta, f, indent=2)
        if log is not None:
            elapsed = time.perf_counter() - t0
            log.info(
                "cache complete n=%d skipped=%d dir=%s (%.1fs, %.2f spec/s aggregate)",
                n_valid,
                n_skipped,
                cache_dir,
                elapsed,
                n_valid / max(elapsed, 1e-6),
            )

    _dist_barrier(device)

    return meta
