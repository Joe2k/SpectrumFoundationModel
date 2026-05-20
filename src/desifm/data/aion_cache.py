"""Precomputed AION spectrum token cache for fast transformer training."""

from __future__ import annotations

import json
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


def collect_valid_indices(
    ds: DR1StreamDataset,
    *,
    rank: int = 0,
    world_size: int = 1,
    log: Any = None,
) -> list[int]:
    """Dataset indices with non-None spectra, in ascending order (DDP-safe)."""
    import torch.distributed as dist

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
        valid = merge_valid_indices(gathered)
        if log is not None and rank == 0:
            log.info(
                "index scan complete: %d valid / %d rows in %.1fs",
                len(valid),
                n,
                time.perf_counter() - t0,
            )
        return valid

    if log is not None:
        log.info("index scan complete: %d valid / %d rows in %.1fs", len(local), n, time.perf_counter() - t0)
    return local


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

    valid_indices = collect_valid_indices(ds, rank=rank, world_size=world_size, log=log)
    n_valid = len(valid_indices)
    n_skipped = len(ds) - n_valid
    if n_valid == 0:
        raise RuntimeError("no valid spectra in manifest — check FITS paths")

    if rank == 0:
        if log is not None:
            log.info("allocating cache for %d spectra (skipped %d)", n_valid, n_skipped)
        _init_cache_files(paths, n_valid)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    codes_mmap = np.memmap(paths["codes"], dtype=np.uint16, mode="r+", shape=(n_valid, N_LATENT_TOKENS))
    z_mmap = np.memmap(paths["z"], dtype=np.float32, mode="r+", shape=(n_valid,))
    hp_mmap = np.memmap(paths["healpix"], dtype=np.int32, mode="r+", shape=(n_valid,))

    my_indices = valid_indices[rank::world_size]
    n_assigned = len(my_indices)

    if log is not None and rank == 0:
        per_rank = (n_valid + world_size - 1) // world_size if world_size > 0 else n_valid
        log.info(
            "encoding %d spectra (%d per GPU, batch_size=%d, device=%s)",
            n_valid,
            per_rank,
            batch_size,
            device,
        )

    if log is not None and rank == 0:
        log.info("loading AION on %s — GPU encode starting", device)
    tok = AionSpectrumTokenizer(device, hf_repo=aion_hf_repo)
    pending: list[dict] = []
    pending_hp: list[int] = []
    pending_rows: list[int] = []
    local_written = 0
    t0 = time.perf_counter()

    def flush_pending() -> None:
        nonlocal local_written
        if not pending:
            return
        batch = collate_spectra(pending)
        if batch is None:
            pending.clear()
            pending_hp.clear()
            pending_rows.clear()
            return
        codes, _meta = tok.encode_batch(batch)
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

    for j, ds_idx in enumerate(my_indices):
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
        dist.barrier()
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

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    return meta
