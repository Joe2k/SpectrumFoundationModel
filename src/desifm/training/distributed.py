"""torch.distributed helpers for multi-GPU training."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize process group if launched with torchrun. Returns rank, world_size, local_rank, device."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def wrap_ddp(model: torch.nn.Module, device: torch.device, world_size: int) -> torch.nn.Module:
    if world_size <= 1:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def all_ranks_agree_skip(local_skip: bool, device: torch.device) -> bool:
    """True if any rank wants to skip the step (DDP-safe)."""
    if not dist.is_initialized():
        return local_skip
    flag = torch.tensor([1.0 if local_skip else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0)
