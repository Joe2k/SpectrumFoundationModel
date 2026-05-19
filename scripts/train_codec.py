#!/usr/bin/env python3
"""Pretrain the spectrum codec. Supports DDP via torchrun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, load_manifest
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    unwrap,
    wrap_ddp,
)
from desifm.training.paths import require_scratch_manifest, scratch_root
from desifm.training.wandb_log import finish, init_run, log_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--run-name", default="codec_v1")
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    rank, world_size, _local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)

    if args.smoke:
        args.steps, args.max_spectra, args.batch_size = 50, 100, 4
        args.synthetic = True

    if args.synthetic:
        ds = SyntheticSpectrumDataset(n_spectra=args.max_spectra or 512, seed=42)
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        manifest = require_scratch_manifest(args.manifest)
        ds = DR1StreamDataset(load_manifest(manifest), max_spectra=args.max_spectra)

    sampler = DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_spectra,
        drop_last=True,
    )

    out_root = args.scratch_out or scratch_root() / "checkpoints"
    run_dir = out_root / args.run_name
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)

    model = SpectrumCodec().to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec") if main_proc else None

    step, best = 0, float("inf")
    it = iter(loader)
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if batch is None:
            continue
        flux, ivar = batch["flux"].to(device), batch["ivar"].to(device)
        x = torch.stack([flux, torch.sqrt(ivar.clamp(min=1e-10))], dim=1)
        out = unwrap(model)(x)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if main_proc and step % 10 == 0 and wb:
            log_metrics(wb, {"train/loss": loss.item(), "train/recon": out["recon_loss"].item()}, step)
        if main_proc and loss.item() < best:
            best = loss.item()
            torch.save({"model": unwrap(model).state_dict(), "step": step}, run_dir / "best.pt")
        step += 1

    if main_proc:
        torch.save({"model": unwrap(model).state_dict(), "step": step}, run_dir / "final.pt")
        finish(wb)
        print(f"done -> {run_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
