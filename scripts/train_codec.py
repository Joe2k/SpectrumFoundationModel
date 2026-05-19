#!/usr/bin/env python3
"""Pretrain the spectrum codec on a DR1 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, load_manifest
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.paths import require_scratch_manifest, scratch_root
from desifm.training.wandb_log import finish, init_run, log_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run-name", default="codec_v1")
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps, args.max_spectra, args.batch_size = 50, 100, 4

    manifest = require_scratch_manifest(args.manifest) if not args.smoke else args.manifest
    out_root = args.scratch_out or scratch_root() / "checkpoints"
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DR1StreamDataset(load_manifest(manifest), max_spectra=args.max_spectra)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_spectra)

    model = SpectrumCodec().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec")

    step, best = 0, float("inf")
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if batch is None:
            continue
        flux, ivar = batch["flux"].to(device), batch["ivar"].to(device)
        x = torch.stack([flux, torch.sqrt(ivar.clamp(min=1e-10))], dim=1)
        out = model(x)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            log_metrics(wb, {"train/loss": loss.item(), "train/recon": out["recon_loss"].item()}, step)
        if loss.item() < best:
            best = loss.item()
            torch.save({"model": model.state_dict(), "step": step}, run_dir / "best.pt")
        step += 1

    torch.save({"model": model.state_dict(), "step": step}, run_dir / "final.pt")
    finish(wb)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
