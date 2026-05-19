#!/usr/bin/env python3
"""Pretrain the spectrum codec. Supports DDP via torchrun."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("train_codec")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def append_metrics(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


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
    p.add_argument("--log-every", type=int, default=50, help="Log to stdout/jsonl every N steps")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)

    if args.smoke:
        args.steps, args.max_spectra, args.batch_size = 50, 100, 4
        args.synthetic = True
        args.log_every = 10

    if args.synthetic:
        ds = SyntheticSpectrumDataset(n_spectra=args.max_spectra or 512, seed=42)
        manifest_path = None
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        manifest_path = require_scratch_manifest(args.manifest)
        ds = DR1StreamDataset(load_manifest(manifest_path), max_spectra=args.max_spectra)

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
    log = setup_logging(run_dir) if main_proc else None
    metrics_path = run_dir / "metrics.jsonl"

    if main_proc:
        n_params = sum(p.numel() for p in SpectrumCodec().parameters())
        log.info("=== spectrum codec (tokenizer) training ===")
        log.info("run_dir=%s", run_dir)
        log.info("manifest=%s", manifest_path or "(synthetic)")
        log.info("dataset_size=%d batch_size=%d steps=%d lr=%g", len(ds), args.batch_size, args.steps, args.lr)
        log.info("device=%s world_size=%d local_rank=%d", device, world_size, local_rank)
        log.info("model_params=%d log_every=%d wandb_mode=%s", n_params, args.log_every, args.wandb_mode)
        log.info("metrics_jsonl=%s train_log=%s", metrics_path, run_dir / "train.log")

    model = SpectrumCodec().to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec") if main_proc else None
    if main_proc and wb is not None:
        log.info("wandb run: %s", getattr(wb, "url", wb.name))

    step, best = 0, float("inf")
    t0 = time.perf_counter()
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

        if main_proc:
            loss_f = loss.item()
            recon_f = out["recon_loss"].item()
            q_f = out["q_loss"].item()
            metrics = {
                "kind": "train",
                "step": step,
                "loss": loss_f,
                "recon_loss": recon_f,
                "q_loss": q_f,
                "lr": opt.param_groups[0]["lr"],
                "best_loss": min(best, loss_f),
            }
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                metrics["steps_per_sec"] = sps
                metrics["elapsed_sec"] = elapsed
                append_metrics(metrics_path, metrics)
                log.info(
                    "step %d/%d loss=%.4f recon=%.4f q=%.4f best=%.4f (%.2f step/s)",
                    step,
                    args.steps,
                    loss_f,
                    recon_f,
                    q_f,
                    min(best, loss_f),
                    sps,
                )
            if step % args.log_every == 0:
                log_metrics(wb, {"train/loss": loss_f, "train/recon": recon_f, "train/q_loss": q_f}, step)
            if loss_f < best:
                best = loss_f
                torch.save({"model": unwrap(model).state_dict(), "step": step, "loss": best}, run_dir / "best.pt")
                if step % args.log_every == 0:
                    log.info("saved best checkpoint step=%d loss=%.4f", step, best)
        step += 1

    if main_proc:
        elapsed = time.perf_counter() - t0
        torch.save({"model": unwrap(model).state_dict(), "step": step, "loss": best}, run_dir / "final.pt")
        summary = {
            "kind": "summary",
            "steps": step,
            "best_loss": best,
            "elapsed_sec": elapsed,
            "run_dir": str(run_dir),
        }
        append_metrics(metrics_path, summary)
        finish(wb)
        log.info("=== done === best_loss=%.4f elapsed=%.1fs -> %s", best, elapsed, run_dir)
        log.info("checkpoints: %s %s", run_dir / "best.pt", run_dir / "final.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()
