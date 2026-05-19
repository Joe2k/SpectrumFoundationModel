#!/usr/bin/env python3
"""Pretrain the spectrum codec (AION Tier-A preprocessing). Supports DDP via torchrun."""

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
from desifm.training.codec_input import prepare_codec_batch
from desifm.training.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    unwrap,
    wrap_ddp,
)
from desifm.training.loss_tracker import LossTracker
from desifm.training.paths import require_scratch_manifest, scratch_root
from desifm.training.wandb_log import finish, init_run, log_metrics, replace_best_artifact


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
    p.add_argument("--run-name", default="codec_v3")
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-loss", type=float, default=50.0, help="Skip steps with loss above this")
    p.add_argument(
        "--checkpoint-metric",
        choices=["median", "mean"],
        default="median",
        help="Aggregate last log_every steps for best checkpoint",
    )
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-wandb-artifact", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)

    if args.smoke:
        args.steps, args.max_spectra, args.batch_size = 50, 100, 4
        args.synthetic = True

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
    best_ckpt = run_dir / "best.pt"
    artifact_state: dict[str, str | None] = {"qualified": None}
    artifact_name = f"{args.run_name}-codec-best".replace("/", "_")
    tracker = LossTracker(window=args.log_every, max_loss=args.max_loss) if main_proc else None

    if main_proc:
        n_params = sum(p.numel() for p in SpectrumCodec().parameters())
        log.info("=== spectrum codec training (AION Tier-A input) ===")
        log.info("run_dir=%s", run_dir)
        log.info("manifest=%s", manifest_path or "(synthetic)")
        log.info("dataset_size=%d batch_size=%d steps=%d lr=%g", len(ds), args.batch_size, args.steps, args.lr)
        log.info("device=%s world_size=%d", device, world_size)
        log.info("norm=arcsinh+mask-aware checkpoint=%s max_loss=%g", args.checkpoint_metric, args.max_loss)

    model = SpectrumCodec().to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec") if main_proc else None
    if main_proc and wb is not None:
        log.info("wandb run: %s", getattr(wb, "url", wb.name))

    step, best_metric = 0, float("inf")
    skipped = 0
    t0 = time.perf_counter()
    it = iter(loader)

    def checkpoint_score() -> float:
        assert tracker is not None
        return tracker.window_median() if args.checkpoint_metric == "median" else tracker.window_mean()

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

        batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        x, denorm, mask = prepare_codec_batch(batch_d)
        out = unwrap(model)(x, denorm, mask)
        loss = out["loss"]

        if not torch.isfinite(loss):
            skipped += 1
            step += 1
            continue

        loss_f = loss.item()
        if main_proc and not tracker.update(loss_f):
            skipped += 1
            if step % args.log_every == 0:
                log.warning("skip step %d loss=%.4g (max_loss=%g)", step, loss_f, args.max_loss)
            step += 1
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.grad_clip)
        opt.step()

        if main_proc:
            assert tracker is not None
            do_log = step % args.log_every == 0
            if do_log:
                avg_f = tracker.window_mean()
                med_f = tracker.window_median()
                score = checkpoint_score()
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                append_metrics(
                    metrics_path,
                    {
                        "kind": "train",
                        "step": step,
                        "loss_batch": loss_f,
                        "loss_avg": avg_f,
                        "loss_median": med_f,
                        "recon_loss": out["recon_loss"].item(),
                        "q_loss": out["q_loss"].item(),
                        "best_metric": min(best_metric, score),
                        "skipped": skipped,
                        "steps_per_sec": sps,
                    },
                )
                log.info(
                    "step %d/%d batch=%.4f med=%.4f recon=%.4f q=%.4f best_%s=%.4f skip=%d (%.2f step/s)",
                    step,
                    args.steps,
                    loss_f,
                    med_f,
                    out["recon_loss"].item(),
                    out["q_loss"].item(),
                    args.checkpoint_metric,
                    min(best_metric, score),
                    skipped,
                    sps,
                )
                log_metrics(
                    wb,
                    {
                        "train/loss_batch": loss_f,
                        "train/loss_avg": avg_f,
                        "train/loss_median": med_f,
                        "train/recon": out["recon_loss"].item(),
                        "train/q_loss": out["q_loss"].item(),
                        "train/skipped": skipped,
                    },
                    step,
                )

                if score < best_metric:
                    best_metric = score
                    torch.save(
                        {
                            "model": unwrap(model).state_dict(),
                            "step": step,
                            "loss": best_metric,
                            "input_style": "aion_tier_a",
                        },
                        best_ckpt,
                    )
                    log.info("saved best step=%d %s=%.4f -> %s", step, args.checkpoint_metric, best_metric, best_ckpt)
                    if wb and not args.no_wandb_artifact:
                        replace_best_artifact(wb, best_ckpt, artifact_name, step, best_metric, artifact_state)

        step += 1

    if main_proc:
        elapsed = time.perf_counter() - t0
        torch.save(
            {"model": unwrap(model).state_dict(), "step": step, "loss": best_metric, "input_style": "aion_tier_a"},
            run_dir / "final.pt",
        )
        append_metrics(
            metrics_path,
            {"kind": "summary", "steps": step, "best_metric": best_metric, "skipped": skipped, "elapsed_sec": elapsed},
        )
        finish(wb)
        log.info("=== done === best_%s=%.4f skipped=%d -> %s", args.checkpoint_metric, best_metric, skipped, run_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
