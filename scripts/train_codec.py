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

from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, healpix_split, load_manifest
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.codec_input import (
    INPUT_STYLE_V3,
    INPUT_STYLE_V4,
    prepare_codec_batch,
    prepare_codec_batch_for_style,
    prepare_codec_batch_v4,
)
from desifm.training.codec_loss import flux_rms, flux_std_ratio
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


def prepare_batch(batch: dict, input_style: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if input_style == INPUT_STYLE_V4:
        return prepare_codec_batch_v4(batch)
    if input_style == INPUT_STYLE_V3:
        return prepare_codec_batch(batch)
    return prepare_codec_batch_for_style(batch, input_style)


@torch.no_grad()
def run_validation(
    model: SpectrumCodec,
    loader: DataLoader,
    device: torch.device,
    input_style: str,
    *,
    lambda_phys: float,
    lambda_entropy: float,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    n = 0
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "phys": 0.0,
        "q": 0.0,
        "entropy": 0.0,
        "rms": 0.0,
        "std_ratio": 0.0,
    }
    for batch in loader:
        if batch is None:
            continue
        batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        x, denorm, mask = prepare_batch(batch_d, input_style)
        out = model(
            x,
            denorm,
            mask,
            lambda_phys=lambda_phys,
            lambda_entropy=lambda_entropy,
        )
        mask_g = mask
        totals["loss"] += float(out["loss"].item())
        totals["recon"] += float(out["recon_loss"].item())
        totals["phys"] += float(out["phys_loss"].item())
        totals["q"] += float(out["q_loss"].item())
        totals["entropy"] += float(out["entropy_loss"].item())
        totals["rms"] += float(flux_rms(out["recon_phys"], out["target_phys"], mask_g).item())
        totals["std_ratio"] += float(
            flux_std_ratio(out["recon_phys"], out["target_phys"], mask_g).item()
        )
        n += 1
        if n >= max_batches:
            break
    model.train()
    if n == 0:
        return {k: float("inf") for k in totals}
    return {k: v / n for k, v in totals.items()}


def infer_codec_version(run_name: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if run_name.startswith("codec_v4"):
        return "v4"
    return "v3"


def apply_version_defaults(args: argparse.Namespace) -> None:
    if args.codec_version != "v4":
        return
    if args.steps == 5000:
        args.steps = 20_000
    if args.batch_size == 16:
        args.batch_size = 32
    if args.lr == 3e-4:
        args.lr = 1e-4
    if args.checkpoint_metric == "median":
        args.checkpoint_metric = "val_rms"
    if args.healpix_holdout_frac == 0.0:
        args.healpix_holdout_frac = 0.05
    if args.val_every == 0:
        args.val_every = 500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--run-name", default="codec_v3")
    p.add_argument("--codec-version", choices=["v3", "v4"], default=None)
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup (v4 default 1000)")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-loss", type=float, default=50.0, help="Skip steps with loss above this")
    p.add_argument(
        "--checkpoint-metric",
        choices=["median", "mean", "val_rms"],
        default="median",
        help="Train window aggregate (v3) or val physical RMS (v4)",
    )
    p.add_argument("--healpix-holdout-frac", type=float, default=0.0)
    p.add_argument("--healpix-split-seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=0, help="Run val every N steps (0=disabled)")
    p.add_argument("--val-max-batches", type=int, default=32)
    p.add_argument("--lambda-phys", type=float, default=0.0)
    p.add_argument("--lambda-entropy", type=float, default=0.0)
    p.add_argument("--commitment-weight", type=float, default=None, help="LFQ beta (v4 default 0.05)")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-wandb-artifact", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    args.codec_version = infer_codec_version(args.run_name, args.codec_version)
    apply_version_defaults(args)

    input_style = INPUT_STYLE_V4 if args.codec_version == "v4" else INPUT_STYLE_V3
    if args.codec_version == "v4":
        if args.lambda_phys == 0.0:
            args.lambda_phys = 0.5
        if args.lambda_entropy == 0.0:
            args.lambda_entropy = 0.1
        if args.warmup_steps == 0:
            args.warmup_steps = 1000
    commitment = args.commitment_weight
    if commitment is None:
        commitment = 0.05 if args.codec_version == "v4" else 0.25

    rank, world_size, local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)

    if args.smoke:
        args.steps, args.max_spectra, args.batch_size = 50, 100, 4
        args.synthetic = True
        args.val_every = 10
        args.healpix_holdout_frac = 0.1

    val_loader = None
    if args.synthetic:
        ds = SyntheticSpectrumDataset(n_spectra=args.max_spectra or 512, seed=42)
        manifest_path = None
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        manifest_path = require_scratch_manifest(args.manifest)
        records = load_manifest(manifest_path)
        if args.healpix_holdout_frac > 0:
            train_rec, val_rec = healpix_split(records, args.healpix_holdout_frac, args.healpix_split_seed)
            if main_proc and not val_rec:
                raise SystemExit("healpix holdout produced empty val set — increase manifest or lower holdout")
            ds = DR1StreamDataset(train_rec, max_spectra=args.max_spectra)
            if val_rec:
                val_loader = DataLoader(
                    DR1StreamDataset(val_rec, max_spectra=args.max_spectra),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_spectra,
                    drop_last=False,
                )
        else:
            ds = DR1StreamDataset(records, max_spectra=args.max_spectra)

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
    use_train_tracker = args.checkpoint_metric in ("median", "mean")
    tracker = LossTracker(window=args.log_every, max_loss=args.max_loss) if main_proc and use_train_tracker else None

    if main_proc:
        n_params = sum(p.numel() for p in SpectrumCodec(commitment_weight=commitment).parameters())
        log.info("=== spectrum codec %s (%s) ===", args.codec_version, input_style)
        log.info("run_dir=%s", run_dir)
        log.info("manifest=%s", manifest_path or "(synthetic)")
        log.info(
            "dataset_size=%d batch_size=%d steps=%d lr=%g λ_phys=%g λ_ent=%g commit=%g",
            len(ds),
            args.batch_size,
            args.steps,
            args.lr,
            args.lambda_phys,
            args.lambda_entropy,
            commitment,
        )
        log.info(
            "device=%s world_size=%d checkpoint_metric=%s val_every=%d holdout=%g",
            device,
            world_size,
            args.checkpoint_metric,
            args.val_every,
            args.healpix_holdout_frac,
        )

    model = SpectrumCodec(commitment_weight=commitment).to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec") if main_proc else None
    if main_proc and wb is not None:
        log.info("wandb run: %s", getattr(wb, "url", wb.name))

    step, best_metric = 0, float("inf")
    skipped = 0
    t0 = time.perf_counter()
    it = iter(loader)

    def lr_scale(s: int) -> float:
        if args.warmup_steps <= 0 or s >= args.warmup_steps:
            return 1.0
        return max(s / args.warmup_steps, 1e-8)

    def checkpoint_score() -> float:
        assert tracker is not None
        return tracker.window_median() if args.checkpoint_metric == "median" else tracker.window_mean()

    def save_best(metric: float, val_stats: dict[str, float] | None = None) -> None:
        nonlocal best_metric
        if metric >= best_metric:
            return
        best_metric = metric
        payload = {
            "model": unwrap(model).state_dict(),
            "step": step,
            "loss": best_metric,
            "input_style": input_style,
            "codec_version": args.codec_version,
            "lambda_phys": args.lambda_phys,
            "lambda_entropy": args.lambda_entropy,
            "commitment_weight": commitment,
        }
        if val_stats:
            payload["val"] = val_stats
        torch.save(payload, best_ckpt)
        log.info("saved best step=%d metric=%.4f -> %s", step, best_metric, best_ckpt)
        if wb and not args.no_wandb_artifact:
            replace_best_artifact(wb, best_ckpt, artifact_name, step, best_metric, artifact_state)

    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)

        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_scale(step)

        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if batch is None:
            continue

        batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        x, denorm, mask = prepare_batch(batch_d, input_style)
        out = unwrap(model)(
            x,
            denorm,
            mask,
            lambda_phys=args.lambda_phys,
            lambda_entropy=args.lambda_entropy,
        )
        loss = out["loss"]

        if not torch.isfinite(loss):
            skipped += 1
            step += 1
            continue

        loss_f = loss.item()
        if main_proc and tracker is not None and not tracker.update(loss_f):
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
            do_log = step % args.log_every == 0
            if do_log:
                avg_f = tracker.window_mean() if tracker else loss_f
                med_f = tracker.window_median() if tracker else loss_f
                elapsed = time.perf_counter() - t0
                sps = (step + 1) / max(elapsed, 1e-6)
                rec = {
                    "kind": "train",
                    "step": step,
                    "loss_batch": loss_f,
                    "recon_loss": out["recon_loss"].item(),
                    "phys_loss": out.get("phys_loss", torch.tensor(0.0)).item(),
                    "q_loss": out["q_loss"].item(),
                    "entropy_loss": out.get("entropy_loss", torch.tensor(0.0)).item(),
                    "best_metric": best_metric,
                    "skipped": skipped,
                    "steps_per_sec": sps,
                    "lr": opt.param_groups[0]["lr"],
                }
                if tracker:
                    rec["loss_avg"] = avg_f
                    rec["loss_median"] = med_f
                append_metrics(metrics_path, rec)
                log.info(
                    "step %d/%d loss=%.4f recon=%.4f phys=%.4f q=%.4f ent=%.4f skip=%d (%.2f step/s)",
                    step,
                    args.steps,
                    loss_f,
                    out["recon_loss"].item(),
                    rec["phys_loss"],
                    out["q_loss"].item(),
                    rec["entropy_loss"],
                    skipped,
                    sps,
                )
                wb_payload = {
                    "train/loss_batch": loss_f,
                    "train/recon": out["recon_loss"].item(),
                    "train/phys": rec["phys_loss"],
                    "train/q_loss": out["q_loss"].item(),
                    "train/entropy": rec["entropy_loss"],
                    "train/skipped": skipped,
                    "train/lr": opt.param_groups[0]["lr"],
                }
                if tracker:
                    wb_payload["train/loss_avg"] = avg_f
                    wb_payload["train/loss_median"] = med_f
                log_metrics(wb, wb_payload, step)

                if use_train_tracker and tracker:
                    score = checkpoint_score()
                    if score < best_metric:
                        save_best(score)

            if (
                val_loader is not None
                and args.val_every > 0
                and step > 0
                and step % args.val_every == 0
            ):
                val_stats = run_validation(
                    unwrap(model),
                    val_loader,
                    device,
                    input_style,
                    lambda_phys=args.lambda_phys,
                    lambda_entropy=args.lambda_entropy,
                    max_batches=args.val_max_batches,
                )
                append_metrics(metrics_path, {"kind": "val", "step": step, **val_stats})
                log.info(
                    "val step=%d rms=%.4f recon=%.4f std_ratio=%.3f q=%.4f ent=%.4f",
                    step,
                    val_stats["rms"],
                    val_stats["recon"],
                    val_stats["std_ratio"],
                    val_stats["q"],
                    val_stats["entropy"],
                )
                log_metrics(
                    wb,
                    {
                        "val/loss": val_stats["loss"],
                        "val/recon_arcsinh": val_stats["recon"],
                        "val/phys": val_stats["phys"],
                        "val/rms_flux": val_stats["rms"],
                        "val/std_ratio": val_stats["std_ratio"],
                        "val/q_loss": val_stats["q"],
                        "val/entropy_penalty": val_stats["entropy"],
                    },
                    step,
                )
                if args.checkpoint_metric == "val_rms":
                    save_best(val_stats["rms"], val_stats)

        step += 1

    if main_proc:
        elapsed = time.perf_counter() - t0
        torch.save(
            {
                "model": unwrap(model).state_dict(),
                "step": step,
                "loss": best_metric,
                "input_style": input_style,
                "codec_version": args.codec_version,
            },
            run_dir / "final.pt",
        )
        append_metrics(
            metrics_path,
            {"kind": "summary", "steps": step, "best_metric": best_metric, "skipped": skipped, "elapsed_sec": elapsed},
        )
        finish(wb)
        log.info("=== done === best=%.4f skipped=%d -> %s", best_metric, skipped, run_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
