#!/usr/bin/env python3
"""Pretrain the spectrum codec. Supports DDP via torchrun."""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    INPUT_STYLE_V5,
    prepare_codec_batch,
    prepare_codec_batch_for_style,
    prepare_codec_batch_v4,
)
from desifm.training.codec_loss import (
    code_usage_passes_gate,
    code_usage_stats,
    flux_rms,
    flux_std_ratio,
    flux_std_ratio_per_sample,
)
from desifm.training.distributed import (
    all_ranks_agree_skip,
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
    if input_style in (INPUT_STYLE_V4, INPUT_STYLE_V5):
        return prepare_codec_batch_v4(batch)
    if input_style == INPUT_STYLE_V3:
        return prepare_codec_batch(batch)
    return prepare_codec_batch_for_style(batch, input_style)


def model_forward_kw(codec_version: str) -> dict:
    if codec_version in ("v5a", "v5"):
        return {"use_batch_entropy": True}
    return {}


def n_codes_for_model(model: torch.nn.Module) -> int:
    if hasattr(model, "quant") and hasattr(model.quant, "n_codes"):
        return int(model.quant.n_codes)
    if hasattr(model, "latent_dim"):
        return 2 ** int(model.latent_dim)
    return 256


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_style: str,
    *,
    codec_version: str,
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
        "std_ratio_per_spec_median": 0.0,
        "n_unique_codes": 0.0,
        "code_usage_fraction": 0.0,
        "code_usage_fraction_gate": 0.0,
    }
    per_spec_ratios: list[float] = []
    fwd_extra = model_forward_kw(codec_version)
    n_codes = n_codes_for_model(model)
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
            **fwd_extra,
        )
        mask_g = mask
        usage = code_usage_stats(out["indices"], n_codes=n_codes)
        totals["loss"] += float(out["loss"].item())
        totals["recon"] += float(out["recon_loss"].item())
        totals["phys"] += float(out["phys_loss"].item())
        totals["q"] += float(out["q_loss"].item())
        totals["entropy"] += float(out["entropy_loss"].item())
        totals["rms"] += float(flux_rms(out["recon_phys"], out["target_phys"], mask_g).item())
        totals["std_ratio"] += float(
            flux_std_ratio(out["recon_phys"], out["target_phys"], mask_g).item()
        )
        totals["n_unique_codes"] += float(usage["n_unique"])
        totals["code_usage_fraction"] += float(usage["usage_fraction"])
        totals["code_usage_fraction_gate"] += float(usage["usage_fraction_gate"])
        per_spec_ratios.extend(
            flux_std_ratio_per_sample(out["recon_phys"], out["target_phys"], mask_g).tolist()
        )
        n += 1
        if n >= max_batches:
            break
    model.train()
    if n == 0:
        return {k: float("inf") for k in totals}
    out_stats = {k: v / n for k, v in totals.items()}
    if per_spec_ratios:
        out_stats["std_ratio_per_spec_median"] = float(torch.tensor(per_spec_ratios).median().item())
    else:
        out_stats["std_ratio_per_spec_median"] = 0.0
    return out_stats


def learning_rate_scale(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    schedule: str = "constant",
    min_lr: float = 1e-6,
) -> float:
    """Multiplier for base LR at ``step`` (0-based): linear warmup, then constant or cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return max(step / warmup_steps, 1e-8)
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unknown lr schedule: {schedule!r}")

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    min_ratio = min(min_lr / max(base_lr, 1e-12), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def lambda_ramp_scale(step: int, ramp_steps: int) -> float:
    """Linear 0→1 multiplier for λ_phys / optional λ_ent over the first ``ramp_steps``."""
    if ramp_steps <= 0:
        return 1.0
    return min(max(step / ramp_steps, 0.0), 1.0)


def effective_lambda_phys(
    step: int,
    target: float,
    ramp_steps: int,
    *,
    delay_until_code_usage: bool,
    phys_unlocked: bool,
    phys_ramp_origin: int | None,
) -> float:
    """Training λ_phys: zero until code-usage gate passes, then linear ramp from unlock step."""
    if delay_until_code_usage and not phys_unlocked:
        return 0.0
    origin = phys_ramp_origin if phys_ramp_origin is not None else 0
    ramp = lambda_ramp_scale(max(step - origin, 0), ramp_steps)
    return target * ramp


def infer_codec_version(run_name: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if "v5b" in run_name or run_name.endswith("_v5"):
        return "v5"
    if run_name.startswith("codec_v5"):
        return "v5a"
    if run_name.startswith("codec_v4"):
        return "v4"
    return "v3"


def _apply_tier1_training_defaults(args: argparse.Namespace) -> None:
    if args.steps == 5000:
        args.steps = 20_000
    if args.batch_size == 16:
        args.batch_size = 32
    if args.lr == 3e-4:
        args.lr = 1e-4
    if args.healpix_holdout_frac == 0.0:
        args.healpix_holdout_frac = 0.05
    if args.val_every == 0:
        args.val_every = 500
    if args.lr_schedule == "constant":
        args.lr_schedule = "cosine"
    if args.lambda_phys_ramp_steps == 0:
        args.lambda_phys_ramp_steps = 4000
    if getattr(args, "warmup_steps", 0) == 0:
        args.warmup_steps = 1000


def apply_version_defaults(args: argparse.Namespace) -> None:
    if args.codec_version == "v4":
        _apply_tier1_training_defaults(args)
        if args.checkpoint_metric == "median":
            args.checkpoint_metric = "val_rms"
    elif args.codec_version == "v5a":
        _apply_tier1_training_defaults(args)
        if args.checkpoint_metric == "median":
            args.checkpoint_metric = "val_std_ratio_per_spec_median"
        if args.min_code_usage_fraction == 0.0:
            args.min_code_usage_fraction = 0.3
        if args.delay_lambda_phys_until_code_usage is None:
            args.delay_lambda_phys_until_code_usage = True
    elif args.codec_version == "v5":
        _apply_tier1_training_defaults(args)
        if args.checkpoint_metric == "median":
            args.checkpoint_metric = "val_std_ratio_per_spec_median"
        if args.min_code_usage_fraction == 0.0:
            args.min_code_usage_fraction = 0.3
        if args.weight_decay == 0.0:
            args.weight_decay = 0.05
        if args.delay_lambda_phys_until_code_usage is None:
            args.delay_lambda_phys_until_code_usage = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--run-name", default="codec_v3")
    p.add_argument("--codec-version", choices=["v3", "v4", "v5a", "v5"], default=None)
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup (v4 default 1000)")
    p.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="After warmup: flat LR or cosine decay to --min-lr (v4 default: cosine)",
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Floor LR for cosine schedule (absolute, not a ratio)",
    )
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-loss", type=float, default=50.0, help="Skip steps with loss above this")
    p.add_argument(
        "--checkpoint-metric",
        choices=["median", "mean", "val_rms", "val_std_ratio_per_spec_median"],
        default="median",
        help="Train window (v3), val RMS (v4), or per-spec std_ratio median (v5a/v5)",
    )
    p.add_argument(
        "--min-code-usage-fraction",
        type=float,
        default=0.0,
        help="Reject val checkpoints when unique-code fraction is below this (v5a/v5 default 0.3)",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--healpix-holdout-frac", type=float, default=0.0)
    p.add_argument("--healpix-split-seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=0, help="Run val every N steps (0=disabled)")
    p.add_argument("--val-max-batches", type=int, default=32)
    p.add_argument("--lambda-phys", type=float, default=0.0)
    p.add_argument(
        "--lambda-phys-ramp-steps",
        type=int,
        default=0,
        help="Linearly ramp λ_phys from 0 to target over N steps (v4 default 4000)",
    )
    p.add_argument(
        "--delay-lambda-phys-until-code-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hold λ_phys at 0 until val code_usage_fraction ≥ min-code-usage-fraction (v5a/v5 default: on)",
    )
    p.add_argument("--lambda-entropy", type=float, default=0.0)
    p.add_argument("--commitment-weight", type=float, default=None, help="LFQ beta (v4 default 0.05)")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-wandb-artifact", action="store_true")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per GPU")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    args.codec_version = infer_codec_version(args.run_name, args.codec_version)
    apply_version_defaults(args)

    if args.codec_version == "v5":
        input_style = INPUT_STYLE_V5
    elif args.codec_version in ("v4", "v5a"):
        input_style = INPUT_STYLE_V4
    else:
        input_style = INPUT_STYLE_V3

    if args.codec_version in ("v4", "v5a", "v5"):
        if args.lambda_phys == 0.0:
            args.lambda_phys = 0.5
        if args.lambda_entropy == 0.0:
            if args.codec_version == "v5":
                args.lambda_entropy = 1.5
            elif args.codec_version == "v5a":
                args.lambda_entropy = 0.75
            else:
                args.lambda_entropy = 0.1
    commitment = args.commitment_weight
    if commitment is None:
        commitment = 0.05 if args.codec_version in ("v4", "v5a", "v5") else 0.25

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
        num_workers=args.num_workers,
        collate_fn=collate_spectra,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
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
    delay_phys = bool(args.delay_lambda_phys_until_code_usage)
    phys_unlocked = not delay_phys
    phys_ramp_origin: int | None = 0 if phys_unlocked else None

    if main_proc:
        log.info("=== spectrum codec %s (%s) ===", args.codec_version, input_style)
        log.info("run_dir=%s", run_dir)
        log.info("manifest=%s", manifest_path or "(synthetic)")
        log.info(
            "dataset_size=%d batch_size=%d steps=%d lr=%g schedule=%s min_lr=%g warmup=%d "
            "λ_phys=%g ramp=%d delay_phys_until_usage=%s (gate=%.2f) λ_ent=%g commit=%g",
            len(ds),
            args.batch_size,
            args.steps,
            args.lr,
            args.lr_schedule,
            args.min_lr,
            args.warmup_steps,
            args.lambda_phys,
            args.lambda_phys_ramp_steps,
            delay_phys,
            args.min_code_usage_fraction,
            args.lambda_entropy,
            commitment,
        )
        eff_batch = args.batch_size * world_size
        log.info(
            "device=%s world_size=%d per_gpu_batch=%d effective_batch=%d "
            "checkpoint_metric=%s val_every=%d holdout=%g num_workers=%d",
            device,
            world_size,
            args.batch_size,
            eff_batch,
            args.checkpoint_metric,
            args.val_every,
            args.healpix_holdout_frac,
            args.num_workers,
        )

    if args.codec_version == "v5":
        from desifm.tokenization.spectrum_codec_v5 import SpectrumCodecV5

        model = SpectrumCodecV5(commitment_weight=commitment).to(device)
    else:
        model = SpectrumCodec(commitment_weight=commitment).to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(
        unwrap(model).parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    fwd_extra = model_forward_kw(args.codec_version)
    wb = init_run(args.wandb_mode, args.run_name, vars(args), run_dir / "wandb", group="phase2-codec") if main_proc else None
    if main_proc and wb is not None:
        log.info("wandb run: %s", getattr(wb, "url", wb.name))

    step, best_metric = 0, float("inf")
    skipped = 0
    t0 = time.perf_counter()
    it = iter(loader)

    def lr_scale(s: int) -> float:
        return learning_rate_scale(
            s,
            total_steps=args.steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            schedule=args.lr_schedule,
            min_lr=args.min_lr,
        )

    def checkpoint_score() -> float:
        assert tracker is not None
        return tracker.window_median() if args.checkpoint_metric == "median" else tracker.window_mean()

    def checkpoint_metric_value(val_stats: dict[str, float]) -> float:
        if args.checkpoint_metric == "val_std_ratio_per_spec_median":
            return -val_stats["std_ratio_per_spec_median"]
        return val_stats["rms"]

    def save_best(metric: float, val_stats: dict[str, float] | None = None) -> None:
        nonlocal best_metric
        if metric >= best_metric:
            return
        if val_stats is not None and args.min_code_usage_fraction > 0:
            n_codes_gate = int(val_stats.get("n_codes", 256))
            n_unique = float(val_stats.get("n_unique_codes", 0))
            if not code_usage_passes_gate(n_unique, n_codes_gate, args.min_code_usage_fraction):
                if main_proc and log:
                    log.info(
                        "skip checkpoint: unique=%.0f / gate_bins=%d (%.3f) < %.3f",
                        n_unique,
                        min(n_codes_gate, 256),
                        val_stats.get("usage_fraction_gate", 0.0),
                        args.min_code_usage_fraction,
                    )
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
        lambda_phys_eff = effective_lambda_phys(
            step,
            args.lambda_phys,
            args.lambda_phys_ramp_steps,
            delay_until_code_usage=delay_phys,
            phys_unlocked=phys_unlocked,
            phys_ramp_origin=phys_ramp_origin,
        )
        ramp = (
            lambda_phys_eff / args.lambda_phys
            if args.lambda_phys > 0
            else 0.0
        )
        out = unwrap(model)(
            x,
            denorm,
            mask,
            lambda_phys=lambda_phys_eff,
            lambda_entropy=args.lambda_entropy,
            **fwd_extra,
        )
        loss = out["loss"]

        loss_f = loss.item()
        local_skip = not torch.isfinite(loss) or (args.max_loss > 0 and loss_f > args.max_loss)
        if all_ranks_agree_skip(local_skip, device):
            if main_proc:
                skipped += 1
                if step % args.log_every == 0:
                    log.warning("skip step %d loss=%.4g (max_loss=%g)", step, loss_f, args.max_loss)
            step += 1
            continue

        if main_proc and tracker is not None:
            tracker.update(loss_f)

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
                    "lambda_phys_eff": lambda_phys_eff,
                    "lambda_ramp": ramp,
                    "lambda_phys_unlocked": phys_unlocked,
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
                    "train/lambda_phys_eff": lambda_phys_eff,
                    "train/lambda_ramp": ramp,
                    "train/lambda_phys_unlocked": float(phys_unlocked),
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
                val_lambda_phys = effective_lambda_phys(
                    step,
                    args.lambda_phys,
                    args.lambda_phys_ramp_steps,
                    delay_until_code_usage=delay_phys,
                    phys_unlocked=phys_unlocked,
                    phys_ramp_origin=phys_ramp_origin,
                )
                val_stats = run_validation(
                    unwrap(model),
                    val_loader,
                    device,
                    input_style,
                    codec_version=args.codec_version,
                    lambda_phys=val_lambda_phys,
                    lambda_entropy=args.lambda_entropy,
                    max_batches=args.val_max_batches,
                )
                if (
                    delay_phys
                    and not phys_unlocked
                    and code_usage_passes_gate(
                        val_stats["n_unique_codes"],
                        int(val_stats.get("n_codes", 256)),
                        args.min_code_usage_fraction,
                    )
                ):
                    phys_unlocked = True
                    phys_ramp_origin = step
                    log.info(
                        "λ_phys unlocked at step %d (unique=%.0f, gate_frac=%.3f ≥ %.3f); "
                        "ramp %d steps to target %.3f",
                        step,
                        val_stats["n_unique_codes"],
                        val_stats.get("code_usage_fraction_gate", 0.0),
                        args.min_code_usage_fraction,
                        args.lambda_phys_ramp_steps,
                        args.lambda_phys,
                    )
                append_metrics(metrics_path, {"kind": "val", "step": step, **val_stats})
                log.info(
                    "val step=%d rms=%.4f recon=%.4f std_ratio=%.3f std_med=%.3f "
                    "codes=%.0f usage=%.3f q=%.4f ent=%.4f",
                    step,
                    val_stats["rms"],
                    val_stats["recon"],
                    val_stats["std_ratio"],
                    val_stats["std_ratio_per_spec_median"],
                    val_stats["n_unique_codes"],
                    val_stats["code_usage_fraction"],
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
                        "val/std_ratio_per_spec_median": val_stats["std_ratio_per_spec_median"],
                        "val/n_unique_codes": val_stats["n_unique_codes"],
                        "val/code_usage_fraction": val_stats["code_usage_fraction"],
                        "val/code_usage_fraction_gate": val_stats.get("code_usage_fraction_gate", 0.0),
                        "val/q_loss": val_stats["q"],
                        "val/entropy_penalty": val_stats["entropy"],
                    },
                    step,
                )
                if args.checkpoint_metric in ("val_rms", "val_std_ratio_per_spec_median"):
                    save_best(checkpoint_metric_value(val_stats), val_stats)

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
        if delay_phys and not phys_unlocked:
            log.warning(
                "λ_phys never unlocked (code usage stayed below %.3f); "
                "consider higher λ_ent or v5b architecture",
                args.min_code_usage_fraction,
            )
        log.info("=== done === best=%.4f skipped=%d -> %s", best_metric, skipped, run_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()
