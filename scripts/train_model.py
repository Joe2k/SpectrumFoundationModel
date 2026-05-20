#!/usr/bin/env python3
"""Train DesiFoundationModel (Approach A or B). Supports DDP via torchrun."""

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
from desifm.model.transformer import DesiFoundationModel
from desifm.tokenization.aion_bridge import AionSpectrumTokenizer
from desifm.tokenization.redshift_codec import RedshiftCodec
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.batching import build_sequences, tokenize_batch
from desifm.training.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    unwrap,
    wrap_ddp,
)
from desifm.training.metrics import accuracy, masked_spec_accuracy
from desifm.training.env import load_project_env
from desifm.training.paths import require_scratch_manifest, scratch_root
from desifm.training.wandb_log import find_wandb_run_id, finish, init_run, log_metrics, save_wandb_run_id


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("train_model")
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


def _loss_parts_float(parts: dict | None) -> dict[str, float]:
    if not parts:
        return {}
    out = {}
    for k, v in parts.items():
        out[k] = float(v.item()) if torch.is_tensor(v) else float(v)
    if "loss_spec" in out:
        out["perplexity_spec"] = float(math.exp(min(out["loss_spec"], 20.0)))
    return out


def lr_schedule(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def build_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    z_codec: RedshiftCodec,
    step: int,
    best_val: float,
    args: argparse.Namespace,
    wandb_id: str | None = None,
) -> dict:
    out = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "z_codec": z_codec.state_dict(),
        "step": step,
        "best_val": best_val,
        "approach": args.approach,
        "spectrum_tokenizer": args.spectrum_tokenizer,
        "d_model": args.d_model,
        "lr": args.lr,
        "steps": args.steps,
        "z_weight": args.z_weight,
        "aux_z_weight": args.aux_z_weight,
        "encoder_mask_ratio": args.encoder_mask_ratio,
        "run_name": args.run_name,
    }
    if wandb_id:
        out["wandb_id"] = wandb_id
    return out


def save_checkpoint(path: Path, payload: dict, log: logging.Logger | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    if log is not None:
        log.info("saved checkpoint step=%s -> %s", payload.get("step"), path)


def _validate_checkpoint_meta(ckpt: dict, args: argparse.Namespace) -> None:
    for key in ("approach", "spectrum_tokenizer", "d_model"):
        if key in ckpt and ckpt[key] != getattr(args, key):
            raise SystemExit(
                f"checkpoint {key}={ckpt[key]!r} does not match CLI {getattr(args, key)!r}; "
                "use a new --run-name or --no-resume"
            )


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    last = run_dir / "last.pt"
    if last.is_file():
        return last
    best = run_dir / "best.pt"
    if best.is_file():
        return best
    return None


def load_resume_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    z_codec: RedshiftCodec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float, bool, str | None]:
    """Load checkpoint; return (start_step, best_val, full_resume, wandb_id)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _validate_checkpoint_meta(ckpt, args)
    unwrap(model).load_state_dict(ckpt["model"])
    z_codec.load_state_dict(ckpt["z_codec"])
    full = "optimizer" in ckpt
    if full:
        optimizer.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("step", -1)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    wandb_id = ckpt.get("wandb_id")
    return start, best_val, full, str(wandb_id) if wandb_id else None


def collect_z(records: list[dict], max_files: int = 100) -> torch.Tensor:
    import numpy as np
    from astropy.io import fits

    parts = []
    for rec in records[:max_files]:
        with fits.open(rec["redrock"], memmap=True) as h:
            z = h["REDSHIFTS"].data["Z"]
            w = h["REDSHIFTS"].data["ZWARN"]
            good = w == 0
            if good.any():
                parts.append(z[good])
    if not parts:
        return torch.tensor([0.0], dtype=torch.float32)
    return torch.tensor(np.concatenate(parts), dtype=torch.float32)


def collect_z_synthetic(n: int = 5000, seed: int = 0) -> torch.Tensor:
    ds = SyntheticSpectrumDataset(n_spectra=n, seed=seed)
    return torch.stack([ds[i]["z"] for i in range(n)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument(
        "--spectrum-tokenizer",
        choices=["aion", "desifm"],
        default="aion",
        help="Spectrum tokenizer: official AION (default) or desifm SpectrumCodec checkpoint",
    )
    p.add_argument("--codec-ckpt", type=Path, default=None, help="Required when --spectrum-tokenizer desifm")
    p.add_argument("--aion-hf-repo", default="polymathic-ai/aion-base")
    p.add_argument("--approach", choices=["a", "b"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers per GPU rank. DDP total = world_size * num_workers; "
        "use 2 on 4-GPU NERSC (8 loaders) to avoid host-RAM OOM killing workers.",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-weight", type=float, default=20.0)
    p.add_argument("--aux-z-weight", type=float, default=0.5)
    p.add_argument("--encoder-mask-ratio", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--log-every", type=int, default=10, help="Log train metrics every N steps")
    p.add_argument("--val-every", type=int, default=500, help="Run validation every N steps (0 = disable)")
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=32,
        help="Cap validation batches per eval (faster; full val if 0)",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use random spectra (no FITS/manifest)")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore run_dir/last.pt (or best.pt) and train from step 0",
    )
    args = p.parse_args()
    load_project_env()

    if args.spectrum_tokenizer == "aion":
        try:
            import aion  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Missing package 'aion' (install polymathic-aion). On NERSC run:\n"
                "  bash scripts/bootstrap_venv.sh\n"
                "  # or: .venv/bin/pip install -e .\n"
                "  .venv/bin/python -c \"import aion; print('ok')\""
            )

    if args.spectrum_tokenizer == "desifm" and args.codec_ckpt is None:
        raise SystemExit("--codec-ckpt required when --spectrum-tokenizer desifm")

    rank, world_size, _local_rank, device = setup_distributed()
    main_proc = is_main_process(rank)

    if args.smoke:
        args.steps, args.batch_size, args.d_model = 30, 2, 128
        args.synthetic = True
        args.num_workers = 0
        args.log_every = 5
        args.val_every = 10

    if args.synthetic:
        train_ds = SyntheticSpectrumDataset(n_spectra=400, seed=42)
        val_ds = SyntheticSpectrumDataset(n_spectra=80, seed=99)
        z_samples = collect_z_synthetic(2000)
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        manifest = require_scratch_manifest(args.manifest)
        records = load_manifest(manifest)
        train_rec, val_rec = healpix_split(records, holdout=0.05, seed=42)
        train_ds = DR1StreamDataset(train_rec, max_spectra=200 if args.smoke else None)
        val_ds = DR1StreamDataset(val_rec, max_spectra=50 if args.smoke else None)
        z_samples = collect_z(train_rec)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    use_cuda = device.type == "cuda"
    loader_kw = dict(
        num_workers=args.num_workers,
        collate_fn=collate_spectra,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_kw,
    )

    out_root = args.scratch_out or scratch_root() / "checkpoints"
    run_dir = out_root / args.run_name
    log = setup_logging(run_dir) if main_proc else logging.getLogger("train_model")
    log.setLevel(logging.INFO)
    metrics_path = run_dir / "metrics.jsonl"
    val_every = args.val_every

    if main_proc:
        log.info("=== DesiFoundationModel approach %s ===", args.approach)
        log.info("run_dir=%s", run_dir)
        log.info(
            "spectrum_tokenizer=%s steps=%d batch_size=%d num_workers=%d d_model=%d device=%s world_size=%d",
            args.spectrum_tokenizer,
            args.steps,
            args.batch_size,
            args.num_workers,
            args.d_model,
            device,
            world_size,
        )
        log.info("manifest=%s", args.manifest if not args.synthetic else "(synthetic)")
        log.info("log_every=%d val_every=%d metrics=%s train.log=%s", args.log_every, val_every, metrics_path, run_dir / "train.log")
        sys.stdout.flush()

    if args.spectrum_tokenizer == "aion":
        spectrum_tok = AionSpectrumTokenizer(device, hf_repo=args.aion_hf_repo)
    else:
        spectrum_tok = SpectrumCodec().to(device)
        spectrum_tok.load_state_dict(
            torch.load(args.codec_ckpt, map_location=device, weights_only=False)["model"]
        )
        spectrum_tok.eval()

    model = DesiFoundationModel(d_model=args.d_model).to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)

    z_codec = RedshiftCodec()
    step, best_val = 0, float("inf")
    resume_ckpt: Path | None = None if args.no_resume else find_resume_checkpoint(run_dir)

    ckpt_wandb_id: str | None = None
    if resume_ckpt is not None:
        if main_proc:
            start_step, best_val, full, ckpt_wandb_id = load_resume_checkpoint(
                resume_ckpt, model, opt, z_codec, args, device
            )
            step = start_step
            if not full:
                log.warning(
                    "checkpoint %s has no optimizer state — model/z_codec restored; optimizer fresh",
                    resume_ckpt,
                )
            log.info(
                "resumed from %s at step %d (target steps=%d) best_val=%.4f",
                resume_ckpt.name,
                step,
                args.steps,
                best_val,
            )
        if world_size > 1:
            import torch.distributed as dist

            if not main_proc:
                start_step, best_val, _, ckpt_wandb_id = load_resume_checkpoint(
                    resume_ckpt, model, opt, z_codec, args, device
                )
                step = start_step
            meta = torch.tensor([step, best_val], device=device, dtype=torch.float64)
            dist.broadcast(meta, src=0)
            step = int(meta[0].item())
            best_val = float(meta[1].item())
    else:
        if main_proc:
            z_codec.fit(z_samples)
        if world_size > 1:
            import torch.distributed as dist

            state = [z_codec.state_dict()] if rank == 0 else [None]
            dist.broadcast_object_list(state, src=0)
            if rank != 0:
                z_codec.load_state_dict(state[0])

    if resume_ckpt is not None and step >= args.steps:
        if main_proc:
            log.info("checkpoint step %d >= --steps %d; nothing to do", step, args.steps)
            print(f"done {args.run_name} already at step {step}", flush=True)
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
        cleanup_distributed()
        return

    wandb_resume_id = None
    if resume_ckpt is not None:
        wandb_resume_id = find_wandb_run_id(run_dir, resume_ckpt) or ckpt_wandb_id

    wb = None
    if main_proc:
        wb = init_run(
            args.wandb_mode,
            args.run_name,
            vars(args),
            run_dir / "wandb",
            group=f"phase5-approach-{args.approach}",
            tags=[
                "final-2026",
                f"approach-{args.approach}",
                f"tokenizer-{args.spectrum_tokenizer}",
            ],
            resume_id=wandb_resume_id,
            resume_step=step if resume_ckpt is not None else None,
        )
        if wb is not None and getattr(wb, "id", None):
            save_wandb_run_id(run_dir, str(wb.id))
            if resume_ckpt is not None:
                log_metrics(wb, {"resume/from_step": step}, step=step)
    train_it = iter(train_loader)
    warmup = min(200, max(1, args.steps // 5))
    t0 = time.perf_counter()

    try:
        while step < args.steps:
            if train_sampler is not None:
                train_sampler.set_epoch(step)
            try:
                batch = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                batch = next(train_it)
            if batch is None:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_schedule(step, args.lr, warmup=warmup, total=args.steps)
            spec_idx, z_idx = tokenize_batch(batch, spectrum_tok, z_codec, device)
            enc, dec, tgt, mask_pos = build_sequences(
                spec_idx, z_idx, args.approach, encoder_mask_ratio=args.encoder_mask_ratio
            )
            logits, loss, loss_parts = unwrap(model)(
                enc,
                dec,
                targets=tgt,
                z_weight=args.z_weight,
                aux_z_weight=args.aux_z_weight,
                approach=args.approach,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.grad_clip)
            opt.step()
            lp = _loss_parts_float(loss_parts)

            if main_proc and step % args.log_every == 0:
                m = accuracy(logits.detach(), tgt)
                ms = masked_spec_accuracy(logits.detach(), tgt, mask_pos)
                n_unique = int(spec_idx.unique().numel())
                lr = opt.param_groups[0]["lr"]
                sps = (step + 1) / max(time.perf_counter() - t0, 1e-6)
                rec = {
                    "kind": "train",
                    "step": step,
                    "loss": loss.item(),
                    "grad_norm": float(grad_norm),
                    "lr": lr,
                    "steps_per_sec": sps,
                    "spectrum_tokenizer": args.spectrum_tokenizer,
                    "spec_codes_unique": n_unique,
                    **{f"acc/{k}": v for k, v in m.items()},
                    "acc/masked_spec": ms,
                    **lp,
                }
                append_metrics(metrics_path, rec)
                log.info(
                    "train step %d/%d loss=%.4f loss_z=%.4f loss_spec=%.4f z_acc=%.3f spec_acc=%.3f "
                    "masked_spec=%.3f grad_norm=%.3f codes_unique=%d lr=%.2e (%.2f step/s)",
                    step,
                    args.steps,
                    loss.item(),
                    lp.get("loss_z", float("nan")),
                    lp.get("loss_spec", float("nan")),
                    m["z"],
                    m["spec"],
                    ms,
                    float(grad_norm),
                    n_unique,
                    lr,
                    sps,
                )
                sys.stdout.flush()
                if wb:
                    wb_payload = {
                        "train/loss": loss.item(),
                        "train/z_acc": m["z"],
                        "train/spec_acc": m["spec"],
                        "train/overall_acc": m["overall"],
                        "train/masked_spec_acc": ms,
                        "train/spec_codes_unique": n_unique,
                        "train/lr": lr,
                        "train/steps_per_sec": sps,
                        "train/grad_norm": float(grad_norm),
                        "train/loss_z": lp.get("loss_z", 0.0),
                        "train/loss_spec": lp.get("loss_spec", 0.0),
                        "train/perplexity_spec": lp.get("perplexity_spec", 0.0),
                    }
                    if "loss_aux_z" in lp:
                        wb_payload["train/loss_aux_z"] = lp["loss_aux_z"]
                    log_metrics(wb, wb_payload, step)

            if main_proc and val_every > 0 and step > 0 and step % val_every == 0:
                log.info("validation step %d ...", step)
                sys.stdout.flush()
                unwrap(model).eval()
                vloss, vn = 0.0, 0
                vz_acc, vspec_acc, vmasked = 0.0, 0.0, 0.0
                vloss_z, vloss_spec = 0.0, 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        if vb is None:
                            continue
                        si, zi = tokenize_batch(vb, spectrum_tok, z_codec, device)
                        enc_v, dec_v, tgt_v, vmp = build_sequences(
                            si, zi, args.approach, args.encoder_mask_ratio
                        )
                        lg, ls, vparts = unwrap(model)(
                            enc_v,
                            dec_v,
                            targets=tgt_v,
                            z_weight=args.z_weight,
                            aux_z_weight=args.aux_z_weight,
                            approach=args.approach,
                        )
                        vm = accuracy(lg, tgt_v)
                        vms = masked_spec_accuracy(lg, tgt_v, vmp)
                        vlp = _loss_parts_float(vparts)
                        vloss += ls.item()
                        vz_acc += vm["z"]
                        vspec_acc += vm["spec"]
                        vmasked += vms if not math.isnan(vms) else 0.0
                        vloss_z += vlp.get("loss_z", 0.0)
                        vloss_spec += vlp.get("loss_spec", 0.0)
                        vn += 1
                        if args.val_max_batches > 0 and vn >= args.val_max_batches:
                            break
                vloss /= max(vn, 1)
                vz_acc /= max(vn, 1)
                vspec_acc /= max(vn, 1)
                vmasked /= max(vn, 1)
                vloss_z /= max(vn, 1)
                vloss_spec /= max(vn, 1)
                val_rec = {
                    "kind": "val",
                    "step": step,
                    "loss": vloss,
                    "loss_z": vloss_z,
                    "loss_spec": vloss_spec,
                    "perplexity_spec": float(math.exp(min(vloss_spec, 20.0))),
                    "acc/z": vz_acc,
                    "acc/spec": vspec_acc,
                    "acc/masked_spec": vmasked,
                    "n_batches": vn,
                    "best_val_loss": min(best_val, vloss),
                }
                append_metrics(metrics_path, val_rec)
                saved = vloss < best_val
                if saved:
                    best_val = vloss
                wb_id = str(wb.id) if wb is not None and getattr(wb, "id", None) else None
                ckpt = build_checkpoint(model, opt, z_codec, step, best_val, args, wandb_id=wb_id)
                save_checkpoint(run_dir / "last.pt", ckpt, log)
                if saved:
                    save_checkpoint(run_dir / "best.pt", ckpt, log)
                log.info(
                    "val step %d loss=%.4f loss_z=%.4f loss_spec=%.4f z_acc=%.3f spec_acc=%.3f "
                    "masked_spec=%.3f batches=%d best_val=%.4f%s",
                    step,
                    vloss,
                    vloss_z,
                    vloss_spec,
                    vz_acc,
                    vspec_acc,
                    vmasked,
                    vn,
                    best_val,
                    " -> saved best.pt" if saved else "",
                )
                sys.stdout.flush()
                if wb:
                    log_metrics(
                        wb,
                        {
                            "val/loss": vloss,
                            "val/loss_z": vloss_z,
                            "val/loss_spec": vloss_spec,
                            "val/perplexity_spec": val_rec["perplexity_spec"],
                            "val/z_acc": vz_acc,
                            "val/spec_acc": vspec_acc,
                            "val/masked_spec_acc": vmasked,
                        },
                        step,
                    )
                unwrap(model).train()

            step += 1
    finally:
        if main_proc and step > 0:
            try:
                wb_id = str(wb.id) if wb is not None and getattr(wb, "id", None) else None
                save_checkpoint(
                    run_dir / "last.pt",
                    build_checkpoint(model, opt, z_codec, step - 1, best_val, args, wandb_id=wb_id),
                    log,
                )
            except Exception:
                pass

    if main_proc:
        finish(wb)
        log.info("=== done === best_val=%.4f -> %s", best_val, run_dir)
        print(f"done {args.run_name} best_val={best_val:.4f} -> {run_dir}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
