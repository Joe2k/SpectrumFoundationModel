#!/usr/bin/env python3
"""Train DesiFoundationModel (Approach A or B). Supports DDP via torchrun."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from desifm.training.wandb_log import finish, init_run, log_metrics


def lr_schedule(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


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
        help="DataLoader workers per process (FITS I/O). Try 4–8 on NERSC per GPU rank.",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-weight", type=float, default=20.0)
    p.add_argument("--aux-z-weight", type=float, default=0.5)
    p.add_argument("--encoder-mask-ratio", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use random spectra (no FITS/manifest)")
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
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    if args.spectrum_tokenizer == "aion":
        spectrum_tok = AionSpectrumTokenizer(device, hf_repo=args.aion_hf_repo)
    else:
        spectrum_tok = SpectrumCodec().to(device)
        spectrum_tok.load_state_dict(
            torch.load(args.codec_ckpt, map_location=device, weights_only=False)["model"]
        )
        spectrum_tok.eval()

    z_codec = RedshiftCodec()
    if main_proc:
        z_codec.fit(z_samples)
    if world_size > 1:
        import torch.distributed as dist

        state = [z_codec.state_dict()] if rank == 0 else [None]
        dist.broadcast_object_list(state, src=0)
        if rank != 0:
            z_codec.load_state_dict(state[0])

    model = DesiFoundationModel(d_model=args.d_model).to(device)
    model = wrap_ddp(model, device, world_size)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr)

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
        )

    step, best_val = 0, float("inf")
    train_it = iter(train_loader)
    warmup = min(200, max(1, args.steps // 5))

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
        logits, loss = unwrap(model)(
            enc,
            dec,
            targets=tgt,
            z_weight=args.z_weight,
            aux_z_weight=args.aux_z_weight,
            approach=args.approach,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
        opt.step()

        if main_proc and step % 20 == 0:
            m = accuracy(logits.detach(), tgt)
            ms = masked_spec_accuracy(logits.detach(), tgt, mask_pos)
            n_unique = int(spec_idx.unique().numel())
            rec = {
                "kind": "train",
                "step": step,
                "loss": loss.item(),
                "spectrum_tokenizer": args.spectrum_tokenizer,
                "spec_codes_unique": n_unique,
                **{f"acc/{k}": v for k, v in m.items()},
                "acc/masked_spec": ms,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if wb:
                log_metrics(
                    wb,
                    {
                        "train/loss": loss.item(),
                        "train/z_acc": m["z"],
                        "train/spec_acc": m["spec"],
                        "train/spec_codes_unique": n_unique,
                    },
                    step,
                )

        if main_proc and step > 0 and step % max(20, args.steps // 3) == 0:
            unwrap(model).eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    if vb is None:
                        continue
                    si, zi = tokenize_batch(vb, spectrum_tok, z_codec, device)
                    enc_v, dec_v, tgt_v, _mp = build_sequences(
                        si, zi, args.approach, args.encoder_mask_ratio
                    )
                    _lg, ls = unwrap(model)(
                        enc_v,
                        dec_v,
                        targets=tgt_v,
                        z_weight=args.z_weight,
                        aux_z_weight=args.aux_z_weight,
                        approach=args.approach,
                    )
                    vloss += ls.item()
                    vn += 1
            vloss /= max(vn, 1)
            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {
                        "model": unwrap(model).state_dict(),
                        "z_codec": z_codec.state_dict(),
                        "approach": args.approach,
                        "spectrum_tokenizer": args.spectrum_tokenizer,
                        "step": step,
                    },
                    run_dir / "best.pt",
                )
            if wb:
                log_metrics(wb, {"val/loss": vloss}, step)
            unwrap(model).train()
        step += 1

    if main_proc:
        finish(wb)
        print(f"done {args.run_name} best_val={best_val:.4f} -> {run_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
