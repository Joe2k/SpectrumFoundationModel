#!/usr/bin/env python3
"""Train DesiFoundationModel (Approach A or B)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, healpix_split, load_manifest
from desifm.model.transformer import DesiFoundationModel
from desifm.tokenization.redshift_codec import RedshiftCodec
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.batching import build_sequences, tokenize_batch
from desifm.training.metrics import accuracy, masked_spec_accuracy
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--codec-ckpt", type=Path, required=True)
    p.add_argument("--approach", choices=["a", "b"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--scratch-out", type=Path, default=None)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-weight", type=float, default=20.0)
    p.add_argument("--aux-z-weight", type=float, default=0.5)
    p.add_argument("--encoder-mask-ratio", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps, args.batch_size, args.d_model = 30, 2, 128

    manifest = require_scratch_manifest(args.manifest) if not args.smoke else args.manifest
    records = load_manifest(manifest)
    train_rec, val_rec = healpix_split(records, holdout=0.05, seed=42)

    out_root = args.scratch_out or scratch_root() / "checkpoints"
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = SpectrumCodec().to(device)
    codec.load_state_dict(torch.load(args.codec_ckpt, map_location=device, weights_only=False)["model"])
    codec.eval()

    z_codec = RedshiftCodec()
    z_codec.fit(collect_z(train_rec))

    train_ds = DR1StreamDataset(train_rec, max_spectra=200 if args.smoke else None)
    val_ds = DR1StreamDataset(val_rec, max_spectra=50 if args.smoke else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_spectra)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_spectra)

    model = DesiFoundationModel(d_model=args.d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    wb = init_run(
        args.wandb_mode,
        args.run_name,
        vars(args),
        run_dir / "wandb",
        group=f"phase5-approach-{args.approach}",
        tags=["final-2026", f"approach-{args.approach}"],
    )

    step, best_val = 0, float("inf")
    train_it = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_loader)
            batch = next(train_it)
        if batch is None:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_schedule(step, args.lr, warmup=200, total=args.steps)
        spec_idx, z_idx = tokenize_batch(batch, codec, z_codec, device)
        enc, dec, tgt, mask_pos = build_sequences(
            spec_idx, z_idx, args.approach, encoder_mask_ratio=args.encoder_mask_ratio
        )
        logits, loss = model(enc, dec, targets=tgt, z_weight=args.z_weight, aux_z_weight=args.aux_z_weight, approach=args.approach)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            m = accuracy(logits.detach(), tgt)
            ms = masked_spec_accuracy(logits.detach(), tgt, mask_pos)
            rec = {"kind": "train", "step": step, "loss": loss.item(), **{f"acc/{k}": v for k, v in m.items()}, "acc/masked_spec": ms}
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            log_metrics(wb, {"train/loss": loss.item(), "train/z_acc": m["z"], "train/spec_acc": m["spec"]}, step)

        if step > 0 and step % 100 == 0:
            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    if vb is None:
                        continue
                    si, zi = tokenize_batch(vb, codec, z_codec, device)
                    enc, dec, tgt, mp = build_sequences(si, zi, args.approach, args.encoder_mask_ratio)
                    lg, ls = model(enc, dec, targets=tgt, z_weight=args.z_weight, aux_z_weight=args.aux_z_weight, approach=args.approach)
                    vloss += ls.item()
                    vn += 1
            vloss /= max(vn, 1)
            if vloss < best_val:
                best_val = vloss
                torch.save(
                    {"model": model.state_dict(), "z_codec": z_codec.state_dict(), "approach": args.approach, "step": step},
                    run_dir / "best.pt",
                )
            log_metrics(wb, {"val/loss": vloss}, step)
            model.train()
        step += 1

    finish(wb)
    print(f"done {args.run_name} best_val={best_val:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
