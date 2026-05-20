#!/usr/bin/env python3
"""Smoke-test official AION DESI spectrum tokenizer (encode one batch)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from desifm.constants import N_LATENT_TOKENS, N_SPECTRUM_CODES
from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, load_manifest
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.tokenization.aion_bridge import AionSpectrumTokenizer
from desifm.training.env import load_project_env
from desifm.training.paths import scratch_root


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("smoke_aion")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(run_dir / "smoke.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def main():
    load_project_env()
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default="smoke_aion_tokenizer")
    p.add_argument(
        "--scratch-out",
        type=Path,
        default=None,
        help="Log directory root (default: $SCRATCH/checkpoints or ./checkpoints)",
    )
    args = p.parse_args()

    out_root = args.scratch_out or scratch_root() / "checkpoints"
    run_dir = out_root / args.run_name
    log = setup_logging(run_dir)
    log.info("=== AION spectrum tokenizer smoke ===")
    log.info("run_dir=%s", run_dir)
    log.info("device=%s batch_size=%d synthetic=%s", args.device, args.batch_size, args.synthetic)
    if args.manifest:
        log.info("manifest=%s", args.manifest)
    sys.stdout.flush()

    t0 = time.time()
    if args.synthetic:
        log.info("building synthetic batch (length=4096)...")
        ds = SyntheticSpectrumDataset(n_spectra=max(4, args.batch_size), length=4096)
        items = [ds[i] for i in range(args.batch_size)]
        batch = collate_spectra(items)
    else:
        if args.manifest is None:
            raise SystemExit("--manifest required unless --synthetic")
        log.info("loading manifest %s ...", args.manifest)
        records = load_manifest(args.manifest)
        ds = DR1StreamDataset(records, max_spectra=args.batch_size * 4)
        items = []
        for i in range(len(ds)):
            it = ds[i]
            if it is not None:
                items.append(it)
            if len(items) >= args.batch_size:
                break
        if len(items) < 1:
            raise SystemExit("no valid spectra from manifest")
        batch = collate_spectra(items)
        log.info("loaded %d spectra from manifest", len(items))

    log.info(
        "batch shapes flux=%s wavelength=%s",
        tuple(batch["flux"].shape),
        tuple(batch["wavelength"].shape),
    )
    import os

    hf_home = os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")
    log.info("HF_HOME=%s", hf_home)
    log.info(
        "next: AION loads DESI codec from Hugging Face on first encode "
        "(1–5 min on login node; compute nodes may hang without cache — see scripts/prefetch_aion_codec.py)"
    )
    sys.stdout.flush()

    desifm_log = logging.getLogger("desifm")
    desifm_log.setLevel(logging.INFO)
    for h in log.handlers:
        if h not in desifm_log.handlers:
            desifm_log.addHandler(h)

    device = torch.device(args.device)
    tok = AionSpectrumTokenizer(device)
    t_enc0 = time.time()
    spec_idx, meta = tok.encode_batch(batch)
    encode_sec = time.time() - t_enc0

    n_unique = int(spec_idx.unique().numel())
    record = {
        "shape": list(spec_idx.shape),
        "min": int(spec_idx.min()),
        "max": int(spec_idx.max()),
        "n_unique": n_unique,
        "meta": meta,
        "encode_sec": round(encode_sec, 3),
        "total_sec": round(time.time() - t0, 3),
        "device": str(device),
        "synthetic": args.synthetic,
    }
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("shape=%s (expected B,%d)", tuple(spec_idx.shape), N_LATENT_TOKENS)
    log.info("min=%d max=%d (codes 0..%d)", int(spec_idx.min()), int(spec_idx.max()), N_SPECTRUM_CODES - 1)
    log.info("n_unique=%d meta=%s", n_unique, meta)
    log.info("encode_time=%.2fs total_time=%.2fs", encode_sec, record["total_sec"])
    log.info("wrote %s", metrics_path)
    log.info("wrote %s", run_dir / "smoke.log")

    assert spec_idx.shape[1] == N_LATENT_TOKENS
    assert spec_idx.max() < N_SPECTRUM_CODES
    log.info("OK")
    print(f"OK -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
