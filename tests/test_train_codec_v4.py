"""Smoke tests for codec v4 training CLI defaults."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_codec as tc  # noqa: E402


def test_infer_codec_version():
    assert tc.infer_codec_version("codec_v4_main", None) == "v4"
    assert tc.infer_codec_version("codec_v3", None) == "v3"
    assert tc.infer_codec_version("codec_v5a_antollapse", None) == "v5a"
    assert tc.infer_codec_version("codec_v5b_main", None) == "v5"


def test_apply_version_defaults_v5a():
    args = tc.argparse.Namespace(
        codec_version="v5a",
        steps=5000,
        batch_size=16,
        lr=3e-4,
        checkpoint_metric="median",
        healpix_holdout_frac=0.0,
        val_every=0,
        lr_schedule="constant",
        lambda_phys_ramp_steps=0,
        warmup_steps=0,
        min_code_usage_fraction=0.0,
        weight_decay=0.0,
    )
    tc.apply_version_defaults(args)
    assert args.checkpoint_metric == "val_std_ratio_per_spec_median"
    assert args.min_code_usage_fraction == 0.3


def test_apply_version_defaults():
    args = tc.argparse.Namespace(
        codec_version="v4",
        steps=5000,
        batch_size=16,
        lr=3e-4,
        checkpoint_metric="median",
        healpix_holdout_frac=0.0,
        val_every=0,
        lr_schedule="constant",
        lambda_phys_ramp_steps=0,
        warmup_steps=0,
    )
    tc.apply_version_defaults(args)
    assert args.steps == 20_000
    assert args.batch_size == 32
    assert args.lr == 1e-4
    assert args.checkpoint_metric == "val_rms"
    assert args.healpix_holdout_frac == 0.05
    assert args.val_every == 500
    assert args.lr_schedule == "cosine"
    assert args.lambda_phys_ramp_steps == 4000


def test_lambda_ramp_scale():
    assert tc.lambda_ramp_scale(0, 4000) == 0.0
    assert abs(tc.lambda_ramp_scale(2000, 4000) - 0.5) < 1e-6
    assert tc.lambda_ramp_scale(4000, 4000) == 1.0
    assert tc.lambda_ramp_scale(9000, 4000) == 1.0
    assert tc.lambda_ramp_scale(100, 0) == 1.0


def test_learning_rate_scale_warmup_and_cosine():
    base = 1e-4
    warmup = 1000
    total = 20_000
    assert tc.learning_rate_scale(0, total_steps=total, base_lr=base, warmup_steps=warmup, schedule="cosine") < 0.01
    assert abs(
        tc.learning_rate_scale(999, total_steps=total, base_lr=base, warmup_steps=warmup, schedule="cosine") - 0.999
    ) < 0.01
    assert abs(
        tc.learning_rate_scale(1000, total_steps=total, base_lr=base, warmup_steps=warmup, schedule="cosine") - 1.0
    ) < 1e-6
    end = tc.learning_rate_scale(
        total - 1, total_steps=total, base_lr=base, warmup_steps=warmup, schedule="cosine", min_lr=1e-6
    )
    assert end < 0.02
    assert tc.learning_rate_scale(5000, total_steps=total, base_lr=base, warmup_steps=0, schedule="constant") == 1.0
