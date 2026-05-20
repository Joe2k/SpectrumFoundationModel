"""Smoke tests for codec v4 training CLI defaults."""

import sys
from pathlib import Path

import pytest

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
        delay_lambda_phys_until_code_usage=None,
    )
    tc.apply_version_defaults(args)
    assert args.checkpoint_metric == "val_std_ratio_per_spec_median"
    assert args.min_code_usage_fraction == 0.3
    assert args.delay_lambda_phys_until_code_usage is True


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


def test_effective_lambda_phys_delayed_until_unlock():
    assert (
        tc.effective_lambda_phys(
            5000,
            0.5,
            4000,
            delay_until_code_usage=True,
            phys_unlocked=False,
            phys_ramp_origin=None,
        )
        == 0.0
    )
    assert (
        tc.effective_lambda_phys(
            6000,
            0.5,
            4000,
            delay_until_code_usage=True,
            phys_unlocked=True,
            phys_ramp_origin=5000,
        )
        == pytest.approx(0.125, rel=1e-4)
    )


def test_effective_lambda_phys_no_delay():
    assert tc.effective_lambda_phys(
        2000,
        0.5,
        4000,
        delay_until_code_usage=False,
        phys_unlocked=False,
        phys_ramp_origin=None,
    ) == pytest.approx(0.25, rel=1e-4)


def test_apply_version_defaults_v5_fm_profile():
    args = tc.argparse.Namespace(
        codec_version="v5",
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
        delay_lambda_phys_until_code_usage=None,
        loss_profile=None,
        diversity_loss_weight=None,
        lambda_arcsinh=None,
        quant_temperature_start=None,
        quant_temperature_min=None,
        quant_temperature_anneal_steps=None,
    )
    tc.apply_version_defaults(args)
    assert args.loss_profile == "fm"
    assert args.delay_lambda_phys_until_code_usage is False
    assert args.lambda_phys_ramp_steps == 0
    assert args.diversity_loss_weight == 2.0
    assert args.lambda_arcsinh == 0.1
    assert args.quant_temperature_anneal_steps == 2000


def test_apply_version_defaults_v5_desifm_profile():
    args = tc.argparse.Namespace(
        codec_version="v5",
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
        delay_lambda_phys_until_code_usage=None,
        loss_profile="desifm",
        diversity_loss_weight=None,
        lambda_arcsinh=None,
    )
    tc.apply_version_defaults(args)
    assert args.delay_lambda_phys_until_code_usage is True
    assert args.diversity_loss_weight == 0.0


def test_model_forward_kw_v5_fm():
    kw = tc.model_forward_kw("v5", loss_profile="fm", diversity_loss_weight=1.0, lambda_arcsinh=0.1)
    assert kw["loss_profile"] == "fm"
    assert kw["lambda_diversity"] == 1.0
    assert "use_batch_entropy" not in kw


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
