"""Smoke tests for codec v4 training CLI defaults."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_codec as tc  # noqa: E402


def test_infer_codec_version():
    assert tc.infer_codec_version("codec_v4_main", None) == "v4"
    assert tc.infer_codec_version("codec_v3", None) == "v3"


def test_apply_version_defaults():
    args = tc.argparse.Namespace(
        codec_version="v4",
        steps=5000,
        batch_size=16,
        lr=3e-4,
        checkpoint_metric="median",
        healpix_holdout_frac=0.0,
        val_every=0,
    )
    tc.apply_version_defaults(args)
    assert args.steps == 20_000
    assert args.batch_size == 32
    assert args.lr == 1e-4
    assert args.checkpoint_metric == "val_rms"
    assert args.healpix_holdout_frac == 0.05
    assert args.val_every == 500
