"""Regression: train_model.py with desifm SpectrumCodec checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


@pytest.mark.skipif(not PYTHON.is_file(), reason="repo .venv required")
def test_train_model_desifm_smoke(tmp_path: Path):
    codec_out = tmp_path / "codec"
    codec_out.mkdir()
    train_codec = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_codec.py"),
        "--synthetic",
        "--smoke",
        "--run-name",
        "pytest_codec",
        "--scratch-out",
        str(codec_out),
        "--wandb-mode",
        "disabled",
    ]
    subprocess.run(train_codec, cwd=ROOT, check=True, timeout=300)
    codec_ckpt = codec_out / "pytest_codec" / "best.pt"
    assert codec_ckpt.is_file()

    out = tmp_path / "fm"
    cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_model.py"),
        "--synthetic",
        "--smoke",
        "--spectrum-tokenizer",
        "desifm",
        "--codec-ckpt",
        str(codec_ckpt),
        "--approach",
        "a",
        "--run-name",
        "pytest_desifm_a",
        "--scratch-out",
        str(out),
        "--wandb-mode",
        "disabled",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=300)
    assert (out / "pytest_desifm_a" / "best.pt").is_file()
