"""Resume training from run_dir/last.pt with the same --run-name."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "scripts" / "train_model.py"


def _codec_ckpt(tmp_path: Path) -> Path:
    codec_out = tmp_path / "codec"
    codec_out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
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
        ],
        cwd=ROOT,
        check=True,
        timeout=300,
    )
    path = codec_out / "pytest_codec" / "best.pt"
    assert path.is_file()
    return path


def _run_train(
    tmp_path: Path,
    run_name: str,
    codec_ckpt: Path,
    steps: int,
    extra: list[str] | None = None,
) -> Path:
    out = tmp_path / "ckpt"
    cmd = [
        str(PYTHON),
        str(SCRIPT),
        "--synthetic",
        "--spectrum-tokenizer",
        "desifm",
        "--codec-ckpt",
        str(codec_ckpt),
        "--approach",
        "b",
        "--run-name",
        run_name,
        "--scratch-out",
        str(out),
        "--steps",
        str(steps),
        "--batch-size",
        "2",
        "--d-model",
        "128",
        "--val-every",
        "5",
        "--log-every",
        "1",
        "--wandb-mode",
        "disabled",
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=300)
    return out / run_name


@pytest.mark.skipif(not PYTHON.is_file(), reason="repo .venv required")
def test_train_model_resumes_from_last_pt(tmp_path: Path):
    codec_ckpt = _codec_ckpt(tmp_path)
    run = "pytest_resume"
    run_dir = _run_train(tmp_path, run, codec_ckpt, steps=12)
    last = run_dir / "last.pt"
    assert last.is_file(), "expected last.pt after validation"
    ckpt = torch.load(last, weights_only=False)
    first_step = int(ckpt["step"])
    assert first_step >= 10
    assert "optimizer" in ckpt

    _run_train(tmp_path, run, codec_ckpt, steps=20)
    ckpt2 = torch.load(last, weights_only=False)
    assert int(ckpt2["step"]) > first_step


@pytest.mark.skipif(not PYTHON.is_file(), reason="repo .venv required")
def test_no_resume_starts_from_zero(tmp_path: Path):
    codec_ckpt = _codec_ckpt(tmp_path / "codec_only")
    run = "pytest_no_resume"
    run_dir = _run_train(tmp_path, run, codec_ckpt, steps=8)
    last = run_dir / "last.pt"
    assert last.is_file()

    _run_train(tmp_path, run, codec_ckpt, steps=16, extra=["--no-resume"])
    ckpt = torch.load(last, weights_only=False)
    assert ckpt["step"] == 15
