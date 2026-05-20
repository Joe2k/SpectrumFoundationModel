"""Subprocess smoke for train_model.py with AION spectrum tokenizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"

pytest.importorskip("aion")


@pytest.mark.aion
@pytest.mark.skipif(not PYTHON.is_file(), reason="repo .venv required")
def test_train_model_aion_smoke(tmp_path: Path):
    out = tmp_path / "smoke_aion"
    cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_model.py"),
        "--synthetic",
        "--smoke",
        "--spectrum-tokenizer",
        "aion",
        "--approach",
        "b",
        "--run-name",
        "pytest_aion_b",
        "--scratch-out",
        str(out),
        "--wandb-mode",
        "disabled",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=600)
    ckpt = out / "pytest_aion_b" / "best.pt"
    metrics = out / "pytest_aion_b" / "metrics.jsonl"
    assert ckpt.is_file(), "best.pt missing"
    assert metrics.is_file() and metrics.stat().st_size > 0
