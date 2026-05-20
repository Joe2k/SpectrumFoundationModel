"""W&B run id discovery for training resume."""

from __future__ import annotations

from pathlib import Path

import torch

from desifm.training.wandb_log import find_wandb_run_id, save_wandb_run_id


def test_find_wandb_run_id_from_txt(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_wandb_run_id(run_dir, "abc12345")
    assert find_wandb_run_id(run_dir) == "abc12345"


def test_find_wandb_run_id_from_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = run_dir / "last.pt"
    torch.save({"step": 10, "wandb_id": "xyz98765"}, ckpt)
    assert find_wandb_run_id(run_dir, ckpt) == "xyz98765"


def test_find_wandb_run_id_from_wandb_dir(tmp_path: Path):
    run_dir = tmp_path / "run"
    wandb_run = run_dir / "wandb" / "run-20250519_220935-2c13i7w7"
    wandb_run.mkdir(parents=True)
    assert find_wandb_run_id(run_dir) == "2c13i7w7"
