"""Weights & Biases helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

WANDB_PROJECT = "desi-fm-2026"


def init_run(
    mode: str,
    name: str,
    config: dict,
    dir: Path,
    group: str | None = None,
    tags: list[str] | None = None,
):
    if mode == "disabled":
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        mode = "offline"
    os.environ["WANDB_MODE"] = mode
    try:
        import wandb
    except ImportError:
        return None
    Path(dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=WANDB_PROJECT,
        name=name,
        config=config,
        dir=str(dir),
        group=group,
        tags=tags or ["final-2026"],
    )


def log_metrics(run, metrics: dict[str, Any], step: int) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass


def finish(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
