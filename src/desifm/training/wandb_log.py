"""Weights & Biases helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

WANDB_PROJECT = "desi-fm-2026"


def _project_root() -> Path:
    """Repo root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def _load_project_dotenv() -> bool:
    """Load .env from repo root. Returns True if file exists."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = _project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        return True
    load_dotenv(override=False)
    return False


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
    had_env_file = _load_project_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    if mode == "online" and not api_key:
        print(
            "[wandb] WANDB_API_KEY not set — falling back to offline. "
            "Add WANDB_API_KEY=... to .env at repo root, or run `wandb login`.",
            flush=True,
        )
        if had_env_file:
            print("[wandb] Found .env but WANDB_API_KEY is missing or empty.", flush=True)
        mode = "offline"
    elif mode == "online" and api_key:
        src = str(_project_root() / ".env") if had_env_file and (_project_root() / ".env").is_file() else "environment"
        print(f"[wandb] API key loaded from {src}; mode=online project={WANDB_PROJECT}", flush=True)
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
