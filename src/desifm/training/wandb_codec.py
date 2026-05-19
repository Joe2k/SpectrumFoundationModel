"""Download spectrum codec checkpoints from Weights & Biases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

WANDB_ENTITY = "jjayaseelan-university-of-san-francisco"
WANDB_PROJECT = "desi-fm-2026"


def _load_dotenv_optional() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[3]
    p = root / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def codec_artifact_name(run_name: str) -> str:
    """W&B artifact logical name (no entity/project prefix)."""
    return f"{run_name.replace('/', '_')}-codec-best"


def download_codec_best_pt(
    run_name: str,
    cache_dir: Path,
    *,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    alias: str = "best",
) -> Path:
    """Download ``{run_name}-codec-best:{alias}`` and return path to ``best.pt`` inside the run root.

    Requires ``WANDB_API_KEY`` (or ``wandb login``) for online access.
    """
    _load_dotenv_optional()
    import wandb

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logical = codec_artifact_name(run_name)
    qualified = f"{entity}/{project}/{logical}:{alias}"
    api = wandb.Api()
    artifact = api.artifact(qualified)
    root = Path(artifact.download(root=str(cache_dir / run_name)))
    ckpt = root / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Downloaded artifact missing best.pt under {root}")
    return ckpt


def wandb_run_history_df(
    run_id: str,
    keys: list[str],
    *,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
    samples: int | None = 2000,
) -> Any:
    """Return a pandas DataFrame of scalar metrics vs step (for learning-curve plots).

    Some wandb versions return a **list** of dicts from ``run.history``; we always coerce
    to a :class:`pandas.DataFrame` for a stable ``.columns`` API.
    """
    import pandas as pd
    import wandb

    _load_dotenv_optional()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    raw = run.history(keys=keys, samples=samples or 10000)
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    return pd.DataFrame(list(raw))


def wandb_run_url(run_id: str, *, entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def ensure_wandb_auth() -> bool:
    """True if WANDB_API_KEY is set or wandb is already logged in."""
    _load_dotenv_optional()
    return bool(os.environ.get("WANDB_API_KEY"))
