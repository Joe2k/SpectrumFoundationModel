"""Load repo-root .env for W&B, Hugging Face, etc."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_project_env() -> bool:
    """Load ``.env`` from repo root. Returns True if the file exists.

    Ensures ``HF_TOKEN`` is visible to ``huggingface_hub`` (also sets
    ``HUGGING_FACE_HUB_TOKEN`` when unset).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
    return env_path.is_file()
