"""SCRATCH-only filesystem helpers for NERSC."""

from __future__ import annotations

import os
from pathlib import Path


def scratch_root() -> Path:
    return Path(os.environ.get("NERSC_SCRATCH_ROOT", os.environ.get("SCRATCH", "/tmp"))) / "deepsrch"


def require_scratch_manifest(path: Path) -> Path:
    p = Path(path)
    if not p.name.endswith("_scratch.jsonl") and not os.environ.get("ALLOW_CFS_MANIFEST"):
        raise ValueError(
            f"Training requires a staged manifest (*_scratch.jsonl), got {p}. "
            "Run scripts/stage_data.py first."
        )
    return p
