"""SCRATCH-only filesystem helpers for NERSC."""

from __future__ import annotations

import os
from pathlib import Path


def scratch_root() -> Path:
    """Return deepsrch workspace root on SCRATCH.

    ``NERSC_SCRATCH_ROOT`` is often already ``$SCRATCH/deepsrch`` (manifests live there).
    Do not append ``deepsrch`` again in that case.
    """
    root = Path(os.environ.get("NERSC_SCRATCH_ROOT", os.environ.get("SCRATCH", "/tmp")))
    if (root / "manifests").is_dir() or (root / "checkpoints").is_dir():
        return root
    nested = root / "deepsrch"
    if nested.is_dir():
        return nested
    return nested


def require_scratch_manifest(path: Path) -> Path:
    p = Path(path)
    if not p.name.endswith("_scratch.jsonl") and not os.environ.get("ALLOW_CFS_MANIFEST"):
        raise ValueError(
            f"Training requires a staged manifest (*_scratch.jsonl), got {p}. "
            "Run scripts/stage_data.py first."
        )
    return p
