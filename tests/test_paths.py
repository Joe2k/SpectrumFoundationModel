import os
import pytest
from pathlib import Path
from desifm.training.paths import require_scratch_manifest, scratch_root


def test_scratch_manifest_policy():
    with pytest.raises(ValueError):
        require_scratch_manifest(Path("/tmp/dr1.jsonl"))
    require_scratch_manifest(Path("/tmp/dr1_scratch.jsonl"))
