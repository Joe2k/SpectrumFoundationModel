import os
from pathlib import Path

from desifm.training.env import load_project_env, project_root


def test_load_project_env_maps_hf_token(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text('HF_TOKEN="hf_test_token_abc"\n')
    monkeypatch.setattr("desifm.training.env.project_root", lambda: tmp_path)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_project_env() is True
    assert os.environ.get("HF_TOKEN") == "hf_test_token_abc"
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == "hf_test_token_abc"


def test_project_root_is_repo():
    root = project_root()
    assert (root / "pyproject.toml").is_file()
