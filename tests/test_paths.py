import os

from desifm.training.paths import scratch_root


def test_scratch_root_when_nersc_root_is_deepsrch(tmp_path, monkeypatch):
    deepsrch = tmp_path / "deepsrch"
    (deepsrch / "manifests").mkdir(parents=True)
    monkeypatch.setenv("NERSC_SCRATCH_ROOT", str(deepsrch))
    monkeypatch.delenv("SCRATCH", raising=False)
    assert scratch_root() == deepsrch


def test_scratch_root_when_scratch_is_parent(tmp_path, monkeypatch):
    scratch = tmp_path / "pscratch"
    deepsrch = scratch / "deepsrch"
    (deepsrch / "manifests").mkdir(parents=True)
    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.delenv("NERSC_SCRATCH_ROOT", raising=False)
    assert scratch_root() == deepsrch
