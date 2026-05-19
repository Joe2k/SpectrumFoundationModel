from pathlib import Path

from desifm.training.wandb_log import replace_best_artifact


def test_replace_best_artifact_noop_without_run(tmp_path: Path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"x")
    state: dict[str, str | None] = {"qualified": None}
    replace_best_artifact(None, ckpt, "test-codec-best", 0, 1.0, state)
    assert state["qualified"] is None
