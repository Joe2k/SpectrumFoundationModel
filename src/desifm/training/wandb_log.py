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
    from desifm.training.env import load_project_env

    return load_project_env()


def _configure_wandb_dirs(log_dir: Path) -> None:
    """Use SCRATCH-local dirs; avoids wandb service failures on compute nodes."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(log_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(log_dir / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(log_dir / "config"))
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "120")


def find_wandb_run_id(run_dir: Path, checkpoint_path: Path | None = None) -> str | None:
    """Resolve W&B run id for resume: wandb_id.txt, checkpoint, then local wandb/ dir."""
    id_file = run_dir / "wandb_id.txt"
    if id_file.is_file():
        rid = id_file.read_text().strip()
        if rid:
            return rid

    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            import torch

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            wid = ckpt.get("wandb_id")
            if wid:
                return str(wid)
        except Exception:
            pass

    wandb_dir = run_dir / "wandb"
    if not wandb_dir.is_dir():
        return None

    latest = wandb_dir / "latest-run"
    if latest.exists():
        target = latest.resolve() if latest.is_symlink() else latest
        rid = _run_id_from_wandb_dirname(target.name)
        if rid:
            return rid

    runs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_path in runs:
        rid = _run_id_from_wandb_dirname(run_path.name)
        if rid:
            return rid
    return None


def _run_id_from_wandb_dirname(dirname: str) -> str | None:
    # e.g. run-20250519_220935-2c13i7w7 -> 2c13i7w7
    if not dirname.startswith("run-"):
        return None
    rid = dirname.rsplit("-", 1)[-1]
    return rid if len(rid) >= 8 else None


def save_wandb_run_id(run_dir: Path, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb_id.txt").write_text(run_id.strip())


def _wandb_init(
    mode: str,
    name: str,
    config: dict,
    log_dir: Path,
    group: str | None,
    tags: list[str] | None,
    *,
    resume_id: str | None = None,
):
    import wandb

    # Avoid deprecated Settings fields (_disable_service, start_method) that break
    # recent wandb pydantic validation on NERSC.
    kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "name": name,
        "config": config,
        "dir": str(log_dir),
        "group": group,
        "tags": tags or ["final-2026"],
        "mode": mode,
    }
    if resume_id:
        kwargs["id"] = resume_id
        kwargs["resume"] = "must"
    return wandb.init(**kwargs)


def init_run(
    mode: str,
    name: str,
    config: dict,
    dir: Path,
    group: str | None = None,
    tags: list[str] | None = None,
    resume_id: str | None = None,
    resume_step: int | None = None,
):
    if mode == "disabled":
        return None
    had_env_file = _load_project_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    requested = mode
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
        print(f"[wandb] API key loaded from {src}; trying mode=online project={WANDB_PROJECT}", flush=True)
    try:
        import wandb
    except ImportError:
        return None

    log_dir = Path(dir)
    _configure_wandb_dirs(log_dir)

    run_dir = log_dir.parent

    def _do_init(rid: str | None, resume_mode: str | None) -> Any:
        import wandb

        kwargs: dict[str, Any] = {
            "project": WANDB_PROJECT,
            "name": name,
            "config": config,
            "dir": str(log_dir),
            "group": group,
            "tags": tags or ["final-2026"],
            "mode": mode,
        }
        if rid:
            kwargs["id"] = rid
            kwargs["resume"] = resume_mode or "must"
        return wandb.init(**kwargs)

    try:
        try:
            run = _do_init(resume_id, "must" if resume_id else None)
        except Exception as exc:
            if not resume_id:
                raise
            print(f"[wandb] resume=must failed ({exc}); retrying with resume=allow", flush=True)
            run = _do_init(resume_id, "allow")
        if resume_id:
            save_wandb_run_id(run_dir, resume_id)
            msg = f"[wandb] resumed id={resume_id} mode={mode} dir={log_dir}"
            if resume_step is not None:
                msg += f" from_step={resume_step}"
            print(msg, flush=True)
        else:
            print(f"[wandb] started mode={mode} dir={log_dir}", flush=True)
        if run is not None and getattr(run, "id", None):
            save_wandb_run_id(run_dir, str(run.id))
        return run
    except Exception as exc:
        if mode == "offline":
            print(f"[wandb] init failed ({exc}); continuing without W&B", flush=True)
            return None
        print(
            f"[wandb] online init failed ({exc}); falling back to offline. "
            f"Sync later: wandb sync {log_dir}",
            flush=True,
        )
        os.environ["WANDB_MODE"] = "offline"
        try:
            run = _wandb_init("offline", name, config, log_dir, group, tags, resume_id=resume_id)
            print(f"[wandb] offline run dir={log_dir} (requested {requested})", flush=True)
            if run is not None and getattr(run, "id", None):
                save_wandb_run_id(log_dir.parent, str(run.id))
            return run
        except Exception as exc2:
            print(f"[wandb] offline init also failed ({exc2}); continuing without W&B", flush=True)
            return None


def log_metrics(run, metrics: dict[str, Any], step: int) -> None:
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:
        pass


def replace_best_artifact(
    run,
    checkpoint_path: Path,
    artifact_name: str,
    step: int,
    loss: float,
    state: dict[str, str | None],
) -> None:
    """Upload best.pt to W&B; alias 'best' moves to the new version, then drop the old one."""
    if run is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return
    if os.environ.get("WANDB_MODE") == "offline":
        return
    try:
        import wandb

        old_qualified = state.get("qualified")
        art = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={"step": step, "loss": float(loss)},
        )
        art.add_file(str(checkpoint_path), name="best.pt")
        run.log_artifact(art, aliases=["best"])
        art.wait()
        qualified = getattr(art, "qualified_name", None) or (
            f"{run.entity}/{run.project}/{art.name}:v{art.version}"
        )
        state["qualified"] = qualified
        run.log({"train/best_loss": loss, "checkpoint/step": step}, step=step)
        print(f"[wandb] uploaded best artifact {qualified} (step={step} loss={loss:.4f})", flush=True)

        if old_qualified and old_qualified != qualified:
            try:
                wandb.Api().artifact(old_qualified).delete()
            except Exception:
                pass  # alias may still point at old version briefly; safe to leave extra versions
    except Exception as exc:
        print(f"[wandb] artifact upload failed: {exc}", flush=True)


def finish(run, artifact_state: dict[str, str | None] | None = None) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
