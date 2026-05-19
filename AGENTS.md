# Agent notes (local Python)

For **local** work in this repository, use the virtualenv at `.venv/` only:

- `bash scripts/bootstrap_venv.sh` — create/update `.venv` and `pip install -e ".[dev]"`.
- Prefer `.venv/bin/python`, `.venv/bin/pip`, `.venv/bin/pytest` in commands and docs examples.

Do not install project dependencies with the system/global `python` / `pip` for this repo. NERSC/cluster flows follow `docs/NERSC_INTERACTIVE.md` and may use different environments.
