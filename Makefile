.PHONY: venv bootstrap test smoke download

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

# Create .venv and install package (editable + dev deps)
bootstrap venv:
	bash scripts/bootstrap_venv.sh

test:
	@test -x $(PYTHON) || (echo "Run: make bootstrap" >&2; exit 1)
	$(PYTHON) -m pytest

smoke:
	bash scripts/run_smoke_local.sh

download:
	@test -x $(PYTHON) || (echo "Run: make bootstrap" >&2; exit 1)
	$(PYTHON) scripts/download_dr1_local.py
