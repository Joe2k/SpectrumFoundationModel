#!/usr/bin/env bash
# Create/update the repo-local virtualenv and install this package (editable).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  echo "Creating ${ROOT}/.venv ..."
  python3 -m venv "${ROOT}/.venv"
fi

"${ROOT}/.venv/bin/python" -m pip install -U pip
"${ROOT}/.venv/bin/pip" install -e ".[dev,aion]"

echo "Done. Use this interpreter for all work in this repo:"
echo "  ${ROOT}/.venv/bin/python"
echo "Or: source ${ROOT}/.venv/bin/activate"
