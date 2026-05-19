#!/usr/bin/env bash
# Local end-to-end smoke: codec pretrain + Approach A/B (synthetic data, no FITS).
# Requires repo .venv (see scripts/bootstrap_venv.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: missing ${PYTHON}" >&2
  echo "Run: bash ${ROOT}/scripts/bootstrap_venv.sh" >&2
  exit 1
fi
export PYTHONPATH="${ROOT}/src"
OUT="${ROOT}/checkpoints/smoke_local"
mkdir -p "$OUT"

echo "== codec smoke =="
"$PYTHON" "${ROOT}/scripts/train_codec.py" \
  --synthetic --smoke \
  --run-name smoke_codec \
  --scratch-out "$OUT" \
  --wandb-mode disabled

CODEC="${OUT}/smoke_codec/best.pt"
echo "== approach A smoke =="
"$PYTHON" "${ROOT}/scripts/train_model.py" \
  --synthetic --smoke \
  --codec-ckpt "$CODEC" \
  --approach a \
  --run-name smoke_approach_a \
  --scratch-out "$OUT" \
  --wandb-mode disabled

echo "== approach B smoke =="
"$PYTHON" "${ROOT}/scripts/train_model.py" \
  --synthetic --smoke \
  --codec-ckpt "$CODEC" \
  --approach b \
  --run-name smoke_approach_b \
  --scratch-out "$OUT" \
  --wandb-mode disabled

echo "OK -> $OUT"
