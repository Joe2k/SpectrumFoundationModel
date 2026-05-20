#!/usr/bin/env bash
# Local end-to-end smoke: AION tokenizer + Approach A/B (synthetic data, no FITS).
# Optional legacy desifm codec path: SMOKE_DESIFM_CODEC=1
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

if [[ "${SMOKE_DESIFM_CODEC:-0}" == "1" ]]; then
  echo "== legacy desifm codec smoke =="
  "$PYTHON" "${ROOT}/scripts/train_codec.py" \
    --synthetic --smoke \
    --run-name smoke_codec \
    --scratch-out "$OUT" \
    --wandb-mode disabled
  CODEC="${OUT}/smoke_codec/best.pt"
  echo "== approach A (desifm codec) =="
  "$PYTHON" "${ROOT}/scripts/train_model.py" \
    --synthetic --smoke \
    --spectrum-tokenizer desifm \
    --codec-ckpt "$CODEC" \
    --approach a \
    --run-name smoke_approach_a_desifm \
    --scratch-out "$OUT" \
    --wandb-mode disabled
fi

echo "== AION spectrum tokenizer encode smoke =="
"$PYTHON" "${ROOT}/scripts/smoke_aion_tokenizer.py" --synthetic --scratch-out "$OUT"

echo "== approach A (AION tokenizer) =="
"$PYTHON" "${ROOT}/scripts/train_model.py" \
  --synthetic --smoke \
  --spectrum-tokenizer aion \
  --approach a \
  --run-name smoke_approach_a_aion \
  --scratch-out "$OUT" \
  --wandb-mode disabled

echo "== approach B (AION tokenizer) =="
"$PYTHON" "${ROOT}/scripts/train_model.py" \
  --synthetic --smoke \
  --spectrum-tokenizer aion \
  --approach b \
  --run-name smoke_approach_b_aion \
  --scratch-out "$OUT" \
  --wandb-mode disabled

echo "OK -> $OUT"
