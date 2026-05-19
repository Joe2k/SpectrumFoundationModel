#!/bin/bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH -J codec-v4-ddp
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# 4-GPU DDP codec_v4 Tier 1 on one Perlmutter node.
#
# Submit from repo root:
#   export NERSC_SCRATCH_ROOT=$SCRATCH/deepsrch
#   sbatch scripts/train_codec_ddp.slurm
#
# Overrides:
#   RUN_NAME=codec_v4_tier1_ddp STEPS=20000 BATCH_SIZE=32 sbatch scripts/train_codec_ddp.slurm

set -euo pipefail
# Submit from repo root: cd $HOME/SpectrumFoundationModel && sbatch scripts/train_codec_ddp.slurm
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/SpectrumFoundationModel}}"
cd "$REPO"
module load "${PYTORCH_MODULE:-pytorch/2.8.0}"

export NERSC_SCRATCH_ROOT="${NERSC_SCRATCH_ROOT:-$SCRATCH/deepsrch}"
export WANDB_DIR="${WANDB_DIR:-$NERSC_SCRATCH_ROOT/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-desi-fm-2026}"

RUN_NAME="${RUN_NAME:-codec_v4_tier1_ddp}"
MANIFEST="${MANIFEST:-$NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPUS="${GPUS:-4}"

echo "[codec-ddp] run=$RUN_NAME manifest=$MANIFEST steps=$STEPS batch/GPU=$BATCH_SIZE effective=$((BATCH_SIZE * GPUS))"

srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  scripts/train_codec.py \
  --manifest "$MANIFEST" \
  --run-name "$RUN_NAME" \
  --codec-version v4 \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --wandb-mode online

echo "[codec-ddp] done -> $NERSC_SCRATCH_ROOT/checkpoints/$RUN_NAME"
