# NERSC interactive workflow (SCRATCH only)

```bash
export NERSC_SCRATCH_ROOT=$SCRATCH/deepsrch
mkdir -p $NERSC_SCRATCH_ROOT/{manifests,dr1_staged,checkpoints,wandb,logs}

# W&B (`.env` is gitignored — not on NERSC after clone)
export WANDB_API_KEY=...   # or: wandb login
export WANDB_PROJECT=desi-fm-2026
```

## Reuse staged manifests (if present)

```bash
ls $NERSC_SCRATCH_ROOT/manifests/dr1_*_scratch.jsonl
```

## Stage (once per manifest)

```bash
python scripts/build_manifest.py --max-healpix 200 \
  --out $NERSC_SCRATCH_ROOT/manifests/dr1_200.jsonl

python scripts/stage_data.py \
  --src-manifest $NERSC_SCRATCH_ROOT/manifests/dr1_200.jsonl \
  --dst-root $NERSC_SCRATCH_ROOT/dr1_staged \
  --dst-manifest $NERSC_SCRATCH_ROOT/manifests/dr1_200_scratch.jsonl
```

## Train codec (1 GPU)

```bash
salloc -A deepsrch_g -C gpu -q interactive -t 02:00:00 --nodes=1 --gpus=1 --cpus-per-task=32
pip install -e .
python scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v1 --wandb-mode online

# Logs (stdout + files under run dir):
#   $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v1/train.log
#   $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v1/metrics.jsonl
```

## Train model (4 GPU DDP, Approach A then B)

```bash
python -m torch.distributed.run --nproc_per_node=4 scripts/train_model.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_10k_scratch.jsonl \
  --codec-ckpt $NERSC_SCRATCH_ROOT/checkpoints/codec_v1/best.pt \
  --approach a --run-name p5_approach_a --wandb-mode online

python -m torch.distributed.run --nproc_per_node=4 scripts/train_model.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_10k_scratch.jsonl \
  --codec-ckpt $NERSC_SCRATCH_ROOT/checkpoints/codec_v1/best.pt \
  --approach b --run-name p5_approach_b --wandb-mode online
```

## Local smoke (no FITS, synthetic data)

```bash
bash scripts/run_smoke_local.sh
```
