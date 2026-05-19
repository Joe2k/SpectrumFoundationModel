# NERSC interactive workflow (SCRATCH only)

On your **laptop**, use the repo virtualenv (`bash scripts/bootstrap_venv.sh`); this page is for **Perlmutter / NERSC** only.

```bash
export NERSC_SCRATCH_ROOT=$SCRATCH/deepsrch
mkdir -p $NERSC_SCRATCH_ROOT/{manifests,dr1_staged,checkpoints,wandb,logs}

# W&B (`.env` is gitignored — create on NERSC or export manually)
export WANDB_API_KEY=...
export WANDB_PROJECT=desi-fm-2026
export WANDB_DIR=$NERSC_SCRATCH_ROOT/wandb
# Do not set WANDB_DISABLE_SERVICE — breaks wandb init on recent versions.

# If online init fails on compute nodes (ServicePollForTokenError), use offline:
#   --wandb-mode offline
# then from a login node: wandb sync $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v3/wandb
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

Default manifest: **dr1_1k_scratch.jsonl** (1k healpix). Stage to SCRATCH first for I/O.

### codec_v4 (recommended)

Dual loss (arcsinh + physical flux), 5% healpix holdout, checkpoint on **val/rms_flux**.

```bash
salloc -A deepsrch_g -C gpu -q interactive -t 04:00:00 --nodes=1 --gpus=1 --cpus-per-task=32
pip install -e .
python scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v4 \
  --codec-version v4 \
  --wandb-mode online
```

Defaults with `--run-name codec_v4`: 20k steps, batch 32, lr 1e-4, warmup 1k, `λ_phys=0.5`, `λ_ent=0.1`, val every 500 steps.

### codec_v3 (legacy)

Mask-aware arcsinh + Huber; checkpoint on train loss median.

```bash
python scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v3 \
  --steps 5000 \
  --batch-size 16 \
  --checkpoint-metric median \
  --wandb-mode online
# or: --wandb-mode offline   # if ServicePollForTokenError on compute node

# Logs:
#   $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v3/train.log
#   $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v3/metrics.jsonl
```

**Note:** `codec_v2` checkpoints used a simpler median norm; retrain for `codec_v3` (incompatible input pipeline).

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
