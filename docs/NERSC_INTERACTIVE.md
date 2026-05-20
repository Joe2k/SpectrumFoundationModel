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

Defaults with `--codec-version v4`: 20k steps, batch 32, lr 1e-4, warmup 1k, **cosine decay to min_lr 1e-6**, **λ_phys ramp 0→0.5 over 4k steps**, `λ_ent=0.1`, val every 500. W&B logs `val/std_ratio_per_spec_median` (honest collapse metric) alongside pooled `val/std_ratio`.

Suggested new run after `git pull`:

```bash
export RUN_NAME=codec_v4_tier1_cosine_ramp
sbatch scripts/train_codec_ddp.slurm
```

Disable cosine or ramp: `--lr-schedule constant` and/or `--lambda-phys-ramp-steps 0`.

### codec_v4 DDP (4 GPUs, recommended for 20k steps)

`train_codec.py` supports `torchrun` / DDP. On one full GPU node:

```bash
export NERSC_SCRATCH_ROOT=$SCRATCH/deepsrch
sbatch scripts/train_codec_ddp.slurm
```

Or interactively:

```bash
salloc -A deepsrch_g -C gpu -q interactive -t 04:00:00 --nodes=1 --gpus=4 --cpus-per-task=128
cd $HOME/SpectrumFoundationModel   # repo root — required before torchrun/sbatch
git pull
module load pytorch/2.8.0
pip install -e .
export NERSC_SCRATCH_ROOT=$SCRATCH/deepsrch

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v4_tier1_ddp \
  --codec-version v4 \
  --batch-size 32 \
  --num-workers 8 \
  --wandb-mode online
```

Per-GPU batch 32 → **effective batch 128** across 4 GPUs. Checkpoints: `$NERSC_SCRATCH_ROOT/checkpoints/<run_name>/`.

### codec_v5a (anti-collapse, v4 backbone)

Same architecture as v4; FM-style **batch entropy**, stronger `λ_ent`, checkpoint on **per-spec** `std_ratio` median, skip save if code usage &lt; 30%. **`λ_phys` delayed** until code usage ≥ 30% (default for v5a).

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v5a_antollapse \
  --codec-version v5a \
  --batch-size 32 \
  --num-workers 8 \
  --wandb-mode online
```

### codec_v5b (FM V2–inspired tokenizer)

`SpectrumCodecV5`: U-Net skips, cross-attention, `latent_dim=10`.

**Stop `codec_v5b_r2` / `codec_v5b_r3`** — batch histogram entropy (`bincount` on discrete indices) has **no gradient** to the encoder; scaling `λ_ent` only multiplies a saturated ~1.0 penalty.

**Recommended run (`codec_v5b_r4`, FM loss profile):** physical MSE from step 0, differentiable `latent_bit_balance_loss` on pre-quant `z`, in-quant histogram entropy at weight **0.1** (monitor via `train/entropy`).

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v5b_r4 \
  --codec-version v5 \
  --loss-profile fm \
  --diversity-loss-weight 1.0 \
  --no-delay-lambda-phys-until-code-usage \
  --batch-size 32 \
  --num-workers 8 \
  --wandb-mode online
```

Legacy **desifm** profile (delayed `λ_phys`, external batch entropy): `--loss-profile desifm` (v5a still uses v4 backbone + desifm defaults).

Watch W&B (r4): `val/n_unique_codes` (target ≥77), `train/diversity_loss` (should decrease), `train/entropy` (histogram monitor, target &lt;0.9 sustained), `val/std_ratio_per_spec_median` (&gt;0.5), `train/phys`, `train/q_loss`.

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
