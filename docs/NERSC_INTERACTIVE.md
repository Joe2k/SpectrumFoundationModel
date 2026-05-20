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

**Stop `codec_v5b_r2`–`r5`** — marginal bit-balance + 4k `λ_phys` ramp still collapsed to ~3–5 codes.

**Recommended run (`codec_v5b_r6`):** v5 **fm** defaults after `git pull`:

| Default | Value |
|---------|--------|
| `λ_phys` | **0.5 from step 0** (no 4k ramp) |
| `diversity_loss_weight` | **2.0** |
| LFQ temperature | **1.0 → 0.1** over **2000** steps (`sign(z/τ)` STE) |

```bash
cd $HOME/SpectrumFoundationModel
git pull
module load pytorch/2.8.0
pip install -e .

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v5b_r6 \
  --codec-version v5 \
  --loss-profile fm \
  --no-delay-lambda-phys-until-code-usage \
  --batch-size 32 \
  --num-workers 8 \
  --wandb-mode online
```

Explicit overrides (usually not needed; fm defaults match r6):

```bash
  --lambda-phys-ramp-steps 0 \
  --diversity-loss-weight 2.0 \
  --quant-temperature-start 1.0 \
  --quant-temperature-min 0.1 \
  --quant-temperature-anneal-steps 2000
```

Legacy **desifm** profile: `--loss-profile desifm` (delayed `λ_phys`, 4k ramp, external batch entropy).

**Gate @ val step 500:** `val/n_unique_codes` **> 20** (on track to 77); if still ≤ 5, stop and debug.

Watch W&B: `val/n_unique_codes`, `train/diversity_loss`, `train/quant_temperature`, `train/entropy` (&lt;0.9), `train/lambda_phys_eff` (should be **0.5** immediately), `train/q_loss` (should stay **> 0.01** early).

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

## Train model (4 GPU DDP, official AION spectrum tokenizer)

**Prereq:** `pip install -e ".[dev,aion]"` in project venv. HF auth for gated `polymathic-ai/aion-base`: either copy repo `.env` with `HF_TOKEN=...` to NERSC, or `huggingface-cli login`.

```bash
# Encode smoke (1 process)
python scripts/smoke_aion_tokenizer.py --synthetic

# Transformer smoke (~30 steps)
python scripts/train_model.py --synthetic --smoke --spectrum-tokenizer aion \
  --approach b --run-name p5_smoke_b_aion --wandb-mode online

# Production — Approach B first (assignment focus)
python -m torch.distributed.run --nproc_per_node=4 scripts/train_model.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_10k_scratch.jsonl \
  --spectrum-tokenizer aion \
  --approach b --run-name p5_approach_b_aion \
  --steps 10000 --batch-size 8 --wandb-mode online

# Approach A (if time)
python -m torch.distributed.run --nproc_per_node=4 scripts/train_model.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_10k_scratch.jsonl \
  --spectrum-tokenizer aion \
  --approach a --run-name p5_approach_a_aion \
  --steps 10000 --batch-size 8 --wandb-mode online
```

**Legacy** (desifm `codec_v2` only): add `--spectrum-tokenizer desifm --codec-ckpt $NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v2/best.pt`.

**Note:** AION encode per batch is heavier than a small desifm codec; use `batch-size 4–8` if OOM or low step/s.

## Local smoke (no FITS, synthetic data)

```bash
bash scripts/run_smoke_local.sh
```
