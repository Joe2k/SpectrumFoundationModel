# Research Log (fresh implementation)

All code in `src/desifm/` is an original implementation for the final submission.
Earlier experiments in `../FoundationModel/` are reference only — not imported.

W&B project: **desi-fm-2026** (`jjayaseelan-university-of-san-francisco/desi-fm-2026`)

## 2026-05-18: Rewrite from scratch

- Deleted prior copied tree; rebuilt package `desifm` v0.2.0
- Approach B: `REDMASK` token in decoder on every step
- SCRATCH-only training policy; W&B project `desi-fm-2026`

## 2026-05-18: DDP, notebooks, local smoke

- `train_codec.py` / `train_model.py`: `torchrun` DDP via `desifm.training.distributed`
- `--synthetic` flag + `SyntheticSpectrumDataset` for laptop smoke (no FITS)
- `bash scripts/run_smoke_local.sh` — pytest + smoke A/B under `checkpoints/smoke_local/`
- Phase notebooks `notebooks/01_phase_data.ipynb` … `08_phase_submission.ipynb`

## 2026-05-18: Codec training observability (NERSC)

### Code changes

| Change | Purpose |
|--------|---------|
| `train.log` + `metrics.jsonl` under run dir | Persistent logs on SCRATCH |
| `--log-every 10` (default) | Frequent stdout + W&B metrics |
| `replace_best_artifact()` | Upload `best.pt` to W&B; delete prior version |
| Repo-root `.env` → `WANDB_API_KEY` | Online sync from laptop / NERSC `.env` |
| `prepare_codec_input()` | Per-spectrum median normalization (stable loss scale) |
| `LossTracker` (batch / avg / ema) | Readable metrics; checkpoint on **10-step avg** |
| `--grad-clip 1.0` | Training stability |

### Why `codec_v1` loss spiked to 400+

Raw-flux MSE scales with brightness; a single bright minibatch dominates `train/loss` while `best` tracked one lucky faint batch (~0.32). **Not model collapse** — logging artifact.

`codec_v2` uses normalized flux + avg-loss checkpoints; batch/avg losses stay O(0.01–10).

---

## W&B runs — spectrum codec (phase 2)

### `codec_v2` (latest, **running**)

| Field | Value |
|-------|-------|
| Run ID | `nv7py9b1` |
| URL | https://wandb.ai/jjayaseelan-university-of-san-francisco/desi-fm-2026/runs/nv7py9b1 |
| State | running (as of 2026-05-18 fetch) |
| Manifest | `dr1_1k_scratch.jsonl` |
| Config | steps=5000, batch=16, lr=3e-4, log_every=10, grad_clip=1 |
| Group | `phase2-codec` |

**Summary metrics @ step 170** (from W&B):

| Metric | Value |
|--------|-------|
| `train/best_avg` | **1.665** |
| `train/loss_avg` | 1.719 (last logged window) |
| `train/loss_batch` | 1.758 |
| `train/loss_ema` | 3.362 |
| `train/recon` | 1.753 |
| `train/q_loss` | 0.005 |

**Training curve (sampled, every ~10 steps):**

| Step | loss_batch | loss_avg | best_avg | recon |
|------|------------|----------|----------|-------|
| 0 | 3.19 | 3.19 | 3.19 | 2.46 |
| 30 | 1.17 | 2.11 | **2.11** | 1.13 |
| 40 | 0.66 | **1.98** | **1.98** | 0.64 |
| 80 | 2.34 | 1.67 | **1.67** | 2.32 |
| 90 | 1.76 | 1.72 | 1.67 | 1.75 |
| 120 | 1.35 | 9.31 | 1.67 | 1.33 |
| 170 | 2.15 | 2.06 | 1.67 | 2.15 |

`loss_avg` windows can spike when one hard batch enters the 10-step window; `best_avg` is monotonic and is the checkpoint criterion.

### `codec_v1` (killed @ step 1000 — pre-normalization)

| Field | Value |
|-------|-------|
| Run ID | `efvc5599` |
| URL | https://wandb.ai/jjayaseelan-university-of-san-francisco/desi-fm-2026/runs/efvc5599 |
| State | killed |
| Issue | Raw-flux MSE; `log_every=100`; single-batch `train/loss` |

**Sampled history (shows spikes):**

| Step | train/loss | train/recon |
|------|------------|-------------|
| 0 | 214.1 | 213.5 |
| 200 | 18.6 | 18.6 |
| 300 | **117.0** | 117.0 |
| 600 | **467.2** | 467.2 |
| 900 | 1.75 | 1.75 |
| 1000 | 53.0 | 53.0 |

Do **not** use `codec_v1` checkpoint for downstream transformer training; use `codec_v2` when complete.

---

### Next

1. Let `codec_v2` finish (5000 steps on `dr1_1k_scratch.jsonl`)
2. Scale to `dr1_10k_scratch.jsonl` if time permits
3. `train_model.py --codec-ckpt .../codec_v2/best.pt` Approach A then B (4× GPU DDP)
4. Record transformer run IDs in `TRAINING_REGISTRY.yaml`
