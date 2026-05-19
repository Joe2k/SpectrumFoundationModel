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
| `replace_best_artifact()` | Upload `best.pt` to W&B; delete prior version (`art.wait()`) |
| Repo-root `.env` → `WANDB_API_KEY` | Online sync from laptop / NERSC `.env` |
| `prepare_codec_input()` | Per-spectrum median normalization (stable loss scale) |
| `LossTracker` (batch / avg / ema) | Readable metrics; checkpoint on **10-step avg** |
| `--grad-clip 1.0` | Training stability |

### Why `codec_v1` loss spiked to 400+

Raw-flux MSE scales with brightness; a single bright minibatch dominates `train/loss` while `best` tracked one lucky faint batch (~0.32). **Not model collapse** — logging artifact.

`codec_v2` uses normalized flux + avg-loss checkpoints; batch/avg losses stay O(0.01–10).

### Why `train/loss_ema` looked high mid-run

EMA uses every step’s batch loss (decay 0.98). Rare outlier batches (~step 180, ~4190) inject huge values; EMA decays slowly. **Ignore EMA for decisions**; use `train/best_avg`, `train/loss_batch`, and `train/recon`.

---

## W&B runs — spectrum codec (phase 2)

### `codec_v2` — **finished** (production tokenizer)

| Field | Value |
|-------|-------|
| Run ID | `nv7py9b1` |
| URL | https://wandb.ai/jjayaseelan-university-of-san-francisco/desi-fm-2026/runs/nv7py9b1 |
| State | **finished** |
| Manifest | `dr1_1k_scratch.jsonl` |
| Config | steps=5000, batch=16, lr=3e-4, log_every=10, grad_clip=1 |
| Runtime | ~63 min (3763 s) |
| Steps | 4990 |
| Group | `phase2-codec` |

**Checkpoint (use for transformer):**

```text
$NERSC_SCRATCH_ROOT/deepsrch/checkpoints/codec_v2/best.pt
```

Saved when **10-step `loss_avg`** improved; final `best_avg` = **1.375** (@ step ~4770).

**Final summary metrics (W&B @ step 4990):**

| Metric | Value |
|--------|-------|
| `train/best_avg` | **1.375** |
| `train/loss_avg` | 1.81 |
| `train/loss_batch` | 3.34 |
| `train/loss_ema` | 2.47 |
| `train/recon` | 3.34 |
| `train/q_loss` | 4.0×10⁻⁵ |

**Training arc (`best_avg`):**

| Phase | Step (approx) | best_avg |
|-------|---------------|----------|
| Warmup | 0–40 | 3.19 → 1.98 |
| Early | 80–410 | 1.67 → 1.50 |
| Mid | 1400–2090 | 1.45 → 1.44 |
| Late | 4770 | **1.375** (final best) |

VQ (`q_loss`) negligible throughout; reconstruction (`recon`) drives loss.

**Outlier batches (did not update `best.pt`):**

| Step | loss_batch | Notes |
|------|------------|-------|
| ~180 | batch ~2, **avg ~17M** | Bad row in 10-step window; EMA polluted for hundreds of steps |
| ~4190 | **~2.2×10⁶** | Numerical blow-up (recon dominated); `best_avg` unchanged at 1.436 |
| ~4770+ | batch ~1–2 | Normal; new best saved |

**Verdict:** Training healthy. Normalized MSE ~1–3 typical; `best.pt` from avg-loss criterion is valid for phase 5.

### `codec_v1` (killed @ step 1000 — pre-normalization)

| Field | Value |
|-------|-------|
| Run ID | `efvc5599` |
| URL | https://wandb.ai/jjayaseelan-university-of-san-francisco/desi-fm-2026/runs/efvc5599 |
| State | killed |
| Issue | Raw-flux MSE; `log_every=100`; single-batch `train/loss` |

| Step | train/loss | train/recon |
|------|------------|-------------|
| 0 | 214.1 | 213.5 |
| 300 | 117.0 | 117.0 |
| 600 | 467.2 | 467.2 |
| 1000 | 53.0 | 53.0 |

**Do not use** `codec_v1` for downstream training.

---

### Next

1. **Phase 5:** `train_model.py` with `--codec-ckpt .../codec_v2/best.pt` — Approach A then B (4× GPU DDP on `dr1_10k_scratch.jsonl` if staged)
2. Optional: codec eval notebook (`03_phase_codec_eval.ipynb`) on held-out healpix
3. Optional: retrain codec on `dr1_10k` if 1k tokenizer quality insufficient
4. Record transformer W&B run IDs in `TRAINING_REGISTRY.yaml`
5. Code hardening: skip non-finite / clipped loss batches; robust EMA (future)
