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

## 2026-05-19: codec_v3 Tier A (AION-style preprocessing)

Implemented in `desifm.training.codec_input` + updated `SpectrumCodec` / `train_codec.py`:

| Feature | Detail |
|---------|--------|
| Norm | Mask-aware mean flux → `log10` → `denorm`; `(flux/denorm - 1)×0.2`; **arcsinh** |
| Sanitize | `nan_to_num`, ivar clip ≤ 100 |
| Loss | Huber (smooth L1) on **arcsinh flux**, mask-weighted pixels |
| Decode | `sinh` + inverse affine with per-spectrum `denorm` |
| Stability | Skip steps with `loss > 50` or non-finite; **median** over 10 steps for `best.pt` |
| Data | `mask` in DR1 dataset + collate (pad = masked) |
| Default run | `codec_v3` on **`dr1_1k_scratch.jsonl`** |

### `codec_v3` — **running** (Tier A)

| Field | Value |
|-------|-------|
| Run ID | `yqzndkc0` |
| URL | https://wandb.ai/jjayaseelan-university-of-san-francisco/desi-fm-2026/runs/yqzndkc0 |
| State | **running** (as of 2026-05-19) |
| Manifest | `dr1_1k_scratch.jsonl` |
| Group | `phase2-codec` |

**Train on NERSC (1k healpix):**

```bash
python scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v3 --steps 5000 --wandb-mode online
```

`codec_v2` checkpoints are **not compatible** (different input pipeline). Use `codec_v2` for a quick FM baseline or retrain `codec_v3` for AION-aligned tokens.

### Next (superseded by v4 — see below)

1. ~~Train **`codec_v3`**~~ — running; eval shows **physical collapse** despite low `train/recon`
2. **Phase 5** should wait for **`codec_v4`** with held-out healpix + physical metrics
3. Record transformer W&B run IDs in `TRAINING_REGISTRY.yaml`

---

## 2026-05-19: Local codec eval + training-tile download

### Public DR1 for laptop eval

| Piece | Detail |
|-------|--------|
| Discovery | `discover_public_training_tiles()` — same survey/program order as `nersc/build_dr1_index.py` |
| Portal gap | **sv3** not on `data.desi.lbl.gov`; first available tile ≈ **`main/bright/0/0`** |
| Manifest | `data/manifests/train_eval_dr1.jsonl` (1 healpix, coadd + redrock) |
| Notebooks | `01_phase_data.ipynb`, `03_phase_codec_eval.ipynb` — self-contained download cells |
| CLI | `python scripts/download_dr1_local.py --training-order` |

### Eval findings (codec_v2 / codec_v3 on real coadd)

| Check | Result |
|-------|--------|
| Stitched coadd | OK |
| v3 input roundtrip | OK → preprocessing/denorm correct |
| v2/v3 recon on real DR1 | **Flat** (decoder ≈ constant flux) |
| `std(recon)/std(coadd)` | ~0.01–0.02 vs ~1.6 on `main/bright` tile |
| v2 per-spec `recon_loss` | ~0.99 (consistent with `best_avg` ≈ 1.37) |
| v3 `train/recon` ~0.03 | Arcsinh Huber — **misleading** vs physical plots |
| Synthetic spectra | v2/v3 retain ~12–17% variance ratio — eval code OK, **OOD real DR1** fails |

**Root causes:** (1) train manifest mostly **sv3** on NERSC vs local **main** tile; (2) loss allows **mean prediction** in normalized space; (3) v2 undertrained; (4) no **healpix holdout** or **physical RMS** checkpoint gate.

### Helpers added

- `desifm.training.codec_eval` — v2 legacy linear forward, per-spectrum eval, W&B download
- `desifm.training.wandb_codec` — artifact + history helpers
- `scripts/render_codec_notebooks.py` — regenerate `02` / `03`

---

## 2026-05-19: **codec_v4** plan (in progress)

**Goal:** Spectrum tokenizer that reconstructs real DESI coadds on **held-out healpix**, not just low arcsinh loss.

### Tier 1 (this implementation)

| Feature | Detail |
|---------|--------|
| `input_style` | `mask_arcsinh_v4` (v3 norm + optional 5-pixel top-hat on flux) |
| Loss | Arcsinh Huber + **physical flux Huber** (`λ_phys` default 0.5) |
| VQ | Commitment **β=0.05**; **code-usage entropy** penalty (`λ_ent` default 0.1) |
| Data split | `healpix_split` 5% val healpix (entire tiles) |
| Checkpoint | Best on **val physical RMS** (not train arcsinh alone) |
| Scale | Default **20k** steps, batch 32, lr 1e-4, warmup 1000, cosine decay → min_lr 1e-6 (NERSC) |
| λ_phys | Target 0.5, **linear ramp over 4000 steps** (arcsinh-first, then physical) |
| Eval | `val/rms_flux`, `val/recon_arcsinh`, `val/std_ratio` (pooled), **`val/std_ratio_per_spec_median`**, `val/q_loss` |

### Tier 2 (after Tier 1 gates)

- U-Net skip connections + light decoder cross-attention (FoundationModel V2 recipe)
- Scale manifest 2k–10k healpix; SCRATCH-staged I/O

### Success gates (held-out healpix)

- `val/std_ratio` > 0.5 (recon flux std / target std on good pixels)
- `val/rms_flux` < 0.5 × median(σ) per spectrum (tune on val plots)
- Code usage > 30% of 256 latent indices
- Visual: emission/continuum visible in overlay (not flat line)

### Train on NERSC

```bash
python scripts/train_codec.py \
  --manifest $NERSC_SCRATCH_ROOT/manifests/dr1_1k_scratch.jsonl \
  --run-name codec_v4 \
  --steps 20000 \
  --healpix-holdout-frac 0.05 \
  --val-every 500 \
  --lambda-phys 0.5 \
  --lambda-entropy 0.1 \
  --checkpoint-metric val_rms \
  --wandb-mode online
```
