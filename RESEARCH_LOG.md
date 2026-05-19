# Research Log (fresh implementation)

All code in `src/desifm/` is an original implementation for the final submission.
Earlier experiments in `../FoundationModel/` are reference only — not imported.

## 2026-05-18: Rewrite from scratch

- Deleted prior copied tree; rebuilt package `desifm` v0.2.0
- Approach B: `REDMASK` token in decoder on every step
- SCRATCH-only training policy; W&B project `desi-fm-2026`

## 2026-05-18: DDP, notebooks, local smoke

- `train_codec.py` / `train_model.py`: `torchrun` DDP via `desifm.training.distributed`
- `--synthetic` flag + `SyntheticSpectrumDataset` for laptop smoke (no FITS)
- `bash scripts/run_smoke_local.sh` — 21 pytest passed; smoke A/B completed under `checkpoints/smoke_local/`
- Phase notebooks `notebooks/01_phase_data.ipynb` … `08_phase_submission.ipynb`

### Next

1. Stage manifest on Perlmutter if needed
2. Production: `torchrun` codec + Approach A/B on `dr1_10k_scratch.jsonl`
3. Update `TRAINING_REGISTRY.yaml` with NERSC run IDs
