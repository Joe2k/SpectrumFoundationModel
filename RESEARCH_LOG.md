# Research Log (fresh implementation)

All code in `src/desifm/` is an original implementation for the final submission.
Earlier experiments in `../FoundationModel/` are reference only — not imported.

## 2026-05-18: Rewrite from scratch

- Deleted prior copied tree; rebuilt package `desifm` v0.2.0
- Approach B: `REDMASK` token in decoder on every step
- SCRATCH-only training policy; W&B project `desi-fm-2026`

### Next

1. Stage manifest on Perlmutter if needed
2. `scripts/train_codec.py` then `scripts/train_model.py --approach a|b`
3. Fill notebooks + `TRAINING_REGISTRY.yaml`
