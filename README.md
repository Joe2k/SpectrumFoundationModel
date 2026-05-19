# DESI Foundation Model (`desifm`)

**From-scratch** unimodal foundation model for DESI spectra + redshift (PHYS303 2026).

This codebase was rewritten cleanly for the final project. It is **not** a copy of the earlier `FoundationModel/` tree.

## Design

- **Spectrum codec**: 1D ConvNeXt-style encoder/decoder + lookup-free quantization → 273 discrete tokens
- **Redshift codec**: empirical CDF → Gaussian → uniform bins (256 levels)
- **Transformer**: encoder-decoder with modality embeddings
- **Approach A**: encoder sees `z`; auxiliary MLP on pooled encoder for joint redshift learning
- **Approach B**: encoder is spectrum-only; decoder always uses `REDMASK` at the redshift slot

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Local smoke (no FITS)

```bash
bash scripts/run_smoke_local.sh
```

Uses `--synthetic` random spectra; checkpoints land in `checkpoints/smoke_local/`.

## NERSC (interactive, SCRATCH-only)

See [docs/NERSC_INTERACTIVE.md](docs/NERSC_INTERACTIVE.md).

- W&B project: **`desi-fm-2026`**
- Train only on `*_scratch.jsonl` manifests
- Logs: `RESEARCH_LOG.md`, `TRAINING_REGISTRY.yaml`

## Layout

```
src/desifm/          # library code (all original)
scripts/             # CLI entry points
tests/               # tests written for this package
notebooks/           # phase notebooks (fill during runs)
```
