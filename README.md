# DESI Foundation Model (`desifm`)

**From-scratch** unimodal foundation model for DESI spectra + redshift (PHYS303 2026).

This codebase was rewritten cleanly for the final project. It is **not** a copy of the earlier `FoundationModel/` tree.

## Design

- **Spectrum codec**: ConvNeXt + LFQ; mask-aware norm, arcsinh compression, Huber loss → 273 tokens
- **Redshift codec**: empirical CDF → Gaussian → uniform bins (256 levels)
- **Transformer**: encoder-decoder with modality embeddings
- **Approach A**: encoder sees `z`; auxiliary MLP on pooled encoder for joint redshift learning
- **Approach B**: encoder is spectrum-only; decoder always uses `REDMASK` at the redshift slot

## Install (local: venv only)

Use a **repo-local** virtualenv. Do not `pip install` this project into your system Python; that environment is shared with other tools and cannot be cleaned up “for this repo only” without manual `pip uninstall` decisions on your side.

```bash
bash scripts/bootstrap_venv.sh
```

Then use `.venv/bin/python` (or `source .venv/bin/activate`). Shortcuts:

```bash
make bootstrap   # same as the script above
make test        # pytest inside .venv
make smoke       # local training smoke (requires .venv)
make download    # DR1 sample download script
```

In VS Code / Cursor, use the interpreter **`./.venv/bin/python`** (`.vscode/settings.json` points there).

## Local smoke (no FITS)

```bash
make smoke
```

Uses `--synthetic` random spectra; checkpoints land in `checkpoints/smoke_local/`.

## Local DR1 sample (real FITS, laptop)

`notebooks/01_phase_data.ipynb` downloads **one** iron healpix tile by default (two FITS: coadd + redrock) from the [DESI public portal](https://data.desi.lbl.gov/public/) into `data/dr1_public/` (gitignored) and builds `data/manifests/local_dr1.jsonl`. That notebook is **local-only** (no NERSC paths). Increase `N_HEALPIX` in the notebook (or pass `--max-tiles`) to pull more tiles from the built-in catalog.

```bash
make download
```

## NERSC (interactive, SCRATCH-only)

See [docs/NERSC_INTERACTIVE.md](docs/NERSC_INTERACTIVE.md).

- W&B project: **`desi-fm-2026`** — put `WANDB_API_KEY=...` in `.env` at repo root (local only; not committed)
- Train only on `*_scratch.jsonl` manifests
- Logs: `RESEARCH_LOG.md`, `TRAINING_REGISTRY.yaml`

## Layout

```
src/desifm/          # library code (all original)
scripts/             # CLI entry points (bootstrap_venv.sh, run_smoke_local.sh, …)
tests/               # tests written for this package
notebooks/           # phase notebooks (use ../.venv/bin/python kernel)
Makefile             # venv-scoped shortcuts (make test, smoke, download)
.cursor/rules/       # agent conventions (venv-only Python)
```
