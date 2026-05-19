# FoundationModel vs `desifm` — data & plotting

This compares the sibling repo **`../FoundationModel`** (course-era tree) with this project’s **`desifm`** pipeline. Paths below are relative to each repo root.

## Band stitching (B / R / Z)

| | FoundationModel | `desifm` |
|---|------------------|----------|
| **Implementation** | `src/utils/data.py` → `stitch_bands` | `src/desifm/data/stitch.py` → `stitch_bands` |
| **Logic** | Concatenate bands, sort by λ, merge duplicates within **0.1 Å** using **ivar-weighted** flux; combined `ivar` is sum of ivar on good pixels; mask if all masked in bin | Same |

The stitched arrays are effectively the same physical quantities.

## Quality cuts & datasets

| | FoundationModel | `desifm` |
|---|------------------|----------|
| **Main dataset** | `DESISpectrumDataset` in `src/utils/data.py` — reads one coadd (+ optional redrock), **builds a Python list** of all good spectra in memory | `DR1StreamDataset` in `src/desifm/data/dr1_stream.py` — **JSONL manifest**, opens FITS **on demand** with `memmap` |
| **Cuts** | `COADD_FIBERSTATUS`, `ZWARN`, nonzero total flux | Same |
| **Batch fields** | Also returns **`wavelength`**, **`ra`**, **`dec`** per item; `collate_desi_batch` stacks them | Returns **`flux`**, **`ivar`**, **`mask`**, **`z`** only (wavelength implied by fixed stitched grid; no RA/DEC in batch) |
| **NERSC DR1 stream** | `nersc/dr1_dataset.py` reuses `stitch_bands` from `src/utils/data.py` | Native `dr1_stream` + `public_dr1` for local downloads |

## Plotting (why curves can *look* different)

| | FoundationModel | `desifm` |
|---|------------------|----------|
| **Entry point** | `src/utils/plotting.py` → `plot_spectrum`, `plot_spectrum_grid` | `src/desifm/viz/spectrum_plot.py` → `plot_spectrum_with_lines` (+ notebook panels) |
| **Masked pixels** | Good flux: main line; bad: separate **faint red** trace | Gray **`fill_between`** band between min/max flux |
| **±1σ from `ivar`** | `sigma = 1/sqrt(ivar + 1e-20)`; **`fill_between` only where `~mask`** | Full-wavelength σ (with **display cap** on huge σ so y-limits stay readable) |
| **Y-axis limits** | **`ax.set_ylim(median(good) ± 5·std(good))`** on unmasked flux — **strongly stabilizes** the view so continuum slope and red-side rise are visible | Same idea: **`median ± ylim_k_std·std`** on good pixels (default **5**), expanded to include the **capped ±1σ** band; disable with ``adaptive_ylim=False`` |
| **Y-axis label** | Explicit **10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹** label | **“Coadd native units”** (same order of magnitude as DESI docs; not read from FITS `TUNIT*` automatically) |
| **Extra overlays** | — | **Emission / absorption** rest lines at λ_obs = λ_rest (1+z) |

So: FoundationModel plots were tuned to **avoid autoscale traps** (good-only error band + fixed y-window). `desifm` now caps the σ band for display and uses sci notation; for an even closer match to FM, you could additionally set **`ylim` from median ± k·std** on good pixels only (same as `plot_spectrum`).

## Codec / training preprocessing (not the same as exploratory plots)

| | FoundationModel (`src/tokenizers/spectrum.py` v1) | FoundationModel (`spectrum_v2.py`) | `desifm` (`training/codec_input.py`) |
|---|----------------------------------|--------------------------------------|----------------------------------------|
| **Norm scale** | Mean of flux on **`flux > 0`** only (+ clamps, log10 denorm, ×0.2, **arcsinh**) | Same family + **5-pixel top-hat** on normalized grid before encoder | Mean on **good pixels** `~mask` (+ `nan_to_num`, **ivar ≤ 100**, same log10 denorm + arcsinh) |
| **Smoothing** | None in v1 | **Top-hat** pre-encoder | **None** in current Tier-A codec |
| **Grid** | Interpolate to **8704** inside tokenizer | Same | **`GRID_SIZE` 8704** in `SpectrumCodec` |

Exploratory notebooks should compare **raw stitched flux** (or FM-style `plot_spectrum`) to avoid mixing in codec-normalized panels, which are **supposed** to look flat-ish in arcsinh space.

## Where to look in FoundationModel

- Data + stitch: `FoundationModel/src/utils/data.py`
- Plots: `FoundationModel/src/utils/plotting.py`
- Example driver: `FoundationModel/scripts/visualize_spectra.py`
- Notebook: `FoundationModel/notebooks/01_explore_desi_spectra.ipynb`

## Where to look in `desifm`

- Stitch: `src/desifm/data/stitch.py`
- Stream: `src/desifm/data/dr1_stream.py`
- Codec input: `src/desifm/training/codec_input.py`
- Plot + lines: `src/desifm/viz/spectrum_plot.py`
- Notebook: `notebooks/01_phase_data.ipynb`
