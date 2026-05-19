#!/usr/bin/env python3
"""Regenerate phase-2/3 codec notebooks (run from repo root)."""

from __future__ import annotations

import json
from pathlib import Path


def _src(s: str) -> list[str]:
    if s and not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def _nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python3",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def build_phase2() -> dict:
    cells: list[dict] = []
    cells.append(
        _md(
            """# Phase 2 — Spectrum codec (architecture & forward pass)

ConvNeXt-style 1D encoder + lookup-free quantizer (LFQ) + decoder; fixed **8704** grid and **273** latent tokens.

**Kernel:** choose **Python 3.x (.venv)** from `final-project/.venv` (not a random system Python). Re-run `pip install -e .` from the repo root if imports fail.

The first code cell adds `src/` to `sys.path` whether Jupyter’s working directory is the repo root or `notebooks/`. It does **not** use raw `%matplotlib inline` (that breaks outside IPython)."""
        )
    )
    cells.append(
        _code(
            r"""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_cwd = Path.cwd().resolve()
if (_cwd / "src" / "desifm").is_dir():
    REPO = _cwd
elif (_cwd.parent / "src" / "desifm").is_dir():
    REPO = _cwd.parent
else:
    raise FileNotFoundError(f"Could not find repo root (src/desifm). cwd={_cwd}")
sys.path.insert(0, str(REPO / "src"))

try:
    from IPython import get_ipython

    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

plt.rcParams["figure.figsize"] = (11, 4)

from desifm.constants import GRID_SIZE, N_LATENT_TOKENS
from desifm.tokenization.spectrum_codec import SpectrumCodec
from desifm.training.codec_input import denormalize_spectrum_output, normalize_spectrum_input

print("GRID_SIZE =", GRID_SIZE, "  N_LATENT_TOKENS =", N_LATENT_TOKENS)"""
        )
    )
    cells.append(_md("## Model footprint\n\nParameter count and default channel widths (see `SpectrumCodec` in `src/desifm/tokenization/spectrum_codec.py`)."))
    cells.append(
        _code(
            r"""model = SpectrumCodec()
n_params = sum(p.numel() for p in model.parameters())
print(f"SpectrumCodec parameters: {n_params:,}")
print(model)"""
        )
    )
    cells.append(_md("## Synthetic forward pass\n\nRandom `(B, 2, L)` arcsinh-normalized tensor (flux + istd channels). `forward` returns reconstruction in **arcsinh** space, Huber `recon_loss` on flux vs input, and LFQ `q_loss`."))
    cells.append(
        _code(
            r"""torch.manual_seed(0)
B, L = 4, 4096
flux = torch.rand(B, L) * 0.5 + 0.5
ivar = torch.ones(B, L) * 10.0
mask = torch.zeros(B, L, dtype=torch.bool)
x, denorm = normalize_spectrum_input(flux, ivar, mask)
with torch.no_grad():
    out = model(x, denorm, mask=mask)

print("x shape:", tuple(x.shape), " denorm:", tuple(denorm.shape))
print("indices shape:", tuple(out["indices"].shape), "(B, n_tokens)")
for k in ("recon_loss", "q_loss", "loss"):
    print(f"  {k}: {float(out[k].item()):.6f}")

# Physical-space round-trip of the *input* (not quantized recon)
phys = denormalize_spectrum_output(x, denorm)
print("physical flux shape:", tuple(phys.shape), " max |flux_err|:", float((phys[:, 0] - flux).abs().max()))"""
        )
    )
    cells.append(_md("## Loss breakdown (bar)\n\nSingle minibatch — useful for smoke; training curves live in W&B / `metrics.jsonl`."))
    cells.append(
        _code(
            r"""fig, ax = plt.subplots()
keys = ["recon_loss", "q_loss"]
vals = [float(out[k].item()) for k in keys]
ax.bar(keys, vals, color=["steelblue", "coral"])
ax.set_ylabel("loss (arcsinh flux / LFQ)")
ax.set_title("Codec forward — loss components (one batch)")
plt.show()"""
        )
    )
    cells.append(_md("## Encode → decode token path\n\nIndices are packed LFQ binary codes per latent position."))
    cells.append(
        _code(
            r"""with torch.no_grad():
    idx, _ = model.encode(x, denorm)
    recon_from_idx = model.decode(idx, denorm, to_physical=True)
L_in = flux.shape[1]
# Decoder runs on the codec's fixed GRID_SIZE; resample flux channel back to input length for comparison.
recon_flux_grid = recon_from_idx[:, 0:1, :]
recon_at_in = torch.nn.functional.interpolate(
    recon_flux_grid, size=L_in, mode="linear", align_corners=False
).squeeze(1)
print("token index min/max:", int(idx.min()), int(idx.max()))
print("decode physical shape (B, 2, GRID_SIZE):", tuple(recon_from_idx.shape))
print("max |flux decode - GT| @ input length:", float((recon_at_in - flux).abs().max()))"""
        )
    )
    cells.append(
        _md(
            """## Train on real data

From repo root (or NERSC), see `scripts/train_codec.py` and `docs/NERSC_INTERACTIVE.md`.

```bash
python scripts/train_codec.py --manifest /path/to/manifest.jsonl --run-name codec_v3 --wandb-mode online
```"""
        )
    )
    return _nb(cells)


def build_phase3() -> dict:
    cells: list[dict] = []
    cells.append(
        _md(
            """# Phase 3 — Codec evaluation (v2 / v3 / **codec_v4 Tier 1**)

Compare spectrum codecs (see `RESEARCH_LOG.md` and `TRAINING_REGISTRY.yaml`):

| Run | Eval path | Notes |
|-----|-----------|--------|
| **codec_v2** | `codec_v2_linear` | Legacy median linear scale |
| **codec_v3** | `mask_arcsinh_v3` | Tier-A arcsinh + mask-aware denorm |
| **codec_v4_tier1_ddp** | `mask_arcsinh_v4` | Tier 1: dual loss, top-hat, val `rms_flux` checkpoint |

**Requirements**

1. `pip install -e ".[dev]"` for `pyyaml`.
2. `WANDB_API_KEY` in repo `.env` or `wandb login`.
3. W&B artifacts `{run_name}-codec-best:best` or local `checkpoints/<run_name>/best.pt`.

Checkpoints cache under `checkpoints/wandb_cache/<run_name>/`.

**Local DR1:** next section downloads one healpix tile (training walk order on the [public portal](https://data.desi.lbl.gov/public/dr1/))."""
        )
    )
    cells.append(
        _code(
            r"""import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_cwd = Path.cwd().resolve()
if (_cwd / "src" / "desifm").is_dir():
    REPO = _cwd
elif (_cwd.parent / "src" / "desifm").is_dir():
    REPO = _cwd.parent
else:
    raise FileNotFoundError(f"Could not find repo root. cwd={_cwd}")
sys.path.insert(0, str(REPO / "src"))

try:
    from IPython import get_ipython

    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

plt.rcParams["figure.figsize"] = (12, 5)

CACHE = REPO / "checkpoints" / "wandb_cache"
CACHE.mkdir(parents=True, exist_ok=True)

with open(REPO / "TRAINING_REGISTRY.yaml") as f:
    registry = yaml.safe_load(f)
runs = {r["run_name"]: r for r in registry["runs"]}
for r in runs.values():
    wid = r.get("wandb_id")
    if isinstance(wid, str) and wid.strip().lower() in ("", "null", "none", "~"):
        r["wandb_id"] = None
print("Registry runs:", ", ".join(sorted(runs)))"""
        )
    )
    cells.append(
        _code(
            r"""from desifm.training.wandb_codec import (
    download_codec_best_pt,
    ensure_wandb_auth,
    wandb_run_history_df,
)

def resolve_ckpt(run_name: str) -> Path:
    local = REPO / "checkpoints" / run_name / "best.pt"
    if local.is_file():
        print(run_name, "using local", local)
        return local
    if not ensure_wandb_auth():
        raise SystemExit("Set WANDB_API_KEY or run wandb login to download checkpoints.")
    path = download_codec_best_pt(run_name, CACHE)
    print(run_name, "downloaded to", path)
    return path

# Runs to compare (edit list if you only want v4)
EVAL_RUN_NAMES = ["codec_v2", "codec_v3", "codec_v4_tier1_ddp"]
CHECKPOINTS: dict[str, Path] = {}
for run_name in EVAL_RUN_NAMES:
    try:
        CHECKPOINTS[run_name] = resolve_ckpt(run_name)
    except Exception as exc:
        print(f"{run_name} unavailable:", exc)
print("loaded:", list(CHECKPOINTS))"""
        )
    )
    cells.append(
        _md(
            "## Download eval DR1 tile (minimal)\n\n"
            "One healpix in **training manifest walk order** (`sv3`→`main`, `bright`→`dark`; "
            "skips surveys not on the public portal). Writes "
            "`data/manifests/train_eval_dr1.jsonl`. Idempotent: skips files already on disk."
        )
    )
    cells.append(
        _code(
            r"""from desifm.data.public_dr1 import discover_public_training_tiles, ensure_dr1_tiles_local
from desifm.data.dr1_stream import load_manifest

DATA_ROOT = REPO / "data" / "dr1_public"
MANIFEST = REPO / "data" / "manifests" / "train_eval_dr1.jsonl"
N_HEALPIX = 1  # minimal: one tile = coadd + redrock

tiles = discover_public_training_tiles(N_HEALPIX)
print("tiles (training walk order, public portal):", tiles)
ensure_dr1_tiles_local(DATA_ROOT, MANIFEST, tiles)
for r in load_manifest(MANIFEST):
    print(
        f"  healpix {r['healpix']} ({r['survey']}/{r['program']}) n_rows={r.get('n_rows')}"
    )"""
        )
    )
    cells.append(
        _md(
            "## Training curves (W&B)\n\n"
            "Left: `train/recon`. Right: `val/rms_flux` (Tier 1 v4 only; checkpoint metric)."
        )
    )
    cells.append(
        _code(
            r"""PLOT_RUNS = [
    ("codec_v2", "steelblue"),
    ("codec_v3", "coral"),
    ("codec_v4_tier1_ddp", "mediumseagreen"),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for run_name, color in PLOT_RUNS:
    rid = (runs.get(run_name) or {}).get("wandb_id")
    if not rid:
        print(f"skip curves: no wandb_id for {run_name}")
        continue
    for ax, key in zip(axes, ["train/recon", "val/rms_flux"]):
        try:
            df = wandb_run_history_df(str(rid), keys=["_step", key])
        except Exception as exc:
            print(run_name, key, "failed:", exc)
            continue
        if key not in df.columns:
            continue
        sub = df.dropna(subset=[key])
        if sub.empty:
            continue
        ax.plot(sub["_step"], sub[key], color=color, label=run_name, alpha=0.9)
for ax, title, ylab in zip(
    axes,
    ["train/recon (arcsinh)", "val/rms_flux (held-out healpix)"],
    ["train/recon", "val/rms_flux"],
):
    ax.set_xlabel("step")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
        )
    )
    cells.append(
        _md(
            "## Load models + input styles\n\n"
            "- **codec_v2**: force `codec_v2_linear` (not in checkpoint metadata).\n"
            "- **codec_v3 / v4**: `input_style` from `best.pt` (`mask_arcsinh_v3` / `mask_arcsinh_v4`).\n"
            "- Per-spectrum plots use **unpadded** stitched coadds, not a single padded batch row."
        )
    )
    cells.append(
        _code(
            r"""from desifm.training.codec_eval import (
    forward_physical,
    forward_physical_from_spec,
    load_spectrum_codec,
)

PLOT_STYLE = {
    "codec_v2": ("codec_v2_linear", "darkorange", "codec_v2"),
    "codec_v3": (None, "crimson", "codec_v3"),
    "codec_v4_tier1_ddp": (None, "mediumseagreen", "codec_v4 tier1"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

MODELS: dict[str, tuple] = {}
for run_name, ckpt_path in CHECKPOINTS.items():
    model, style_ckpt = load_spectrum_codec(ckpt_path, device)
    style_override, color, label = PLOT_STYLE.get(run_name, (None, "gray", run_name))
    style = style_override or style_ckpt
    MODELS[run_name] = (model, style, color, label)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"{run_name}: style={style}  step={blob.get('step')}  loss={blob.get('loss')}")"""
        )
    )
    cells.append(
        _md(
            "## Evaluation batch\n\n"
            "Training metrics use a **padded collated batch**. "
            "Plots below use the **stitched coadd** (real λ, flux, ivar, mask) from "
            "`train_eval_dr1.jsonl` when present (run the download cell above), else "
            "`local_dr1.jsonl`, else synthetic data."
        )
    )
    cells.append(
        _code(
            r"""from desifm.data.dr1_stream import DR1StreamDataset, collate_spectra, load_manifest
from desifm.data.synthetic import SyntheticSpectrumDataset
from desifm.viz.spectrum_plot import load_stitched_spectrum, plot_spectrum_with_lines

for candidate in (
    REPO / "data" / "manifests" / "train_eval_dr1.jsonl",
    REPO / "data" / "manifests" / "local_dr1.jsonl",
):
    if candidate.is_file():
        MANIFEST = candidate
        break
else:
    MANIFEST = REPO / "data" / "manifests" / "train_eval_dr1.jsonl"

raw_specs: list[dict] = []
n_eval = 8

if MANIFEST.is_file():
    recs = load_manifest(MANIFEST)
    ds = DR1StreamDataset(recs, max_spectra=64)
    items = []
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        rec_idx, row = ds.index[i]
        rec = recs[rec_idx]
        sp = load_stitched_spectrum(rec["coadd"], rec["redrock"], row, require_good_z=True)
        if sp is None:
            continue
        sp["healpix"] = rec.get("healpix")
        sp["row"] = row
        raw_specs.append(sp)
        items.append(item)
        if len(items) >= n_eval:
            break
    if not items:
        raise RuntimeError(
            "No good spectra in manifest; run the Download eval DR1 tile section above."
        )
    batch = collate_spectra(items)
    print("manifest:", MANIFEST.name)
    print("batch from DR1 manifest, shape", tuple(batch["flux"].shape))
    print("raw stitched spectra:", len(raw_specs), "pixels[0]", len(raw_specs[0]["wavelength"]))
else:
    ds = SyntheticSpectrumDataset(n_spectra=n_eval, length=4096, seed=1)
    items = [ds[i] for i in range(n_eval)]
    batch = collate_spectra(items)
    for i in range(n_eval):
        L = int(items[i]["flux"].shape[0])
        raw_specs.append(
            {
                "wavelength": np.linspace(3600, 9800, L),
                "flux": items[i]["flux"].numpy(),
                "ivar": items[i]["ivar"].numpy(),
                "mask": items[i]["mask"].numpy(),
                "z": float(items[i]["z"].item()),
            }
        )
    print("batch synthetic, shape", tuple(batch["flux"].shape))

for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)

print("eval models:", {k: v[1] for k, v in MODELS.items()})"""
        )
    )
    cells.append(
        _md(
            "## Scalar metrics (padded batch)\n\n"
            "These match **training** (collated, padded minibatch). "
            "They can disagree with the per-spectrum plots below. Lower `recon_loss` is better (Huber in arcsinh space)."
        )
    )
    cells.append(
        _code(
            r"""rows = []
for run_name, (model, style, _color, label) in MODELS.items():
    out = forward_physical(model, batch, style)
    rows.append((label, out["recon_loss"], out["q_loss"], out["loss_total"]))

print(f"{'run':16s} {'recon':>10s} {'q_loss':>10s} {'total':>10s}")
for name, a, b, c in rows:
    print(f"{name:16s} {a:10.5f} {b:10.5f} {c:10.5f}")"""
        )
    )
    cells.append(
        _md(
            "## Per-spectrum collapse check\n\n"
            "Training logs `val/std_ratio` by pooling **all good pixels in the validation "
            "minibatch** (std across the batch tensor). That can look healthier than this "
            "table, which measures **within each spectrum** — the same view as the flux plots.\n\n"
            "**Tier 1 gate:** `std_ratio` > 0.5 on good pixels (`RESEARCH_LOG.md`). "
            "Values well below 0.5 mean a flat reconstruction (mode collapse)."
        )
    )
    cells.append(
        _code(
            r"""from desifm.training.codec_loss import flux_std_ratio

COLLAPSE_GATE = 0.5


def rms_numpy(true_np, recon_np, mask_np):
    g = ~np.asarray(mask_np, dtype=bool)
    d = true_np[g].astype("float64") - recon_np[g].astype("float64")
    return float(np.sqrt(np.mean(d * d)))


def per_spec_std_ratio(model, style, spec: dict) -> tuple[float, float, float]:
    # Within-spectrum std(recon)/std(coadd) on good pixels + RMS + arcsinh recon_loss.
    fwd = forward_physical_from_spec(model, spec, style, device)
    rec = fwd["flux_recon_native"]
    flux_t = torch.from_numpy(np.asarray(spec["flux"], dtype=np.float32)).unsqueeze(0)
    mask_t = torch.from_numpy(np.asarray(spec["mask"], dtype=bool)).unsqueeze(0)
    ratio = float(flux_std_ratio(rec.unsqueeze(0), flux_t, mask_t).item())
    rms = rms_numpy(spec["flux"], rec.numpy(), spec["mask"])
    return ratio, rms, float(fwd["recon_loss"])


model_labels = [(run_name, label) for run_name, (_, _, _, label) in MODELS.items()]
hdr = f"{'idx':>3s} {'z':>7s} {'hp':>4s} {'row':>4s}"
for _, label in model_labels:
    short = label.replace("codec_", "").replace(" tier1", " t1")[:10]
    hdr += f" {short + '_ratio':>11s}"
print(hdr)
print("-" * len(hdr))

per_model_ratios: dict[str, list[float]] = {label: [] for _, label in model_labels}
n_collapsed: dict[str, int] = {label: 0 for _, label in model_labels}

for idx, spec in enumerate(raw_specs):
    z = float(spec["z"])
    hp = spec.get("healpix", "?")
    row = spec.get("row", "?")
    line = f"{idx:3d} {z:7.4f} {str(hp):>4s} {str(row):>4s}"
    for run_name, label in model_labels:
        model, style, _, _ = MODELS[run_name]
        ratio, _rms, _loss = per_spec_std_ratio(model, style, spec)
        per_model_ratios[label].append(ratio)
        if ratio < COLLAPSE_GATE:
            n_collapsed[label] += 1
        flag = "!" if ratio < COLLAPSE_GATE else " "
        line += f" {ratio:10.3f}{flag}"
    print(line)

print()
print(f"Gate: std_ratio > {COLLAPSE_GATE}  (! = collapsed)")
for _, label in model_labels:
    vals = np.asarray(per_model_ratios[label], dtype=np.float64)
    print(
        f"  {label:16s}  mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
        f"min={vals.min():.3f}  collapsed={n_collapsed[label]}/{len(vals)}"
    )

print("\nBatch-pooled std_ratio (same aggregation as W&B val/std_ratio on this minibatch):")
for run_name, (model, style, _c, label) in MODELS.items():
    out = forward_physical(model, batch, style)
    pooled = float(
        flux_std_ratio(out["flux_recon"], out["flux_true"], batch["mask"].to(device)).item()
    )
    flag = "  <- can exceed per-spec median when spectra differ in level" if pooled > COLLAPSE_GATE else ""
    print(f"  {label:16s}  pooled={pooled:.3f}{flag}")"""
        )
    )
    cells.append(
        _md(
            "## Flux overlays (per spectrum, unpadded)\n\n"
            "| Trace | Meaning |\n"
            "|-------|--------|\n"
            "| **stitched coadd** | Real DESI flux from FITS |\n"
            "| **codec recon traces** | Model output in physical units |\n\n"
            "Bottom panel: internal units (v2 linear scale; v3/v4 arcsinh on 8704 grid)."
        )
    )
    cells.append(
        _code(
            r"""from desifm.constants import GRID_SIZE
from desifm.training.codec_input import prepare_codec_batch_for_style
from desifm.training.codec_loss import flux_std_ratio


def _ylim_overlay(flux_np, *extra, good, k_std=5.0):
    arrs = [flux_np[good]] + [a[good] for a in extra if a is not None]
    vals = np.concatenate([a[np.isfinite(a)] for a in arrs if a.size])
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    std = float(np.std(vals))
    if not np.isfinite(std) or std <= 0:
        std = max(float(np.percentile(vals, 99) - np.percentile(vals, 1)) / 4.0, 1e-30)
    return med - float(k_std) * std, med + float(k_std) * std


def plot_compare(idx: int = 0):
    sp = raw_specs[idx]
    wave = sp["wavelength"]
    flux = np.asarray(sp["flux"])
    mask = sp["mask"]
    ivar = sp.get("ivar")
    z = float(sp["z"])
    good = ~np.asarray(mask, dtype=bool)

    forwards: dict[str, dict] = {}
    recons = []
    roundtrips = []
    for run_name, (model, style, color, label) in MODELS.items():
        fwd = forward_physical_from_spec(model, sp, style, device)
        forwards[run_name] = fwd
        rec = fwd["flux_recon_native"].numpy()
        recons.append(rec)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    plot_spectrum_with_lines(
        ax,
        wave,
        flux,
        z,
        mask=mask,
        ivar=ivar,
        show_lines=False,
        flux_label="stitched coadd (FITS)",
        adaptive_ylim=False,
    )
    for run_name, (model, style, color, label) in MODELS.items():
        fwd = forwards[run_name]
        rec = fwd["flux_recon_native"].numpy()
        ax.plot(wave[good], rec[good], "--", color=color, lw=1.4, zorder=10, label=f"{label} recon")
        if style != "codec_v2_linear":
            rt = fwd["flux_roundtrip_native"].numpy()
            roundtrips.append(rt)
            if run_name == "codec_v4_tier1_ddp":
                ax.plot(
                    wave[good],
                    rt[good],
                    ":",
                    color="silver",
                    lw=1.0,
                    zorder=9,
                    label="v4 input roundtrip",
                )
    ylim = _ylim_overlay(flux, *recons, *roundtrips, good=good)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(f"spectrum {idx}  z={z:.4f}  healpix={sp.get('healpix', '?')} row={sp.get('row', '?')}")
    ax.legend(loc="upper right", fontsize=7)

    ax2 = axes[1]
    g = np.linspace(wave.min(), wave.max(), GRID_SIZE)
    for run_name, (model, style, color, label) in MODELS.items():
        fwd = forwards[run_name]
        if style == "codec_v2_linear":
            ax2.plot(g, fwd["target_norm"][0].numpy(), "k-", lw=0.5, alpha=0.5)
            ax2.plot(g, fwd["recon_norm"][0].numpy(), "--", color=color, lw=0.8, label=f"{label} (flux/scale)")
        else:
            batch1 = {
                "flux": torch.from_numpy(flux).float().unsqueeze(0),
                "ivar": torch.from_numpy(np.asarray(sp["ivar"], dtype=np.float32)).unsqueeze(0),
                "mask": torch.from_numpy(np.asarray(mask, dtype=bool)).unsqueeze(0),
            }
            x, den, _ = prepare_codec_batch_for_style(batch1, style)
            xg = torch.nn.functional.interpolate(x, size=GRID_SIZE, mode="linear", align_corners=False)
            with torch.no_grad():
                out = model(xg.to(device), den.to(device))
            ax2.plot(g, xg[0, 0].cpu().numpy(), ":", color=color, lw=0.4, alpha=0.5)
            ax2.plot(g, out["recon"][0, 0].cpu().numpy(), "--", color=color, lw=0.8, label=f"{label} (arcsinh)")
    ax2.set_ylabel("codec internal units")
    ax2.set_xlabel("wavelength (Å)")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    flux_t = torch.from_numpy(flux).float().unsqueeze(0)
    mask_t = torch.from_numpy(np.asarray(mask, dtype=bool)).unsqueeze(0)
    for run_name, (model, style, color, label) in MODELS.items():
        rec = forwards[run_name]["flux_recon_native"].numpy()
        ratio = float(flux_std_ratio(torch.from_numpy(rec).unsqueeze(0), flux_t, mask_t).item())
        rms = rms_numpy(flux, rec, mask)
        print(f"{label:16s}  RMS={rms:.4f}  std_ratio={ratio:.3f}  recon_loss={forwards[run_name]['recon_loss']:.4f}")


plot_compare(0)  # change index to inspect other rows from the table above"""
        )
    )
    return _nb(cells)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks"
    (nb_dir / "02_phase_codec.ipynb").write_text(json.dumps(build_phase2(), indent=2) + "\n")
    (nb_dir / "03_phase_codec_eval.ipynb").write_text(json.dumps(build_phase3(), indent=2) + "\n")
    print("Wrote", nb_dir / "02_phase_codec.ipynb")
    print("Wrote", nb_dir / "03_phase_codec_eval.ipynb")


if __name__ == "__main__":
    main()
