"""Token layout and special IDs for the DESI foundation model."""

from __future__ import annotations

# Special tokens (shared vocabulary)
SOS = 0
EOS = 1
PAD = 2
MASK = 3
REDMASK = 4  # always-mask redshift (Approach B decoder)

SPECTRUM_OFFSET = 8
REDSHIFT_OFFSET = 1032
VOCAB_SIZE = 2056

N_SPECTRUM_CODES = 1024
N_REDSHIFT_BINS = 256

# Spectrum codec fixed wavelength grid (8704 pixels)
GRID_SIZE = 8704
N_LATENT_TOKENS = 273
