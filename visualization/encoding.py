"""Shared visual-encoding rules (Visualization layer).

These encodings are deliberate, documented design choices, kept identical
across every view so coordinated multiple views share one visual language:

* **Frequency -> colour** using a *perceptually uniform sequential* colormap
  (``cividis`` by default: perceptually uniform and colour-vision-deficiency
  friendly -- rainbow/jet are rejected per Szafir).
* **Uncertainty -> mark size** via ``size = 1 - confidence`` (error-ellipse
  convention: a BIG mark means MORE uncertain).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..domain.confidence import mark_size, sigmoid

# One perceptually-uniform sequential colormap, referenced everywhere.
FREQUENCY_CMAP = "cividis"
# Plotly's equivalent name (Cividis), used by the Plotly backend.
FREQUENCY_CMAP_PLOTLY = "Cividis"
# Density (source-count) colormap: dark -> bright, so a high-count cell glows
# against the near-empty background (the Splatterplots look). Kept distinct from
# the frequency map so the two channels never share a meaning on one scale.
DENSITY_CMAP = "magma"
DENSITY_CMAP_PLOTLY = "Magma"


def confidence_sizes(
    confidence: Sequence[float],
    min_px: float = 4.0,
    max_px: float = 18.0,
) -> np.ndarray:
    """Map confidence to marker size in pixels via ``size = 1 - confidence``.
    Largest mark (``max_px``) = least confident."""
    s = mark_size(np.asarray(confidence, dtype=float))
    return min_px + s * (max_px - min_px)


def resolve_confidence(confidence: Sequence[float], snr: Sequence[float]) -> np.ndarray:
    """Use catalog confidence where finite/non-zero; otherwise fall back to
    ``sigmoid(SNR)``. Older runs ship an all-zero confidence column."""
    conf = np.asarray(confidence, dtype=float)
    snr = np.asarray(snr, dtype=float)
    missing = ~np.isfinite(conf) | (conf == 0.0)
    if missing.any():
        conf = conf.copy()
        conf[missing] = sigmoid(snr[missing])
    return conf
