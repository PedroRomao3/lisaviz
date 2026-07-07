"""Confidence-score recomputation and mark-size encoding (Domain service).

The catalog ``confidence`` score, used to drive sky-map mark size, is defined
as a pipeline of three steps:

1. **start** = ``sigmoid(SNR)``;
2. if the source was seen in a previous iteration, **replace** that value with
   the cross-iteration waveform overlap;
3. **subtract** a penalty proportional to the cumulative overlap with
   neighbouring sources (confusion); finally **floor at 0**.

Mark size on the sky map is ``1 - confidence`` (error-ellipse convention: a
BIG mark means MORE uncertain).

This module is pure (stdlib + numpy only) so it is testable without any IO.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# Defaults chosen so that the SNR detection threshold (~7) maps near 0.5.
SIGMOID_MIDPOINT = 7.0
SIGMOID_SCALE = 1.5
CONFUSION_WEIGHT = 1.0


def sigmoid(snr: np.ndarray, midpoint: float = SIGMOID_MIDPOINT, scale: float = SIGMOID_SCALE) -> np.ndarray:
    """Logistic map from SNR to an initial (0, 1) confidence."""
    z = (np.asarray(snr, dtype=float) - midpoint) / scale
    return 1.0 / (1.0 + np.exp(-z))


def compute_confidence(
    snr: float,
    previous_overlap: Optional[float] = None,
    neighbour_overlaps: Iterable[float] = (),
    confusion_weight: float = CONFUSION_WEIGHT,
) -> float:
    """Compute a single source's confidence per the three-step rule above."""
    base = float(sigmoid(snr)) if previous_overlap is None else float(previous_overlap)
    penalty = confusion_weight * float(np.sum(list(neighbour_overlaps)))
    return max(0.0, base - penalty)


def mark_size(confidence: np.ndarray) -> np.ndarray:
    """Sky-map mark size = ``1 - confidence`` (clipped to [0, 1])."""
    return np.clip(1.0 - np.asarray(confidence, dtype=float), 0.0, 1.0)


def neighbour_confusion(
    frequencies: Sequence[float],
    amplitudes: Optional[Sequence[float]] = None,
    bandwidth: float = 1e-7,
) -> np.ndarray:
    """Estimate, per source, the cumulative spectral overlap with neighbours.

    A Gaussian kernel in frequency separation approximates how much each pair of
    sources overlaps spectrally (the dominant confusion axis for GBs). The
    self-term is excluded. Optionally weighted by the neighbour amplitude so
    that a faint source near a loud one is penalised more.
    """
    f = np.asarray(frequencies, dtype=float)
    n = f.size
    if n == 0:
        return np.zeros(0)
    df = f[:, None] - f[None, :]
    kernel = np.exp(-0.5 * (df / bandwidth) ** 2)
    np.fill_diagonal(kernel, 0.0)
    if amplitudes is not None:
        a = np.asarray(amplitudes, dtype=float)
        kernel = kernel * (a[None, :] / (a[:, None] + np.finfo(float).tiny))
    return kernel.sum(axis=1)
