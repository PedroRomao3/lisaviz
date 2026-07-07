"""Module 3 -- Signal Reconstruction: frequency-domain waveform overlay with a
match / inner-product score, and the PSD residual plot (FR-06).

These functions are array-based so they are callable standalone in a notebook;
pass in TDI spectra (e.g. from ``signal.h5``: ``TDIA``, ``TDIE``, ``freq``) and
optionally a noise PSD (e.g. evaluated from a domain :class:`NoiseModel`).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .backends import LineSeries, PlotlyBackend, RenderBackend


def noise_weighted_inner_product(a: np.ndarray, b: np.ndarray, psd: np.ndarray, df: float) -> float:
    """4 Re sum(a conj(b) / S) df -- the standard GW noise-weighted inner product."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(4.0 * np.sum((a * np.conj(b) / psd).real) * df)


def match(a: np.ndarray, b: np.ndarray, psd: np.ndarray, df: float) -> float:
    """Normalised overlap (match) in [0, 1] between two frequency-domain
    signals: ``<a,b> / sqrt(<a,a><b,b>)``."""
    aa = noise_weighted_inner_product(a, a, psd, df)
    bb = noise_weighted_inner_product(b, b, psd, df)
    if aa <= 0 or bb <= 0:
        return float("nan")
    return noise_weighted_inner_product(a, b, psd, df) / np.sqrt(aa * bb)


def _asd(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x))


def plot_waveform_overlay(
    freq: np.ndarray,
    observed: np.ndarray,
    recovered: Sequence[np.ndarray],
    injection: Optional[np.ndarray] = None,
    psd: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    backend: Optional[RenderBackend] = None,
):
    """Overlay the ASD of the observed TDI with one or more recovered waveforms.
    If ``injection`` and ``psd`` are given, each recovered waveform's match
    against the injection is reported in its legend label."""
    backend = backend or PlotlyBackend()
    df = float(np.median(np.diff(freq)))
    series = [LineSeries(label="observed", x=freq, y=_asd(observed), color="#1f4e79", width=1.5)]
    palette = ["#c00000", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, rec in enumerate(recovered):
        lab = labels[i] if labels else f"recovered {i}"
        if injection is not None and psd is not None:
            lab += f"  (match={match(rec, injection, psd, df):.3f})"
        series.append(LineSeries(label=lab, x=freq, y=_asd(rec), color=palette[i % len(palette)], width=1.2, dash="dash"))
    if injection is not None:
        series.append(LineSeries(label="injection", x=freq, y=_asd(injection), color="#888888", width=1.0, dash="dot"))
    return backend.spectral_overlay(series, axis_labels=("Frequency [Hz]", "ASD"), title="Waveform overlay")


def plot_psd_residual(
    freq: np.ndarray,
    observed_psd: np.ndarray,
    model_psd: np.ndarray,
    noise_psd: Optional[np.ndarray] = None,
    noise_evolution: Optional[Sequence[Tuple[str, np.ndarray]]] = None,
    backend: Optional[RenderBackend] = None,
):
    """PSD overlay (observed vs reconstructed sources + noise) with a fractional
    residual panel. ``noise_evolution`` optionally overlays the noise PSD at
    several iterations to show how the noise model evolves."""
    backend = backend or PlotlyBackend()
    observed_psd = np.asarray(observed_psd, dtype=float)
    model_psd = np.asarray(model_psd, dtype=float)
    series = [
        LineSeries(label="observed", x=freq, y=observed_psd, color="#1f4e79", width=1.5),
        LineSeries(label="model", x=freq, y=model_psd, color="#c00000", width=1.5, dash="dash"),
    ]
    if noise_psd is not None:
        series.append(LineSeries(label="noise", x=freq, y=np.asarray(noise_psd, dtype=float),
                                 color="#888888", width=1.0, dash="dot"))
    if noise_evolution:
        shades = np.linspace(0.3, 0.85, len(noise_evolution))
        for (lab, psd_i), shade in zip(noise_evolution, shades):
            grey = f"rgba(120,120,120,{shade:.2f})"
            series.append(LineSeries(label=lab, x=freq, y=np.asarray(psd_i, dtype=float), color=grey, width=0.8))
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.where(model_psd > 0, observed_psd / model_psd, np.nan)
    return backend.spectral_overlay(series, residual=(freq, residual),
                                    axis_labels=("Frequency [Hz]", "PSD"), title="PSD residual")
