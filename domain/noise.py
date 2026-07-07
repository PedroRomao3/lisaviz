"""Evaluate a :class:`NoiseModel` into a power spectral density curve (Domain).

Pure stdlib + numpy. This turns the 7 fitted noise parameters
(``Sacc_log10, Soms_log10, A_log10, f1, f2, alpha, fknee``) into a PSD over a
frequency grid, so the signal-reconstruction view (FR-06) can draw the fitted
noise curve and form a residual. The functional form is the standard LISA
decomposition -- an instrument floor (acceleration + optical-metrology terms)
plus the galactic-confusion foreground that cuts off above the knee -- evaluated
analytically; it is a faithful diagnostic curve, not a calibration-grade model.
"""

from __future__ import annotations

import numpy as np

from .models import NoiseModel


def evaluate_noise_psd(model: NoiseModel, freq: np.ndarray) -> np.ndarray:
    """PSD of ``model`` on ``freq`` [Hz]: instrument floor + galactic confusion."""
    f = np.asarray(freq, dtype=float)
    p = model.parameters
    sacc = 10.0 ** p["Sacc_log10"]
    soms = 10.0 ** p["Soms_log10"]
    amp = 10.0 ** p["A_log10"]
    f1 = float(p["f1"])
    f2 = float(p["f2"])
    alpha = float(p["alpha"])
    fknee = float(p["fknee"])

    # Instrument: acceleration noise rises at low f, OMS noise rises at high f.
    instrument = sacc * (1.0 + (4.0e-4 / f) ** 2) + soms * (f / 3.0e-3) ** 2
    # Galactic confusion foreground: red-tilted, exponential roll-off, knee cutoff.
    with np.errstate(over="ignore", invalid="ignore"):
        confusion = (amp * (f / 1.0e-3) ** (-7.0 / 3.0)
                     * np.exp(-((f / f1) ** alpha))
                     * 0.5 * (1.0 + np.tanh((fknee - f) / f2)))
    confusion = np.where(np.isfinite(confusion), confusion, 0.0)
    return instrument + confusion
