"""Sampling-space <-> physical-space transforms for galactic binaries.

This mirrors, exactly, the parametrisation used by the pipeline in
``globalfit/gb/likelihood.py`` (``GBLogLik.pars_to_ldc`` /
``ldc_to_pars`` / base ``rescale``). It is reproduced here -- rather than
imported -- to keep the Domain layer free of the heavy ``globalfit`` runtime
(NFR-01). The authoritative reference is:

    sampling names : amp_log10, fr, fdot, dec_sin, alpha, iota_cos, phiL, phiR
    physical names : Amplitude, Frequency, FrequencyDerivative, Declination,
                     RightAscension, Inclination, Polarization, InitialPhase

The on-disk F-statistic chains store the four *intrinsic* parameters
``fr, fdot, dec_sin, alpha`` in the unit hypercube [0, 1]. Recovering physical
values is a two-step process: (1) rescale the hypercube value into the band's
prior interval, then (2) apply the trigonometric fold.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .models import FrequencyBand

# Sampling-space (inner MCMC) parameter order, matching PARS_NAMES.
PARS_NAMES = ("amp_log10", "fr", "fdot", "dec_sin", "alpha", "iota_cos", "phiL", "phiR")
# Physical (LDC) parameter order, matching LDC_NAMES.
LDC_NAMES = (
    "Frequency",
    "FrequencyDerivative",
    "Declination",
    "RightAscension",
    "Amplitude",
    "Inclination",
    "Polarization",
    "InitialPhase",
)
# The four intrinsic sampling parameters present in the F-stat search chains.
INTRINSIC_PARS = ("fr", "fdot", "dec_sin", "alpha")


def rescale(hc: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a unit-hypercube value to its physical prior interval.

    Mirrors ``base.likelihood.LogLik.rescale``: ``phys = (hc - 1)*(hi - lo) + hi``
    which is algebraically ``lo + hc*(hi - lo)`` for the usual [0, 1] hypercube.
    """
    return (np.asarray(hc, dtype=float) - 1.0) * (hi - lo) + hi


def scale(phys: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of :func:`rescale` (physical -> unit hypercube)."""
    return (np.asarray(phys, dtype=float) - hi) / (hi - lo) + 1.0


def _fold_declination(dec_sin_phys: np.ndarray) -> np.ndarray:
    """sin(dec) (folded to [-1, 1]) -> declination in radians."""
    folded = (np.asarray(dec_sin_phys, dtype=float) + 1.0) % 2.0 - 1.0
    return np.arcsin(folded)


def hc_to_frequency(fr_hc: np.ndarray, band: FrequencyBand) -> np.ndarray:
    return rescale(fr_hc, band.f_min, band.f_max)


def hc_to_fdot(fdot_hc: np.ndarray, band: FrequencyBand) -> Optional[np.ndarray]:
    if band.fdot_min is None or band.fdot_max is None:
        return None
    return rescale(fdot_hc, band.fdot_min, band.fdot_max)


def hc_to_declination(dec_sin_hc: np.ndarray) -> np.ndarray:
    return _fold_declination(rescale(dec_sin_hc, -1.0, 1.0))


def hc_to_right_ascension(alpha_hc: np.ndarray) -> np.ndarray:
    return rescale(alpha_hc, 0.0, 2.0 * np.pi) % (2.0 * np.pi)


def intrinsic_chain_to_physical(
    samples: Dict[str, np.ndarray], band: FrequencyBand
) -> Dict[str, np.ndarray]:
    """Convert the hypercube sampling-space parameters (intrinsic and extrinsic)
    of a chain into physical arrays keyed by physical names.
    """
    out: Dict[str, np.ndarray] = {}
    if "fr" in samples:
        out["Frequency"] = hc_to_frequency(samples["fr"], band)
    if "dec_sin" in samples:
        out["Declination"] = hc_to_declination(samples["dec_sin"])
    if "alpha" in samples:
        out["RightAscension"] = hc_to_right_ascension(samples["alpha"])
    if "fdot" in samples:
        fdot_phys = hc_to_fdot(samples["fdot"], band)
        if fdot_phys is not None:
            out["FrequencyDerivative"] = fdot_phys
        else:
            out["FrequencyDerivative_hc"] = np.asarray(samples["fdot"], dtype=float)

    # Extrinsic parameters
    if "amp_log10" in samples:
        out["Amplitude"] = 10.0 ** np.asarray(samples["amp_log10"], dtype=float)
    if "iota_cos" in samples:
        out["Inclination"] = np.arccos(np.clip(np.asarray(samples["iota_cos"], dtype=float), -1.0, 1.0))
    if "phiL" in samples and "phiR" in samples:
        phiL = np.asarray(samples["phiL"], dtype=float) * 2.0 * np.pi
        phiR = np.asarray(samples["phiR"], dtype=float) * 2.0 * np.pi
        out["Polarization"] = 0.5 * (phiR - phiL) % np.pi
        out["InitialPhase"] = 0.5 * (phiR + phiL) % (2.0 * np.pi)

    return out


def hc_axis_to_physical(name: str, values: np.ndarray, band: FrequencyBand):
    """Map a single sampling-axis (``fr``/``fdot``/``dec_sin``/``alpha``) from
    hypercube to physical values. Returns ``(physical_name, physical_values)``.
    Falls back to the raw hypercube values (suffixed ``_hc``) when no physical
    mapping is available (e.g. ``fdot`` without band bounds)."""
    values = np.asarray(values, dtype=float)
    if name == "fr":
        return "Frequency", hc_to_frequency(values, band)
    if name == "dec_sin":
        return "Declination", hc_to_declination(values)
    if name == "alpha":
        return "RightAscension", hc_to_right_ascension(values)
    if name == "fdot":
        phys = hc_to_fdot(values, band)
        if phys is not None:
            return "FrequencyDerivative", phys
        return "FrequencyDerivative_hc", values
    return name, values


def physical_label(name: str) -> str:
    """Human-readable axis label for a physical parameter name."""
    return {
        "Frequency": "f0 [Hz]",
        "FrequencyDerivative": "fdot [Hz/s]",
        "FrequencyDerivative_hc": "fdot [hypercube]",
        "Declination": "delta [rad]",
        "RightAscension": "alpha [rad]",
        "Amplitude": "A",
        "Inclination": "iota [rad]",
        "Polarization": "psi [rad]",
        "InitialPhase": "phi0 [rad]",
    }.get(name, name)
