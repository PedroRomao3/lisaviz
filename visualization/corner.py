"""Module 2 -- Posterior Inference: corner plot (FR-03) with publication export
(FR-07).

Samples are transformed out of the inner MCMC *sampling* space (hypercube +
``log10(A)``, ``sin(delta)`` etc.) into *physical* space before rendering, using
the authoritative transforms in :mod:`lisaviz.domain.sampling`. For an intrinsic
F-statistic chain this yields a 4x4 corner over physical
(Frequency, FrequencyDerivative, Declination, RightAscension); an 8-parameter
GB chain yields 8x8.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..domain.models import MCMCChain
from ..domain import sampling
from .backends import MatplotlibBackend, RenderBackend

# Preferred physical-parameter ordering for the corner axes.
_PHYS_ORDER = [
    "Frequency",
    "FrequencyDerivative",
    "FrequencyDerivative_hc",
    "Declination",
    "RightAscension",
    "Amplitude",
    "Inclination",
    "Polarization",
    "InitialPhase",
]


def chain_to_physical_samples(chain: MCMCChain, burn_in: int = 0) -> Dict[str, np.ndarray]:
    """Pool all walkers' post-burn-in draws and convert to physical space."""
    if chain.band is None:
        raise ValueError("Chain has no frequency band; cannot rescale sampling -> physical.")
    pooled: Dict[str, list] = {}
    for w in chain.walkers:
        sliced = {k: np.asarray(v, dtype=float)[burn_in:] for k, v in w.samples.items()}
        phys = sampling.intrinsic_chain_to_physical(sliced, chain.band)
        for k, v in phys.items():
            pooled.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in pooled.items()}


def plot_corner(
    chain: MCMCChain,
    burn_in: int = 0,
    backend: Optional[RenderBackend] = None,
    truths: Optional[Dict[str, float]] = None,
    order: Optional[Sequence[str]] = None,
    source_index: Optional[int] = None,
):
    """Render the corner plot in physical units. Defaults to the Matplotlib
    backend for publication-quality static export."""
    backend = backend or MatplotlibBackend()
    samples = chain_to_physical_samples(chain, burn_in=burn_in)
    if order is None:
        order = [n for n in _PHYS_ORDER if n in samples]
    labels = {n: sampling.physical_label(n) for n in order}
    # Rename keys to human labels for axis display while preserving order.
    relabelled = {labels[n]: samples[n] for n in order}
    truth_relabelled = None
    if truths is not None:
        truth_relabelled = {labels.get(k, k): v for k, v in truths.items()}
    band = chain.band.label if chain.band else ""
    src = "" if source_index is None else f" -- source slot {source_index}"
    return backend.corner(relabelled, list(relabelled.keys()),
                          truths=truth_relabelled, title=f"Posterior -- {band}{src}")
