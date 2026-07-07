"""FR-05 convergence-candidate heuristic (Domain service).

This computes *our own* heuristic flag; it deliberately does NOT read
``run_diag.pkl`` (which only stores ``ar_t``, ``jumps_t``, ``nswap`` -- no
convergence verdict). The rule:

* summarise each chain by the **mean log-likelihood of its final ~20%**;
* a band is flagged as a *candidate* when its score is anomalously low
  relative to neighbouring bands (robust median / MAD outlier test).

We flag CANDIDATES for inspection (via the trace plot / sky-map highlight); we
do not certify convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def tail_mean_loglik(log_likelihood: Sequence[float], frac: float = 0.2) -> float:
    """Mean log-likelihood over the final ``frac`` of a chain."""
    arr = np.asarray(log_likelihood, dtype=float)
    if arr.size == 0:
        return float("nan")
    n_tail = max(1, int(round(frac * arr.size)))
    return float(np.mean(arr[-n_tail:]))


@dataclass(frozen=True)
class ConvergenceReport:
    """Per-band tail score and the set of flagged candidate bands."""

    scores: Dict[str, float]
    flagged: List[str]
    threshold: float

    def is_flagged(self, band_label: str) -> bool:
        return band_label in self.flagged


def flag_low_bands(
    band_scores: Dict[str, float],
    n_mad: float = 3.0,
) -> ConvergenceReport:
    """Flag bands whose tail-mean log-likelihood is anomalously low.

    Uses a robust (median + MAD) lower threshold: a band is a candidate when its
    score is below ``median - n_mad * 1.4826 * MAD``. With too few bands to form
    a robust baseline, nothing is flagged.
    """
    labels = [k for k, v in band_scores.items() if np.isfinite(v)]
    values = np.array([band_scores[k] for k in labels], dtype=float)
    if values.size < 3:
        return ConvergenceReport(scores=dict(band_scores), flagged=[], threshold=float("nan"))

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 0 else 0.0
    threshold = median - n_mad * scale if scale > 0 else float(np.min(values)) - 1.0

    flagged = [lab for lab, val in zip(labels, values) if val < threshold]
    return ConvergenceReport(scores=dict(band_scores), flagged=flagged, threshold=threshold)
