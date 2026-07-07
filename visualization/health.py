"""Module 1 -- MCMC Health: trace plot (FR-04) + autocorrelation diagnostic.

Every plotting function is callable on its own with a domain ``MCMCChain`` (e.g.
from a Jupyter notebook) without touching h5py handles or chunk layouts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..domain.models import MCMCChain
from . import stats
from .backends import LineSeries, MatplotlibBackend, PlotlyBackend, RenderBackend, TracePanel

# Bright, high-separation walker colours that stay legible on both white
# (publication export) and the dark dashboard background (#0e1117).
_WALKER_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
# Flagged (stuck) walkers use the darker half of a ColorBrewer YlOrRd sequential
# ramp so that *several* flagged walkers stay mutually discriminable even on thin
# marks, while reading clearly as "hot"/anomalous against the cool converged
# palette. The darker half (orange -> dark red) keeps them high-contrast on both
# the white publication export and the dark dashboard (pale yellows wash out on
# white, so they are deliberately excluded).
_FLAG_COLORS = ["#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
_FLAG_COLOR = _FLAG_COLORS[3]  # representative red (kept for back-compat)

# Visual hierarchy: converged walkers are thin + translucent so overlap reads as
# density; flagged walkers are thicker + opaque and drawn on top.
_CONVERGED_WIDTH = 1.5
_CONVERGED_OPACITY = 0.4
_FLAGGED_WIDTH = 2.5
_FLAGGED_OPACITY = 1.0


def _walker_color(walker_id: int, flagged: bool, flag_rank: int = 0) -> str:
    """Colour for a walker. ``flag_rank`` indexes the sequential flag ramp so
    multiple flagged walkers get distinct hot shades."""
    if flagged:
        return _FLAG_COLORS[flag_rank % len(_FLAG_COLORS)]
    return _WALKER_COLORS[walker_id % len(_WALKER_COLORS)]


def _walker_style(walker_id: int, flagged: bool, flag_rank: int = 0):
    """(color, width, opacity) establishing the converged-vs-flagged hierarchy."""
    if flagged:
        return _walker_color(walker_id, True, flag_rank), _FLAGGED_WIDTH, _FLAGGED_OPACITY
    return _walker_color(walker_id, False), _CONVERGED_WIDTH, _CONVERGED_OPACITY


def plot_traces(
    chain: MCMCChain,
    burn_in: int = 0,
    backend: Optional[RenderBackend] = None,
    flagged_walkers: Optional[List[int]] = None,
    walker_ids: Optional[List[int]] = None,
    source_index: Optional[int] = None,
):
    """Per-parameter walker trajectories, plus a log-likelihood panel and -- when
    available -- the trans-dimensional ``nsource`` hyperparameter panel (which
    reveals the sampler's jumps between source counts).

    ``walker_ids`` optionally restricts the plot to a subset of walkers (by
    walker id), so a dashboard checkbox / notebook call can isolate one or a few
    walkers. ``None`` (the default) draws all walkers."""
    backend = backend or PlotlyBackend()
    flagged = set(flagged_walkers or [])
    selected = None if walker_ids is None else set(walker_ids)
    # Stable rank for each flagged walker -> distinct shade on the hot ramp.
    flag_rank = {wid: r for r, wid in enumerate(sorted(flagged))}
    panels: List[TracePanel] = []

    def _shown(walker_id: int) -> bool:
        return selected is None or walker_id in selected

    def _series(walker_id: int, y: np.ndarray) -> LineSeries:
        is_flagged = walker_id in flagged
        color, width, opacity = _walker_style(walker_id, is_flagged, flag_rank.get(walker_id, 0))
        return LineSeries(label=f"walker {walker_id}", x=np.arange(y.size), y=y,
                          color=color, width=width, opacity=opacity)

    def _ordered(walker_ids):
        """Selected walkers only; converged first, flagged last (highest z-order)."""
        return sorted((wid for wid in walker_ids if _shown(wid)),
                      key=lambda wid: (wid in flagged, wid))

    by_id = {w.walker_id: w for w in chain.walkers}
    for name in chain.param_names:
        panel = TracePanel(name=name)
        for wid in _ordered(by_id):
            panel.series.append(_series(wid, np.asarray(by_id[wid].samples[name], dtype=float)))
        panels.append(panel)

    loglik_panel = TracePanel(name="log L")
    for wid in _ordered(by_id):
        loglik_panel.series.append(_series(wid, np.asarray(by_id[wid].log_likelihood, dtype=float)))
    panels.append(loglik_panel)

    if chain.nsource_trace is not None:
        nst = np.atleast_2d(chain.nsource_trace)
        nsrc_panel = TracePanel(name="nsource", is_integer=True)
        for i in _ordered(range(nst.shape[0])):
            nsrc_panel.series.append(_series(i, nst[i].astype(float)))
        panels.append(nsrc_panel)

    band = chain.band.label if chain.band else ""
    src = "" if source_index is None else f" -- source slot {source_index}"
    return backend.trace_panels(panels, burn_in=burn_in, title=f"Trace -- {band}{src}")


def diagnostics_table(chain: MCMCChain, burn_in: int = 0) -> Dict[str, stats.ParamDiagnostics]:
    """ESS / R-hat / MCSE / tau per sampling parameter (arviz-stats)."""
    return stats.diagnostics(chain, burn_in=burn_in)


def plot_autocorrelation(
    chain: MCMCChain,
    max_lag: int = 200,
    backend: Optional[RenderBackend] = None,
):
    """Autocorrelation curve per parameter, annotated with the integrated
    autocorrelation time tau, plus the chain acceptance fraction in the title."""
    backend = backend or PlotlyBackend()
    curves = {name: stats.autocorr_curves(chain, name, max_lag=max_lag) for name in chain.param_names}
    diag = stats.diagnostics(chain)
    tau = {name: diag[name].tau for name in chain.param_names}
    accept = stats.acceptance_fraction(chain)
    band = chain.band.label if chain.band else ""
    return backend.autocorr_panels(curves, tau, title=f"Autocorrelation -- {band}  (accept frac ~ {accept:.2f})")
