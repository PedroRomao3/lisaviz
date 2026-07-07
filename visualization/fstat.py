"""FR-09 -- F-statistic degeneracy contours.

Renders 2-D contour surfaces of the F-statistic grid evaluations
(``fstats.pkl``) to visualize parameter degeneracies *before* committing to a
full MCMC run (US-05: the pipeline developer inspecting the search surface). The
F-statistic is profiled (max over the other intrinsic axes) onto the chosen
parameter pair, and the axes are mapped from sampling (hypercube) space to
physical units for interpretability.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..domain.models import FstatGrid
from ..domain import sampling
from .backends import MatplotlibBackend, RenderBackend
from .encoding import FREQUENCY_CMAP, FREQUENCY_CMAP_PLOTLY


def _default_pair(grid: FstatGrid) -> Tuple[str, str]:
    """Pick the two parameters with the most grid resolution (a fixed axis
    cannot form a contour)."""
    varying = grid.variable_axes()
    if len(varying) < 2:
        raise ValueError(f"F-stat grid has fewer than two varying axes: {varying}")
    counts = {n: np.unique(np.round(grid.grid[:, grid.param_names.index(n)], 9)).size for n in varying}
    ordered = sorted(varying, key=lambda n: counts[n], reverse=True)
    return ordered[0], ordered[1]


def plot_fstat_contour(
    grid: FstatGrid,
    x: Optional[str] = None,
    y: Optional[str] = None,
    bins: int = 60,
    backend: Optional[RenderBackend] = None,
    physical: bool = True,
):
    """Profile the F-statistic onto the (x, y) intrinsic-parameter plane and
    render filled contours. ``x``/``y`` are sampling-axis names
    (``fr``/``fdot``/``dec_sin``/``alpha``); when omitted, the two
    highest-resolution axes are used. Defaults to the Matplotlib backend for a
    publication-quality static figure."""
    backend = backend or MatplotlibBackend()
    if x is None or y is None:
        x, y = _default_pair(grid)

    xc, yc, z = grid.project_pair(x, y, bins=bins)

    # Mark the profiled maximum (the F-stat peak) in the same plane.
    flat_idx = int(np.nanargmax(z))
    yi, xi = np.unravel_index(flat_idx, z.shape)
    peak_hc = (xc[xi], yc[yi])

    if physical and grid.band is not None:
        xname, xc = sampling.hc_axis_to_physical(x, xc, grid.band)
        yname, yc = sampling.hc_axis_to_physical(y, yc, grid.band)
        _, peak_x = sampling.hc_axis_to_physical(x, np.array([peak_hc[0]]), grid.band)
        _, peak_y = sampling.hc_axis_to_physical(y, np.array([peak_hc[1]]), grid.band)
        peak = (float(peak_x[0]), float(peak_y[0]))
        labels = (sampling.physical_label(xname), sampling.physical_label(yname))
    else:
        peak = peak_hc
        labels = (f"{x} [hypercube]", f"{y} [hypercube]")

    cmap = FREQUENCY_CMAP if backend.name == "matplotlib" else FREQUENCY_CMAP_PLOTLY
    band = grid.band.label if grid.band else ""
    return backend.fstat_contour(xc, yc, z, axis_labels=labels, color_label="F-statistic",
                                 title=f"F-stat degeneracy -- {band}", cmap=cmap, peak=peak)
